"""
Baseline Encoder Wrappers for MM-LeJEPA Experiments.

Wraps pre-trained or custom architectures to conform to the MM-LeJEPA
encoder interface, returning (embeddings, projections) tuples.

Encoders:
1. DINOv3FrozenEncoder   – Frozen DINOv3 ViT, train probes only (RGB).
2. DINOv3ScratchEncoder  – DINOv3 architecture trained from scratch (RGB).
3. ImageBindEncoder      – Separate ImageBind encoders per modality,
                           optionally backed by official pretrained weights.
4. MultiMAEEncoder       – Multi-modal Masked Autoencoder with shared decoder.

All encoders return the same interface:
    forward(cam_views, range_views) -> (all_emb, proj)
    where all_emb is (N, embed_dim) and proj is (V, B, proj_dim).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import importlib
from torchvision.ops import MLP
from typing import Tuple, Optional, Dict, List

from src.encoder import (
    get_vit_config,
    VIT_CONFIGS,
    initialize_module_from_dinov3,
    initialize_module_from_timm_vit,
)


def _resize_pos_embed_2d(pos_embed: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Resize a learned 2D positional embedding to a new patch grid when needed."""
    if pos_embed.shape[1] == target_tokens:
        return pos_embed

    base_grid = int(round(math.sqrt(pos_embed.shape[1])))
    target_grid = int(round(math.sqrt(target_tokens)))
    if base_grid * base_grid != pos_embed.shape[1] or target_grid * target_grid != target_tokens:
        raise ValueError(
            f"Expected square patch grids, got base={pos_embed.shape[1]} target={target_tokens}"
        )

    pos = pos_embed.transpose(1, 2).reshape(1, pos_embed.shape[2], base_grid, base_grid)
    pos = F.interpolate(pos, size=(target_grid, target_grid), mode='bicubic', align_corners=False)
    return pos.flatten(2).transpose(1, 2)


class _BackboneInitWrapper(nn.Module):
    """Minimal wrapper exposing a timm-style backbone to the DINOv3 init helper."""

    def __init__(self, backbone: nn.Module, range_patch_embed: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        if range_patch_embed is not None:
            self.range_patch_embed = range_patch_embed


def _initialize_timm_backbone_from_source(
    backbone: nn.Module,
    vit_size: str,
    source: str = 'dinov3',
    range_patch_embed: Optional[nn.Module] = None,
    verbose: bool = True,
):
    """Initialize a timm-style encoder backbone from a supported pretrained source."""
    wrapper = _BackboneInitWrapper(backbone, range_patch_embed=range_patch_embed)
    if source == 'dinov3':
        return initialize_module_from_dinov3(wrapper, vit_size=vit_size, verbose=verbose)
    if source == 'timm':
        return initialize_module_from_timm_vit(wrapper, vit_size=vit_size, verbose=verbose)
    raise ValueError(f"Unsupported pretrained source for timm-style backbone: {source!r}")


def _initialize_multimae_core_from_source(
    core: nn.Module,
    vit_size: str,
    source: str = 'dinov3',
    verbose: bool = True,
):
    """Seed the MultiMAE encoder core from a shared pretrained ViT trunk."""
    if 'rgb' not in core.input_adapters or 'depth' not in core.input_adapters:
        raise ValueError("MultiMAE DINOv3 init requires rgb and depth input adapters")

    backbone_like = nn.Module()
    backbone_like.blocks = core.encoder_blocks
    backbone_like.norm = core.encoder_norm
    backbone_like.patch_embed = nn.Module()
    backbone_like.patch_embed.proj = core.input_adapters['rgb'].patch_embed
    backbone_like.cls_token = core.global_tokens

    wrapper = _BackboneInitWrapper(
        backbone_like,
        range_patch_embed=core.input_adapters['depth'].patch_embed,
    )
    if source == 'dinov3':
        result = initialize_module_from_dinov3(wrapper, vit_size=vit_size, verbose=False)
    elif source == 'timm':
        result = initialize_module_from_timm_vit(wrapper, vit_size=vit_size, verbose=False)
    else:
        raise ValueError(f"Unsupported pretrained source for MultiMAE core: {source!r}")
    if verbose:
        print(
            f"✅ Initialized {core.__class__.__name__} RGB/depth encoder core from {source.upper()} "
            f"({result['source_name']})"
        )
    return result


# ====================================================================
# 1. DINOv3 Encoder
# ====================================================================

class DINOv3FrozenEncoder(nn.Module):
    """
    Frozen DINOv3 ViT encoder – only probes are trained.

    Uses DINOv3 ViT-S/16 via timm (patch_size=16, 4 register tokens).
    Falls back to a randomly-initialised timm ViT if pretrained weights
    are unavailable.

    The encoder is fully frozen; only the projection MLP is trainable
    (and downstream probes, handled outside this class).

    DINOv3 vs DINOv2:
      - patch_size=16 (not 14) → 14×14=196 patches for 224×224 input
      - 4 register tokens + 1 CLS = 5 prefix tokens
      - No need for patch interpolation or input resizing

    Args:
        proj_dim:  Projection output dimension for SigReg / loss.
        img_size:  Input image size.
        vit_size:  'small' | 'base' | 'large' – selects DINOv3 variant.
    """

    # DINOv3 timm model names (patch-16 family)
    _DINO_MODELS = {
        'small': 'vit_small_patch16_dinov3',
        'base':  'vit_base_patch16_dinov3',
        'large': 'vit_large_patch16_dinov3',
    }
    # DINOv3 hidden dims (same as DINOv2)
    _DINO_DIMS = {
        'small': 384,
        'base':  768,
        'large': 1024,
    }
    # Number of prefix tokens: 1 CLS + 4 register tokens
    _NUM_PREFIX = 5

    def __init__(
        self,
        proj_dim: int = 128,
        img_size: int = 224,
        vit_size: str = 'small',
        pretrained: bool = True,
        freeze_backbone: bool = True,
        allow_random_fallback: bool = True,
    ):
        super().__init__()
        self.vit_size = vit_size
        self.freeze_backbone = freeze_backbone
        self.input_size = img_size
        vit_cfg = get_vit_config(vit_size)
        self.embed_dim = vit_cfg['embed_dim']
        dino_dim = self._DINO_DIMS.get(vit_size, 384)

        # Try loading DINOv3 from timm (pretrained weights available from v1.0.20)
        timm_name = self._DINO_MODELS[vit_size]
        try:
            self.backbone = timm.create_model(
                timm_name, pretrained=pretrained, num_classes=0, img_size=img_size,
            )
            if pretrained:
                print(f"✅ Loaded DINOv3 pre-trained: {timm_name}")
            else:
                print(f"🧱 Initialized DINOv3 from scratch: {timm_name}")
        except Exception as e:
            if pretrained and allow_random_fallback:
                print(f"⚠️  DINOv3 pretrained load failed ({e}), using random init")
                self.backbone = timm.create_model(
                    timm_name, pretrained=False, num_classes=0, img_size=img_size,
                )
            else:
                raise
        dino_dim = self.backbone.embed_dim
        self.num_prefix = getattr(self.backbone, 'num_prefix_tokens', self._NUM_PREFIX)
        patch_size = getattr(getattr(self.backbone, 'patch_embed', None), 'patch_size', 16)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.dino_dim = dino_dim
        # DINOv3 uses patch_size=16 → 14×14=196 patches for 224×224 (same as standard ViT)
        self.n_patches = (img_size // 16) ** 2  # 196 for 224

        # Projection head (trainable)
        self.head = nn.Linear(dino_dim, self.embed_dim)
        self.proj = MLP(self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        self.pre_head_proj = MLP(dino_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def train(self, mode: bool = True):
        """Keep backbone frozen even when .train() is called."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _maybe_resize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve valid multi-crop resolutions; only resize incompatible inputs."""
        height, width = x.shape[-2:]
        if (height, width) == (self.input_size, self.input_size):
            return x
        if height % self.patch_size == 0 and width % self.patch_size == 0:
            return x
        return F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode='bicubic',
            align_corners=False,
        )

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode (N, C, H, W) images through frozen DINOv3.

        Returns:
            cls_emb: (N, dino_dim)
            patch_tokens: (N, n_patches, dino_dim)
        """
        x = self._maybe_resize_input(x)
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone.forward_features(x)
        else:
            out = self.backbone.forward_features(x)

        # CLS is always the first token
        cls_token = out[:, 0]
        # Patch tokens follow the prefix tokens (CLS + 4 registers)
        patch_tokens = out[:, self.num_prefix:]

        return cls_token, patch_tokens

    def forward(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass — processes RGB only (DINOv3 is a unimodal baseline).
        range_views is accepted but ignored.

        Args:
            cam_views: dict{'global': (B, V, 3, H, W), 'local': ...} or (B, V, 3, H, W)
        """
        if isinstance(cam_views, dict):
            c_global = cam_views['global']
            B, V_g, C, H, W = c_global.shape
            flat = c_global.flatten(0, 1)
            cls_g, patches_g = self._encode(flat)
            cls_emb_g = cls_g if getattr(self, 'use_pre_head_features', False) else self.head(cls_g)
            proj_head = self.pre_head_proj if getattr(self, 'use_pre_head_features', False) else self.proj
            proj_g = proj_head(cls_emb_g).reshape(B, V_g, -1).transpose(0, 1)

            c_local = cam_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                flat_l = c_local.flatten(0, 1)
                cls_l, _ = self._encode(flat_l)
                cls_emb_l = cls_l if getattr(self, 'use_pre_head_features', False) else self.head(cls_l)
                V_l = c_local.shape[1]
                proj_l = proj_head(cls_emb_l).reshape(B, V_l, -1).transpose(0, 1)

                cls_emb = torch.cat([cls_emb_g, cls_emb_l], dim=0)
                proj = torch.cat([proj_g, proj_l], dim=0)
            else:
                cls_emb = cls_emb_g
                proj = proj_g
        else:
            B, V, C, H, W = cam_views.shape
            flat = cam_views.flatten(0, 1)
            cls_tok, patches = self._encode(flat)
            cls_emb = cls_tok if getattr(self, 'use_pre_head_features', False) else self.head(cls_tok)
            proj_head = self.pre_head_proj if getattr(self, 'use_pre_head_features', False) else self.proj
            proj = proj_head(cls_emb).reshape(B, V, -1).transpose(0, 1)

        # Duplicate to mimic [cam; lidar] output shape expected by probes
        all_emb = torch.cat([cls_emb, cls_emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        return all_emb, proj_dup

    def forward_with_patches(self, cam_views, range_views=None):
        """Return patch tokens for patch-based probes."""
        if isinstance(cam_views, dict):
            c_global = cam_views['global']
        else:
            c_global = cam_views
        B, V, C, H, W = c_global.shape
        flat = c_global.flatten(0, 1)
        cls_tok, patch_tokens = self._encode(flat)
        cls_emb = cls_tok if getattr(self, 'use_pre_head_features', False) else self.head(cls_tok)
        proj_head = self.pre_head_proj if getattr(self, 'use_pre_head_features', False) else self.proj
        proj = proj_head(cls_emb).reshape(B, V, -1).transpose(0, 1)
        all_emb = torch.cat([cls_emb, cls_emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        # patch_tokens: (B*V, n_patches, dino_dim) — return as (cam, lid) pair
        if patch_tokens is None:
            patch_tokens = torch.zeros(B * V, self.n_patches, self.dino_dim, device=c_global.device)
        return all_emb, proj_dup, (patch_tokens, patch_tokens)


# Keep backward-compatible alias
DINOv2FrozenEncoder = DINOv3FrozenEncoder


class DINOv3ScratchEncoder(DINOv3FrozenEncoder):
    """Trainable DINOv3 architecture baseline initialized from scratch.

    Includes learnable mask token and ``forward_masked_ibot`` for true iBOT
    masked-student training (DINOv2-style): the *student* sees mask tokens
    at randomly chosen patch positions while the *teacher* always processes
    the full, unmasked input.
    """

    def __init__(self, proj_dim: int = 128, img_size: int = 224, vit_size: str = 'small'):
        super().__init__(
            proj_dim=proj_dim,
            img_size=img_size,
            vit_size=vit_size,
            pretrained=False,
            freeze_backbone=False,
            allow_random_fallback=False,
        )
        # Learnable mask token used for iBOT masked-student forward pass
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dino_dim))
        self._init_weights_dinov3()

    def _init_weights_dinov3(self):
        """Apply the official DINOv3 scratch initialization sequence when available."""

        if hasattr(self.backbone, 'init_weights') and callable(self.backbone.init_weights):
            self.backbone.init_weights()
        else:
            def _init_vit_weights(module):
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    return
                if isinstance(module, nn.LayerNorm):
                    module.reset_parameters()
                    return
                module_name = module.__class__.__name__
                if module_name in {'LayerScale', 'PatchEmbed', 'RMSNorm'} and hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

            rope_module = getattr(self.backbone, 'rope_embed', None)
            if rope_module is None:
                rope_module = getattr(self.backbone, 'rope', None)
            if rope_module is not None and hasattr(rope_module, '_init_weights'):
                rope_module._init_weights()
            self.backbone.apply(_init_vit_weights)

        if hasattr(self.backbone, 'cls_token') and self.backbone.cls_token is not None:
            nn.init.normal_(self.backbone.cls_token, std=0.02)
        if hasattr(self.backbone, 'storage_tokens') and self.backbone.storage_tokens is not None:
            nn.init.normal_(self.backbone.storage_tokens, std=0.02)
        if hasattr(self.backbone, 'reg_token') and self.backbone.reg_token is not None:
            nn.init.normal_(self.backbone.reg_token, std=0.02)
        if hasattr(self.backbone, 'register_tokens') and self.backbone.register_tokens is not None:
            nn.init.normal_(self.backbone.register_tokens, std=0.02)
        if hasattr(self.backbone, 'mask_token') and self.backbone.mask_token is not None:
            nn.init.zeros_(self.backbone.mask_token)
        nn.init.zeros_(self.mask_token)

    def forward_masked_ibot(
        self,
        cam_views: torch.Tensor,
        mask_ratio: float = 0.3,
    ):
        """Student forward with true iBOT masking.

        Replaces randomly selected patch positions with ``self.mask_token``
        *before* the transformer blocks – matching the official DINOv2 recipe
        where the student sees a partially-masked sequence and the teacher
        sees the full, unmasked input.

        Follows the exact DINOv3 forward_features pipeline:
          patch_embed → _pos_embed (CLS/reg/RoPE) → [mask] → norm_pre → blocks → norm

        Args:
            cam_views: (B, V, 3, H, W) global views (typically V=2).
            mask_ratio: fraction of patch positions to mask.

        Returns:
            cls_token:      (N, dino_dim) – CLS output per image.
            patch_tokens:   (N, n_patches, dino_dim) – all patch outputs.
            bool_mask:      (N, n_patches) – True at masked positions.
        """
        if isinstance(cam_views, dict):
            cam_views = cam_views['global']
        B, V, C, H, W = cam_views.shape
        flat = cam_views.flatten(0, 1)          # (N, 3, H, W)
        flat = self._maybe_resize_input(flat)
        N = flat.shape[0]

        # --- patch embed (4D for DINOv3) ---
        x = self.backbone.patch_embed(flat)

        # --- _pos_embed: flattens 4D, prepends CLS+reg, applies pos_drop,
        #     returns rotary pos embed for blocks ---
        x, rot_pos_embed = self.backbone._pos_embed(x)

        # --- mask patch positions (after prefix tokens) ---
        n_patches = x.shape[1] - self.num_prefix
        n_mask = max(1, int(round(mask_ratio * n_patches)))
        noise = torch.rand(N, n_patches, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        bool_mask = torch.zeros(N, n_patches, dtype=torch.bool, device=x.device)
        bool_mask.scatter_(1, ids_shuffle[:, :n_mask], True)

        prefix = x[:, :self.num_prefix]
        patches = x[:, self.num_prefix:]
        mask_tokens = self.mask_token.expand(N, n_patches, -1)
        patches = torch.where(bool_mask.unsqueeze(-1), mask_tokens, patches)
        x = torch.cat([prefix, patches], dim=1)

        # --- norm_pre → blocks (with rope) → norm ---
        x = self.backbone.norm_pre(x)
        for blk in self.backbone.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.backbone.norm(x)

        cls_out = x[:, 0]
        patch_out = x[:, self.num_prefix:]      # skip CLS + registers
        return cls_out, patch_out, bool_mask


# ====================================================================
# 2. ImageBind-style Encoder
# ====================================================================

class ImageBindEncoder(nn.Module):
    """
        ImageBind-style baseline for RGB-depth contrastive training.

        For paper-facing baselines, this class now prefers the upstream
        ``ImageBindModel`` pipeline for both modes:
        - ``use_official_weights=True``: official ImageBind Huge with pretrained
            weights or a provided checkpoint.
        - ``use_official_weights=False``: the same upstream ImageBind architecture
            family, but instantiated from scratch with smaller small/base/large
            transformer dimensions.

        A timm-based dual-ViT fallback is kept only as an explicit compatibility
        path when the upstream ImageBind package is unavailable and fallback is
        allowed.

    Args:
        proj_dim:   Projection output dimension.
        img_size:   Input image resolution.
        vit_size:   'small' | 'base' | 'large'.
    """

    _SCRATCH_CONFIGS = {
        'small': {
            'vision_embed_dim': 384,
            'vision_num_blocks': 12,
            'vision_num_heads': 6,
            'depth_embed_dim': 384,
            'depth_num_blocks': 12,
            'depth_num_heads': 6,
            'out_embed_dim': 512,
        },
        'base': {
            'vision_embed_dim': 768,
            'vision_num_blocks': 12,
            'vision_num_heads': 12,
            'depth_embed_dim': 768,
            'depth_num_blocks': 12,
            'depth_num_heads': 12,
            'out_embed_dim': 768,
        },
        'large': {
            'vision_embed_dim': 1024,
            'vision_num_blocks': 24,
            'vision_num_heads': 16,
            'depth_embed_dim': 1024,
            'depth_num_blocks': 24,
            'depth_num_heads': 16,
            'out_embed_dim': 1024,
        },
    }

    def __init__(
        self,
        proj_dim: int = 128,
        img_size: int = 224,
        vit_size: str = 'small',
        use_official_weights: bool = True,
        allow_timm_fallback: bool = True,
        imagebind_ckpt_path: Optional[str] = None,
        force_timm_backbone: bool = False,
    ):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        self.img_size = img_size
        self.embed_dim = vit_cfg['embed_dim']
        self.vit_dim = vit_cfg['vit_dim']
        self.n_patches = (img_size // 16) ** 2
        self.use_official_weights = False
        self._uses_imagebind_model = False
        self.requested_vit_size = vit_size
        self.effective_backbone_name = vit_cfg['model_name']
        self.force_timm_backbone = bool(force_timm_backbone)

        if not self.force_timm_backbone:
            try:
                imagebind_model = importlib.import_module("imagebind.models.imagebind_model")
                self._ib_modality = imagebind_model.ModalityType

                if use_official_weights:
                    self._ib_model = imagebind_model.imagebind_huge(pretrained=(imagebind_ckpt_path is None))
                    self.effective_backbone_name = 'imagebind_huge'
                    if imagebind_ckpt_path is not None:
                        state = torch.load(imagebind_ckpt_path, map_location='cpu', weights_only=True)
                        self._ib_model.load_state_dict(state)
                    self.use_official_weights = True
                    print(
                        "✅ Using official ImageBind Huge vision+depth weights "
                        f"(requested vit_size={vit_size} is ignored by the official path)"
                    )
                else:
                    self._ib_model = self._build_scratch_imagebind_model(imagebind_model, vit_size)
                    self.effective_backbone_name = f'imagebind_{vit_size}_scratch'
                    print(
                        "🧱 Using upstream ImageBind architecture in scratch mode "
                        f"(vit_size={vit_size}, no pretrained weights)"
                    )

                self._configure_imagebind_adapters(proj_dim)
                self._uses_imagebind_model = True
                return
            except Exception as e:
                if not allow_timm_fallback:
                    raise ImportError(
                        "ImageBind architecture requested, but the ImageBind package/checkpoint is unavailable"
                    ) from e
                print(f"⚠️  Upstream ImageBind unavailable ({e}); falling back to timm approximation")
        else:
            print("🧱 Forcing timm-compatible ImageBind fallback backbone")

        fallback_pretrained = bool(use_official_weights) and (not self.force_timm_backbone)

        # RGB encoder fallback.
        self.rgb_encoder = timm.create_model(
            vit_cfg['model_name'],
            pretrained=fallback_pretrained,
            num_classes=self.embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            dynamic_img_size=True,
        )

        # Depth encoder fallback.
        self.depth_encoder = timm.create_model(
            vit_cfg['model_name'],
            pretrained=fallback_pretrained,
            num_classes=self.embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            in_chans=1,
            dynamic_img_size=True,
        )

        # Separate projection heads
        self.proj_rgb = MLP(self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        self.proj_depth = MLP(self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

        # Joint projection for concatenated CLS (optional, for SigReg)
        self.proj_joint = MLP(self.embed_dim * 2, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

        if fallback_pretrained:
            print("⚠️  Using timm fallback with pretrained ViT backbones")
        else:
            print("⚠️  Using timm fallback with scratch ViT backbones")

    def initialize_from_pretrained(self, source: str = 'dinov3', vit_size: Optional[str] = None, verbose: bool = True):
        """Seed the timm-compatible ImageBind fallback from a shared pretrained ViT source."""
        if self._uses_imagebind_model:
            raise ValueError("Pretrained init is only supported for the timm-compatible ImageBind fallback")

        init_vit_size = vit_size or self.requested_vit_size
        rgb_result = _initialize_timm_backbone_from_source(
            self.rgb_encoder,
            vit_size=init_vit_size,
            source=source,
            verbose=False,
        )
        _initialize_timm_backbone_from_source(
            self.depth_encoder,
            vit_size=init_vit_size,
            source=source,
            verbose=False,
        )
        if verbose:
            print(
                f"✅ Initialized ImageBind timm fallback from {source.upper()} ({rgb_result['source_name']})"
            )
        return rgb_result

    def initialize_from_dinov3(self, vit_size: Optional[str] = None, verbose: bool = True):
        return self.initialize_from_pretrained(source='dinov3', vit_size=vit_size, verbose=verbose)

    def _build_scratch_imagebind_model(self, imagebind_model, vit_size: str):
        if vit_size not in self._SCRATCH_CONFIGS:
            raise ValueError(f"Unsupported ImageBind scratch vit_size={vit_size!r}")

        scratch_cfg = self._SCRATCH_CONFIGS[vit_size]
        return imagebind_model.ImageBindModel(
            out_embed_dim=scratch_cfg['out_embed_dim'],
            vision_embed_dim=scratch_cfg['vision_embed_dim'],
            vision_num_blocks=scratch_cfg['vision_num_blocks'],
            vision_num_heads=scratch_cfg['vision_num_heads'],
            depth_embed_dim=scratch_cfg['depth_embed_dim'],
            depth_num_blocks=scratch_cfg['depth_num_blocks'],
            depth_num_heads=scratch_cfg['depth_num_heads'],
        )

    def _configure_imagebind_adapters(self, proj_dim: int):
        vision_head = self._ib_model.modality_heads[self._ib_modality.VISION]
        depth_head = self._ib_model.modality_heads[self._ib_modality.DEPTH]
        vision_shape = vision_head[0].normalized_shape
        depth_shape = depth_head[0].normalized_shape
        self._ib_vision_dim = int(vision_shape[0] if isinstance(vision_shape, (tuple, list)) else vision_shape)
        self._ib_depth_dim = int(depth_shape[0] if isinstance(depth_shape, (tuple, list)) else depth_shape)
        self._ib_out_dim = int(vision_head[-1].out_features)

        self.rgb_embed_adapter = (
            nn.Identity() if self._ib_out_dim == self.embed_dim else nn.Linear(self._ib_out_dim, self.embed_dim)
        )
        self.depth_embed_adapter = (
            nn.Identity() if self._ib_out_dim == self.embed_dim else nn.Linear(self._ib_out_dim, self.embed_dim)
        )
        self.rgb_patch_adapter = (
            nn.Identity() if self._ib_vision_dim == self.vit_dim else nn.Linear(self._ib_vision_dim, self.vit_dim)
        )
        self.depth_patch_adapter = (
            nn.Identity() if self._ib_depth_dim == self.vit_dim else nn.Linear(self._ib_depth_dim, self.vit_dim)
        )
        self.proj_joint = MLP(self.embed_dim * 2, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def _maybe_resize_views(self, views: torch.Tensor) -> torch.Tensor:
        """Resize views to the configured encoder resolution when needed."""
        if views.shape[-2:] == (self.img_size, self.img_size):
            return views
        b, v, c, _, _ = views.shape
        flat = views.flatten(0, 1)
        flat = F.interpolate(
            flat,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False,
        )
        return flat.reshape(b, v, c, self.img_size, self.img_size)

    def _resize_patch_tokens(self, patch_tokens: torch.Tensor, target_tokens: Optional[int] = None) -> torch.Tensor:
        """Resize square patch-token grids to the target token count used by probes."""
        if patch_tokens is None:
            return patch_tokens
        target_tokens = self.n_patches if target_tokens is None else target_tokens
        if patch_tokens.shape[1] == target_tokens:
            return patch_tokens

        src_tokens = patch_tokens.shape[1]
        src_grid = int(src_tokens ** 0.5)
        target_grid = int(target_tokens ** 0.5)
        if src_grid * src_grid != src_tokens or target_grid * target_grid != target_tokens:
            return patch_tokens[:, :target_tokens]

        grid = patch_tokens.reshape(patch_tokens.shape[0], src_grid, src_grid, patch_tokens.shape[-1]).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(target_grid, target_grid), mode='bicubic', align_corners=False)
        return grid.permute(0, 2, 3, 1).reshape(patch_tokens.shape[0], target_tokens, patch_tokens.shape[-1])

    def _encode_imagebind_modality(self, modality_key, input_name: str, views: torch.Tensor):
        """Encode one modality with the upstream ImageBind pipeline."""
        pre = self._ib_model.modality_preprocessors[modality_key](**{input_name: views})
        trunk_out = self._ib_model.modality_trunks[modality_key](**pre['trunk'])
        head_out = self._ib_model.modality_heads[modality_key](trunk_out, **pre['head'])
        emb = self._ib_model.modality_postprocessors[modality_key](head_out)
        cls = trunk_out[:, 0]
        patches = trunk_out[:, 1:]
        return emb, cls, patches

    def _encode_modality(self, encoder, views):
        """Encode (B, V, C, H, W) through a ViT encoder.

        Returns:
            cls_emb: (B*V, embed_dim)
            patch_tokens: (B*V, n_patches, vit_dim)
        """
        B, V, C, H, W = views.shape
        flat = views.flatten(0, 1)
        # timm ViT forward_features
        feats = encoder.forward_features(flat)
        if isinstance(feats, torch.Tensor):
            if feats.dim() == 3:
                cls_tok = feats[:, 0]
                patch_tokens = feats[:, 1:]
            else:
                cls_tok = feats
                patch_tokens = None
        else:
            cls_tok = feats
            patch_tokens = None
        # Pass through classification head
        cls_emb = encoder.forward_head(cls_tok.unsqueeze(1) if cls_tok.dim() == 2 and hasattr(encoder, 'forward_head') else cls_tok)
        if cls_emb.dim() == 3:
            cls_emb = cls_emb[:, 0]
        # Actually, simpler: just use full forward for CLS
        # and forward_features for patches
        return cls_emb, patch_tokens

    def forward(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._uses_imagebind_model:
            return self._forward_imagebind_model(cam_views, range_views)

        if isinstance(cam_views, dict):
            c_global = cam_views['global']
            r_global = range_views['global']
        else:
            c_global = cam_views
            r_global = range_views

        B, V, C_cam, H, W = c_global.shape

        # RGB forward
        rgb_flat = c_global.flatten(0, 1)
        rgb_cls = self.rgb_encoder(rgb_flat)  # (B*V, embed_dim)

        # Depth forward
        depth_flat = r_global.flatten(0, 1)
        depth_cls = self.depth_encoder(depth_flat)  # (B*V, embed_dim)

        # Concatenated CLS for probes
        joint_cls = torch.cat([rgb_cls, depth_cls], dim=-1)  # (B*V, 2*embed_dim)

        # Per-modality projections (for cross-modal contrastive loss)
        proj_rgb = self.proj_rgb(rgb_cls).reshape(B, V, -1).transpose(0, 1)  # (V, B, proj_dim)
        proj_depth = self.proj_depth(depth_cls).reshape(B, V, -1).transpose(0, 1)
        # Store for contrastive loss computation in training loop
        self._last_proj_rgb = proj_rgb
        self._last_proj_depth = proj_depth

        # Joint projection (used as the main proj output)
        proj_joint = self.proj_joint(joint_cls).reshape(B, V, -1).transpose(0, 1)

        # Handle local views
        c_local = cam_views.get('local') if isinstance(cam_views, dict) else None
        r_local = range_views.get('local') if isinstance(range_views, dict) else None
        if c_local is not None and c_local.numel() > 0:
            V_l = c_local.shape[1]
            rgb_cls_l = self.rgb_encoder(c_local.flatten(0, 1))
            depth_cls_l = self.depth_encoder(r_local.flatten(0, 1))
            joint_cls_l = torch.cat([rgb_cls_l, depth_cls_l], dim=-1)
            proj_joint_l = self.proj_joint(joint_cls_l).reshape(B, V_l, -1).transpose(0, 1)
            joint_cls = torch.cat([joint_cls, joint_cls_l], dim=0)
            proj_joint = torch.cat([proj_joint, proj_joint_l], dim=0)

        # Return interface: (all_emb, proj)
        # all_emb should be (N, embed_dim) for probes — use rgb_cls for compatibility
        # Store joint_cls (2*embed_dim) separately if needed
        all_emb = torch.cat([rgb_cls, depth_cls], dim=0)  # (2*B*V, embed_dim) — [cam; lid]
        return all_emb, proj_joint

    def _forward_imagebind_model(self, cam_views, range_views):
        if isinstance(cam_views, dict):
            c_global = cam_views['global']
            r_global = range_views['global']
        else:
            c_global = cam_views
            r_global = range_views

        c_global = self._maybe_resize_views(c_global)
        r_global = self._maybe_resize_views(r_global)

        B, V, _, _, _ = c_global.shape
        rgb_flat = c_global.flatten(0, 1)
        depth_flat = r_global.flatten(0, 1)

        rgb_emb_off, _, _ = self._encode_imagebind_modality(self._ib_modality.VISION, 'vision', rgb_flat)
        depth_emb_off, _, _ = self._encode_imagebind_modality(self._ib_modality.DEPTH, 'depth', depth_flat)

        rgb_cls = self.rgb_embed_adapter(rgb_emb_off)
        depth_cls = self.depth_embed_adapter(depth_emb_off)
        joint_cls = torch.cat([rgb_cls, depth_cls], dim=-1)

        self._last_proj_rgb = rgb_cls.reshape(B, V, -1).transpose(0, 1)
        self._last_proj_depth = depth_cls.reshape(B, V, -1).transpose(0, 1)
        proj_joint = self.proj_joint(joint_cls).reshape(B, V, -1).transpose(0, 1)

        c_local = cam_views.get('local') if isinstance(cam_views, dict) else None
        r_local = range_views.get('local') if isinstance(range_views, dict) else None
        if c_local is not None and c_local.numel() > 0:
            c_local = self._maybe_resize_views(c_local)
            r_local = self._maybe_resize_views(r_local)
            V_l = c_local.shape[1]
            rgb_l_off, _, _ = self._encode_imagebind_modality(
                self._ib_modality.VISION, 'vision', c_local.flatten(0, 1)
            )
            depth_l_off, _, _ = self._encode_imagebind_modality(
                self._ib_modality.DEPTH, 'depth', r_local.flatten(0, 1)
            )
            rgb_l = self.rgb_embed_adapter(rgb_l_off)
            depth_l = self.depth_embed_adapter(depth_l_off)
            joint_l = torch.cat([rgb_l, depth_l], dim=-1)
            proj_joint_l = self.proj_joint(joint_l).reshape(B, V_l, -1).transpose(0, 1)
            proj_joint = torch.cat([proj_joint, proj_joint_l], dim=0)

        all_emb = torch.cat([rgb_cls, depth_cls], dim=0)
        return all_emb, proj_joint

    def forward_with_patches(self, cam_views, range_views):
        """Return concatenated patch tokens for patch probes."""
        if self._uses_imagebind_model:
            return self._forward_with_patches_imagebind_model(cam_views, range_views)

        if isinstance(cam_views, dict):
            c_global = cam_views['global']
            r_global = range_views['global']
        else:
            c_global = cam_views
            r_global = range_views

        B, V, C_cam, H, W = c_global.shape

        # Get patch tokens
        rgb_feats = self.rgb_encoder.forward_features(c_global.flatten(0, 1))
        depth_feats = self.depth_encoder.forward_features(r_global.flatten(0, 1))

        if isinstance(rgb_feats, torch.Tensor) and rgb_feats.dim() == 3:
            rgb_patches = rgb_feats[:, 1:]  # Skip CLS
            rgb_cls = rgb_feats[:, 0]
        else:
            rgb_patches = torch.zeros(B * V, 196, self.vit_dim, device=c_global.device)
            rgb_cls = rgb_feats if isinstance(rgb_feats, torch.Tensor) else rgb_feats[:, 0]

        if isinstance(depth_feats, torch.Tensor) and depth_feats.dim() == 3:
            depth_patches = depth_feats[:, 1:]
        else:
            depth_patches = torch.zeros(B * V, 196, self.vit_dim, device=r_global.device)

        rgb_cls_emb = self.rgb_encoder(c_global.flatten(0, 1))
        depth_cls_emb = self.depth_encoder(r_global.flatten(0, 1))
        joint_cls = torch.cat([rgb_cls_emb, depth_cls_emb], dim=-1)
        proj = self.proj_joint(joint_cls).reshape(B, V, -1).transpose(0, 1)

        all_emb = torch.cat([rgb_cls_emb, depth_cls_emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)

        # Concatenate patch tokens along feature dim
        # concat_patches: (B*V, n_patches, 2*vit_dim)
        concat_patches = torch.cat([rgb_patches, depth_patches], dim=-1)

        return all_emb, proj_dup, (rgb_patches, depth_patches)

    def _forward_with_patches_imagebind_model(self, cam_views, range_views):
        if isinstance(cam_views, dict):
            c_global = cam_views['global']
            r_global = range_views['global']
        else:
            c_global = cam_views
            r_global = range_views

        c_global = self._maybe_resize_views(c_global)
        r_global = self._maybe_resize_views(r_global)

        B, V, _, _, _ = c_global.shape
        rgb_flat = c_global.flatten(0, 1)
        depth_flat = r_global.flatten(0, 1)

        rgb_emb_off, _, rgb_patches_off = self._encode_imagebind_modality(self._ib_modality.VISION, 'vision', rgb_flat)
        depth_emb_off, _, depth_patches_off = self._encode_imagebind_modality(self._ib_modality.DEPTH, 'depth', depth_flat)

        rgb_cls = self.rgb_embed_adapter(rgb_emb_off)
        depth_cls = self.depth_embed_adapter(depth_emb_off)
        joint_cls = torch.cat([rgb_cls, depth_cls], dim=-1)
        proj = self.proj_joint(joint_cls).reshape(B, V, -1).transpose(0, 1)

        rgb_patches = self.rgb_patch_adapter(rgb_patches_off)
        depth_patches = self.depth_patch_adapter(depth_patches_off)
        rgb_patches = self._resize_patch_tokens(rgb_patches)
        depth_patches = self._resize_patch_tokens(depth_patches)

        all_emb = torch.cat([rgb_cls, depth_cls], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        return all_emb, proj_dup, (rgb_patches, depth_patches)

    def get_joint_embed_dim(self):
        """Return the concatenated embedding dimension for probe creation."""
        return self.embed_dim * 2


# ====================================================================
# 3. Multi-Modal Masked Autoencoder (MultiMAE)
# ====================================================================

class _CrossAttentionDecoderLayer(nn.Module):
    """Decoder layer with cross-attention from queries to encoder memory."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x = self.norm1(tgt)
        x = tgt + self.dropout(self.self_attn(x, x, x, need_weights=False)[0])
        # Cross-attention to encoder output
        x2 = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x2, memory, memory, need_weights=False)[0])
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


class MultiMAEEncoder(nn.Module):
    """
    Multi-Modal Masked Autoencoder (MultiMAE-style).

    Improvements over the original implementation, aligning with the
    official MultiMAE recipe (EPFL-VILAB/MultiMAE):

      - **Cross-attention decoder**: decoder queries attend to visible
        encoder outputs via cross-attention, not just self-attention.
      - **Joint token-budget masking**: a single budget of visible tokens
        is allocated across *both* modalities, so masking one modality
        heavily forces the model to reconstruct from the other.
      - **Learnable task weighting**: per-task log-variance parameters
        (uncertainty weighting) balance RGB vs depth losses automatically.
      - **Separate decoder position embeddings** for decoder tokens.

    During training:
      1. Patch-embed RGB and depth separately.
      2. Jointly mask tokens with a shared budget across modalities.
      3. Feed visible tokens through a shared ViT encoder.
      4. Decode ALL tokens (including masked) through a cross-attention decoder.
      5. Reconstruction loss on masked tokens with learnable task balancing.
      6. CLS token used for probes / downstream.

    Args:
        proj_dim:     Projection output dimension.
        img_size:     Input image resolution.
        vit_size:     'small' | 'base' | 'large'.
        mask_ratio:   Fraction of total tokens to mask (default 0.75).
        decoder_depth: Number of decoder transformer layers (default 2).
        decoder_dim:  Decoder hidden dim (default 256).
    """

    def __init__(
        self,
        proj_dim: int = 128,
        img_size: int = 224,
        vit_size: str = 'small',
        mask_ratio: float = 0.75,
        decoder_depth: int = 2,
        decoder_dim: int = 256,
        depth_channels: int = 1,
    ):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        self.embed_dim = vit_cfg['embed_dim']
        self.vit_dim = vit_cfg['vit_dim']
        self.mask_ratio = mask_ratio
        self.depth_channels = depth_channels
        patch_size = 16
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 196

        # Shared ViT encoder
        self.backbone = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.1,
            img_size=img_size,
            dynamic_img_size=True,
        )
        self.vit_embed_dim = self.backbone.embed_dim

        # Separate patch embeddings
        self.cam_patch_embed = self.backbone.patch_embed
        self.depth_patch_embed = nn.Conv2d(
            depth_channels, self.vit_embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # Modality embeddings
        self.cam_modality_embed = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        self.depth_modality_embed = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        nn.init.normal_(self.cam_modality_embed, std=0.02)
        nn.init.normal_(self.depth_modality_embed, std=0.02)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embeddings for encoder (1 CLS + 2 * n_patches)
        total_tokens = 1 + 2 * self.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, self.vit_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Mask tokens (learnable placeholder for masked positions)
        self.mask_token_cam = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.mask_token_depth = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token_cam, std=0.02)
        nn.init.trunc_normal_(self.mask_token_depth, std=0.02)

        # Cross-attention decoder
        self.decoder_embed = nn.Linear(self.vit_embed_dim, decoder_dim)
        self.decoder_layers = nn.ModuleList([
            _CrossAttentionDecoderLayer(
                d_model=decoder_dim,
                nhead=4,
                dim_feedforward=decoder_dim * 4,
                dropout=0.1,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        # Separate positional embeddings for decoder queries
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 2 * self.n_patches, decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Reconstruction heads (predict raw pixel patches)
        self.cam_recon_head = nn.Linear(decoder_dim, 3 * patch_size * patch_size)
        self.depth_recon_head = nn.Linear(decoder_dim, depth_channels * patch_size * patch_size)

        # Learnable task weights (uncertainty weighting: Kendall et al.)
        self.log_sigma_cam = nn.Parameter(torch.zeros(1))
        self.log_sigma_depth = nn.Parameter(torch.zeros(1))

        # CLS → embedding + projection
        self.head = nn.Linear(self.vit_embed_dim, self.embed_dim)
        self.proj = MLP(self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def _generate_joint_masks(self, N: int, n_patches: int, device: torch.device):
        """Sample a shared visible-token budget across RGB and depth.

        This follows the official MultiMAE pattern more closely than the
        previous implementation: sample a fixed-size visible subset from the
        concatenated multimodal token pool, encode only those tokens, then
        unshuffle back into full-length decoder queries.
        """
        total = 2 * n_patches
        n_keep = max(2, int(total * (1 - self.mask_ratio)))
        noise = torch.rand(N, total, device=device)
        ids_shuffle = noise.argsort(dim=1)
        ids_keep = ids_shuffle[:, :n_keep]

        mask_all = torch.ones(N, total, dtype=torch.bool, device=device)
        mask_all.scatter_(1, ids_keep, False)

        return ids_keep, mask_all

    def _patchify(self, imgs: torch.Tensor, patch_size: int = 16):
        """Convert (N, C, H, W) to (N, n_patches, C*p*p)."""
        N, C, H, W = imgs.shape
        h = H // patch_size
        w = W // patch_size
        x = imgs.reshape(N, C, h, patch_size, w, patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(N, h * w, -1)
        return x

    def _forward_batch(self, cam_views, depth_views):
        """Forward with masking and reconstruction."""
        B, V, C_cam, H, W = cam_views.shape
        cam_flat = cam_views.flatten(0, 1)    # (N, 3, H, W)
        depth_flat = depth_views.flatten(0, 1)
        N = cam_flat.shape[0]

        # === Patch embedding ===
        cam_patches = self.cam_patch_embed(cam_flat)
        if cam_patches.dim() == 4:
            if cam_patches.shape[-1] == self.vit_embed_dim:
                cam_patches = cam_patches.flatten(1, 2)
            else:
                cam_patches = cam_patches.flatten(2).transpose(1, 2)
        depth_patches = self.depth_patch_embed(depth_flat)
        depth_patches = depth_patches.flatten(2).transpose(1, 2)
        n_patches = cam_patches.shape[1]

        # Add modality embeddings
        cam_patches = cam_patches + self.cam_modality_embed
        depth_patches = depth_patches + self.depth_modality_embed

        cam_pos = self.pos_embed[:, 1:1 + n_patches]
        depth_pos = self.pos_embed[:, 1 + n_patches:1 + 2 * n_patches]

        if self.mask_ratio > 0:
            # === Joint token-budget masking, batched like official MultiMAE ===
            ids_keep, mask_all = self._generate_joint_masks(N, n_patches, cam_flat.device)

            joint_tokens = torch.cat([
                cam_patches + cam_pos,
                depth_patches + depth_pos,
            ], dim=1)
            gather_idx = ids_keep.unsqueeze(-1).expand(-1, -1, joint_tokens.shape[-1])
            visible_tokens = torch.gather(joint_tokens, dim=1, index=gather_idx)

            cls = self.cls_token.expand(N, -1, -1)
            x = torch.cat([cls + self.pos_embed[:, :1], visible_tokens], dim=1)

            x = self.backbone.pos_drop(x)
            if hasattr(self.backbone, 'norm_pre'):
                x = self.backbone.norm_pre(x)
            x = self.backbone.blocks(x)
            x = self.backbone.norm(x)

            cls_out = x[:, 0]
            visible_encoded = x[:, 1:]
            visible_decoded = self.decoder_embed(visible_encoded)

            full_encoded = joint_tokens.new_zeros(N, 2 * n_patches, self.vit_embed_dim)
            full_encoded.scatter_(1, gather_idx, visible_encoded)
            encoder_cam_patches = full_encoded[:, :n_patches]
            encoder_depth_patches = full_encoded[:, n_patches:]

            cam_queries = self.mask_token_cam.expand(N, n_patches, -1).clone()
            depth_queries = self.mask_token_depth.expand(N, n_patches, -1).clone()
            decoder_queries = torch.cat([cam_queries, depth_queries], dim=1)
            decoder_queries.scatter_(
                1,
                ids_keep.unsqueeze(-1).expand(-1, -1, decoder_queries.shape[-1]),
                visible_decoded.to(decoder_queries.dtype),
            )
            decoder_queries = decoder_queries + self.decoder_pos_embed.expand(N, -1, -1)

            encoder_memory = visible_decoded
            for layer in self.decoder_layers:
                decoder_queries = layer(decoder_queries, encoder_memory)
            decoder_queries = self.decoder_norm(decoder_queries)

            cam_decoded = decoder_queries[:, :n_patches]
            depth_decoded = decoder_queries[:, n_patches:]
        else:
            # No masking (probe mode)
            cls = self.cls_token.expand(N, -1, -1)
            x = torch.cat([
                cls + self.pos_embed[:, :1],
                cam_patches + cam_pos,
                depth_patches + depth_pos,
            ], dim=1)

            x = self.backbone.pos_drop(x)

            # Get RoPE if available (DINOv3), otherwise pass None
            rot_pos_embed = None
            if hasattr(self.backbone, 'rope') and self.backbone.rope is not None:
                h = w = self.n_patches ** 0.5
                h, w = int(h), int(w)
                rot_pos_embed = self.backbone.rope.get_embed(shape=(h, w))
                # Tile for RGB+depth tokens (same spatial positions, both modalities)
                rot_pos_embed = torch.cat([rot_pos_embed, rot_pos_embed], dim=0)

            if hasattr(self.backbone, 'norm_pre'):
                x = self.backbone.norm_pre(x)
            if rot_pos_embed is not None:
                for blk in self.backbone.blocks:
                    x = blk(x, rope=rot_pos_embed)
            else:
                x = self.backbone.blocks(x)
            x = self.backbone.norm(x)

            cls_out = x[:, 0]
            encoder_cam_patches = x[:, 1:1 + n_patches]
            encoder_depth_patches = x[:, 1 + n_patches:1 + 2 * n_patches]
            encoder_memory = self.decoder_embed(x[:, 1:])  # skip CLS for decoder memory

            # === Cross-attention decoder ===
            cam_decoder_pos = self.decoder_pos_embed[:, :n_patches].expand(N, -1, -1)
            depth_decoder_pos = self.decoder_pos_embed[:, n_patches:2 * n_patches].expand(N, -1, -1)
            cam_queries = self.decoder_embed(x[:, 1:1 + n_patches])
            depth_queries = self.decoder_embed(x[:, 1 + n_patches:1 + 2 * n_patches])
            decoder_queries = torch.cat([
                cam_queries + cam_decoder_pos,
                depth_queries + depth_decoder_pos,
            ], dim=1)

            for layer in self.decoder_layers:
                decoder_queries = layer(decoder_queries, encoder_memory)
            decoder_queries = self.decoder_norm(decoder_queries)

            cam_decoded = decoder_queries[:, :n_patches]
            depth_decoded = decoder_queries[:, n_patches:]

        # Reconstruct pixel values
        cam_recon = self.cam_recon_head(cam_decoded)
        depth_recon = self.depth_recon_head(depth_decoded)

        # Compute reconstruction loss
        cam_target = self._patchify(cam_flat, self.patch_size)
        depth_target = self._patchify(depth_flat, self.patch_size)

        # Normalize targets per patch
        cam_mean = cam_target.mean(dim=-1, keepdim=True)
        cam_var = cam_target.var(dim=-1, keepdim=True)
        cam_target_norm = (cam_target - cam_mean) / (cam_var + 1e-6).sqrt()

        depth_mean = depth_target.mean(dim=-1, keepdim=True)
        depth_var = depth_target.var(dim=-1, keepdim=True)
        depth_target_norm = (depth_target - depth_mean) / (depth_var + 1e-6).sqrt()

        if self.mask_ratio > 0:
            # Masked-only reconstruction loss on the sampled masked tokens.
            cam_loss_per_patch = (cam_recon - cam_target_norm).pow(2).mean(dim=-1)
            depth_loss_per_patch = (depth_recon - depth_target_norm).pow(2).mean(dim=-1)

            cam_mask = mask_all[:, :n_patches]
            depth_mask = mask_all[:, n_patches:]

            cam_valid = cam_mask.sum(dim=1).clamp_min(1)
            depth_valid = depth_mask.sum(dim=1).clamp_min(1)

            cam_recon_loss = (cam_loss_per_patch * cam_mask.float()).sum(dim=1) / cam_valid
            depth_recon_loss = (depth_loss_per_patch * depth_mask.float()).sum(dim=1) / depth_valid
            cam_recon_loss = cam_recon_loss.mean()
            depth_recon_loss = depth_recon_loss.mean()
        else:
            cam_recon_loss = (cam_recon - cam_target_norm).pow(2).mean()
            depth_recon_loss = (depth_recon - depth_target_norm).pow(2).mean()

        # Uncertainty-weighted task balancing (Kendall et al.)
        recon_loss = (
            cam_recon_loss / (2 * self.log_sigma_cam.exp()) + self.log_sigma_cam / 2
            + depth_recon_loss / (2 * self.log_sigma_depth.exp()) + self.log_sigma_depth / 2
        ).squeeze()

        # CLS embedding and projection
        cls_emb = self.head(cls_out)
        cls_proj = self.proj(cls_emb)
        proj = cls_proj.reshape(B, V, -1).transpose(0, 1)

        return cls_emb, proj, recon_loss, encoder_cam_patches, encoder_depth_patches

    def forward(self, cam_views, range_views):
        """Standard forward interface."""
        if isinstance(cam_views, dict):
            c_global = cam_views['global']
            r_global = range_views['global']
            cls_emb, proj, recon_loss, _, _ = self._forward_batch(c_global, r_global)
            B = c_global.shape[0]

            c_local = cam_views.get('local')
            r_local = range_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                cls_l, proj_l, recon_l, _, _ = self._forward_batch(c_local, r_local)
                cls_emb = torch.cat([cls_emb, cls_l], dim=0)
                proj = torch.cat([proj, proj_l], dim=0)
                recon_loss = (recon_loss + recon_l) / 2
        else:
            cls_emb, proj, recon_loss, _, _ = self._forward_batch(cam_views, range_views)

        self._last_recon_loss = recon_loss

        all_emb = torch.cat([cls_emb, cls_emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        return all_emb, proj_dup

    def forward_with_patches(self, cam_views, range_views):
        """Return patch tokens for probes (uses unmasked forward)."""
        old_ratio = self.mask_ratio
        self.mask_ratio = 0.0
        try:
            if isinstance(cam_views, dict):
                c_global = cam_views['global']
                r_global = range_views['global']
            else:
                c_global = cam_views
                r_global = range_views

            cls_emb, proj, _, cam_full, depth_full = self._forward_batch(c_global, r_global)
            all_emb = torch.cat([cls_emb, cls_emb], dim=0)
            proj_dup = torch.cat([proj, proj], dim=0)
            return all_emb, proj_dup, (cam_full, depth_full)
        finally:
            self.mask_ratio = old_ratio

    def get_recon_loss(self):
        """Get the reconstruction loss from the last forward pass."""
        return getattr(self, '_last_recon_loss', torch.tensor(0.0))


class _MultiMAEPatchedInputAdapter(nn.Module):
    """Task-specific patch adapter for image-like modalities."""

    def __init__(self, num_channels: int, dim_tokens: int, patch_size: int, image_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(num_channels, dim_tokens, kernel_size=patch_size, stride=patch_size)
        self.n_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim_tokens))
        self.task_embed = nn.Parameter(torch.zeros(1, 1, dim_tokens))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.task_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        pos_embed = _resize_pos_embed_2d(self.pos_embed, tokens.shape[1])
        return tokens + pos_embed + self.task_embed


class _MultiMAESemSegInputAdapter(nn.Module):
    """Task-specific semantic segmentation adapter using class embeddings."""

    def __init__(self, num_classes: int, dim_tokens: int, patch_size: int, image_size: int, dim_class_emb: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.class_emb = nn.Embedding(num_classes, dim_class_emb)
        self.proj = nn.Conv2d(dim_class_emb, dim_tokens, kernel_size=1)
        self.n_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim_tokens))
        self.task_embed = nn.Parameter(torch.zeros(1, 1, dim_tokens))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.task_embed, std=0.02)

    def forward(self, seg: torch.Tensor) -> torch.Tensor:
        if seg.dim() == 4 and seg.shape[1] == 1:
            seg = seg[:, 0]
        seg = seg.long().clamp_min(0)
        emb = self.class_emb(seg).permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(emb, kernel_size=self.patch_size, stride=self.patch_size)
        tokens = self.proj(pooled).flatten(2).transpose(1, 2)
        pos_embed = _resize_pos_embed_2d(self.pos_embed, tokens.shape[1])
        return tokens + pos_embed + self.task_embed


class _MultiMAESpatialOutputAdapter(nn.Module):
    """Task-specific spatial decoder that queries the shared encoder memory."""

    def __init__(self, out_dim: int, dim_tokens_enc: int, patch_size: int, image_size: int, decoder_dim: int, decoder_depth: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.query_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, decoder_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.task_proj = nn.Linear(dim_tokens_enc, decoder_dim)
        self.context_proj = nn.Linear(dim_tokens_enc, decoder_dim)
        self.decoder_layers = nn.ModuleList([
            _CrossAttentionDecoderLayer(
                d_model=decoder_dim,
                nhead=4,
                dim_feedforward=decoder_dim * 4,
                dropout=0.1,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.pred_head = nn.Linear(decoder_dim, out_dim)
        nn.init.trunc_normal_(self.query_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        full_task_tokens: Optional[torch.Tensor],
        task_mask: Optional[torch.Tensor],
        encoder_visible_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = encoder_visible_tokens.shape[0]
        pos_embed = _resize_pos_embed_2d(self.query_pos_embed, self.n_patches)
        queries = self.mask_token.expand(batch_size, self.n_patches, -1).clone()

        if full_task_tokens is not None and task_mask is not None:
            task_context = self.task_proj(full_task_tokens)
            queries = torch.where(task_mask.unsqueeze(-1), queries, task_context)

        queries = queries + pos_embed.expand(batch_size, -1, -1)
        memory = self.context_proj(encoder_visible_tokens)
        for layer in self.decoder_layers:
            queries = layer(queries, memory)
        queries = self.decoder_norm(queries)
        return self.pred_head(queries)


class _MultiMAECore(nn.Module):
    """Modular MultiMAE-style encoder/decoder core with task adapters."""

    def __init__(
        self,
        input_adapters: Dict[str, nn.Module],
        output_adapters: Dict[str, nn.Module],
        vit_size: str,
        img_size: int,
        mask_ratio: float,
        num_global_tokens: int = 1,
        alphas: float = 1.0,
        sample_tasks_uniformly: bool = False,
    ):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        self.mask_ratio = mask_ratio
        self.alphas = alphas
        self.sample_tasks_uniformly = sample_tasks_uniformly
        self.input_adapters = nn.ModuleDict(input_adapters)
        self.output_adapters = nn.ModuleDict(output_adapters)

        backbone = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.1,
            img_size=img_size,
            dynamic_img_size=True,
        )
        self.dim_tokens = backbone.embed_dim
        self.encoder_blocks = backbone.blocks
        self.encoder_norm = backbone.norm
        self.pos_drop = backbone.pos_drop
        self.norm_pre = getattr(backbone, 'norm_pre', nn.Identity())

        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, self.dim_tokens))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)

    def _sample_task_counts(self, batch_size: int, task_names: List[str], num_encoded_tokens: int, device: torch.device) -> torch.Tensor:
        num_tasks = len(task_names)
        if isinstance(self.alphas, (list, tuple)):
            alpha_tensor = torch.tensor(self.alphas, dtype=torch.float32, device=device)
        else:
            alpha_tensor = torch.full((num_tasks,), float(self.alphas), dtype=torch.float32, device=device)
        alpha_tensor = alpha_tensor.clamp_min(1e-4)

        if self.sample_tasks_uniformly:
            active = torch.zeros(batch_size, num_tasks, device=device)
            for batch_idx in range(batch_size):
                num_active = torch.randint(1, num_tasks + 1, (1,), device=device).item()
                keep = torch.randperm(num_tasks, device=device)[:num_active]
                active[batch_idx, keep] = 1.0
            proportions = torch.distributions.Dirichlet(alpha_tensor.unsqueeze(0).expand(batch_size, -1) * active.clamp_min(1e-4)).sample()
            proportions = proportions * active
            proportions = proportions / proportions.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            proportions = torch.distributions.Dirichlet(alpha_tensor).sample((batch_size,))

        float_counts = proportions * float(num_encoded_tokens)
        counts = torch.floor(float_counts).long()
        remainders = num_encoded_tokens - counts.sum(dim=1)
        fractions = float_counts - counts.float()

        for batch_idx in range(batch_size):
            if num_encoded_tokens >= num_tasks:
                counts[batch_idx].clamp_(min=1)
            while counts[batch_idx].sum().item() > num_encoded_tokens:
                candidate_idx = torch.argmax(counts[batch_idx]).item()
                if counts[batch_idx, candidate_idx] > 1:
                    counts[batch_idx, candidate_idx] -= 1
                else:
                    break
            if remainders[batch_idx] > 0:
                top_idx = torch.topk(fractions[batch_idx], k=int(remainders[batch_idx].item())).indices
                counts[batch_idx, top_idx] += 1

        return counts

    def generate_random_masks(self, input_tokens: Dict[str, torch.Tensor], num_encoded_tokens: int):
        task_names = list(input_tokens.keys())
        batch_size = next(iter(input_tokens.values())).shape[0]
        device = next(iter(input_tokens.values())).device
        task_counts = self._sample_task_counts(batch_size, task_names, num_encoded_tokens, device)

        task_masks = []
        task_sizes = []
        for task_idx, task_name in enumerate(task_names):
            num_tokens = input_tokens[task_name].shape[1]
            task_sizes.append(num_tokens)
            noise = torch.rand(batch_size, num_tokens, device=device)
            ids_shuffle = noise.argsort(dim=1)
            ids_rank = torch.argsort(ids_shuffle, dim=1)
            task_mask = ids_rank >= task_counts[:, task_idx:task_idx + 1]
            task_masks.append(task_mask)

        mask_all = torch.cat(task_masks, dim=1)
        ids_shuffle = torch.argsort(mask_all.float() + torch.rand_like(mask_all.float()) * 1e-3, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens]

        final_mask_all = torch.ones_like(mask_all, dtype=torch.bool)
        final_mask_all.scatter_(1, ids_keep, False)

        split_masks = torch.split(final_mask_all, task_sizes, dim=1)
        return {task: mask for task, mask in zip(task_names, split_masks)}, ids_keep

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        mask_inputs: bool = True,
        num_encoded_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        input_task_tokens = {task: self.input_adapters[task](tensor) for task, tensor in x.items() if task in self.input_adapters}
        if not input_task_tokens:
            raise ValueError("MultiMAE core requires at least one input modality")

        task_names = list(input_task_tokens.keys())
        task_slices = {}
        start_idx = 0
        for task_name in task_names:
            num_tokens = input_task_tokens[task_name].shape[1]
            task_slices[task_name] = (start_idx, start_idx + num_tokens)
            start_idx += num_tokens
        total_tokens = start_idx

        if num_encoded_tokens is None:
            num_encoded_tokens = total_tokens if not mask_inputs else max(1, int(round((1.0 - self.mask_ratio) * total_tokens)))

        if mask_inputs:
            task_masks, ids_keep = self.generate_random_masks(input_task_tokens, num_encoded_tokens)
        else:
            task_masks = {task: torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device) for task, tokens in input_task_tokens.items()}
            ids_keep = torch.arange(total_tokens, device=next(iter(input_task_tokens.values())).device).unsqueeze(0).expand(next(iter(input_task_tokens.values())).shape[0], -1)

        joint_tokens = torch.cat([input_task_tokens[task] for task in task_names], dim=1)
        gather_idx = ids_keep.unsqueeze(-1).expand(-1, -1, joint_tokens.shape[-1])
        visible_tokens = torch.gather(joint_tokens, dim=1, index=gather_idx)

        batch_size = visible_tokens.shape[0]
        global_tokens = self.global_tokens.expand(batch_size, -1, -1)
        encoder_input = torch.cat([visible_tokens, global_tokens], dim=1)
        encoder_input = self.pos_drop(encoder_input)
        encoder_input = self.norm_pre(encoder_input)
        encoder_output = self.encoder_blocks(encoder_input)
        encoder_output = self.encoder_norm(encoder_output)

        visible_encoded = encoder_output[:, :-self.num_global_tokens]
        global_encoded = encoder_output[:, -self.num_global_tokens:]

        full_encoded = joint_tokens.new_zeros(batch_size, total_tokens, self.dim_tokens)
        full_encoded.scatter_(1, gather_idx, visible_encoded)

        preds = {}
        for task_name, adapter in self.output_adapters.items():
            if task_name in task_slices:
                start_idx, end_idx = task_slices[task_name]
                preds[task_name] = adapter(
                    full_task_tokens=full_encoded[:, start_idx:end_idx],
                    task_mask=task_masks[task_name],
                    encoder_visible_tokens=visible_encoded,
                )
            else:
                preds[task_name] = adapter(
                    full_task_tokens=None,
                    task_mask=None,
                    encoder_visible_tokens=visible_encoded,
                )

        return {
            'preds': preds,
            'task_masks': task_masks,
            'task_slices': task_slices,
            'full_encoded': full_encoded,
            'visible_encoded': visible_encoded,
            'global_encoded': global_encoded,
        }


class MultiMAEExactEncoder(nn.Module):
    """A more faithful MultiMAE-style baseline with modular task adapters."""

    def __init__(
        self,
        proj_dim: int = 128,
        img_size: int = 224,
        vit_size: str = 'small',
        mask_ratio: float = 0.75,
        decoder_depth: int = 2,
        decoder_dim: int = 256,
        depth_channels: int = 1,
        enable_semseg: bool = False,
        enable_panoptic: bool = False,
        num_semseg_classes: int = 8,
        semseg_aux_weight: float = 1.0,
        panoptic_aux_weight: float = 1.0,
    ):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        self.vit_size = vit_size
        self.embed_dim = vit_cfg['embed_dim']
        self.input_size = img_size
        self.mask_ratio = mask_ratio
        self.patch_size = 16
        self.depth_channels = depth_channels
        self.enable_semseg = enable_semseg
        self.enable_panoptic = enable_panoptic
        self.num_semseg_classes = num_semseg_classes
        self.semseg_aux_weight = semseg_aux_weight
        self.panoptic_aux_weight = panoptic_aux_weight

        backbone = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,
        )
        dim_tokens = backbone.embed_dim
        del backbone

        input_adapters = {
            'rgb': _MultiMAEPatchedInputAdapter(3, dim_tokens, self.patch_size, img_size),
            'depth': _MultiMAEPatchedInputAdapter(1, dim_tokens, self.patch_size, img_size),
        }
        if enable_semseg:
            input_adapters['semseg'] = _MultiMAESemSegInputAdapter(
                num_classes=num_semseg_classes,
                dim_tokens=dim_tokens,
                patch_size=self.patch_size,
                image_size=img_size,
            )
        if enable_panoptic:
            input_adapters['panoptic'] = _MultiMAESemSegInputAdapter(
                num_classes=num_semseg_classes,
                dim_tokens=dim_tokens,
                patch_size=self.patch_size,
                image_size=img_size,
            )

        patch_dim_rgb = 3 * self.patch_size * self.patch_size
        patch_dim_depth = 1 * self.patch_size * self.patch_size
        output_adapters = {
            'rgb': _MultiMAESpatialOutputAdapter(patch_dim_rgb, dim_tokens, self.patch_size, img_size, decoder_dim, decoder_depth),
            'depth': _MultiMAESpatialOutputAdapter(patch_dim_depth, dim_tokens, self.patch_size, img_size, decoder_dim, decoder_depth),
        }
        if enable_semseg:
            output_adapters['semseg'] = _MultiMAESpatialOutputAdapter(
                num_semseg_classes * self.patch_size * self.patch_size,
                dim_tokens,
                self.patch_size,
                img_size,
                decoder_dim,
                decoder_depth,
            )
        if enable_panoptic:
            output_adapters['panoptic'] = _MultiMAESpatialOutputAdapter(
                num_semseg_classes * self.patch_size * self.patch_size,
                dim_tokens,
                self.patch_size,
                img_size,
                decoder_dim,
                decoder_depth,
            )

        self.model = _MultiMAECore(
            input_adapters=input_adapters,
            output_adapters=output_adapters,
            vit_size=vit_size,
            img_size=img_size,
            mask_ratio=mask_ratio,
            num_global_tokens=1,
            alphas=1.0,
            sample_tasks_uniformly=False,
        )

        self.head = nn.Linear(dim_tokens, self.embed_dim)
        self.proj = MLP(self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def initialize_from_pretrained(self, source: str = 'dinov3', vit_size: Optional[str] = None, verbose: bool = True):
        """Seed the shared RGB/depth MultiMAE encoder core from a pretrained ViT source."""
        init_vit_size = vit_size or self.vit_size
        return _initialize_multimae_core_from_source(self.model, vit_size=init_vit_size, source=source, verbose=verbose)

    def initialize_from_dinov3(self, vit_size: Optional[str] = None, verbose: bool = True):
        return self.initialize_from_pretrained(source='dinov3', vit_size=vit_size, verbose=verbose)

    def _patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = imgs.shape
        patch = self.patch_size
        grid_h, grid_w = height // patch, width // patch
        tensor = imgs.reshape(batch_size, channels, grid_h, patch, grid_w, patch)
        return tensor.permute(0, 2, 4, 3, 5, 1).reshape(batch_size, grid_h * grid_w, -1)

    def _patchify_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        batch_size, height, width = labels.shape
        patch = self.patch_size
        grid_h, grid_w = height // patch, width // patch
        tensor = labels.reshape(batch_size, grid_h, patch, grid_w, patch)
        return tensor.permute(0, 1, 3, 2, 4).reshape(batch_size, grid_h * grid_w, patch * patch)

    def _reduce_patch_loss(
        self,
        patch_loss: torch.Tensor,
        task_mask: Optional[torch.Tensor],
        sample_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if task_mask is None:
            task_loss = patch_loss.mean(dim=1)
        else:
            weights = task_mask.float()
            denom = weights.sum(dim=1).clamp_min(1.0)
            task_loss = (patch_loss * weights).sum(dim=1) / denom
        if sample_valid_mask is not None:
            valid = sample_valid_mask.float()
            if valid.sum() <= 0:
                return patch_loss.new_tensor(0.0)
            return (task_loss * valid).sum() / valid.sum()
        return task_loss.mean()

    def _build_depth_tensor(self, depth_flat: torch.Tensor) -> torch.Tensor:
        if depth_flat.shape[1] > 1:
            return depth_flat[:, :1]
        return depth_flat

    def _resize_view_tensor(self, tensor: Optional[torch.Tensor], mode: str) -> Optional[torch.Tensor]:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return tensor
        target_hw = (self.input_size, self.input_size)
        if tensor.dim() == 5:
            if tensor.shape[-2:] == target_hw:
                return tensor
            batch_size, num_views = tensor.shape[:2]
            flat = tensor.flatten(0, 1)
            flat = F.interpolate(flat, size=target_hw, mode=mode, align_corners=False if mode != 'nearest' else None)
            return flat.reshape(batch_size, num_views, *flat.shape[1:])
        if tensor.dim() == 4:
            if tensor.shape[-2:] == target_hw:
                return tensor
            return F.interpolate(tensor, size=target_hw, mode=mode, align_corners=False if mode != 'nearest' else None)
        if tensor.dim() == 3:
            if tensor.shape[-2:] == target_hw:
                return tensor
            resized = F.interpolate(tensor.unsqueeze(1).float(), size=target_hw, mode=mode, align_corners=False if mode != 'nearest' else None)
            resized = resized[:, 0]
            return resized.long() if mode == 'nearest' else resized.type_as(tensor)
        return tensor

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        rgb_flat: torch.Tensor,
        depth_flat: torch.Tensor,
        semseg: Optional[torch.Tensor] = None,
        semseg_valid_mask: Optional[torch.Tensor] = None,
        panoptic: Optional[torch.Tensor] = None,
        panoptic_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        preds = outputs['preds']
        task_masks = outputs['task_masks']

        rgb_target = self._patchify(rgb_flat)
        rgb_mean = rgb_target.mean(dim=-1, keepdim=True)
        rgb_var = rgb_target.var(dim=-1, keepdim=True)
        rgb_target = (rgb_target - rgb_mean) / (rgb_var + 1e-6).sqrt()
        rgb_loss_patch = (preds['rgb'] - rgb_target).pow(2).mean(dim=-1)
        total_loss = self._reduce_patch_loss(rgb_loss_patch, task_masks.get('rgb'))

        depth_input = self._build_depth_tensor(depth_flat)
        depth_target = self._patchify(depth_input)
        depth_loss_patch = (preds['depth'] - depth_target).abs().mean(dim=-1)
        total_loss = total_loss + self._reduce_patch_loss(depth_loss_patch, task_masks.get('depth'))

        if self.enable_semseg and semseg is not None and 'semseg' in preds:
            semseg_target = self._patchify_labels(semseg)
            semseg_logits = preds['semseg'].view(
                semseg_target.shape[0],
                semseg_target.shape[1],
                self.num_semseg_classes,
                self.patch_size * self.patch_size,
            )
            semseg_logits = semseg_logits.permute(0, 1, 3, 2).reshape(-1, self.num_semseg_classes)
            semseg_loss = F.cross_entropy(semseg_logits, semseg_target.reshape(-1), reduction='none')
            semseg_loss = semseg_loss.view(semseg_target.shape[0], semseg_target.shape[1], -1).mean(dim=-1)
            total_loss = total_loss + self._reduce_patch_loss(
                semseg_loss,
                task_masks.get('semseg'),
                sample_valid_mask=semseg_valid_mask,
            )

        if self.enable_panoptic and panoptic is not None and 'panoptic' in preds:
            panoptic_target = self._patchify_labels(panoptic)
            panoptic_logits = preds['panoptic'].view(
                panoptic_target.shape[0],
                panoptic_target.shape[1],
                self.num_semseg_classes,
                self.patch_size * self.patch_size,
            )
            panoptic_logits = panoptic_logits.permute(0, 1, 3, 2).reshape(-1, self.num_semseg_classes)
            panoptic_loss = F.cross_entropy(panoptic_logits, panoptic_target.reshape(-1), reduction='none')
            panoptic_loss = panoptic_loss.view(panoptic_target.shape[0], panoptic_target.shape[1], -1).mean(dim=-1)
            total_loss = total_loss + self.panoptic_aux_weight * self._reduce_patch_loss(
                panoptic_loss,
                task_masks.get('panoptic'),
                sample_valid_mask=panoptic_valid_mask,
            )

        return total_loss

    def _encode_views(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
        semseg: Optional[torch.Tensor] = None,
        semseg_valid_mask: Optional[torch.Tensor] = None,
        panoptic: Optional[torch.Tensor] = None,
        panoptic_valid_mask: Optional[torch.Tensor] = None,
        mask_inputs: bool = True,
    ):
        cam_views = self._resize_view_tensor(cam_views, mode='bilinear')
        range_views = self._resize_view_tensor(range_views, mode='bilinear')
        semseg = self._resize_view_tensor(semseg, mode='nearest')
        panoptic = self._resize_view_tensor(panoptic, mode='nearest')
        batch_size, num_views = cam_views.shape[:2]
        rgb_flat = cam_views.flatten(0, 1)
        depth_flat = range_views.flatten(0, 1)

        input_dict = {
            'rgb': rgb_flat,
            'depth': self._build_depth_tensor(depth_flat),
        }
        if semseg is not None:
            if semseg.dim() == 4 and semseg.shape[0] == batch_size and semseg.shape[1] == num_views:
                semseg = semseg.flatten(0, 1)
            elif semseg.dim() == 3 and semseg.shape[0] == batch_size:
                semseg = semseg.repeat_interleave(num_views, dim=0)
            input_dict['semseg'] = semseg
            if semseg_valid_mask is not None and semseg_valid_mask.dim() == 2 and semseg_valid_mask.shape[0] == batch_size and semseg_valid_mask.shape[1] == num_views:
                semseg_valid_mask = semseg_valid_mask.flatten(0, 1)
            elif semseg_valid_mask is not None and semseg_valid_mask.shape[0] == batch_size:
                semseg_valid_mask = semseg_valid_mask.repeat_interleave(num_views, dim=0)
        if panoptic is not None:
            if panoptic.dim() == 4 and panoptic.shape[0] == batch_size and panoptic.shape[1] == num_views:
                panoptic = panoptic.flatten(0, 1)
            elif panoptic.dim() == 3 and panoptic.shape[0] == batch_size:
                panoptic = panoptic.repeat_interleave(num_views, dim=0)
            input_dict['panoptic'] = panoptic
            if panoptic_valid_mask is not None and panoptic_valid_mask.dim() == 2 and panoptic_valid_mask.shape[0] == batch_size and panoptic_valid_mask.shape[1] == num_views:
                panoptic_valid_mask = panoptic_valid_mask.flatten(0, 1)
            elif panoptic_valid_mask is not None and panoptic_valid_mask.shape[0] == batch_size:
                panoptic_valid_mask = panoptic_valid_mask.repeat_interleave(num_views, dim=0)

        outputs = self.model(input_dict, mask_inputs=mask_inputs)
        cls_tokens = outputs['global_encoded'].mean(dim=1)
        cls_emb = self.head(cls_tokens)
        proj = self.proj(cls_emb).reshape(batch_size, num_views, -1).transpose(0, 1)
        recon_loss = self._compute_losses(
            outputs,
            rgb_flat,
            depth_flat,
            semseg=semseg,
            semseg_valid_mask=semseg_valid_mask,
            panoptic=panoptic,
            panoptic_valid_mask=panoptic_valid_mask,
        )

        task_slices = outputs['task_slices']
        rgb_tokens = outputs['full_encoded'][:, task_slices['rgb'][0]:task_slices['rgb'][1]]
        depth_tokens = outputs['full_encoded'][:, task_slices['depth'][0]:task_slices['depth'][1]]
        return cls_emb, proj, recon_loss, rgb_tokens, depth_tokens

    def _compute_probe_multitask_loss(
        self,
        probe_cam_views: Optional[torch.Tensor],
        probe_range_views: Optional[torch.Tensor],
        probe_semseg: Optional[torch.Tensor],
        has_seg_map: Optional[torch.Tensor],
        probe_panoptic: Optional[torch.Tensor] = None,
        has_panoptic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not (self.enable_semseg or self.enable_panoptic) or probe_cam_views is None or probe_range_views is None:
            return self.head.weight.new_tensor(0.0)

        if isinstance(probe_cam_views, dict) or isinstance(probe_range_views, dict):
            return self.head.weight.new_tensor(0.0)

        _, _, probe_loss, _, _ = self._encode_views(
            probe_cam_views,
            probe_range_views,
            semseg=probe_semseg,
            semseg_valid_mask=has_seg_map,
            panoptic=probe_panoptic,
            panoptic_valid_mask=has_panoptic,
            mask_inputs=True,
        )
        return probe_loss

    def forward(self, cam_views, range_views, probe_aux: Optional[Dict[str, torch.Tensor]] = None):
        if isinstance(cam_views, dict):
            c_global = cam_views['global']
            r_global = range_views['global']
            cls_emb, proj, recon_loss, _, _ = self._encode_views(
                c_global,
                r_global,
                semseg=None if probe_aux is None else probe_aux.get('global_seg_map'),
                semseg_valid_mask=None if probe_aux is None else probe_aux.get('global_has_seg_map'),
                panoptic=None if probe_aux is None else probe_aux.get('global_panoptic_seg_map'),
                panoptic_valid_mask=None if probe_aux is None else probe_aux.get('global_has_panoptic_seg_map'),
                mask_inputs=True,
            )

            c_local = cam_views.get('local')
            r_local = range_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                cls_local, proj_local, recon_local, _, _ = self._encode_views(c_local, r_local, mask_inputs=True)
                cls_emb = torch.cat([cls_emb, cls_local], dim=0)
                proj = torch.cat([proj, proj_local], dim=0)
                recon_loss = 0.5 * (recon_loss + recon_local)
        else:
            cls_emb, proj, recon_loss, _, _ = self._encode_views(cam_views, range_views, mask_inputs=True)

        if probe_aux is not None:
            recon_loss = recon_loss + self._compute_probe_multitask_loss(
                probe_aux.get('cam_views'),
                probe_aux.get('range_views'),
                probe_aux.get('seg_map'),
                probe_aux.get('has_seg_map'),
                probe_aux.get('panoptic_seg_map'),
                probe_aux.get('has_panoptic_seg_map'),
            )

        self._last_recon_loss = recon_loss
        all_emb = torch.cat([cls_emb, cls_emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        return all_emb, proj_dup

    def forward_with_patches(self, cam_views, range_views):
        if isinstance(cam_views, dict):
            c_global = cam_views['global']
            r_global = range_views['global']
        else:
            c_global = cam_views
            r_global = range_views

        cls_emb, proj, _, rgb_tokens, depth_tokens = self._encode_views(c_global, r_global, mask_inputs=False)
        all_emb = torch.cat([cls_emb, cls_emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        return all_emb, proj_dup, (rgb_tokens, depth_tokens)

    def get_recon_loss(self):
        return getattr(self, '_last_recon_loss', torch.tensor(0.0))


# ====================================================================
# 3b. Masked Depth Modeling Encoder (LingBot-Depth inspired)
# ====================================================================

class MaskedDepthModelEncoder(nn.Module):
    """
    Masked Depth Modeling (MDM) encoder inspired by LingBot-Depth.

        Uses a DINOv2-style register ViT backbone with LingBot-depth style RGB-D fusion:
      1. Separate patch embeddings for RGB (pretrained) and depth (learned).
            2. Shared positional encodings with modality type offsets (1+ for RGB, 2+ for depth).
      3. Concatenated token sequence [CLS, RGB_tokens, depth_tokens].
            4. Only depth tokens are masked during training.
      5. Reconstruction loss on masked depth patches.

    Key differences from MultiMAE:
    - Uses a register-token DINOv2-style backbone instead of the custom LingBot codebase.
    - Uses the first range-view channel as the depth map when the dataset provides 5 channels.
    - Applies log-depth remapping before tokenization, matching the LingBot input convention.
    - Keeps the lightweight reconstruction head used for this repo's training loop.

    Reference: Tan et al., "Masked Depth Modeling for Spatial Perception",
    arXiv:2601.17895, 2026.

    Args:
        proj_dim:     Projection output dimension.
        img_size:     Input image resolution.
        vit_size:     'small' | 'base' | 'large'.
        depth_mask_ratio: Fraction of depth tokens to mask (default 0.6).
        freeze_rgb_backbone: Whether to freeze the RGB backbone weights.
        decoder_dim:  Decoder hidden dim (default 256).
    """

    _DINO_MODELS = {
        'small': [
            'vit_small_patch14_reg4_dinov2.lvd142m',
            'vit_small_patch14_dinov2.lvd142m',
        ],
        'base': [
            'vit_base_patch14_reg4_dinov2.lvd142m',
            'vit_base_patch14_dinov2.lvd142m',
        ],
        'large': [
            'vit_large_patch14_reg4_dinov2.lvd142m',
            'vit_large_patch14_dinov2.lvd142m',
        ],
    }
    _DINO_DIMS = {
        'small': 384,
        'base':  768,
        'large': 1024,
    }

    def __init__(
        self,
        proj_dim: int = 128,
        img_size: int = 224,
        vit_size: str = 'small',
        depth_mask_ratio: float = 0.6,
        freeze_rgb_backbone: bool = False,
        decoder_dim: int = 256,
        depth_channels: int = 1,
    ):
        super().__init__()
        self.depth_mask_ratio = depth_mask_ratio
        self.freeze_rgb_backbone = freeze_rgb_backbone
        self.depth_channels = depth_channels
        self.depth_token_channels = 1
        self.input_size = img_size

        vit_cfg = get_vit_config(vit_size)
        self.embed_dim = vit_cfg['embed_dim']

        # RGB backbone (register-token DINOv2 from timm when available)
        self.backbone = None
        model_names = self._DINO_MODELS[vit_size]
        load_error = None
        for timm_name in model_names:
            try:
                self.backbone = timm.create_model(
                    timm_name, pretrained=True, num_classes=0, img_size=img_size,
                )
                print(f"✅ MDM: Loaded DINOv2 backbone: {timm_name}")
                break
            except Exception as exc:
                load_error = exc
        if self.backbone is None:
            timm_name = model_names[0]
            self.backbone = timm.create_model(
                timm_name, pretrained=False, num_classes=0, img_size=img_size,
            )
            print(f"⚠️  MDM: DINOv2 pretrained unavailable ({load_error}), using random init: {timm_name}")

        vit_dim = self.backbone.embed_dim
        self.dino_dim = vit_dim
        self.num_prefix = getattr(self.backbone, 'num_prefix_tokens', 5)
        patch_size = getattr(self.backbone.patch_embed, 'patch_size', 16)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        if freeze_rgb_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # Depth patch embedding uses a single depth-like channel, mirroring LingBot-depth.
        self.depth_patch_embed = nn.Conv2d(
            1, vit_dim, kernel_size=patch_size, stride=patch_size,
        )
        nn.init.xavier_uniform_(self.depth_patch_embed.weight)
        nn.init.zeros_(self.depth_patch_embed.bias)

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Modality type codes added to position embeddings
        # Following lingbot-depth: rgb_type_code = 1, depth_type_code = 2
        self.rgb_type_embed = nn.Parameter(torch.ones(1, 1, vit_dim))
        self.depth_type_embed = nn.Parameter(2 * torch.ones(1, 1, vit_dim))

        # Learnable mask token for masked depth positions
        self.depth_mask_token = nn.Parameter(torch.zeros(1, 1, vit_dim))
        nn.init.trunc_normal_(self.depth_mask_token, std=0.02)

        # Lightweight depth decoder for reconstruction
        self.depth_decoder = nn.Sequential(
            nn.Linear(vit_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, self.depth_token_channels * patch_size * patch_size),
        )

        # CLS heads
        self.head = nn.Linear(vit_dim, self.embed_dim)
        self.proj = MLP(self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_rgb_backbone:
            self.backbone.eval()
        return self

    def _maybe_resize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == (self.input_size, self.input_size):
            return x
        return F.interpolate(x, size=(self.input_size, self.input_size),
                             mode='bicubic', align_corners=False)

    def _patchify(self, imgs: torch.Tensor):
        N, C, H, W = imgs.shape
        p = self.patch_size
        h, w = H // p, W // p
        x = imgs.reshape(N, C, h, p, w, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(N, h * w, -1)

    def _resize_patch_tokens_for_probes(self, tokens: torch.Tensor, target_grid: int = 14) -> torch.Tensor:
        """Project variable patch grids onto the repo's standard 14x14 probe layout."""
        if tokens is None:
            return tokens
        grid = int(round(tokens.shape[1] ** 0.5))
        if grid * grid != tokens.shape[1] or grid == target_grid:
            return tokens
        tokens_2d = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid, grid)
        tokens_2d = F.interpolate(tokens_2d, size=(target_grid, target_grid), mode='bicubic', align_corners=False)
        return tokens_2d.flatten(2).transpose(1, 2)

    def _prepare_depth_input(self, depth_flat: torch.Tensor) -> torch.Tensor:
        """Select a single depth channel and apply LingBot-style log remapping."""
        if depth_flat.shape[1] > 1:
            depth_flat = depth_flat[:, :1]
        depth_flat = depth_flat.clone()
        invalid = (~torch.isfinite(depth_flat)) | (depth_flat <= 0.01)
        depth_flat[invalid] = 0.0
        valid = depth_flat > 0.01
        depth_log = torch.zeros_like(depth_flat)
        depth_log[valid] = torch.log(depth_flat[valid])
        return torch.nan_to_num(depth_log, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_register_tokens(self, N: int) -> Optional[torch.Tensor]:
        reg_tokens = getattr(self.backbone, 'reg_token', None)
        if reg_tokens is None:
            reg_tokens = getattr(self.backbone, 'register_tokens', None)
        if reg_tokens is None:
            return None
        return reg_tokens.expand(N, -1, -1)

    def _depth_mask(self, depth_patches: torch.Tensor, depth_flat: torch.Tensor, N: int):
        """Depth-aware masking: always mask invalid depth, randomly mask valid depth."""
        n_patches = depth_patches.shape[1]
        # Identify invalid (near-zero) depth patches
        depth_patchified = self._patchify(depth_flat)  # (N, n_patches, p*p)
        valid_per_patch = (depth_patchified.abs() > 0.01).float().mean(dim=-1)  # (N, n_patches)
        is_valid = valid_per_patch > 0.3  # at least 30% valid pixels

        n_mask = max(1, int(round(self.depth_mask_ratio * n_patches)))
        bool_mask = torch.zeros(N, n_patches, dtype=torch.bool, device=depth_patches.device)

        for i in range(N):
            valid_idx = is_valid[i].nonzero(as_tuple=True)[0]
            invalid_idx = (~is_valid[i]).nonzero(as_tuple=True)[0]
            # Always mask all invalid patches
            bool_mask[i, invalid_idx] = True
            # Randomly mask some valid patches to fill budget
            n_remaining = max(0, n_mask - len(invalid_idx))
            if n_remaining > 0 and len(valid_idx) > 0:
                perm = torch.randperm(len(valid_idx), device=depth_patches.device)[:n_remaining]
                bool_mask[i, valid_idx[perm]] = True

        return bool_mask

    def _forward_batch(self, cam_views, depth_views):
        """Forward with depth masking and reconstruction."""
        B, V, C, H, W = cam_views.shape
        cam_flat = self._maybe_resize_input(cam_views.flatten(0, 1))
        depth_flat = self._maybe_resize_input(depth_views.flatten(0, 1))
        depth_input = self._prepare_depth_input(depth_flat)
        N = cam_flat.shape[0]

        cam_flat = (cam_flat - self.image_mean) / self.image_std

        # === RGB patch embedding (through pretrained backbone) ===
        rgb_out = self.backbone.patch_embed(cam_flat)
        if rgb_out.dim() == 4:
            if rgb_out.shape[-1] == self.dino_dim:
                rgb_patches = rgb_out.flatten(1, 2)
            else:
                rgb_patches = rgb_out.flatten(2).transpose(1, 2)
        else:
            rgb_patches = rgb_out
        n_patches = rgb_patches.shape[1]

        # === Depth patch embedding ===
        depth_patches = self.depth_patch_embed(depth_input)
        depth_patches = depth_patches.flatten(2).transpose(1, 2)

        # === Add modality type codes (and positional embeddings if available) ===
        # DINOv3 uses RoPE (applied inside blocks), so pos_embed may be None.
        pos_embed = self.backbone.pos_embed
        prefix_pos = None
        patch_pos = None
        if pos_embed is not None:
            if pos_embed.shape[1] == n_patches:
                patch_pos = pos_embed[:, :n_patches]
            elif pos_embed.shape[1] >= self.num_prefix + n_patches:
                prefix_pos = pos_embed[:, :self.num_prefix]
                patch_pos = pos_embed[:, self.num_prefix:self.num_prefix + n_patches]
            else:
                patch_pos = pos_embed[:, -n_patches:]

        if patch_pos is not None:
            rgb_patches = rgb_patches + patch_pos + self.rgb_type_embed
            depth_patches = depth_patches + patch_pos + self.depth_type_embed
        else:
            # RoPE model: positional encoding applied inside blocks; only add type codes
            rgb_patches = rgb_patches + self.rgb_type_embed
            depth_patches = depth_patches + self.depth_type_embed

        # === Depth masking ===
        if self.depth_mask_ratio > 0 and self.training:
            depth_mask = self._depth_mask(depth_patches, depth_input, N)
            mask_tokens = self.depth_mask_token.expand(N, n_patches, -1)
            depth_patches_masked = torch.where(depth_mask.unsqueeze(-1), mask_tokens, depth_patches)
        else:
            depth_mask = None
            depth_patches_masked = depth_patches

        # === Build token sequence: [CLS, registers, RGB, depth] ===
        cls_tokens = self.backbone.cls_token.expand(N, -1, -1)
        register_tokens = self._get_register_tokens(N)
        if register_tokens is not None:
            prefix_tokens = torch.cat([cls_tokens, register_tokens], dim=1)
        else:
            prefix_tokens = cls_tokens
        if prefix_pos is not None:
            prefix_tokens = prefix_tokens + prefix_pos[:, :prefix_tokens.shape[1]]
        x = torch.cat([prefix_tokens, rgb_patches, depth_patches_masked], dim=1)

        x = self.backbone.pos_drop(x)
        if hasattr(self.backbone, 'patch_drop') and self.backbone.patch_drop is not None:
            x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)

        # Get RoPE embeddings and tile for doubled (RGB+depth) sequence
        rot_pos_embed = None
        if hasattr(self.backbone, 'rope') and self.backbone.rope is not None:
            h = w = self.input_size // self.patch_size
            rot_pos_embed = self.backbone.rope.get_embed(shape=(h, w))
            # Tile: same spatial positions for both RGB and depth tokens
            rot_pos_embed = torch.cat([rot_pos_embed, rot_pos_embed], dim=0)

        if rot_pos_embed is not None:
            for blk in self.backbone.blocks:
                x = blk(x, rope=rot_pos_embed)
        else:
            x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        cls_out = x[:, 0]
        rgb_out = x[:, self.num_prefix:self.num_prefix + n_patches]
        depth_out = x[:, -n_patches:]

        # === Depth reconstruction loss ===
        depth_recon = self.depth_decoder(depth_out)  # (N, n_patches, p*p)
        depth_target = self._patchify(depth_input)
        
        # NOTE: Do NOT normalize depth patches! Masked Depth Modeling (LingBot-Depth) 
        # requires predicting absolute (log) depth values to retain geometric meaning.
        if depth_mask is not None and depth_mask.any():
            recon_loss = (depth_recon - depth_target).pow(2).mean(dim=-1)[depth_mask].mean()
        else:
            recon_loss = (depth_recon - depth_target).pow(2).mean()

        # CLS embedding and projection
        cls_emb = self.head(cls_out)
        cls_proj = self.proj(cls_emb)
        proj = cls_proj.reshape(B, V, -1).transpose(0, 1)

        return cls_emb, proj, recon_loss, rgb_out, depth_out

    def forward(self, cam_views, range_views):
        if isinstance(cam_views, dict):
            c_global = cam_views['global']
            r_global = range_views['global']
            cls_emb, proj, recon_loss, _, _ = self._forward_batch(c_global, r_global)

            c_local = cam_views.get('local')
            r_local = range_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                cls_l, proj_l, recon_l, _, _ = self._forward_batch(c_local, r_local)
                cls_emb = torch.cat([cls_emb, cls_l], dim=0)
                proj = torch.cat([proj, proj_l], dim=0)
                recon_loss = (recon_loss + recon_l) / 2
        else:
            cls_emb, proj, recon_loss, _, _ = self._forward_batch(cam_views, range_views)

        self._last_recon_loss = recon_loss
        all_emb = torch.cat([cls_emb, cls_emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        return all_emb, proj_dup

    def forward_with_patches(self, cam_views, range_views):
        old_ratio = self.depth_mask_ratio
        self.depth_mask_ratio = 0.0
        try:
            if isinstance(cam_views, dict):
                c_global = cam_views['global']
                r_global = range_views['global']
            else:
                c_global = cam_views
                r_global = range_views
            cls_emb, proj, _, rgb_tokens, depth_tokens = self._forward_batch(c_global, r_global)
            all_emb = torch.cat([cls_emb, cls_emb], dim=0)
            proj_dup = torch.cat([proj, proj], dim=0)
            return all_emb, proj_dup, (rgb_tokens, depth_tokens)
        finally:
            self.depth_mask_ratio = old_ratio

    def get_recon_loss(self):
        return getattr(self, '_last_recon_loss', torch.tensor(0.0))


# ====================================================================
# 4. Late Fusion Encoder (Explicit concatenation)
# ====================================================================

class LateFusionEncoder(nn.Module):
    """
    Late Fusion: Separate ViT encoders for RGB and depth.
    Embeddings are concatenated at the END for probes.

    Similar to arch E but makes concatenation explicit and configurable.

    The CLS embeddings are concatenated for probes (2*embed_dim).
    The patch embeddings are concatenated for patch probes (2*vit_dim).
    Each modality has its own SigReg projection.

    Args:
        proj_dim:   Projection output dim.
        img_size:   Input resolution.
        vit_size:   'small' | 'base' | 'large'.
    """

    def __init__(
        self,
        proj_dim: int = 128,
        img_size: int = 224,
        vit_size: str = 'small',
    ):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        self.embed_dim = vit_cfg['embed_dim']
        self.vit_dim = vit_cfg['vit_dim']

        # Separate ViT encoders
        self.rgb_encoder = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=self.embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            dynamic_img_size=True,
        )

        self.depth_encoder = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=self.embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            in_chans=1,
            dynamic_img_size=True,
        )

        # Per-modality projections (for per-modality SigReg)
        self.proj_rgb = MLP(self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        self.proj_depth = MLP(self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

        # Joint projection (concatenated CLS → proj_dim)
        self.proj_joint = MLP(self.embed_dim * 2, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def _forward_views(self, cam_tensor, depth_tensor):
        """Process (B, V, C, H, W) views through separate encoders."""
        B, V, C_cam, H, W = cam_tensor.shape

        rgb_cls = self.rgb_encoder(cam_tensor.flatten(0, 1))      # (B*V, embed_dim)
        depth_cls = self.depth_encoder(depth_tensor.flatten(0, 1))  # (B*V, embed_dim)

        # Joint CLS via concatenation
        joint_cls = torch.cat([rgb_cls, depth_cls], dim=-1)  # (B*V, 2*embed_dim)

        # Projections
        proj_rgb = self.proj_rgb(rgb_cls).reshape(B, V, -1).transpose(0, 1)   # (V, B, D)
        proj_depth = self.proj_depth(depth_cls).reshape(B, V, -1).transpose(0, 1)

        # Concatenated SigReg projection
        proj_joint = self.proj_joint(joint_cls).reshape(B, V, -1).transpose(0, 1)

        return rgb_cls, depth_cls, joint_cls, proj_rgb, proj_depth, proj_joint

    def forward(self, cam_views, range_views):
        if isinstance(cam_views, dict):
            c_global = cam_views['global']
            r_global = range_views['global']
        else:
            c_global = cam_views
            r_global = range_views

        rgb_cls, depth_cls, joint_cls, proj_r, proj_d, proj_j = self._forward_views(c_global, r_global)
        B = c_global.shape[0]

        # Handle local views
        c_local = cam_views.get('local') if isinstance(cam_views, dict) else None
        r_local = range_views.get('local') if isinstance(range_views, dict) else None
        if c_local is not None and c_local.numel() > 0:
            _, _, _, _, _, proj_j_l = self._forward_views(c_local, r_local)
            proj_j = torch.cat([proj_j, proj_j_l], dim=0)

        # Return concatenated CLS as embedding, joint proj for SigReg
        # Store modality-specific info for later access
        self._last_rgb_cls = rgb_cls
        self._last_depth_cls = depth_cls
        self._last_proj_rgb = proj_r
        self._last_proj_depth = proj_d

        # all_emb: [rgb_cls; depth_cls] for compatibility with probe code
        all_emb = torch.cat([rgb_cls, depth_cls], dim=0)
        return all_emb, proj_j

    def forward_with_patches(self, cam_views, range_views):
        """Return separate patch tokens."""
        if isinstance(cam_views, dict):
            c_global = cam_views['global']
            r_global = range_views['global']
        else:
            c_global = cam_views
            r_global = range_views

        B, V, C, H, W = c_global.shape
        rgb_feats = self.rgb_encoder.forward_features(c_global.flatten(0, 1))
        depth_feats = self.depth_encoder.forward_features(r_global.flatten(0, 1))

        if isinstance(rgb_feats, torch.Tensor) and rgb_feats.dim() == 3:
            rgb_patches = rgb_feats[:, 1:]
        else:
            rgb_patches = torch.zeros(B * V, 196, self.vit_dim, device=c_global.device)

        if isinstance(depth_feats, torch.Tensor) and depth_feats.dim() == 3:
            depth_patches = depth_feats[:, 1:]
        else:
            depth_patches = torch.zeros(B * V, 196, self.vit_dim, device=r_global.device)

        all_emb, proj = self.forward(cam_views, range_views)
        return all_emb, proj, (rgb_patches, depth_patches)

    def get_joint_embed_dim(self):
        return self.embed_dim * 2
