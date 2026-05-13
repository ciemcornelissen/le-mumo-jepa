"""
Multi-Modal Encoders for MM-LeJEPA.

Option A: Separate encoders (ViT + PointMLP) with late fusion
Option B: Unified ViT with separate passes (shared weights, no interaction)
Option C: True Fusion - camera + range patches combined in ONE forward pass
"""

import copy
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
import timm
from typing import Tuple, Optional


# ============================================
# ViT Size Configuration
# ============================================
# Maps vit_size ('small', 'base', 'large') to timm model name,
# hidden dimension (vit_dim), and default CLS head output dimension.
VIT_CONFIGS = {
    'small': {
        'model_name': 'vit_small_patch16_224',
        'vit_dim': 384,       # Hidden dimension of ViT-S
        'embed_dim': 512,     # CLS head output (num_classes)
    },
    'base': {
        'model_name': 'vit_base_patch16_224',
        'vit_dim': 768,       # Hidden dimension of ViT-B
        'embed_dim': 768,     # CLS head output
    },
    'large': {
        'model_name': 'vit_large_patch16_224',
        'vit_dim': 1024,      # Hidden dimension of ViT-L
        'embed_dim': 1024,    # CLS head output
    },
}


_DINOV3_MODELS = {
    'small': 'vit_small_patch16_dinov3',
    'base': 'vit_base_patch16_dinov3',
    'large': 'vit_large_patch16_dinov3',
}


def get_vit_config(vit_size: str = 'small') -> dict:
    """Get ViT configuration for a given size.
    
    Args:
        vit_size: One of 'small', 'base', 'large'
        
    Returns:
        Dict with 'model_name', 'vit_dim', 'embed_dim'
    """
    if vit_size not in VIT_CONFIGS:
        raise ValueError(f"Unknown vit_size='{vit_size}'. Must be one of {list(VIT_CONFIGS.keys())}")
    return VIT_CONFIGS[vit_size]


def _adapt_patch_embed_weight(weight: torch.Tensor, target_in_chans: int) -> torch.Tensor:
    """Adapt a 3-channel ViT patch embedding to arbitrary input channels."""
    if weight.shape[1] == target_in_chans:
        return weight

    mean_weight = weight.mean(dim=1, keepdim=True)
    if target_in_chans == 1:
        return mean_weight

    scaled = mean_weight.repeat(1, target_in_chans, 1, 1)
    return scaled * (weight.shape[1] / float(target_in_chans))


def initialize_module_from_dinov3(
    module: nn.Module,
    vit_size: str = 'small',
    verbose: bool = True,
) -> dict:
    """Seed compatible encoder weights from a pretrained timm DINOv3 backbone.

    Loads transformer trunk weights wherever shapes match, adapts patch embedding
    weights for non-RGB inputs, preserves the target model's absolute positional
    embeddings (DINOv3 uses RoPE but target ViTs rely on learned absolute
    positions), zeros QKV biases absent in DINOv3, and seeds custom fusion
    tokens from DINOv3 register tokens when available.
    """
    if vit_size not in _DINOV3_MODELS:
        raise ValueError(f"Unsupported vit_size='{vit_size}' for DINOv3 init")
    if not hasattr(module, 'backbone'):
        raise ValueError("initialize_module_from_dinov3 expects module.backbone")

    source_name = _DINOV3_MODELS[vit_size]
    source = timm.create_model(source_name, pretrained=True, num_classes=0, img_size=224)
    source_state = source.state_dict()
    target_backbone = module.backbone
    target_state = target_backbone.state_dict()

    loaded_keys = []
    skipped_keys = []
    update_state = {}

    for key, value in source_state.items():
        if key == 'patch_embed.proj.weight':
            target_weight = target_state.get(key)
            if target_weight is None:
                skipped_keys.append(key)
                continue
            adapted = _adapt_patch_embed_weight(value, target_weight.shape[1])
            if adapted.shape == target_weight.shape:
                update_state[key] = adapted.to(dtype=target_weight.dtype)
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
            continue

        if key == 'patch_embed.proj.bias':
            if key in target_state and target_state[key].shape == value.shape:
                update_state[key] = value.to(dtype=target_state[key].dtype)
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
            continue

        if key.startswith('blocks.') or key.startswith('norm.') or key == 'cls_token':
            if key in target_state and target_state[key].shape == value.shape:
                update_state[key] = value.to(dtype=target_state[key].dtype)
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)

    # ── Fuse DINOv3 LayerScale gammas into output weights ──────────
    # DINOv3 uses LayerScale: x = x + gamma * sublayer(x).  The target
    # ViT has no LayerScale (implicit gamma=1).  To preserve DINOv3
    # behaviour we absorb gamma into the last linear of each sub-layer:
    #   attn.proj.weight  ← diag(gamma_1) @ attn.proj.weight
    #   attn.proj.bias    ← gamma_1 * attn.proj.bias
    #   mlp.fc2.weight    ← diag(gamma_2) @ mlp.fc2.weight
    #   mlp.fc2.bias      ← gamma_2 * mlp.fc2.bias
    _fused_gamma = 0
    for key in list(update_state.keys()):
        # Detect block index from keys already staged for loading
        import re as _re
        m = _re.match(r'blocks\.(\d+)\.attn\.proj\.(weight|bias)', key)
        if m:
            blk = m.group(1)
            gamma_key = f'blocks.{blk}.gamma_1'
            gamma = source_state.get(gamma_key)
            if gamma is not None:
                if m.group(2) == 'weight':
                    update_state[key] = gamma.unsqueeze(1) * update_state[key]
                else:  # bias
                    update_state[key] = gamma * update_state[key]
                _fused_gamma += 1
            continue
        m = _re.match(r'blocks\.(\d+)\.mlp\.fc2\.(weight|bias)', key)
        if m:
            blk = m.group(1)
            gamma_key = f'blocks.{blk}.gamma_2'
            gamma = source_state.get(gamma_key)
            if gamma is not None:
                if m.group(2) == 'weight':
                    update_state[key] = gamma.unsqueeze(1) * update_state[key]
                else:  # bias
                    update_state[key] = gamma * update_state[key]
                _fused_gamma += 1

    target_backbone.load_state_dict(update_state, strict=False)

    with torch.no_grad():
        # Keep the target model's absolute positional embeddings intact.
        # DINOv3 itself uses rotary embeddings, but the target timm ViTs in this
        # codepath rely on absolute position parameters for spatial structure.
        # Zeroing them removes position information entirely and is especially
        # harmful for frozen or lightly tuned detection backbones.
        pos_embed = getattr(target_backbone, 'pos_embed', None)

        # DINOv3 has no QKV bias; zero the target's biases to match.
        for name, param in target_backbone.named_parameters():
            if name.endswith('attn.qkv.bias'):
                param.zero_()

        if hasattr(module, 'range_patch_embed'):
            range_weight = _adapt_patch_embed_weight(
                source_state['patch_embed.proj.weight'],
                module.range_patch_embed.weight.shape[1],
            )
            module.range_patch_embed.weight.copy_(range_weight.to(dtype=module.range_patch_embed.weight.dtype))
            if module.range_patch_embed.bias is not None and 'patch_embed.proj.bias' in source_state:
                module.range_patch_embed.bias.copy_(
                    source_state['patch_embed.proj.bias'].to(dtype=module.range_patch_embed.bias.dtype)
                )

        reg_token = source_state.get('reg_token')
        cls_token = source_state.get('cls_token')

        if hasattr(module, 'cls_token') and cls_token is not None and module.cls_token.shape == cls_token.shape:
            module.cls_token.copy_(cls_token.to(dtype=module.cls_token.dtype))

        if hasattr(module, 'pos_embed') and isinstance(module.pos_embed, torch.nn.Parameter):
            pass

        if hasattr(module, 'fusion_tokens') and reg_token is not None:
            # Keep per-position diversity from the target init while injecting a
            # DINOv3 latent-token prior. Copying one identical register-token
            # average to every fusion token was numerically brittle on FLIR.
            fusion_tokens = module.fusion_tokens
            random_seed = fusion_tokens.detach().clone().float()
            fusion_seed = reg_token.mean(dim=1, keepdim=True).to(device=random_seed.device, dtype=torch.float32)
            fusion_seed = fusion_seed - fusion_seed.mean()
            fusion_seed = fusion_seed / fusion_seed.std().clamp_min(1e-6)
            fusion_seed = fusion_seed * random_seed.std().clamp_min(1e-6)
            fusion_seed = fusion_seed.expand_as(random_seed)
            blended_seed = 0.5 * random_seed + 0.5 * fusion_seed
            fusion_tokens.copy_(blended_seed.to(dtype=fusion_tokens.dtype))

    if verbose:
        print(
            f"✅ Initialized {module.__class__.__name__} from DINOv3 ({source_name}); "
            f"loaded {len(loaded_keys)} trunk tensors, fused {_fused_gamma} LayerScale gammas"
        )
        if skipped_keys:
            print(f"ℹ️  DINOv3 init skipped {len(skipped_keys)} non-matching tensors")

    return {
        'source_name': source_name,
        'loaded_keys': loaded_keys,
        'skipped_keys': skipped_keys,
    }


def initialize_module_from_timm_vit(
    module: nn.Module,
    vit_size: str = 'small',
    verbose: bool = True,
) -> dict:
    """Seed compatible encoder weights from a pretrained plain timm ViT backbone."""
    vit_cfg = get_vit_config(vit_size)
    if not hasattr(module, 'backbone'):
        raise ValueError("initialize_module_from_timm_vit expects module.backbone")

    source_name = vit_cfg['model_name']
    source = timm.create_model(source_name, pretrained=True, num_classes=0, img_size=224)
    source_state = source.state_dict()
    target_backbone = module.backbone
    target_state = target_backbone.state_dict()

    loaded_keys = []
    skipped_keys = []
    update_state = {}

    for key, value in source_state.items():
        if key == 'patch_embed.proj.weight':
            target_weight = target_state.get(key)
            if target_weight is None:
                skipped_keys.append(key)
                continue
            adapted = _adapt_patch_embed_weight(value, target_weight.shape[1])
            if adapted.shape == target_weight.shape:
                update_state[key] = adapted.to(dtype=target_weight.dtype)
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
            continue

        if key == 'patch_embed.proj.bias':
            if key in target_state and target_state[key].shape == value.shape:
                update_state[key] = value.to(dtype=target_state[key].dtype)
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
            continue

        if key.startswith('blocks.') or key.startswith('norm.') or key in {'cls_token', 'pos_embed'}:
            if key in target_state and target_state[key].shape == value.shape:
                update_state[key] = value.to(dtype=target_state[key].dtype)
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)

    target_backbone.load_state_dict(update_state, strict=False)

    with torch.no_grad():
        if hasattr(module, 'range_patch_embed'):
            range_weight = _adapt_patch_embed_weight(
                source_state['patch_embed.proj.weight'],
                module.range_patch_embed.weight.shape[1],
            )
            module.range_patch_embed.weight.copy_(range_weight.to(dtype=module.range_patch_embed.weight.dtype))
            if module.range_patch_embed.bias is not None and 'patch_embed.proj.bias' in source_state:
                module.range_patch_embed.bias.copy_(
                    source_state['patch_embed.proj.bias'].to(dtype=module.range_patch_embed.bias.dtype)
                )

    if verbose:
        print(
            f"✅ Initialized {module.__class__.__name__} from timm ViT ({source_name}); "
            f"loaded {len(loaded_keys)} tensors"
        )
        if skipped_keys:
            print(f"ℹ️  timm init skipped {len(skipped_keys)} non-matching tensors")

    return {
        'source_name': source_name,
        'loaded_keys': loaded_keys,
        'skipped_keys': skipped_keys,
    }


class MMEncoderA(nn.Module):
    """
    Option A: Separate encoders for camera and LiDAR.
    
    Architecture:
        Camera → ViT-S/16 → 512-dim ──┐
                                       ├──→ Shared MLP → proj_dim
        LiDAR → PointMLP → 512-dim ───┘
    """
    
    def __init__(self, proj_dim: int = 128, img_size: int = 224, embed_dim: int = 512, vit_size: str = 'small'):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        embed_dim = vit_cfg['embed_dim']
        
        self.cam_encoder = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            dynamic_img_size=True,
        )
        
        from point_encoder import PointMLPEncoder
        self.lidar_encoder = PointMLPEncoder(embed_dim=embed_dim)
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
    
    def forward(
        self, 
        cam_views: torch.Tensor, 
        lidar_points: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        
        # --- Camera Path ---
        if isinstance(cam_views, dict):
            # Process Global
            cam_global = cam_views.get('global')
            B, V_g, C, H_g, W_g = cam_global.shape
            cg_flat = cam_global.flatten(0, 1)
            cg_emb = self.cam_encoder(cg_flat)
            cg_proj = self.proj(cg_emb).reshape(B, V_g, -1).transpose(0, 1)
            
            # Process Local
            cam_local = cam_views.get('local')
            if cam_local is not None and cam_local.numel() > 0:
                B, V_l, C, H_l, W_l = cam_local.shape
                cl_flat = cam_local.flatten(0, 1)
                cl_emb = self.cam_encoder(cl_flat)
                cl_proj = self.proj(cl_emb).reshape(B, V_l, -1).transpose(0, 1)
                
                cam_emb = torch.cat([cg_emb, cl_emb], dim=0)
                cam_proj = torch.cat([cg_proj, cl_proj], dim=0)
            else:
                cam_emb = cg_emb
                cam_proj = cg_proj
        else:
            B, V, C, H, W = cam_views.shape
            cam_flat = cam_views.flatten(0, 1)
            cam_emb = self.cam_encoder(cam_flat)
            cam_proj = self.proj(cam_emb).reshape(B, V, -1).transpose(0, 1)
        
        # --- LiDAR Path ---
        if isinstance(lidar_points, dict):
             # Process Global
            l_global = lidar_points.get('global')
            B, V_g, N_g, C_g = l_global.shape
            lg_flat = l_global.flatten(0, 1) # (B*Vg, N, 5)
            lg_emb = self.lidar_encoder(lg_flat)
            lg_proj = self.proj(lg_emb).reshape(B, V_g, -1).transpose(0, 1)
            
            # Process Local
            l_local = lidar_points.get('local')
            if l_local is not None and l_local.numel() > 0:
                B, V_l, N_l, C_l = l_local.shape
                ll_flat = l_local.flatten(0, 1)
                ll_emb = self.lidar_encoder(ll_flat)
                ll_proj = self.proj(ll_emb).reshape(B, V_l, -1).transpose(0, 1)
                
                lidar_emb = torch.cat([lg_emb, ll_emb], dim=0)
                lidar_proj = torch.cat([lg_proj, ll_proj], dim=0)
            else:
                lidar_emb = lg_emb
                lidar_proj = lg_proj
        else:
            if lidar_points.dim() == 3: # (B, N, 5)
                 B, N, C = lidar_points.shape
                 lidar_emb = self.lidar_encoder(lidar_points)
                 lidar_proj = self.proj(lidar_emb).unsqueeze(0) # (1, B, D)
            elif lidar_points.dim() == 4: # (B, V, N, 5)
                B, V, N, C = lidar_points.shape
                lidar_flat = lidar_points.flatten(0, 1)
                lidar_emb = self.lidar_encoder(lidar_flat)
                lidar_proj = self.proj(lidar_emb).reshape(B, V, -1).transpose(0, 1)
            else: # Fallback
                lidar_emb = self.lidar_encoder(lidar_points)
                lidar_proj = self.proj(lidar_emb).unsqueeze(0)
        
        return (cam_emb, lidar_emb), (cam_proj, lidar_proj)



class MMEncoderB(nn.Module):
    """
    Option B: Unified ViT with SEPARATE forward passes.
    
    Camera and range share ViT weights but are processed independently.
    Alignment happens only at the SIGReg loss level.
    
    With aligned_mode=True, uses 1-channel projected depth (camera-aligned)
    instead of 5-channel range images (panoramic, not aligned).
    """
    
    def __init__(
        self, 
        proj_dim: int = 128, 
        img_size: int = 224,
        range_channels: int = 5,
        embed_dim: int = 512,
        aligned_mode: bool = False,  # NEW: Use 1-channel aligned depth instead of range
        vit_size: str = 'small',
    ):
        super().__init__()
        self.aligned_mode = aligned_mode
        vit_cfg = get_vit_config(vit_size)
        embed_dim = vit_cfg['embed_dim']
        
        self.backbone = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            in_chans=3,
            dynamic_img_size=True,
        )
        
        patch_size = 16
        self.cam_patch_embed = self.backbone.patch_embed
        
        # Depth/range input channels depend on mode
        if aligned_mode:
            # 1-channel for aligned camera-projected depth
            lidar_channels = 1
        else:
            # 5-channel for panoramic range images
            lidar_channels = range_channels
        
        self.range_patch_embed = nn.Conv2d(
            lidar_channels, 
            self.backbone.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        
        # Modality embeddings (like positional, but for modality type)
        # Added for Shared Trunk scenarios to help shared weights distinguish input source
        self.cam_modality_embed = nn.Parameter(torch.zeros(1, 1, self.backbone.embed_dim))
        self.range_modality_embed = nn.Parameter(torch.zeros(1, 1, self.backbone.embed_dim))
        nn.init.normal_(self.cam_modality_embed, std=0.02)
        nn.init.normal_(self.range_modality_embed, std=0.02)
    
    def _forward_features(self, x: torch.Tensor, is_range: bool = False) -> torch.Tensor:
        if is_range:
            x = self.range_patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            # Add range modality embedding
            x = x + self.range_modality_embed
        else:
            x = self.cam_patch_embed(x)
            # timm with dynamic_img_size=True returns (B, H, W, D) - channels last
            # We need (B, N, D) where N = H*W
            if x.dim() == 4:
                # Check if channels-last (B, H, W, D) or channels-first (B, D, H, W)
                if x.shape[-1] == self.backbone.embed_dim:
                    # Channels-last: (B, H, W, D) -> flatten H,W -> (B, H*W, D)
                    x = x.flatten(1, 2)
                else:
                    # Channels-first: (B, D, H, W) -> (B, D, H*W) -> (B, H*W, D)
                    x = x.flatten(2).transpose(1, 2)
            
            # Add camera modality embedding
            x = x + self.cam_modality_embed
        
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Interpolate positional embeddings if needed
        pos_embed = self.backbone.pos_embed
        if pos_embed.shape[1] != x.shape[1]:
            # Expecting pos_embed to be (1, N_tokens, D)
            # x is (B, N_current_tokens, D)
            # Exclude cls token for interpolation
            pos_embed_cls = pos_embed[:, :1]
            pos_embed_patches = pos_embed[:, 1:]
            
            # Reshape patches to square Grid for bicubic interpolation
            N_orig = pos_embed_patches.shape[1]
            H_orig = int(N_orig**0.5)
            W_orig = H_orig
            
            N_curr = x.shape[1] - 1
            H_curr = int(N_curr**0.5)
            W_curr = H_curr
            
            # Check square assumption
            if H_orig * W_orig != N_orig:
                 # Should not happen for standard ViT, but safer to warn or fallback
                 pass
            
            if H_curr * W_curr != N_curr:
                 # Non-square input (weird but possible), handle carefully?
                 # Assuming square for now as transforms are square crop.
                 pass

            # Interpolate (1, N, D) -> (1, D, H, W)
            pe = pos_embed_patches.reshape(1, H_orig, W_orig, -1).permute(0, 3, 1, 2)
            pe = torch.nn.functional.interpolate(
                pe, size=(H_curr, W_curr), mode='bicubic', align_corners=False
            )
            pe = pe.permute(0, 2, 3, 1).flatten(1, 2) # (1, N_curr, D)
            
            pos_embed = torch.cat([pos_embed_cls, pe], dim=1)
            
        x = x + pos_embed
        x = self.backbone.pos_drop(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return self.backbone.head(x[:, 0])
    
    def _forward_features_with_patches(self, x: torch.Tensor, is_range: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that returns BOTH CLS token and patch embeddings.
        
        Returns:
            cls_emb: (B, embed_dim) CLS token embedding
            patch_emb: (B, N_patches, vit_dim) Patch embeddings BEFORE the head
        """
        if is_range:
            x = self.range_patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            x = x + self.range_modality_embed
        else:
            x = self.cam_patch_embed(x)
            if x.dim() == 4:
                if x.shape[-1] == self.backbone.embed_dim:
                    x = x.flatten(1, 2)
                else:
                    x = x.flatten(2).transpose(1, 2)
            x = x + self.cam_modality_embed
        
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Interpolate positional embeddings if needed
        pos_embed = self.backbone.pos_embed
        if pos_embed.shape[1] != x.shape[1]:
            pos_embed_cls = pos_embed[:, :1]
            pos_embed_patches = pos_embed[:, 1:]
            N_orig = pos_embed_patches.shape[1]
            H_orig = int(N_orig**0.5)
            W_orig = H_orig
            N_curr = x.shape[1] - 1
            H_curr = int(N_curr**0.5)
            W_curr = H_curr
            pe = pos_embed_patches.reshape(1, H_orig, W_orig, -1).permute(0, 3, 1, 2)
            pe = torch.nn.functional.interpolate(
                pe, size=(H_curr, W_curr), mode='bicubic', align_corners=False
            )
            pe = pe.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = torch.cat([pos_embed_cls, pe], dim=1)
            
        x = x + pos_embed
        x = self.backbone.pos_drop(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        
        # Return both CLS (through head) and patch embeddings (raw)
        cls_emb = self.backbone.head(x[:, 0])  # (B, embed_dim)
        patch_emb = x[:, 1:]  # (B, N_patches, vit_dim) - skip CLS token
        
        return cls_emb, patch_emb
    
    def forward_with_patches(
        self, 
        cam_views: torch.Tensor, 
        range_views: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward that also returns patch embeddings for patch-level alignment.
        
        Returns:
            (cam_emb, range_emb): CLS token embeddings
            (cam_proj, range_proj): Projections for SIGReg
            (cam_patches, range_patches): Patch embeddings (B*V, N_patches, vit_dim)
        """
        # Handle standard tensor inputs (not dict for simplicity)
        if isinstance(cam_views, dict):
            raise NotImplementedError("forward_with_patches doesn't support dict inputs yet")
        
        B, V, _, H, W = cam_views.shape
        cam_flat = cam_views.flatten(0, 1)
        range_flat = range_views.flatten(0, 1)
        
        cam_emb, cam_patches = self._forward_features_with_patches(cam_flat, is_range=False)
        range_emb, range_patches = self._forward_features_with_patches(range_flat, is_range=True)
        
        all_emb = torch.cat([cam_emb, range_emb], dim=0)
        all_proj = self.proj(all_emb)
        cam_proj = all_proj[:B*V].reshape(B, V, -1).transpose(0, 1)
        range_proj = all_proj[B*V:].reshape(B, V, -1).transpose(0, 1)
        proj = torch.cat([cam_proj, range_proj], dim=0)
        
        return (cam_emb, range_emb), proj, (cam_patches, range_patches)
    
    def forward(self, cam_views: torch.Tensor, range_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle dictionary inputs for mixed resolution
        if isinstance(cam_views, dict):
            # Process Global
            cam_global = cam_views.get('global')
            B, V_g, _, H_g, W_g = cam_global.shape
            cg_flat = cam_global.flatten(0, 1)
            cg_emb = self._forward_features(cg_flat, is_range=False)
            
            # Process Local
            cam_local = cam_views.get('local')
            if cam_local is not None and cam_local.numel() > 0:
                B, V_l, _, H_l, W_l = cam_local.shape
                cl_flat = cam_local.flatten(0, 1)
                cl_emb = self._forward_features(cl_flat, is_range=False)
                cam_emb = torch.cat([cg_emb, cl_emb], dim=0)
                V = V_g + V_l
            else:
                cam_emb = cg_emb
                V = V_g
        else:
            B, V, _, H, W = cam_views.shape
            cam_flat = cam_views.flatten(0, 1)
            cam_emb = self._forward_features(cam_flat, is_range=False)
            
            
        if isinstance(range_views, dict):
            # Process Global
            rng_global = range_views.get('global')
            rg_flat = rng_global.flatten(0, 1)
            rg_emb = self._forward_features(rg_flat, is_range=True)
            
            # Process Local
            rng_local = range_views.get('local')
            if rng_local is not None and rng_local.numel() > 0:
                rl_flat = rng_local.flatten(0, 1)
                rl_emb = self._forward_features(rl_flat, is_range=True)
                range_emb = torch.cat([rg_emb, rl_emb], dim=0)
                
                # End of local processing
                range_emb = torch.cat([rg_emb, rl_emb], dim=0)
            else:

                range_emb = rg_emb

        else:
            range_flat = range_views.flatten(0, 1)
            range_emb = self._forward_features(range_flat, is_range=True)
        
        all_emb = torch.cat([cam_emb, range_emb], dim=0)
        
        if isinstance(cam_views, dict) and cam_views.get('local') is not None and cam_views.get('local').numel() > 0:
            # We have Global and Local mixed in the flat embedding tensors.
            # cam_emb = [cam_global_flat; cam_local_flat]
            # range_emb = [rng_global_flat; rng_local_flat]
            
            # Recalculate sizes
            cg_size = cam_views['global'].flatten(0, 1).shape[0]
            
            # Extract
            cg_emb = cam_emb[:cg_size]
            cl_emb = cam_emb[cg_size:]
            rg_emb = range_emb[:cg_size]
            rl_emb = range_emb[cg_size:]
            
            # 1. Global Projection
            # Cat Cam+Range Global
            all_g = torch.cat([cg_emb, rg_emb], dim=0)
            proj_g = self.proj(all_g)
            
            # Split back: proj_g is (2 * B*Vg, D) -> [cam_g; rng_g]
            V_g = cam_views['global'].shape[1]
            cam_proj_g = proj_g[:cg_size].reshape(B, V_g, -1).transpose(0, 1)
            range_proj_g = proj_g[cg_size:].reshape(B, V_g, -1).transpose(0, 1)
            
            # 2. Local Projection
            # Cat Cam+Range Local
            all_l = torch.cat([cl_emb, rl_emb], dim=0)
            proj_l = self.proj(all_l)
            
            # Split back
            cl_size = cam_views['local'].flatten(0, 1).shape[0]
            V_l = cam_views['local'].shape[1]
            cam_proj_l = proj_l[:cl_size].reshape(B, V_l, -1).transpose(0, 1)
            range_proj_l = proj_l[cl_size:].reshape(B, V_l, -1).transpose(0, 1)
            
            # 3. Concatenate (View dim is 0)
            cam_proj = torch.cat([cam_proj_g, cam_proj_l], dim=0)
            range_proj = torch.cat([range_proj_g, range_proj_l], dim=0)
            proj = torch.cat([cam_proj, range_proj], dim=0)
            
        else:
             # Standard case
             all_proj = self.proj(all_emb)
             cam_proj = all_proj[:B*V].reshape(B, V, -1).transpose(0, 1)
             range_proj = all_proj[B*V:].reshape(B, V, -1).transpose(0, 1)
             proj = torch.cat([cam_proj, range_proj], dim=0)

        
        return all_emb, proj

    def forward_single(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Forward single modality and return CLS embedding.
        
        Args:
            x: Input tensor (B, V, C, H, W)
            modality: 'cam' or 'range'
            
        Returns:
            CLS embeddings (B*V, embed_dim)
        """
        B, V, _, H, W = x.shape
        x_flat = x.flatten(0, 1)  # (B*V, C, H, W)
        
        is_range = (modality == 'range')
        emb = self._forward_features(x_flat, is_range=is_range)
        
        return emb  # (B*V, embed_dim)
    
    def forward_camera_only(self, cam_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with camera only - range/depth is zeroed."""
        # Create zero tensor matching the aligned_mode decision in __init__
        range_channels = self.range_patch_embed.in_channels
        
        if isinstance(cam_views, dict):
            zero_range = {}
            for k, v in cam_views.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    B, V, _, H, W = v.shape
                    zero_range[k] = torch.zeros(B, V, range_channels, H, W, device=v.device, dtype=v.dtype)
            
            emb, proj = self.forward(cam_views, zero_range)
            # Output is [Camera, Range] concatenated
            cam_emb = emb[:emb.shape[0]//2]
            cam_proj = proj[:proj.shape[0]//2]
            return cam_emb, cam_proj
        else:
            B, V, _, H, W = cam_views.shape
            zero_range = torch.zeros(B, V, range_channels, H, W, device=cam_views.device, dtype=cam_views.dtype)
            # We only care about the camera embeddings (first B*V elements)
            emb, proj = self.forward(cam_views, zero_range)
            
            N = B * V
            cam_emb = emb[:N]
            cam_proj = proj[:V] # First V rows (transposed from B*V)
            return cam_emb, cam_proj




class MMEncoderC(nn.Module):
    """
    Option C: TRUE FUSION - Camera + Range patches processed TOGETHER.
    
    Architecture:
        Camera → patches ──┐
                           ├──→ Concatenate → Single ViT forward → Fused embeddings
        Range → patches ───┘
    
    Key difference from B: Both modalities go through ViT TOGETHER,
    allowing cross-attention between camera and range patches.
    
    This means zeroing one modality WILL affect the other's output!
    
    With aligned_mode=True, uses 1-channel projected depth (camera-aligned)
    instead of 5-channel range images (panoramic, not aligned).
    """
    
    def __init__(
        self, 
        proj_dim: int = 128, 
        img_size: int = 224,
        range_channels: int = 5,
        embed_dim: int = 512,
        aligned_mode: bool = False,  # NEW: Use 1-channel aligned depth instead of range
        vit_size: str = 'small',
    ):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        embed_dim = vit_cfg['embed_dim']
        self.embed_dim = embed_dim
        self.aligned_mode = aligned_mode
        
        # Backbone ViT
        self.backbone = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            dynamic_img_size=True,
        )
        
        self.vit_embed_dim = self.backbone.embed_dim  # 384 for ViT-S
        patch_size = 16
        self.n_patches = (img_size // patch_size) ** 2  # 196 for 224x224
        
        # Separate patch embeddings
        self.cam_patch_embed = self.backbone.patch_embed
        
        # Depth/range input channels depend on mode
        if aligned_mode:
            # 1-channel for aligned camera-projected depth
            lidar_channels = 1
        else:
            # 5-channel for panoramic range images
            lidar_channels = range_channels
        
        self.range_patch_embed = nn.Conv2d(
            lidar_channels, 
            self.vit_embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Modality embeddings (like positional, but for modality type)
        self.cam_modality_embed = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        self.range_modality_embed = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        nn.init.normal_(self.cam_modality_embed, std=0.02)
        nn.init.normal_(self.range_modality_embed, std=0.02)
        
        # Two cls tokens (one per modality)
        self.cam_cls = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        self.range_cls = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        nn.init.trunc_normal_(self.cam_cls, std=0.02)
        nn.init.trunc_normal_(self.range_cls, std=0.02)
        
        # Extended positional embedding for 2x patches + 2 cls tokens
        # Original: 1 + 196 = 197, New: 2 + 196*2 = 394
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 + self.n_patches * 2, self.vit_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Head for each modality
        self.cam_head = nn.Linear(self.vit_embed_dim, embed_dim)
        self.range_head = nn.Linear(self.vit_embed_dim, embed_dim)
        
        # Projection
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
    
    def _forward_batch(self, cam_views: torch.Tensor, range_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal forward for a batch of known fixed size."""
        B, V, C, H, W = cam_views.shape
        
        # Flatten views
        cam_flat = cam_views.flatten(0, 1)  # (B*V, 3, H, W)
        range_flat = range_views.flatten(0, 1)  # (B*V, 5, H, W)
        N = cam_flat.shape[0]
        
        # Get patches
        cam_patches = self.cam_patch_embed(cam_flat)
        if cam_patches.dim() == 4:
            # (B, H, W, D) -> (B, N, D)
            cam_patches = cam_patches.flatten(1, 2)
            
        range_patches = self.range_patch_embed(range_flat)
        # range_patch_embed is Conv2d -> (B, D, H, W)
        # Flatten to (B, N, D)
        range_patches = range_patches.flatten(2).transpose(1, 2)
        
        # Add modality embeddings
        cam_patches = cam_patches + self.cam_modality_embed
        range_patches = range_patches + self.range_modality_embed
        
        # Prepare cls tokens
        cam_cls = self.cam_cls.expand(N, -1, -1)
        range_cls = self.range_cls.expand(N, -1, -1)
        
        # Concatenate: [cam_cls, cam_patches, range_cls, range_patches]
        x = torch.cat([cam_cls, cam_patches, range_cls, range_patches], dim=1)
        # Shape: (N, 2 + 2*N_patches, vit_dim)
        
        # Add positional embedding
        # pos_embed is (1, 2 + 2*196, D) for 224x224
        if x.shape[1] != self.pos_embed.shape[1]:
            # Interpolate pos_embed
            # Structure: [cls_c, patches_c(N_p), cls_r, patches_r(N_p)]
            # We assume N_patches is square
            
            # Slice original
            # Total patches in original
            N_patches_orig = (self.pos_embed.shape[1] - 2) // 2
            
            pc_cls = self.pos_embed[:, 0:1]
            pc_emb = self.pos_embed[:, 1:1+N_patches_orig]
            pr_cls = self.pos_embed[:, 1+N_patches_orig:2+N_patches_orig]
            pr_emb = self.pos_embed[:, 2+N_patches_orig:]
            
            # Target patches
            N_patches_curr = (x.shape[1] - 2) // 2
            
            # Helper to interpolate
            def interp_pe(pe, n_curr):
                N_orig = pe.shape[1]
                H_orig = int(N_orig**0.5)
                W_orig = H_orig
                
                H_curr = int(n_curr**0.5)
                W_curr = H_curr
                
                pe = pe.reshape(1, H_orig, W_orig, -1).permute(0, 3, 1, 2)
                pe = torch.nn.functional.interpolate(
                    pe, size=(H_curr, W_curr), mode='bicubic', align_corners=False
                )
                pe = pe.permute(0, 2, 3, 1).flatten(1, 2)
                return pe
                
            pc_emb_new = interp_pe(pc_emb, N_patches_curr)
            pr_emb_new = interp_pe(pr_emb, N_patches_curr)
            
            pos_embed = torch.cat([pc_cls, pc_emb_new, pr_cls, pr_emb_new], dim=1)
        else:
            pos_embed = self.pos_embed
            
        x = x + pos_embed
        x = self.backbone.pos_drop(x)
        
        # Transformer blocks (FUSION happens here!)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        
        # Extract cls token outputs (need to account for dynamic length)
        # First is cam_cls (idx 0)
        # range_cls is at idx 1 + N_patches_element
        N_patches_element = (x.shape[1] - 2) // 2
        
        cam_feat = x[:, 0]
        range_feat = x[:, 1 + N_patches_element]
        
        # Project to embedding dim
        cam_emb = self.cam_head(cam_feat)
        range_emb = self.range_head(range_feat)
        
        # Combine embeddings
        all_emb = torch.cat([cam_emb, range_emb], dim=0)  
        
        # Project for SIGReg
        all_proj = self.proj(all_emb)
        cam_proj = all_proj[:N].reshape(B, V, -1).transpose(0, 1)
        range_proj = all_proj[N:].reshape(B, V, -1).transpose(0, 1)
        proj = torch.cat([cam_proj, range_proj], dim=0)
        
        return all_emb, proj

    def _forward_batch_with_patches(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[Tuple, torch.Tensor, Tuple]:
        """Internal forward with patches for a fixed batch size."""
        B, V, C, H, W = cam_views.shape
        cam_flat = cam_views.flatten(0, 1)
        range_flat = range_views.flatten(0, 1)
        N = cam_flat.shape[0]
        
        # Patch embeddings (same as _forward_batch)
        cam_patches = self.cam_patch_embed(cam_flat)
        if cam_patches.dim() == 4:
            cam_patches = cam_patches.flatten(1, 2)
        range_patches = self.range_patch_embed(range_flat)
        range_patches = range_patches.flatten(2).transpose(1, 2)
        
        cam_patches = cam_patches + self.cam_modality_embed
        range_patches = range_patches + self.range_modality_embed
        
        cam_cls = self.cam_cls.expand(N, -1, -1)
        range_cls = self.range_cls.expand(N, -1, -1)
        
        x = torch.cat([cam_cls, cam_patches, range_cls, range_patches], dim=1)
        
        # Positional embedding
        if x.shape[1] != self.pos_embed.shape[1]:
            # Interpolation logic (same as _forward_batch)
            N_patches_orig = (self.pos_embed.shape[1] - 2) // 2
            pc_cls = self.pos_embed[:, 0:1]
            pc_emb = self.pos_embed[:, 1:1+N_patches_orig]
            pr_cls = self.pos_embed[:, 1+N_patches_orig:2+N_patches_orig]
            pr_emb = self.pos_embed[:, 2+N_patches_orig:]
            
            N_patches_curr = (x.shape[1] - 2) // 2
            
            def interp_pe(pe, n_curr):
                N_orig = pe.shape[1]
                H_orig = int(N_orig**0.5)
                W_orig = H_orig
                H_curr = int(n_curr**0.5)
                W_curr = H_curr
                pe = pe.reshape(1, H_orig, W_orig, -1).permute(0, 3, 1, 2)
                pe = torch.nn.functional.interpolate(
                    pe, size=(H_curr, W_curr), mode='bicubic', align_corners=False
                )
                pe = pe.permute(0, 2, 3, 1).flatten(1, 2)
                return pe
                
            pc_emb_new = interp_pe(pc_emb, N_patches_curr)
            pr_emb_new = interp_pe(pr_emb, N_patches_curr)
            pos_embed = torch.cat([pc_cls, pc_emb_new, pr_cls, pr_emb_new], dim=1)
        else:
            pos_embed = self.pos_embed
            
        x = x + pos_embed
        x = self.backbone.pos_drop(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        
        # Extract features
        N_patches_element = (x.shape[1] - 2) // 2
        cam_feat = x[:, 0]
        cam_patch_out = x[:, 1:1+N_patches_element]
        range_feat = x[:, 1 + N_patches_element]
        range_patch_out = x[:, 2+N_patches_element:]
        
        cam_emb = self.cam_head(cam_feat)
        range_emb = self.range_head(range_feat)
        
        all_emb = torch.cat([cam_emb, range_emb], dim=0)
        all_proj = self.proj(all_emb)
        cam_proj = all_proj[:N].reshape(B, V, -1).transpose(0, 1)
        range_proj = all_proj[N:].reshape(B, V, -1).transpose(0, 1)
        proj = torch.cat([cam_proj, range_proj], dim=0)
        
        return all_emb, proj, (cam_patch_out, range_patch_out)

    def forward_with_patches(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Forward pass returning patch embeddings alongside CLS embeddings and projections."""
        if isinstance(cam_views, dict):
            # Global
            c_global = cam_views.get('global')
            r_global = range_views.get('global')
            emb_g, proj_g, patches_g = self._forward_batch_with_patches(c_global, r_global)
            B = c_global.shape[0]
            V_g = c_global.shape[1]
            
            # Local
            c_local = cam_views.get('local')
            r_local = range_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                emb_l, proj_l, patches_l = self._forward_batch_with_patches(c_local, r_local)
                V_l = c_local.shape[1]
                
                # Deconstruct emb
                N_g = B * V_g
                N_l = B * V_l
                cam_emb_g = emb_g[:N_g]; range_emb_g = emb_g[N_g:]
                cam_emb_l = emb_l[:N_l]; range_emb_l = emb_l[N_l:]
                all_emb = torch.cat([cam_emb_g, cam_emb_l, range_emb_g, range_emb_l], dim=0)
                
                # Deconstruct proj
                cam_proj_g = proj_g[:V_g]; range_proj_g = proj_g[V_g:]
                cam_proj_l = proj_l[:V_l]; range_proj_l = proj_l[V_l:]
                proj = torch.cat([cam_proj_g, cam_proj_l, range_proj_g, range_proj_l], dim=0)
                
                # Deconstruct patches
                cam_patches_g, range_patches_g = patches_g
                cam_patches_l, range_patches_l = patches_l
                cam_patches = torch.cat([cam_patches_g, cam_patches_l], dim=0)
                range_patches = torch.cat([range_patches_g, range_patches_l], dim=0)
                
                return all_emb, proj, (cam_patches, range_patches)
            else:
                return emb_g, proj_g, patches_g
        else:
            return self._forward_batch_with_patches(cam_views, range_views)

    def forward(
        self, 
        cam_views: torch.Tensor, 
        range_views: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TRUE FUSION: Camera and range patches processed TOGETHER through ViT.
        Handles Dictionary inputs for Global/Local split.
        """
        if isinstance(cam_views, dict):
            # Global
            c_global = cam_views.get('global')
            r_global = range_views.get('global')
            emb_g, proj_g = self._forward_batch(c_global, r_global)
            B = c_global.shape[0]
            V_g = c_global.shape[1]
            
            # Local
            c_local = cam_views.get('local')
            r_local = range_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                emb_l, proj_l = self._forward_batch(c_local, r_local)
                # Concatenate results logic
                # emb is (2*B*V, D) -> need to be careful with cat
                # emb_g structure: [cam_g_1..N, range_g_1..N]
                # We want concatenated structure to be consistent?
                # Actually, output of forward is (2*B*V_total, D) usually?
                # Let's see how MMEncoderB does it: all_emb = torch.cat([cam_emb, range_emb], dim=0)
                # B iterates: cam_global, cam_local -> cam_emb. range_global, range_local -> range_emb.
                
                # Here _forward_batch returns [cam, range].
                # We need to split them to merge global/local for each modality, then re-cat?
                # Or just return list?
                
                # Deconstruct emb_g
                N_g = B * V_g
                cam_emb_g = emb_g[:N_g]
                range_emb_g = emb_g[N_g:]
                
                # Deconstruct proj_g (V, B, D)
                cam_proj_g = proj_g[:V_g]
                range_proj_g = proj_g[V_g:] # dim 0 is V? proj_g is (2*V_g, B, D)
                
                # Deconstruct emb_l
                V_l = c_local.shape[1]
                N_l = B * V_l
                cam_emb_l = emb_l[:N_l]
                range_emb_l = emb_l[N_l:]
                
                cam_proj_l = proj_l[:V_l]
                range_proj_l = proj_l[V_l:]
                
                # Merge
                cam_emb = torch.cat([cam_emb_g, cam_emb_l], dim=0)
                range_emb = torch.cat([range_emb_g, range_emb_l], dim=0)
                all_emb = torch.cat([cam_emb, range_emb], dim=0)
                
                cam_proj = torch.cat([cam_proj_g, cam_proj_l], dim=0)
                range_proj = torch.cat([range_proj_g, range_proj_l], dim=0)
                proj = torch.cat([cam_proj, range_proj], dim=0)
                
                return all_emb, proj
            else:
                return emb_g, proj_g
        else:
            return self._forward_batch(cam_views, range_views)
    
    def forward_camera_only(self, cam_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with camera only - range patches are zeroed."""
        B, V, C, H, W = cam_views.shape if not isinstance(cam_views, dict) else cam_views['global'].shape
        # Create zero tensor matching the aligned_mode decision in __init__
        range_channels = self.range_patch_embed.in_channels
        
        if isinstance(cam_views, dict):
            zero_range = {}
            for k, v in cam_views.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    b, v_dim, _, h, w = v.shape
                    zero_range[k] = torch.zeros(b, v_dim, range_channels, h, w, device=v.device, dtype=v.dtype)
            return self.forward(cam_views, zero_range)
        else:
             zero_range = torch.zeros(B, V, range_channels, H, W, device=cam_views.device, dtype=cam_views.dtype)
             return self.forward(cam_views, zero_range)
    
    def forward_range_only(self, range_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with range only - camera patches are zeroed."""
        B, V, C, H, W = range_views.shape if not isinstance(range_views, dict) else range_views['global'].shape
        
        if isinstance(range_views, dict):
             zero_cam = {}
             for k, v in range_views.items():
                 if isinstance(v, torch.Tensor) and v.dim() == 5:
                     b, v_dim, _, h, w = v.shape
                     zero_cam[k] = torch.zeros(b, v_dim, 3, h, w, device=v.device, dtype=v.dtype)
             return self.forward(zero_cam, range_views)
        else:
             zero_cam = torch.zeros(B, V, 3, H, W, device=range_views.device, dtype=range_views.dtype)
             return self.forward(zero_cam, range_views)


class MMEncoderD(nn.Module):
    """
    Option D: RGBD Encoder - Single ViT with aligned depth channel.
    
    Unlike Options B/C which use range images (panoramic, not aligned),
    Option D projects LiDAR INTO the camera view, creating RGBD input:
    
    Architecture:
        RGB (3×H×W) + Projected Depth (1×H×W) → RGBD (4×H×W)
        RGBD → ViT-S/16 with 4-channel patch embed → embedding
    
    Key insight: Depth is SPATIALLY ALIGNED with RGB - each pixel's depth
    corresponds to the same 3D location as the RGB pixel.
    
    This is what the lidar_camera_overlay.ipynb does!
    """
    
    def __init__(
        self, 
        proj_dim: int = 128, 
        img_size: int = 224,
        embed_dim: int = 512,
        vit_size: str = 'small',
    ):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        embed_dim = vit_cfg['embed_dim']
        
        # ViT backbone with 4 input channels (RGBD)
        self.backbone = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            in_chans=4,  # RGB + Depth
            dynamic_img_size=True,
        )
        
        # Projection head
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
    
    def forward(
        self, 
        rgbd_views: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgbd_views: (B, V, 4, H, W) RGBD images OR dict {'global': ..., 'local': ...}
            
        Returns:
            emb: (B*V, embed_dim) embeddings
            proj: (V, B, proj_dim) projections for SIGReg
        """
        if isinstance(rgbd_views, dict):
            # Global
            g_views = rgbd_views.get('global')
            B, V_g, C, H_g, W_g = g_views.shape
            gx = g_views.flatten(0, 1)
            emb_g = self.backbone(gx)
            
            # Local
            l_views = rgbd_views.get('local')
            if l_views is not None and l_views.numel() > 0:
                lx = l_views.flatten(0, 1)
                emb_l = self.backbone(lx)
                
                # Project separately
                proj_g = self.proj(emb_g).reshape(B, V_g, -1).transpose(0, 1)
                V_l = l_views.shape[1]
                proj_l = self.proj(emb_l).reshape(B, V_l, -1).transpose(0, 1)
                
                emb = torch.cat([emb_g, emb_l], dim=0)
                proj = torch.cat([proj_g, proj_l], dim=0)
                V = V_g + V_l
            else:
                emb = emb_g
                proj = self.proj(emb_g).reshape(B, V_g, -1).transpose(0, 1)
                V = V_g
        else:
            B, V, C, H, W = rgbd_views.shape
            x = rgbd_views.flatten(0, 1)  # (B*V, 4, H, W)
            emb = self.backbone(x)  # (B*V, embed_dim)
            # Project
            proj = self.proj(emb)  # (B*V, proj_dim)
            proj = proj.reshape(B, V, -1).transpose(0, 1)  # (V, B, proj_dim)
        
        return emb, proj
    
    def _forward_with_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through backbone returning both CLS embedding and patch embeddings.
        
        Args:
            x: (B, 4, H, W) RGBD input
            
        Returns:
            cls_emb: (B, embed_dim) CLS token embedding after head
            patch_emb: (B, N_patches, vit_dim) Patch embeddings before head
        """
        # Get patch embeddings
        x = self.backbone.patch_embed(x)
        if x.dim() == 4:
            if x.shape[-1] == self.backbone.embed_dim:
                x = x.flatten(1, 2)
            else:
                x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Handle positional embedding interpolation
        pos_embed = self.backbone.pos_embed
        if pos_embed.shape[1] != x.shape[1]:
            pos_embed_cls = pos_embed[:, :1]
            pos_embed_patches = pos_embed[:, 1:]
            N_orig = pos_embed_patches.shape[1]
            H_orig = int(N_orig**0.5)
            N_curr = x.shape[1] - 1
            H_curr = int(N_curr**0.5)
            pe = pos_embed_patches.reshape(1, H_orig, H_orig, -1).permute(0, 3, 1, 2)
            pe = torch.nn.functional.interpolate(pe, size=(H_curr, H_curr), mode='bicubic', align_corners=False)
            pe = pe.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = torch.cat([pos_embed_cls, pe], dim=1)
        
        x = x + pos_embed
        x = self.backbone.pos_drop(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        
        cls_emb = self.backbone.head(x[:, 0])  # (B, embed_dim)
        patch_emb = x[:, 1:]  # (B, N_patches, vit_dim)
        
        return cls_emb, patch_emb
    
    def forward_with_patches(
        self, 
        rgbd_views: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward that also returns patch embeddings for patch-level alignment.
        
        For RGBD, we split the patch embeddings into RGB-derived and D-derived parts
        by separating the first 3 channels from the 4th channel contribution.
        Since RGBD uses a single encoder, we return the same patches for both
        but split the input to get RGB-only and D-only patch representations.
        
        Args:
            rgbd_views: (B, V, 4, H, W) RGBD images
            
        Returns:
            emb: (B*V, embed_dim) CLS embeddings
            proj: (V, B, proj_dim) projections for SIGReg
            (rgb_patches, depth_patches): Tuple of patch embeddings from RGB-only and D-only forward passes
        """
        if isinstance(rgbd_views, dict):
            raise NotImplementedError("forward_with_patches doesn't support dict inputs yet")
        
        B, V, C, H, W = rgbd_views.shape
        x = rgbd_views.flatten(0, 1)  # (B*V, 4, H, W)
        
        # Full RGBD forward for CLS and proj
        emb, patch_emb = self._forward_with_patches(x)
        proj = self.proj(emb).reshape(B, V, -1).transpose(0, 1)
        
        # For patch alignment, we need separate RGB and D patch representations
        # Create RGB-only input (zero depth) and D-only input (zero RGB)
        rgb_only = x.clone()
        rgb_only[:, 3:] = 0  # Zero out depth channel
        
        depth_only = x.clone()
        depth_only[:, :3] = 0  # Zero out RGB channels
        
        _, rgb_patches = self._forward_with_patches(rgb_only)
        _, depth_patches = self._forward_with_patches(depth_only)
        
        return emb, proj, (rgb_patches, depth_patches)




class MMEncoderE(nn.Module):
    """
    Option E: Separate ViT encoders for Camera + Aligned Depth.
    
    Like Architecture A (separate encoders, late fusion), but replaces PointMLP
    with a vanilla ViT for aligned depth images.
    
    Architecture:
        Camera RGB (3ch) → ViT-S/16 → 512-dim ──┐
                                                 ├──→ Shared MLP → proj_dim
        Aligned Depth (1ch) → ViT-S/16 → 512-dim ┘
    
    Key features:
    - Both encoders are ViTs (same architecture, different weights)
    - Depth is camera-aligned (spatially corresponds to RGB)
    - Multi-crop augmentation applies to both modalities
    """
    
    def __init__(self, proj_dim: int = 128, img_size: int = 224, embed_dim: int = 512, vit_size: str = 'small'):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        embed_dim = vit_cfg['embed_dim']
        
        # Camera encoder (3-channel RGB)
        self.cam_encoder = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            in_chans=3,
            dynamic_img_size=True,
        )
        
        # Depth encoder (1-channel aligned depth)
        self.depth_encoder = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            in_chans=1,
            dynamic_img_size=True,
        )
        
        # Shared projection head
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
    
    def forward(
        self, 
        cam_views: torch.Tensor, 
        depth_views: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        
        # --- Camera Path ---
        if isinstance(cam_views, dict):
            # Global
            cg = cam_views.get('global')
            B, V_g, _, H_g, W_g = cg.shape
            cg_flat = cg.flatten(0, 1)
            emb_cg = self.cam_encoder(cg_flat)
            
            # Local
            cl = cam_views.get('local')
            if cl is not None and cl.numel() > 0:
                cl_flat = cl.flatten(0, 1)
                emb_cl = self.cam_encoder(cl_flat)
                cam_emb = torch.cat([emb_cg, emb_cl], dim=0)
                V = V_g + cl.shape[1]
                
                # Project
                cg_proj = self.proj(emb_cg).reshape(B, V_g, -1).transpose(0, 1)
                V_l = cl.shape[1]
                cl_proj = self.proj(emb_cl).reshape(B, V_l, -1).transpose(0, 1)
                cam_proj = torch.cat([cg_proj, cl_proj], dim=0)
            else:
                cam_emb = emb_cg
                V = V_g
                cam_proj = self.proj(emb_cg).reshape(B, V_g, -1).transpose(0, 1)
        else:
            B, V, C, H, W = cam_views.shape
            cam_flat = cam_views.flatten(0, 1)
            cam_emb = self.cam_encoder(cam_flat)
            cam_proj = self.proj(cam_emb).reshape(B, V, -1).transpose(0, 1)
            
        
        # --- Depth Path ---
        if isinstance(depth_views, dict):
            # Global
            dg = depth_views.get('global')
            dg_flat = dg.flatten(0, 1)
            emb_dg = self.depth_encoder(dg_flat)
            
            # Local
            dl = depth_views.get('local')
            if dl is not None and dl.numel() > 0:
                dl_flat = dl.flatten(0, 1)
                emb_dl = self.depth_encoder(dl_flat)
                depth_emb = torch.cat([emb_dg, emb_dl], dim=0)
                
                # Project separate parts
                # Re-use V, V_g from cam block or depth block (assumed sync)
                dg_proj = self.proj(emb_dg).reshape(B, V_g, -1).transpose(0, 1)
                V_dl = dl.shape[1]
                dl_proj = self.proj(emb_dl).reshape(B, V_dl, -1).transpose(0, 1)
                depth_proj = torch.cat([dg_proj, dl_proj], dim=0)
            else:
                depth_emb = emb_dg
                depth_proj = self.proj(emb_dg).reshape(B, V_g, -1).transpose(0, 1)
        else:
            depth_flat = depth_views.flatten(0, 1)
            depth_emb = self.depth_encoder(depth_flat)
            depth_proj = self.proj(depth_emb).reshape(B, V, -1).transpose(0, 1)
            
        return (cam_emb, depth_emb), (cam_proj, depth_proj)
    
    def _forward_encoder_with_patches(self, encoder: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through a ViT encoder returning both CLS embedding and patch embeddings.
        
        Returns:
            cls_emb: (B, embed_dim) CLS token embedding after head
            patch_emb: (B, N_patches, vit_dim) Patch embeddings before head
        """
        # Get patch embeddings
        x = encoder.patch_embed(x)
        if x.dim() == 4:
            if x.shape[-1] == encoder.embed_dim:
                x = x.flatten(1, 2)
            else:
                x = x.flatten(2).transpose(1, 2)
        
        cls_token = encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Handle positional embedding interpolation
        pos_embed = encoder.pos_embed
        if pos_embed.shape[1] != x.shape[1]:
            pos_embed_cls = pos_embed[:, :1]
            pos_embed_patches = pos_embed[:, 1:]
            N_orig = pos_embed_patches.shape[1]
            H_orig = int(N_orig**0.5)
            N_curr = x.shape[1] - 1
            H_curr = int(N_curr**0.5)
            pe = pos_embed_patches.reshape(1, H_orig, H_orig, -1).permute(0, 3, 1, 2)
            pe = torch.nn.functional.interpolate(pe, size=(H_curr, H_curr), mode='bicubic', align_corners=False)
            pe = pe.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = torch.cat([pos_embed_cls, pe], dim=1)
        
        x = x + pos_embed
        x = encoder.pos_drop(x)
        x = encoder.blocks(x)
        x = encoder.norm(x)
        
        cls_emb = encoder.head(x[:, 0])  # (B, embed_dim)
        patch_emb = x[:, 1:]  # (B, N_patches, vit_dim)
        
        return cls_emb, patch_emb
    
    def forward_with_patches(
        self, 
        cam_views: torch.Tensor, 
        depth_views: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward that also returns patch embeddings for patch-level alignment.
        
        Returns:
            (cam_emb, depth_emb): CLS token embeddings
            (cam_proj, depth_proj): Projections for SIGReg
            (cam_patches, depth_patches): Patch embeddings (B*V, N_patches, vit_dim)
        """
        if isinstance(cam_views, dict):
            raise NotImplementedError("forward_with_patches doesn't support dict inputs yet")
        
        B, V, C, H, W = cam_views.shape
        cam_flat = cam_views.flatten(0, 1)
        depth_flat = depth_views.flatten(0, 1)
        
        cam_emb, cam_patches = self._forward_encoder_with_patches(self.cam_encoder, cam_flat)
        depth_emb, depth_patches = self._forward_encoder_with_patches(self.depth_encoder, depth_flat)
        
        cam_proj = self.proj(cam_emb).reshape(B, V, -1).transpose(0, 1)
        depth_proj = self.proj(depth_emb).reshape(B, V, -1).transpose(0, 1)
        
        return (cam_emb, depth_emb), (cam_proj, depth_proj), (cam_patches, depth_patches)




class MMEncoderF(nn.Module):
    """
    Option F: Separate encoders + Separate projectors + Cross-modal alignment.
    
    Unlike E which uses a shared projector, F has separate projectors per modality
    and explicitly aligns cross-modal representations.
    
    Architecture:
        Camera RGB (3ch) → ViT-S/16 → 512-dim → MLP_cam → proj_dim
        Aligned Depth (1ch) → ViT-S/16 → 512-dim → MLP_depth → proj_dim
    
    Loss structure:
        1. Intra-modal SIGReg on cam projections (RGB↔RGB views)
        2. Intra-modal SIGReg on depth projections (depth↔depth views)
        3. Cross-modal SIGReg aligning (RGB, depth) pairs
    
    Key features:
    - Each modality has its own projector (no shared weights)
    - Cross-modal alignment via paired SIGReg application
    - Supports learning modality-specific features while maintaining alignment
    """
    
    def __init__(self, proj_dim: int = 128, img_size: int = 224, embed_dim: int = 512, vit_size: str = 'small'):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        embed_dim = vit_cfg['embed_dim']
        
        # Camera encoder (3-channel RGB)
        self.cam_encoder = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            in_chans=3,
            dynamic_img_size=True,
        )
        
        # Depth encoder (1-channel aligned depth)
        self.depth_encoder = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            in_chans=1,
            dynamic_img_size=True,
        )
        
        # SEPARATE projection heads (not shared!)
        self.cam_proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        self.depth_proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
    
    def forward(
        self, 
        cam_views: torch.Tensor, 
        depth_views: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        
        # --- Camera Path ---
        if isinstance(cam_views, dict):
            # Global
            cg = cam_views.get('global')
            B, V_g, _, H_g, W_g = cg.shape
            cg_flat = cg.flatten(0, 1)
            emb_cg = self.cam_encoder(cg_flat)
            
            # Local
            cl = cam_views.get('local')
            if cl is not None and cl.numel() > 0:
                cl_flat = cl.flatten(0, 1)
                emb_cl = self.cam_encoder(cl_flat)
                cam_emb = torch.cat([emb_cg, emb_cl], dim=0)
                V = V_g + cl.shape[1]
            else:
                cam_emb = emb_cg
                V = V_g
        else:
            B, V, C, H, W = cam_views.shape
            cam_flat = cam_views.flatten(0, 1)
            cam_emb = self.cam_encoder(cam_flat)
        
        if isinstance(cam_views, dict) and cam_views.get('local') is not None and cam_views.get('local').numel() > 0:
             # Split embeddings
             cg_size = cam_views['global'].flatten(0, 1).shape[0]
             cg_emb = cam_emb[:cg_size]
             cl_emb = cam_emb[cg_size:]
             
             # Project separate
             # Global
             proj_cg = self.cam_proj(cg_emb)
             V_g = cam_views['global'].shape[1]
             cam_proj_g = proj_cg.reshape(B, V_g, -1).transpose(0, 1)
             
             # Local
             proj_cl = self.cam_proj(cl_emb)
             V_l = cam_views['local'].shape[1]
             cam_proj_l = proj_cl.reshape(B, V_l, -1).transpose(0, 1)
             
             cam_proj_views = torch.cat([cam_proj_g, cam_proj_l], dim=0)
        else:
             cam_proj_out = self.cam_proj(cam_emb)
             cam_proj_views = cam_proj_out.reshape(B, V, -1).transpose(0, 1)
        
        # --- Depth Path ---
        if isinstance(depth_views, dict):
            # Global
            dg = depth_views.get('global')
            dg_flat = dg.flatten(0, 1)
            emb_dg = self.depth_encoder(dg_flat)
            
            # Local
            dl = depth_views.get('local')
            if dl is not None and dl.numel() > 0:
                dl_flat = dl.flatten(0, 1)
                emb_dl = self.depth_encoder(dl_flat)
                depth_emb = torch.cat([emb_dg, emb_dl], dim=0)
            else:
                depth_emb = emb_dg
        else:
            depth_flat = depth_views.flatten(0, 1)
            depth_emb = self.depth_encoder(depth_flat)
            
        if isinstance(depth_views, dict) and depth_views.get('local') is not None and depth_views.get('local').numel() > 0:
             dg_size = depth_views['global'].flatten(0, 1).shape[0]
             dg_emb = depth_emb[:dg_size]
             dl_emb = depth_emb[dg_size:]
             
             proj_dg = self.depth_proj(dg_emb)
             V_g = depth_views['global'].shape[1]
             depth_proj_g = proj_dg.reshape(B, V_g, -1).transpose(0, 1)
             
             proj_dl = self.depth_proj(dl_emb)
             V_l = depth_views['local'].shape[1]
             depth_proj_l = proj_dl.reshape(B, V_l, -1).transpose(0, 1)
             
             depth_proj_views = torch.cat([depth_proj_g, depth_proj_l], dim=0)
        else:
             depth_proj_out = self.depth_proj(depth_emb)
             depth_proj_views = depth_proj_out.reshape(B, V, -1).transpose(0, 1)
        
        # Cross-modal projection: stack mean RGB + mean depth for cross-modal SIGReg
        # Use view-averaged projections for cross-modal alignment
        cam_mean = cam_proj_views.mean(0)  # (B, proj_dim)
        depth_mean = depth_proj_views.mean(0)  # (B, proj_dim)
        cross_proj = torch.stack([cam_mean, depth_mean], dim=0)  # (2, B, proj_dim)
        
        return (cam_emb, depth_emb), (cam_proj_views, depth_proj_views, cross_proj)
    
    def _forward_encoder_with_patches(self, encoder: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through a ViT encoder returning both CLS embedding and patch embeddings.
        
        Returns:
            cls_emb: (B, embed_dim) CLS token embedding after head
            patch_emb: (B, N_patches, vit_dim) Patch embeddings before head
        """
        # Get patch embeddings
        x = encoder.patch_embed(x)
        if x.dim() == 4:
            if x.shape[-1] == encoder.embed_dim:
                x = x.flatten(1, 2)
            else:
                x = x.flatten(2).transpose(1, 2)
        
        cls_token = encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Handle positional embedding interpolation
        pos_embed = encoder.pos_embed
        if pos_embed.shape[1] != x.shape[1]:
            pos_embed_cls = pos_embed[:, :1]
            pos_embed_patches = pos_embed[:, 1:]
            N_orig = pos_embed_patches.shape[1]
            H_orig = int(N_orig**0.5)
            N_curr = x.shape[1] - 1
            H_curr = int(N_curr**0.5)
            pe = pos_embed_patches.reshape(1, H_orig, H_orig, -1).permute(0, 3, 1, 2)
            pe = torch.nn.functional.interpolate(pe, size=(H_curr, H_curr), mode='bicubic', align_corners=False)
            pe = pe.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = torch.cat([pos_embed_cls, pe], dim=1)
        
        x = x + pos_embed
        x = encoder.pos_drop(x)
        x = encoder.blocks(x)
        x = encoder.norm(x)
        
        cls_emb = encoder.head(x[:, 0])  # (B, embed_dim)
        patch_emb = x[:, 1:]  # (B, N_patches, vit_dim)
        
        return cls_emb, patch_emb
    
    def forward_with_patches(
        self, 
        cam_views: torch.Tensor, 
        depth_views: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward that also returns patch embeddings for patch-level alignment.
        
        Returns:
            (cam_emb, depth_emb): CLS token embeddings
            (cam_proj, depth_proj, cross_proj): Projections for SIGReg
            (cam_patches, depth_patches): Patch embeddings (B*V, N_patches, vit_dim)
        """
        if isinstance(cam_views, dict):
            raise NotImplementedError("forward_with_patches doesn't support dict inputs yet")
        
        B, V, C, H, W = cam_views.shape
        cam_flat = cam_views.flatten(0, 1)
        depth_flat = depth_views.flatten(0, 1)
        
        cam_emb, cam_patches = self._forward_encoder_with_patches(self.cam_encoder, cam_flat)
        depth_emb, depth_patches = self._forward_encoder_with_patches(self.depth_encoder, depth_flat)
        
        # Projections
        cam_proj_views = self.cam_proj(cam_emb).reshape(B, V, -1).transpose(0, 1)
        depth_proj_views = self.depth_proj(depth_emb).reshape(B, V, -1).transpose(0, 1)
        
        # Cross-modal projection
        cam_mean = cam_proj_views.mean(0)
        depth_mean = depth_proj_views.mean(0)
        cross_proj = torch.stack([cam_mean, depth_mean], dim=0)
        
        return (cam_emb, depth_emb), (cam_proj_views, depth_proj_views, cross_proj), (cam_patches, depth_patches)


class MMEncoderC_FusionTokens(nn.Module):
    """
    Fusion Tokens Encoder: Each spatial position gets a dedicated fusion token
    that absorbs information from the corresponding RGB + LiDAR patch pair.
    
    Architecture (per-layer attention rules):
        Layer 0:
            - Fusion token[i] attends ONLY to RGB patch[i] + LiDAR patch[i]
            - RGB/LiDAR patches attend to each other normally (full self-attention)
        Layer 1+:
            - Fusion tokens attend ONLY to other fusion tokens + CLS
            - CLS attends ONLY to fusion tokens
            - RGB/LiDAR patches are excluded from attention
    
    Result: Fusion tokens contain merged RGB+LiDAR spatial information.
             CLS aggregates over all fusion tokens.
    """
    
    def __init__(
        self,
        proj_dim: int = 128,
        img_size: int = 224,
        range_channels: int = 5,
        embed_dim: int = 512,
        aligned_mode: bool = False,
        vit_size: str = 'small',
        attention_mode: str = 'prune_after_first',
        fusion_start_layer: int = 0,
    ):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        embed_dim = vit_cfg['embed_dim']
        self.embed_dim = embed_dim
        self.aligned_mode = aligned_mode
        self.attention_mode = str(attention_mode).lower()
        self.fusion_start_layer = max(0, int(fusion_start_layer))
        valid_modes = {'prune_after_first', 'persistent_pair', 'full'}
        if self.attention_mode not in valid_modes:
            print(f"⚠️ Unknown fusion attention_mode='{attention_mode}', falling back to 'prune_after_first'")
            self.attention_mode = 'prune_after_first'
        
        # Backbone ViT (we'll use its blocks individually)
        self.backbone = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            dynamic_img_size=True,
        )
        
        self.vit_embed_dim = self.backbone.embed_dim  # 384 for ViT-S
        patch_size = 16
        self.n_patches = (img_size // patch_size) ** 2  # 196 for 224x224
        
        # Separate patch embeddings for RGB and LiDAR
        self.cam_patch_embed = self.backbone.patch_embed
        lidar_channels = 1 if aligned_mode else range_channels
        self.range_patch_embed = nn.Conv2d(
            lidar_channels, self.vit_embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        
        # Modality embeddings
        self.cam_modality_embed = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        self.range_modality_embed = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        nn.init.normal_(self.cam_modality_embed, std=0.02)
        nn.init.normal_(self.range_modality_embed, std=0.02)
        
        # Fusion tokens: one per spatial patch position
        self.fusion_tokens = nn.Parameter(torch.zeros(1, self.n_patches, self.vit_embed_dim))
        nn.init.trunc_normal_(self.fusion_tokens, std=0.02)
        
        # CLS token (attends only to fusion tokens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Positional embeddings
        # Token layout: [CLS, fusion_0..N-1, cam_0..N-1, range_0..N-1]
        # Total: 1 + N + 2N = 1 + 3N tokens
        total_tokens = 1 + 3 * self.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, self.vit_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Head and projection
        self.head = nn.Linear(self.vit_embed_dim, embed_dim)
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
    
    def _build_attention_masks(self, N_patches: int, device: torch.device, dtype: torch.dtype):
        """
        Build attention masks for layer 0 and layers 1+.
        
        Token layout: [CLS(1), Fusion(N), Cam(N), Range(N)]
        Total = 1 + 3N
        """
        total = 1 + 3 * N_patches
        cls_idx = 0
        fusion_start = 1
        fusion_end = 1 + N_patches
        cam_start = fusion_end
        cam_end = cam_start + N_patches
        range_start = cam_end
        range_end = range_start + N_patches
        
        # === Layer 0 mask ===
        # Fusion[i] attends to cam[i] and range[i] ONLY
        # CLS, cam, range tokens attend to everything (full self-attention among themselves)
        mask_layer0 = torch.zeros(total, total, device=device, dtype=dtype)
        
        # CLS attends to everything
        mask_layer0[cls_idx, :] = 0
        # Everything attends to CLS  
        mask_layer0[:, cls_idx] = 0
        
        # Cam patches: full self-attention among cam + range patches
        mask_layer0[cam_start:cam_end, cam_start:range_end] = 0
        mask_layer0[range_start:range_end, cam_start:range_end] = 0
        
        # Fusion tokens: attend ONLY to their paired cam[i] + range[i]
        # Block everything for fusion tokens first
        mask_layer0[fusion_start:fusion_end, :] = float('-inf')
        for i in range(N_patches):
            fi = fusion_start + i
            ci = cam_start + i
            ri = range_start + i
            mask_layer0[fi, fi] = 0     # self
            mask_layer0[fi, ci] = 0     # attend to cam[i]
            mask_layer0[fi, ri] = 0     # attend to range[i]
        
        # Nobody attends to fusion tokens in layer 0 (except themselves)
        mask_layer0[:, fusion_start:fusion_end] = float('-inf')
        # Fusion self-attention is each token attending to itself only
        for i in range(N_patches):
            fi = fusion_start + i
            mask_layer0[fi, fi] = 0
        
        # === Layer 1+ mask ===
        mask_later = torch.full((total, total), float('-inf'), device=device, dtype=dtype)

        # CLS attends to fusion tokens + self
        mask_later[cls_idx, cls_idx] = 0
        mask_later[cls_idx, fusion_start:fusion_end] = 0

        # Fusion tokens attend to all fusion tokens + CLS
        mask_later[fusion_start:fusion_end, cls_idx] = 0
        mask_later[fusion_start:fusion_end, fusion_start:fusion_end] = 0

        # Cam/Range patches keep self-attention among themselves
        mask_later[cam_start:range_end, cam_start:range_end] = 0

        # Variant: persistent_pair keeps fusion-to-corresponding-patch links after layer 0
        if self.attention_mode == 'persistent_pair':
            for i in range(N_patches):
                fi = fusion_start + i
                ci = cam_start + i
                ri = range_start + i
                mask_later[fi, ci] = 0
                mask_later[fi, ri] = 0
        
        return mask_layer0, mask_later

    def _apply_block_with_mask(self, block, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply one timm ViT block with an additive attention mask."""
        residual = x
        x_norm = block.norm1(x)

        attn = block.attn
        B_attn = x_norm.shape[0]
        qkv = attn.qkv(x_norm).reshape(B_attn, x_norm.shape[1], 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask.unsqueeze(0).unsqueeze(0),
            dropout_p=attn.attn_drop.p if attn.training else 0.0,
            scale=getattr(attn, 'scale', None),
        ).transpose(1, 2).reshape(B_attn, x_norm.shape[1], -1)
        attn_out = attn.proj(attn_out)
        attn_out = attn.proj_drop(attn_out)

        x = residual + block.drop_path1(block.ls1(attn_out))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x

    def _project_qkv(self, attn, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project tokens into per-head Q/K/V tensors."""
        qkv = attn.qkv(x).reshape(x.shape[0], x.shape[1], 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
        return qkv.unbind(0)

    def _attention_from_qkv(self, attn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute dense attention from already projected Q/K/V tensors."""
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=attn.attn_drop.p if attn.training else 0.0,
            scale=getattr(attn, 'scale', None),
        ).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    def _apply_first_fusion_block_sparse(self, block, x: torch.Tensor, n_patches: int) -> torch.Tensor:
        """Apply the first fusion block without materializing a dense masked attention matrix.

        Layer-0 connectivity is sparse:
        - CLS, camera, and range tokens attend only within [CLS, cam, range]
        - fusion_i attends only to [fusion_i, cam_i, range_i]
        """
        residual = x
        x_norm = block.norm1(x)

        cls = x_norm[:, :1]
        fusion = x_norm[:, 1:1 + n_patches]
        cam = x_norm[:, 1 + n_patches:1 + 2 * n_patches]
        range_tokens = x_norm[:, 1 + 2 * n_patches:1 + 3 * n_patches]

        attn = block.attn

        base_tokens = torch.cat([cls, cam, range_tokens], dim=1)
        q_base, k_base, v_base = self._project_qkv(attn, base_tokens)
        base_out = self._attention_from_qkv(attn, q_base, k_base, v_base)

        q_fusion, k_fusion, v_fusion = self._project_qkv(attn, fusion)
        k_cam = k_base[:, :, 1:1 + n_patches, :]
        v_cam = v_base[:, :, 1:1 + n_patches, :]
        k_range = k_base[:, :, 1 + n_patches:1 + 2 * n_patches, :]
        v_range = v_base[:, :, 1 + n_patches:1 + 2 * n_patches, :]

        fusion_keys = torch.stack([k_fusion, k_cam, k_range], dim=3)
        fusion_values = torch.stack([v_fusion, v_cam, v_range], dim=3)

        scale = q_fusion.shape[-1] ** -0.5
        fusion_weights = torch.einsum('bhnd,bhnkd->bhnk', q_fusion.float(), fusion_keys.float()) * scale
        fusion_weights = fusion_weights.softmax(dim=-1)
        fusion_weights = attn.attn_drop(fusion_weights).to(dtype=fusion_values.dtype)
        fusion_out = torch.einsum('bhnk,bhnkd->bhnd', fusion_weights, fusion_values)
        fusion_out = fusion_out.transpose(1, 2).reshape(q_fusion.shape[0], q_fusion.shape[2], -1)

        attn_out = torch.cat(
            [
                base_out[:, :1],
                fusion_out,
                base_out[:, 1:1 + n_patches],
                base_out[:, 1 + n_patches:1 + 2 * n_patches],
            ],
            dim=1,
        )
        attn_out = attn.proj(attn_out)
        attn_out = attn.proj_drop(attn_out)

        x = residual + block.drop_path1(block.ls1(attn_out))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x

    def _interp_token_grid(self, tokens: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """Interpolate a learned token grid to the current patch grid."""
        target_h, target_w = target_hw
        target_tokens = target_h * target_w
        if tokens.shape[1] == target_tokens:
            return tokens.float()

        n_orig = tokens.shape[1]
        h_orig = int(round(n_orig ** 0.5))
        if h_orig * h_orig != n_orig:
            raise RuntimeError(f"Cannot reshape token grid with {n_orig} tokens into a square base grid")

        grid = tokens.reshape(1, h_orig, h_orig, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(target_h, target_w), mode='bicubic', align_corners=False)
        return grid.permute(0, 2, 3, 1).reshape(1, target_tokens, -1)

    def _get_positional_embeddings(
        self,
        grid_hw: Tuple[int, int],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return interpolated positional embeddings for CLS, fusion, camera, and range tokens."""
        total_tokens = 1 + 3 * (grid_hw[0] * grid_hw[1])
        if total_tokens == self.pos_embed.shape[1]:
            p_cls = self.pos_embed[:, :1].float()
            p_fusion = self.pos_embed[:, 1 : 1 + grid_hw[0] * grid_hw[1]].float()
            p_cam = self.pos_embed[:, 1 + grid_hw[0] * grid_hw[1] : 1 + 2 * grid_hw[0] * grid_hw[1]].float()
            p_range = self.pos_embed[:, 1 + 2 * grid_hw[0] * grid_hw[1] :].float()
        else:
            n_patches_orig = (self.pos_embed.shape[1] - 1) // 3
            p_cls = self.pos_embed[:, :1].float()
            p_fusion = self.pos_embed[:, 1 : 1 + n_patches_orig].float()
            p_cam = self.pos_embed[:, 1 + n_patches_orig : 1 + 2 * n_patches_orig].float()
            p_range = self.pos_embed[:, 1 + 2 * n_patches_orig :].float()

            p_fusion = self._interp_token_grid(p_fusion, grid_hw)
            p_cam = self._interp_token_grid(p_cam, grid_hw)
            p_range = self._interp_token_grid(p_range, grid_hw)

        return (
            p_cls.to(dtype=dtype),
            p_fusion.to(dtype=dtype),
            p_cam.to(dtype=dtype),
            p_range.to(dtype=dtype),
        )
    
    def _forward_batch(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            cls_emb: (N, embed_dim) CLS token embedding
            proj: projections for SigReg
            fusion_emb: (N, N_patches, vit_embed_dim) fusion token outputs
        """
        # This custom sparse-attention path becomes numerically unstable under
        # CUDA autocast for some pretrained initializations (notably DINOv3).
        # Run it in fp32 even inside mixed-precision training.
        autocast_ctx = (
            torch.autocast(device_type='cuda', enabled=False)
            if cam_views.is_cuda and torch.is_autocast_enabled()
            else nullcontext()
        )
        with autocast_ctx:
            B, V, C, H, W = cam_views.shape
            cam_flat = cam_views.flatten(0, 1).float()
            range_flat = range_views.flatten(0, 1).float()
            N = cam_flat.shape[0]

            # Patch embeddings
            cam_patches = self.cam_patch_embed(cam_flat)
            if cam_patches.dim() == 4:
                cam_grid_h, cam_grid_w = cam_patches.shape[1], cam_patches.shape[2]
                cam_patches = cam_patches.flatten(1, 2)
            else:
                n_cam = cam_patches.shape[1]
                cam_grid_h = int(round(n_cam ** 0.5))
                if cam_grid_h * cam_grid_h != n_cam:
                    raise RuntimeError(f"Cannot infer square RGB patch grid from {n_cam} tokens")
                cam_grid_w = cam_grid_h

            range_patches = self.range_patch_embed(range_flat)
            range_grid_h, range_grid_w = range_patches.shape[2], range_patches.shape[3]
            range_patches = range_patches.flatten(2).transpose(1, 2)

            if (cam_grid_h, cam_grid_w) != (range_grid_h, range_grid_w):
                raise RuntimeError(
                    "Fusion token encoder requires matching RGB/LiDAR patch grids, "
                    f"got RGB {(cam_grid_h, cam_grid_w)} and LiDAR {(range_grid_h, range_grid_w)}"
                )

            N_patches = cam_patches.shape[1]
            grid_hw = (cam_grid_h, cam_grid_w)

            # Expand learnable tokens
            model_dtype = cam_patches.dtype
            cls = self.cls_token.to(dtype=model_dtype).expand(N, -1, -1)
            fusion = self._interp_token_grid(self.fusion_tokens, grid_hw).to(dtype=model_dtype).expand(N, -1, -1)

            p_cls, p_fusion, p_cam, p_range = self._get_positional_embeddings(
                grid_hw, dtype=cam_patches.dtype
            )

            # Build attention masks
            mask_layer0, mask_later = self._build_attention_masks(
                N_patches, device=cam_patches.device, dtype=cam_patches.dtype
            )
            mask_later = torch.isfinite(mask_later)

            blocks = self.backbone.blocks
            fusion_start_idx = 0
            if self.fusion_start_layer > 0 and len(blocks) > 1:
                fusion_start_idx = min(self.fusion_start_layer, len(blocks) - 1)

                cam_stream = self.backbone.pos_drop(cam_patches + p_cam + self.cam_modality_embed.to(dtype=model_dtype))
                range_stream = self.backbone.pos_drop(range_patches + p_range + self.range_modality_embed.to(dtype=model_dtype))

                for block_idx in range(fusion_start_idx):
                    cam_stream = blocks[block_idx](cam_stream)
                    range_stream = blocks[block_idx](range_stream)

                cam_patches = cam_stream
                range_patches = range_stream
            else:
                cam_patches = cam_patches + self.cam_modality_embed.to(dtype=model_dtype)
                range_patches = range_patches + self.range_modality_embed.to(dtype=model_dtype)

            if fusion_start_idx == 0:
                x = torch.cat([cls, fusion, cam_patches, range_patches], dim=1)
                x = x + torch.cat([p_cls, p_fusion, p_cam, p_range], dim=1)
                x = self.backbone.pos_drop(x)
            else:
                cls = self.backbone.pos_drop(cls + p_cls)
                fusion = self.backbone.pos_drop(fusion + p_fusion)
                x = torch.cat([cls, fusion, cam_patches, range_patches], dim=1)

            if self.attention_mode == 'full':
                for block in blocks[fusion_start_idx:]:
                    x = block(x)
            elif self.attention_mode == 'prune_after_first':
                if fusion_start_idx < len(blocks):
                    x = self._apply_first_fusion_block_sparse(blocks[fusion_start_idx], x, N_patches)

                fusion_start = 1
                fusion_end = 1 + N_patches
                x = torch.cat([x[:, :1], x[:, fusion_start:fusion_end]], dim=1)

                for block in blocks[fusion_start_idx + 1:]:
                    x = block(x)
            else:
                for offset, block in enumerate(blocks[fusion_start_idx:]):
                    if offset == 0:
                        x = self._apply_first_fusion_block_sparse(block, x, N_patches)
                    else:
                        x = self._apply_block_with_mask(block, x, mask_later)

            x = self.backbone.norm(x)

            cls_feat = x[:, 0]
            fusion_out = x[:, 1:1+N_patches]

            cls_emb = self.head(cls_feat)
            cls_proj = self.proj(cls_emb)
            proj = cls_proj.reshape(B, V, -1).transpose(0, 1)

        return cls_emb, proj, fusion_out
    
    def forward(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward returning (emb, proj).
        Uses _forward_batch to get CLS embedding and projection.
        Matches the interface of MMEncoderC.
        """
        if isinstance(cam_views, dict):
            # Global views
            c_global = cam_views.get('global')
            r_global = range_views.get('global')
            cls_emb, proj, _ = self._forward_batch(c_global, r_global)
            B = c_global.shape[0]
            V_g = c_global.shape[1]
            
            c_local = cam_views.get('local')
            r_local = range_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                cls_emb_l, proj_l, _ = self._forward_batch(c_local, r_local)
                cls_emb = torch.cat([cls_emb, cls_emb_l], dim=0)
                proj = torch.cat([proj, proj_l], dim=0)
            
            # Return format: emb has cam and range concatenated for compatibility
            # But fusion encoder only returns CLS (no separate cam/range CLS)
            # We duplicate to match expected interface: all_emb = [cam_cls; range_cls]
            all_emb = torch.cat([cls_emb, cls_emb], dim=0)
            proj_dup = torch.cat([proj, proj], dim=0)
            return all_emb, proj_dup
        else:
            cls_emb, proj, _ = self._forward_batch(cam_views, range_views)
            B = cam_views.shape[0]
            V = cam_views.shape[1]
            all_emb = torch.cat([cls_emb, cls_emb], dim=0)
            proj_dup = torch.cat([proj, proj], dim=0)
            return all_emb, proj_dup
    
    def forward_with_fusion_tokens(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward returning CLS emb, proj, and fusion token embeddings.
        
        Fusion tokens are only returned for global views (local crops have
        different spatial resolution), but CLS emb and proj include ALL views
        (global + local) to keep SigReg and invariance loss consistent with
        forward().
        """
        if isinstance(cam_views, dict):
            c_global = cam_views.get('global')
            r_global = range_views.get('global')
            cls_emb, proj, fusion_out = self._forward_batch(c_global, r_global)
            
            # Also process local views for CLS/proj (fusion tokens are global-only
            # because local crops have different spatial resolution → different
            # number of patches, so fusion tokens can't be concatenated).
            c_local = cam_views.get('local')
            r_local = range_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                cls_emb_l, proj_l, _ = self._forward_batch(c_local, r_local)
                cls_emb = torch.cat([cls_emb, cls_emb_l], dim=0)
                proj = torch.cat([proj, proj_l], dim=0)
            
            return cls_emb, proj, fusion_out
        else:
            return self._forward_batch(cam_views, range_views)

    def forward_with_patches(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compatibility wrapper used by probe extraction paths.

        Returns patch-like tokens as (cam_patches, range_patches). For this encoder,
        we expose fusion tokens as both outputs to satisfy downstream interface.
        """
        cls_emb, proj, fusion_emb = self.forward_with_fusion_tokens(cam_views, range_views)
        all_emb = torch.cat([cls_emb, cls_emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        return all_emb, proj_dup, (fusion_emb, fusion_emb)
    
    def forward_camera_only(self, cam_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with camera only — range patches are zeroed."""
        range_channels = self.range_patch_embed.in_channels
        if isinstance(cam_views, dict):
            zero_range = {}
            for k, v in cam_views.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    b, v_dim, _, h, w = v.shape
                    zero_range[k] = torch.zeros(b, v_dim, range_channels, h, w, device=v.device, dtype=v.dtype)
            return self.forward(cam_views, zero_range)
        else:
            B, V, _, H, W = cam_views.shape
            zero_range = torch.zeros(B, V, range_channels, H, W, device=cam_views.device, dtype=cam_views.dtype)
            return self.forward(cam_views, zero_range)
    
    def forward_range_only(self, range_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with range only — camera patches are zeroed."""
        if isinstance(range_views, dict):
            zero_cam = {}
            for k, v in range_views.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    b, v_dim, _, h, w = v.shape
                    zero_cam[k] = torch.zeros(b, v_dim, 3, h, w, device=v.device, dtype=v.dtype)
            return self.forward(zero_cam, range_views)
        else:
            B, V, _, H, W = range_views.shape
            zero_cam = torch.zeros(B, V, 3, H, W, device=range_views.device, dtype=range_views.dtype)
            return self.forward(zero_cam, range_views)


class _FrustumSlotBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, slots: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        slot_q = self.norm_slots(slots)
        context_kv = self.norm_context(context)
        attn_out, _ = self.attn(slot_q, context_kv, context_kv, need_weights=False)
        slots = slots + attn_out
        slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class MMEncoderC_FrustumSlots(nn.Module):
    """Aligned RGB-depth encoder that pools patch-pair frusta into learned slots."""

    def __init__(
        self,
        proj_dim: int = 128,
        img_size: int = 224,
        range_channels: int = 5,
        embed_dim: int = 512,
        aligned_mode: bool = False,
        vit_size: str = 'small',
        num_slots: int = 8,
        slot_layers: int = 3,
    ):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        embed_dim = vit_cfg['embed_dim']
        self.embed_dim = embed_dim
        self.aligned_mode = aligned_mode
        self.num_slots = num_slots

        backbone = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            dynamic_img_size=True,
        )

        self.vit_embed_dim = backbone.embed_dim
        self.cam_patch_embed = backbone.patch_embed
        self.pos_drop = backbone.pos_drop
        self.output_blocks = nn.ModuleList([copy.deepcopy(block) for block in backbone.blocks[:2]])
        self.output_norm = copy.deepcopy(backbone.norm)
        self.n_patches = (img_size // 16) ** 2

        lidar_channels = 1 if aligned_mode else range_channels
        self.range_patch_embed = nn.Conv2d(
            lidar_channels, self.vit_embed_dim,
            kernel_size=16, stride=16,
        )

        self.cam_modality_embed = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        self.range_modality_embed = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        self.frustum_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, self.vit_embed_dim))
        self.slot_tokens = nn.Parameter(torch.zeros(1, num_slots, self.vit_embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))

        nn.init.trunc_normal_(self.cam_modality_embed, std=0.02)
        nn.init.trunc_normal_(self.range_modality_embed, std=0.02)
        nn.init.trunc_normal_(self.frustum_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.slot_tokens, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        num_heads = backbone.blocks[0].attn.num_heads
        self.frustum_fuser = nn.Sequential(
            nn.LayerNorm(2 * self.vit_embed_dim),
            nn.Linear(2 * self.vit_embed_dim, self.vit_embed_dim),
            nn.GELU(),
            nn.Linear(self.vit_embed_dim, self.vit_embed_dim),
        )
        self.slot_blocks = nn.ModuleList([
            _FrustumSlotBlock(self.vit_embed_dim, num_heads=num_heads)
            for _ in range(slot_layers)
        ])
        self.cls_norm = nn.LayerNorm(self.vit_embed_dim)
        self.slot_context_norm = nn.LayerNorm(self.vit_embed_dim)
        self.cls_attn = nn.MultiheadAttention(self.vit_embed_dim, num_heads, batch_first=True)

        self.head = nn.Linear(self.vit_embed_dim, embed_dim)
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def _maybe_flatten_patches(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if x.shape[-1] == self.vit_embed_dim:
                x = x.flatten(1, 2)
            else:
                x = x.flatten(2).transpose(1, 2)
        return x

    def _interp_pos_embed(self, n_curr: int) -> torch.Tensor:
        if n_curr == self.frustum_pos_embed.shape[1]:
            return self.frustum_pos_embed
        n_orig = self.frustum_pos_embed.shape[1]
        h_orig = int(n_orig ** 0.5)
        h_curr = int(n_curr ** 0.5)
        pe = self.frustum_pos_embed.reshape(1, h_orig, h_orig, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(h_curr, h_curr), mode='bicubic', align_corners=False)
        return pe.permute(0, 2, 3, 1).flatten(1, 2)

    def _forward_batch_with_tokens(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, V, _, _, _ = cam_views.shape
        cam_flat = cam_views.flatten(0, 1)
        range_flat = range_views.flatten(0, 1)
        N = cam_flat.shape[0]

        cam_patches = self._maybe_flatten_patches(self.cam_patch_embed(cam_flat))
        range_patches = self.range_patch_embed(range_flat).flatten(2).transpose(1, 2)
        n_patches = cam_patches.shape[1]

        cam_patches = cam_patches + self.cam_modality_embed
        range_patches = range_patches + self.range_modality_embed

        frustum_tokens = self.frustum_fuser(torch.cat([cam_patches, range_patches], dim=-1))
        frustum_tokens = frustum_tokens + 0.5 * (cam_patches + range_patches)
        frustum_tokens = frustum_tokens + self._interp_pos_embed(n_patches)
        frustum_tokens = self.pos_drop(frustum_tokens)

        slots = self.slot_tokens.expand(N, -1, -1)
        for block in self.slot_blocks:
            slots = block(slots, frustum_tokens)

        cls = self.cls_token.expand(N, -1, -1)
        cls_q = self.cls_norm(cls)
        slot_ctx = self.slot_context_norm(slots)
        cls = cls + self.cls_attn(cls_q, slot_ctx, slot_ctx, need_weights=False)[0]

        joint = torch.cat([cls, slots], dim=1)
        for block in self.output_blocks:
            joint = block(joint)
        joint = self.output_norm(joint)

        cls_feat = joint[:, 0]
        slot_out = joint[:, 1:]
        cls_emb = self.head(cls_feat)
        cls_proj = self.proj(cls_emb).reshape(B, V, -1).transpose(0, 1)
        return cls_emb, cls_proj, slot_out, frustum_tokens

    def _forward_batch(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_emb, cls_proj, slot_out, _ = self._forward_batch_with_tokens(cam_views, range_views)
        return cls_emb, cls_proj, slot_out

    def forward(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(cam_views, dict):
            c_global = cam_views.get('global')
            r_global = range_views.get('global')
            cls_emb, proj, _ = self._forward_batch(c_global, r_global)
            c_local = cam_views.get('local')
            r_local = range_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                cls_emb_l, proj_l, _ = self._forward_batch(c_local, r_local)
                cls_emb = torch.cat([cls_emb, cls_emb_l], dim=0)
                proj = torch.cat([proj, proj_l], dim=0)
            all_emb = torch.cat([cls_emb, cls_emb], dim=0)
            proj_dup = torch.cat([proj, proj], dim=0)
            return all_emb, proj_dup

        cls_emb, proj, _ = self._forward_batch(cam_views, range_views)
        all_emb = torch.cat([cls_emb, cls_emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        return all_emb, proj_dup

    def forward_with_slot_tokens(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(cam_views, dict):
            c_global = cam_views.get('global')
            r_global = range_views.get('global')
            cls_emb, proj, slot_out = self._forward_batch(c_global, r_global)
            c_local = cam_views.get('local')
            r_local = range_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                cls_emb_l, proj_l, _ = self._forward_batch(c_local, r_local)
                cls_emb = torch.cat([cls_emb, cls_emb_l], dim=0)
                proj = torch.cat([proj, proj_l], dim=0)
            return cls_emb, proj, slot_out

        return self._forward_batch(cam_views, range_views)

    def forward_with_patches(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(cam_views, dict):
            c_global = cam_views.get('global')
            r_global = range_views.get('global')
            cls_emb, proj, _, patch_tokens = self._forward_batch_with_tokens(c_global, r_global)
            c_local = cam_views.get('local')
            r_local = range_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                cls_emb_l, proj_l, _, patch_tokens_l = self._forward_batch_with_tokens(c_local, r_local)
                cls_emb = torch.cat([cls_emb, cls_emb_l], dim=0)
                proj = torch.cat([proj, proj_l], dim=0)
                patch_tokens = torch.cat([patch_tokens, patch_tokens_l], dim=0)
        else:
            cls_emb, proj, _, patch_tokens = self._forward_batch_with_tokens(cam_views, range_views)

        all_emb = torch.cat([cls_emb, cls_emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        return all_emb, proj_dup, (patch_tokens, patch_tokens)


class MMEncoderC_LiDARRoPE(nn.Module):
    """
    LiDAR-conditioned RoPE Encoder: RGB-only ViT where LiDAR depth information
    is encoded through Rotary Position Embeddings applied to attention Q/K.
    
    Architecture:
        RGB → ViT patches → standard transformer
        LiDAR depth → per-patch mean depth → frequency modulation → RoPE on Q/K
    
    The depth values modulate the RoPE frequencies such that patches with similar
    depths get similar rotary embeddings, encouraging depth-aware attention patterns.
    
    To force LiDAR usage, a contrastive loss compares the RoPE-conditioned forward
    pass with a depth-zeroed (standard positional) forward pass.
    """
    
    def __init__(
        self,
        proj_dim: int = 128,
        img_size: int = 224,
        embed_dim: int = 512,
        aligned_mode: bool = True,
        vit_size: str = 'small',
        base_freq: float = 10000.0,
    ):
        super().__init__()
        vit_cfg = get_vit_config(vit_size)
        embed_dim = vit_cfg['embed_dim']
        self.embed_dim = embed_dim
        
        # Standard RGB ViT backbone
        self.backbone = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            dynamic_img_size=True,
        )
        
        self.vit_embed_dim = self.backbone.embed_dim  # 384 for ViT-S
        self.head_dim = self.vit_embed_dim // self.backbone.blocks[0].attn.num_heads
        self.num_heads = self.backbone.blocks[0].attn.num_heads
        patch_size = 16
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size  # 14 for 224
        
        # RoPE parameters
        self.base_freq = base_freq
        # Learnable depth modulation scale
        self.depth_alpha = nn.Parameter(torch.tensor(1.0))
        
        # Depth patch embed: extract mean depth per patch via avg pooling
        self.depth_pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        
        # Projection head
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
    
    def _compute_rope_freqs(self, depth_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE sin/cos tables modulated by depth.
        
        Args:
            depth_values: (B, N_patches) per-patch mean depth (0 = no depth)
            
        Returns:
            cos_emb, sin_emb: (B, N_patches+1, head_dim) rotary embeddings
                              +1 for CLS token (uses position 0)
        """
        B, N = depth_values.shape
        device = depth_values.device
        
        # Normalize depth to [0, 1] range for the batch
        depth_max = depth_values.max(dim=1, keepdim=True).values.clamp(min=1.0)
        depth_norm = depth_values / depth_max  # (B, N)
        
        # Compute 2D grid positions
        grid_h = int(N ** 0.5)
        grid_w = grid_h
        
        # Create position indices
        y_pos = torch.arange(grid_h, device=device).unsqueeze(1).expand(-1, grid_w).flatten().float()
        x_pos = torch.arange(grid_w, device=device).unsqueeze(0).expand(grid_h, -1).flatten().float()
        
        # Frequency bands (head_dim // 4 bands for each of: x, y, depth_x, depth_y)
        dim_quarter = self.head_dim // 4
        freq_base = 1.0 / (self.base_freq ** (torch.arange(0, dim_quarter, device=device).float() / dim_quarter))
        
        # Depth modulation: scale frequencies by depth
        # depth_mod: (B, N, 1)
        depth_mod = 1.0 + self.depth_alpha * depth_norm.unsqueeze(-1)  # (B, N, 1)
        
        # Position angles: (N, dim_quarter)
        angles_x = x_pos.unsqueeze(-1) * freq_base.unsqueeze(0)  # (N, dim_quarter)
        angles_y = y_pos.unsqueeze(-1) * freq_base.unsqueeze(0)  # (N, dim_quarter)
        
        # Depth-modulated angles: (B, N, dim_quarter)
        angles_dx = angles_x.unsqueeze(0) * depth_mod  # (B, N, dim_quarter)
        angles_dy = angles_y.unsqueeze(0) * depth_mod  # (B, N, dim_quarter)
        
        # Concatenate: [x_sin/cos, y_sin/cos, dx_sin/cos, dy_sin/cos]
        # Total = 4 * dim_quarter = head_dim
        angles = torch.cat([
            angles_x.unsqueeze(0).expand(B, -1, -1),
            angles_y.unsqueeze(0).expand(B, -1, -1),
            angles_dx,
            angles_dy,
        ], dim=-1)  # (B, N, head_dim)
        
        # Add CLS token position (zeros → no rotation)
        cls_angles = torch.zeros(B, 1, self.head_dim, device=device)
        angles = torch.cat([cls_angles, angles], dim=1)  # (B, N+1, head_dim)
        
        return angles.cos(), angles.sin()
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to tensor.
        
        Args:
            x: (B, num_heads, T, head_dim)
            cos, sin: (B, T, head_dim)
        """
        # Split x into pairs for rotation
        x1 = x[..., ::2]   # even indices
        x2 = x[..., 1::2]  # odd indices
        
        cos = cos[:, None, :, ::2]  # (B, 1, T, head_dim//2)
        sin = sin[:, None, :, ::2]  # (B, 1, T, head_dim//2)
        
        # Rotary operation
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        
        # Interleave back
        out = torch.stack([out1, out2], dim=-1).flatten(-2)
        return out
    
    def _forward_with_rope(
        self,
        rgb: torch.Tensor,
        depth_per_patch: torch.Tensor,
        return_patches: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through ViT with depth-conditioned RoPE.
        
        Args:
            rgb: (B, 3, H, W) RGB images
            depth_per_patch: (B, N_patches) per-patch mean depth
            
        Returns:
            emb: (B, embed_dim) CLS token embeddings
            if return_patches=True, also returns patch tokens (B, N_patches, vit_dim)
        """
        B = rgb.shape[0]
        
        # Patch embedding
        x = self.backbone.patch_embed(rgb)
        if x.dim() == 4:
            x = x.flatten(1, 2)
        
        # Prepend CLS token
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Standard positional embedding (will be augmented by RoPE)
        if x.shape[1] != self.backbone.pos_embed.shape[1]:
            pos_cls = self.backbone.pos_embed[:, :1]
            pos_patches = self.backbone.pos_embed[:, 1:]
            N_orig = pos_patches.shape[1]
            H_orig = int(N_orig ** 0.5)
            N_curr = x.shape[1] - 1
            H_curr = int(N_curr ** 0.5)
            ps = pos_patches.reshape(1, H_orig, H_orig, -1).permute(0, 3, 1, 2)
            ps = torch.nn.functional.interpolate(ps, size=(H_curr, H_curr), mode='bicubic', align_corners=False)
            ps = ps.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = torch.cat([pos_cls, ps], dim=1)
        else:
            pos_embed = self.backbone.pos_embed
        
        x = x + pos_embed
        x = self.backbone.pos_drop(x)
        
        # Compute RoPE embeddings from depth
        rope_cos, rope_sin = self._compute_rope_freqs(depth_per_patch)
        
        # Forward through transformer blocks with RoPE
        for block in self.backbone.blocks:
            residual = x
            x_norm = block.norm1(x)
            
            attn = block.attn
            qkv = attn.qkv(x_norm).reshape(B, x_norm.shape[1], 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            # Apply RoPE to Q and K
            q = self._apply_rope(q, rope_cos, rope_sin)
            k = self._apply_rope(k, rope_cos, rope_sin)
            
            # Standard attention
            scale = q.shape[-1] ** -0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale
            attn_weights = attn_weights.softmax(dim=-1)
            attn_weights = attn.attn_drop(attn_weights)
            
            attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, x_norm.shape[1], -1)
            attn_out = attn.proj(attn_out)
            attn_out = attn.proj_drop(attn_out)
            
            x = residual + block.drop_path1(block.ls1(attn_out))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        
        x = self.backbone.norm(x)
        
        # CLS token embedding through head
        emb = self.backbone.head(x[:, 0])
        if return_patches:
            patch_tokens = x[:, 1:]
            return emb, patch_tokens
        return emb
    
    def forward(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            cam_views: (B, V, 3, H, W) RGB views
            range_views: (B, V, 1, H, W) aligned depth views
        
        Returns:
            all_emb: (2*B*V, embed_dim) - [cam_emb; cam_emb] for interface compatibility
            proj: (2*V, B, proj_dim) projections
        """
        if isinstance(cam_views, dict):
            cam_tensor = cam_views.get('global')
            range_tensor = range_views.get('global')
        else:
            cam_tensor = cam_views
            range_tensor = range_views
        
        B, V, C, H, W = cam_tensor.shape
        cam_flat = cam_tensor.flatten(0, 1)       # (B*V, 3, H, W)
        range_flat = range_tensor.flatten(0, 1)    # (B*V, 1, H, W)
        
        # Extract per-patch mean depth
        depth_per_patch = self.depth_pool(range_flat).flatten(1)  # (B*V, N_patches)
        
        # Forward with depth-conditioned RoPE
        emb = self._forward_with_rope(cam_flat, depth_per_patch)  # (B*V, embed_dim)
        
        # Project for SigReg
        proj_flat = self.proj(emb)  # (B*V, proj_dim)
        proj = proj_flat.reshape(B, V, -1).transpose(0, 1)  # (V, B, proj_dim)
        
        # Interface compatibility: duplicate for [cam; range] expected format
        all_emb = torch.cat([emb, emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        
        # Handle local views
        if isinstance(cam_views, dict):
            c_local = cam_views.get('local')
            r_local = range_views.get('local')
            if c_local is not None and c_local.numel() > 0:
                cl_flat = c_local.flatten(0, 1)
                rl_flat = r_local.flatten(0, 1)
                depth_local = self.depth_pool(rl_flat).flatten(1)
                emb_l = self._forward_with_rope(cl_flat, depth_local)
                proj_l = self.proj(emb_l).reshape(c_local.shape[0], c_local.shape[1], -1).transpose(0, 1)
                
                all_emb = torch.cat([all_emb, emb_l, emb_l], dim=0)
                proj_dup = torch.cat([proj_dup, proj_l, proj_l], dim=0)
        
        return all_emb, proj_dup
    
    def forward_with_depth(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward returning both depth-conditioned and depth-free embeddings.
        
        Returns:
            emb_with_depth: (B*V, embed_dim) embeddings with LiDAR RoPE
            emb_without_depth: (B*V, embed_dim) embeddings without LiDAR RoPE (standard pos)
            proj: (V, B, proj_dim) projections from depth-conditioned pass
        """
        if isinstance(cam_views, dict):
            cam_tensor = cam_views.get('global')
            range_tensor = range_views.get('global')
        else:
            cam_tensor = cam_views
            range_tensor = range_views
        
        B, V, C, H, W = cam_tensor.shape
        cam_flat = cam_tensor.flatten(0, 1)
        range_flat = range_tensor.flatten(0, 1)
        
        # Extract per-patch mean depth
        depth_per_patch = self.depth_pool(range_flat).flatten(1)
        
        # Forward with depth-conditioned RoPE
        emb_with = self._forward_with_rope(cam_flat, depth_per_patch)
        
        # Forward without depth (zero depth → standard positional)
        zero_depth = torch.zeros_like(depth_per_patch)
        emb_without = self._forward_with_rope(cam_flat, zero_depth)
        
        # Project the depth-conditioned version
        proj = self.proj(emb_with).reshape(B, V, -1).transpose(0, 1)
        
        return emb_with, emb_without, proj
    
    def forward_camera_only(self, cam_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with camera only — depth is zeroed (no RoPE modulation)."""
        if isinstance(cam_views, dict):
            cam_tensor = cam_views.get('global')
        else:
            cam_tensor = cam_views
        
        B, V, C, H, W = cam_tensor.shape
        zero_range = torch.zeros(B, V, 1, H, W, device=cam_tensor.device, dtype=cam_tensor.dtype)
        
        if isinstance(cam_views, dict):
            return self.forward(cam_views, {'global': zero_range})
        return self.forward(cam_views, zero_range)

    def forward_with_patches(
        self,
        cam_views: torch.Tensor,
        range_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compatibility wrapper used by probe extraction paths.

        Returns patch tokens from the RGB trunk as cam patches. Range patches are
        mirrored for interface compatibility with existing callers.
        """
        if isinstance(cam_views, dict):
            cam_tensor = cam_views.get('global')
            range_tensor = range_views.get('global')
        else:
            cam_tensor = cam_views
            range_tensor = range_views

        B, V, _, _, _ = cam_tensor.shape
        cam_flat = cam_tensor.flatten(0, 1)
        range_flat = range_tensor.flatten(0, 1)
        depth_per_patch = self.depth_pool(range_flat).flatten(1)

        emb, patch_tokens = self._forward_with_rope(cam_flat, depth_per_patch, return_patches=True)

        proj_flat = self.proj(emb)
        proj = proj_flat.reshape(B, V, -1).transpose(0, 1)

        all_emb = torch.cat([emb, emb], dim=0)
        proj_dup = torch.cat([proj, proj], dim=0)
        return all_emb, proj_dup, (patch_tokens, patch_tokens)


def create_mm_encoder(
    arch: str = "B", 
    proj_dim: int = 128, 
    img_size: int = 224, 
    second_modality_channels: int = 5,
    aligned_mode: bool = False,  # NEW: Use aligned depth for B/C instead of range
    vit_size: str = 'small',     # ViT backbone size: 'small', 'base', 'large'
):
    """Factory function to create MM encoder.
    
    Options:
        A: Separate encoders (ViT + PointMLP for raw points)
        B: Unified ViT, separate passes (no cross-modal interaction)
        C: True fusion - single forward pass with cross-attention
        D: RGBD - single ViT with aligned projected depth channel
        E: Separate encoders (ViT + ViT for aligned depth, shared projector)
        F: Separate encoders + separate projectors + cross-modal alignment
    
    Args:
        arch: Architecture type (A, B, C, D, E, or F)
        proj_dim: Projection dimension for SIGReg loss
        img_size: Input image size
        second_modality_channels: Number of channels for LiDAR input (ignored in aligned mode)
        aligned_mode: If True, B/C use 1-channel aligned depth instead of 5-channel range.
                     A still uses points (filtered to camera FOV), D/E/F are inherently aligned.
        vit_size: ViT backbone size ('small', 'base', 'large')
    """
    if arch == "A":
        return MMEncoderA(proj_dim=proj_dim, img_size=img_size, vit_size=vit_size)
    elif arch == "C":
        return MMEncoderC(
            proj_dim=proj_dim, 
            img_size=img_size, 
            range_channels=second_modality_channels,
            aligned_mode=aligned_mode,
            vit_size=vit_size,
        )
    elif arch == "D":
        return MMEncoderD(proj_dim=proj_dim, img_size=img_size, vit_size=vit_size)
    elif arch == "E":
        return MMEncoderE(proj_dim=proj_dim, img_size=img_size, vit_size=vit_size)
    elif arch == "F":
        return MMEncoderF(proj_dim=proj_dim, img_size=img_size, vit_size=vit_size)
    else:  # Default to B
        return MMEncoderB(
            proj_dim=proj_dim, 
            img_size=img_size, 
            range_channels=second_modality_channels,
            aligned_mode=aligned_mode,
            vit_size=vit_size,
        )


if __name__ == "__main__":
    print("Testing MMEncoderA (Separate)...")
    encoder_a = MMEncoderA(proj_dim=128)
    cam = torch.randn(2, 2, 3, 224, 224)
    lidar = torch.randn(2, 16384, 5)
    (cam_emb, lidar_emb), (cam_proj, lidar_proj) = encoder_a(cam, lidar)
    print(f"Camera emb: {cam_emb.shape}, proj: {cam_proj.shape}")
    print(f"LiDAR emb: {lidar_emb.shape}, proj: {lidar_proj.shape}")
    
    print("\nTesting MMEncoderB (Unified, 5-ch range)...")
    encoder_b = MMEncoderB(proj_dim=128)
    cam = torch.randn(2, 2, 3, 224, 224)
    rng = torch.randn(2, 2, 5, 224, 224)
    emb, proj = encoder_b(cam, rng)
    print(f"Combined emb: {emb.shape}, proj: {proj.shape}")
    
    print("\nTesting MMEncoderB ALIGNED (1-ch depth)...")
    encoder_b_aligned = MMEncoderB(proj_dim=128, aligned_mode=True)
    cam = torch.randn(2, 2, 3, 224, 224)
    depth = torch.randn(2, 2, 1, 224, 224)  # 1 channel aligned depth
    emb, proj = encoder_b_aligned(cam, depth)
    print(f"Combined emb: {emb.shape}, proj: {proj.shape}")
    
    print("\nTesting MMEncoderC (TRUE FUSION, 5-ch range)...")
    encoder_c = MMEncoderC(proj_dim=128)
    cam = torch.randn(2, 2, 3, 224, 224)
    rng = torch.randn(2, 2, 5, 224, 224)
    emb, proj = encoder_c(cam, rng)
    print(f"Fused emb: {emb.shape}, proj: {proj.shape}")
    
    print("\nTesting MMEncoderC ALIGNED (1-ch depth)...")
    encoder_c_aligned = MMEncoderC(proj_dim=128, aligned_mode=True)
    cam = torch.randn(2, 2, 3, 224, 224)
    depth = torch.randn(2, 2, 1, 224, 224)  # 1 channel aligned depth
    emb, proj = encoder_c_aligned(cam, depth)
    print(f"Fused emb: {emb.shape}, proj: {proj.shape}")
    
    # Test that zeroing range affects output
    print("\nTesting modality dropout effect on Option C...")
    encoder_c = MMEncoderC(proj_dim=128)
    cam = torch.randn(2, 2, 3, 224, 224)
    rng = torch.randn(2, 2, 5, 224, 224)
    emb, proj = encoder_c(cam, rng)
    rng_zero = torch.zeros_like(rng)
    emb_zero, _ = encoder_c(cam, rng_zero)
    diff = (emb[:4] - emb_zero[:4]).abs().mean().item()  # Compare camera embeddings
    print(f"Embedding difference when range is zeroed: {diff:.4f}")
    print("(Should be > 0 for Option C, proving cross-modal interaction)")
    
    print("\nTesting MMEncoderD (RGBD - aligned depth)...")
    encoder_d = MMEncoderD(proj_dim=128)
    rgbd = torch.randn(2, 2, 4, 224, 224)  # 4 channels: RGB + Depth
    emb, proj = encoder_d(rgbd)
    print(f"RGBD emb: {emb.shape}, proj: {proj.shape}")
    
    print("\nTesting create_mm_encoder factory with aligned_mode...")
    encoder_b = create_mm_encoder("B", aligned_mode=False)
    print(f"B (unaligned) - aligned_mode: {encoder_b.aligned_mode}")
    encoder_b_al = create_mm_encoder("B", aligned_mode=True)
    print(f"B (aligned) - aligned_mode: {encoder_b_al.aligned_mode}")
    encoder_c_al = create_mm_encoder("C", aligned_mode=True)
    print(f"C (aligned) - aligned_mode: {encoder_c_al.aligned_mode}")
    
    print("\n✓ All encoder tests passed!")


