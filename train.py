"""Le MuMo JEPA training entrypoint.

Example:
    python train.py \
        +dataroot=/path/to/nuscenes_data \
        +arch=C \
        +fusion_tokens_sigreg=true \
        +fusion_tokens_variant=prune_after_first
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

try:
    # Containers and high-throughput multi-worker DataLoaders can hit
    # "received 0 items of ancdata" with the default file_descriptor sharing
    # strategy. file_system is slower but much more robust for large FLIR sweeps.
    torch_mp.set_sharing_strategy('file_system')
except (RuntimeError, AttributeError):
    pass

import math
import sys
import os
import subprocess
import importlib.util
import time
from contextlib import nullcontext

def install_and_import(package_name, import_name=None, pip_args=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package_name}...")
        cmd = [sys.executable, "-m", "pip", "install"]
        if pip_args:
            cmd.extend(pip_args)
        cmd.append(package_name)
        subprocess.check_call(cmd)
        __import__(import_name)


def ensure_imagebind_available(install_all_envs: bool = True):
    """Install ImageBind on demand using no-cache pip installs."""
    if importlib.util.find_spec("imagebind") is not None:
        return

    install_script = Path(__file__).with_name("install_imagebind_envs.sh")
    if install_all_envs and install_script.exists():
        print("Installing ImageBind into base/tfwaymo with --no-cache-dir...")
        subprocess.check_call(["bash", str(install_script)])
    else:
        print("Installing ImageBind into current environment with --no-cache-dir...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "git+https://github.com/facebookresearch/ImageBind.git",
        ])

    if importlib.util.find_spec("imagebind") is None:
        raise ImportError("ImageBind installation completed but the package is still unavailable")

import timm
import hydra
import tqdm
import numpy as np
import copy
from pathlib import Path
from omegaconf import DictConfig

PROJECT_DIR = Path(__file__).resolve().parent
TORCH_CACHE_DIR = PROJECT_DIR / "cache" / "torch_hub"
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))

try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except Exception:
    THOP_AVAILABLE = False

try:
    from torch.utils.flop_counter import FlopCounterMode
    FLOP_COUNTER_AVAILABLE = True
except Exception:
    FLOP_COUNTER_AVAILABLE = False

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available, logging to console only")

# Local imports
from src.dataset import MMNuScenesDataset, mm_collate_fn
from src.encoder import create_mm_encoder, MMEncoderA, MMEncoderB, MMEncoderC_FusionTokens, MMEncoderC_FrustumSlots, MMEncoderC_LiDARRoPE, initialize_module_from_dinov3, initialize_module_from_timm_vit
from torchvision.ops import MLP

# Novel regularizers (beyond SIGReg)
from src.novel_regularizers import (
    GMMRegularizer,
    SinkhornRegularizer, 
    SpectralRegularizer,
    create_regularizer
)

# Baseline losses and encoders
from src.losses import (
    VICRegLoss,
    InfoNCELoss,
    AdaSigNCELoss,
    WhitenedInfoNCELoss,
    EigenNCELoss,
    DINOHead,
    DINOLoss,
    KoLeoLoss,
    IBOTPatchLoss,
)
from src.baseline_encoders import (
    DINOv3FrozenEncoder,
    DINOv3ScratchEncoder,
    ImageBindEncoder,
    MultiMAEEncoder,
    MultiMAEExactEncoder,
    MaskedDepthModelEncoder,
    LateFusionEncoder,
)

# Detection & segmentation probes (patch-based)
from src.detection_probes import (
    BBox3DProbe,
    BBox2DSlotProbe,
    SpatialBBox3DProbe,
    SpatialBBox2DProbe,
    SemanticSegProbe,
    DepthMapProbe,
    OccupancyMapProbe,
    generate_centernet_targets,
    generate_centernet_targets_2d,
    NuScenesDetectionMetrics,
    DetectionMetrics2D,
    SegmentationMetrics,
    FLIR_2D_DETECTION_CLASSES,
    NUM_DETECTION_CLASSES,
    NUM_SIMPLIFIED_SEG_CLASSES,
)
from src.detection_labels import collate_det_seg_labels


WAYMO_PATCH_DET_CLASSES = ['car', 'pedestrian', 'bicycle']
WAYMO_PATCH_CLASS_MAP = {
    0: 0,
    5: 1,
    7: 2,
}
FLIR_BOX_SEG_CLASSES = ['background', *FLIR_2D_DETECTION_CLASSES]
_WAYMO_PATCH_CLASS_TENSOR = None


def remap_waymo_patch_gt_classes(gt_classes: torch.Tensor, gt_mask: torch.Tensor, device):
    """Map sparse Waymo class ids (0,5,7) to dense ids (0,1,2) for patch probes."""
    global _WAYMO_PATCH_CLASS_TENSOR
    if _WAYMO_PATCH_CLASS_TENSOR is None or _WAYMO_PATCH_CLASS_TENSOR.device != device:
        mapping = torch.full((NUM_DETECTION_CLASSES,), -1, dtype=torch.long, device=device)
        for src_idx, dst_idx in WAYMO_PATCH_CLASS_MAP.items():
            mapping[src_idx] = dst_idx
        _WAYMO_PATCH_CLASS_TENSOR = mapping

    remapped = _WAYMO_PATCH_CLASS_TENSOR[gt_classes.clamp(0, NUM_DETECTION_CLASSES - 1)]
    drop_mask = remapped.eq(-1)
    remapped = remapped.clamp(min=0)
    new_mask = gt_mask.clone()
    new_mask[drop_mask] = 0.0
    return remapped, new_mask


def generate_dummy_batch(batch_size, V_global, V_local, img_size=224, local_size=96, 
                         arch='B', device='cuda', num_scenes=10, num_cameras=6, num_locations=4,
                         lidar_mode='depth', aligned_mode=False, occupancy_grid_size=28):
    """
    Generate random dummy data for fast model debugging.
    Bypasses dataloader entirely for rapid iteration on model/loss testing.
    
    Args:
        lidar_mode: 'depth' (1-ch), 'range' (5-ch), 'points' (point cloud), 'aligned_points'
        aligned_mode: If True and lidar_mode is 'depth', use 1-channel aligned depth
    
    Returns:
        cam_views: dict with 'global' and 'local' tensors
        modality2: dict with 'global' and 'local' tensors (depth/range) or point cloud
        labels: dict with all required label tensors
    """
    # Camera views: RGB (3 channels).
    # For arch D (early fusion), the training loop concatenates depth as the 4th
    # channel, so the dataset always provides 3-ch RGB.
    cam_ch = 3
    cam_global = torch.randn(batch_size, V_global, cam_ch, img_size, img_size, device=device)
    cam_local = torch.randn(batch_size, V_local, cam_ch, local_size, local_size, device=device)
    cam_views = {'global': cam_global, 'local': cam_local}
    
    # Modality 2: Based on lidar_mode
    if lidar_mode == 'points' or lidar_mode == 'aligned_points' or arch == 'A':
        # Point cloud format for arch A
        modality2 = torch.randn(batch_size, 1024, 5, device=device)  # (B, N_points, 5)
    elif lidar_mode == 'range':
        # 5-channel range images
        depth_global = torch.randn(batch_size, V_global, 5, img_size, img_size, device=device)
        depth_local = torch.randn(batch_size, V_local, 5, local_size, local_size, device=device)
        modality2 = {'global': depth_global, 'local': depth_local}
    else:
        # 1-channel depth images (aligned mode)
        depth_global = torch.randn(batch_size, V_global, 1, img_size, img_size, device=device)
        depth_local = torch.randn(batch_size, V_local, 1, local_size, local_size, device=device)
        modality2 = {'global': depth_global, 'local': depth_local}
    
    # Labels
    total_views = V_global + V_local
    occupancy_grid_cells = occupancy_grid_size * occupancy_grid_size

    labels = {
        # Global labels (per sample)
        'scene': torch.randint(0, num_scenes, (batch_size,), device=device),
        'camera': torch.randint(0, num_cameras, (batch_size,), device=device),
        'location': torch.randint(0, num_locations, (batch_size,), device=device),
        
        # Spatial labels (per view, stacked)
        'num_cars': torch.randint(0, 10, (batch_size, total_views), device=device).float(),
        'num_pedestrians': torch.randint(0, 5, (batch_size, total_views), device=device).float(),
        'num_peds': torch.randint(0, 5, (batch_size, total_views), device=device).float(),  # alias
        'num_objects': torch.randint(0, 20, (batch_size, total_views), device=device).float(),
        'depth': torch.rand(batch_size, total_views, device=device) * 50,  # 0-50m
        'mean_depth': torch.rand(batch_size, total_views, device=device) * 50,  # Alias for depth
        
        # Depth grid labels (per view)
        'depth_grid': torch.rand(batch_size, total_views, 64, device=device) * 50,
        'depth_grid_mask': torch.ones(batch_size, total_views, 64, device=device),
        
        # High-res depth grid labels (56×56 = 3136)
        'depth_grid_hr': torch.rand(batch_size, total_views, 3136, device=device) * 50,
        'depth_grid_mask_hr': torch.ones(batch_size, total_views, 3136, device=device),
        
        # Grid occupancy labels
        'grid_occupancy': torch.randint(0, 2, (batch_size, 64), device=device).float(),
        'grid_occupancy_car': torch.randint(0, 2, (batch_size, 64), device=device).float(),
        'grid_occupancy_ped': torch.randint(0, 2, (batch_size, 64), device=device).float(),
        'grid_occupancy_hr': torch.randint(0, 2, (batch_size, occupancy_grid_cells), device=device).float(),
        'grid_occupancy_hr_classes': torch.randint(
            0, 2, (batch_size, len(FLIR_2D_DETECTION_CLASSES), occupancy_grid_cells), device=device
        ).float(),
        
        # Distance distribution (for distance_dist probe)
        'distance_dist': torch.rand(batch_size, total_views, 3, device=device),  # 3 distance bins
        
        # Detection / segmentation labels (for patch-based probes)
        'gt_classes': torch.randint(0, NUM_DETECTION_CLASSES, (batch_size, 50), device=device).long(),
        'gt_centers': torch.randn(batch_size, 50, 3, device=device) * 20,
        'gt_sizes': torch.rand(batch_size, 50, 3, device=device) * 5,
        'gt_orientations': torch.randn(batch_size, 50, 2, device=device),
        'gt_mask': (torch.rand(batch_size, 50, device=device) > 0.7).float(),
        'gt_centers_2d': torch.rand(batch_size, 50, 2, device=device),
        'seg_map': torch.randint(0, NUM_SIMPLIFIED_SEG_CLASSES, (batch_size, 224, 224), device=device).long(),
        'box_seg_map': torch.randint(0, len(FLIR_BOX_SEG_CLASSES), (batch_size, 28, 28), device=device).long(),
        'has_box_seg_map': torch.ones(batch_size, device=device, dtype=torch.bool),
    }
    
    return cam_views, modality2, labels


def _resolve_pretrained_encoder_path(checkpoint_path: str | os.PathLike) -> Path:
    checkpoint_path = Path(str(checkpoint_path)).expanduser()
    if checkpoint_path.exists():
        return checkpoint_path

    run_dir = PROJECT_DIR / "saved_models" / checkpoint_path
    if run_dir.is_dir():
        latest_path = run_dir / "latest.pt"
        if latest_path.exists():
            return latest_path.resolve()
        pt_files = sorted(run_dir.glob("*.pt"))
        if pt_files:
            return pt_files[-1]

    raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")


def _load_pretrained_encoder_into_model(model: nn.Module, checkpoint_path: str | os.PathLike):
    resolved_path = _resolve_pretrained_encoder_path(checkpoint_path)
    print(f"📦 Loading pretrained encoder weights from {resolved_path}")
    ckpt = torch.load(resolved_path, map_location='cpu', weights_only=False)
    encoder_state = ckpt.get('encoder', ckpt)
    if not isinstance(encoder_state, dict):
        raise ValueError(f"Checkpoint at {resolved_path} does not contain a valid encoder state_dict")

    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"  ⚠️  Missing encoder keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  ⚠️  Unexpected encoder keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    saved_cfg = ckpt.get('config', {})
    saved_run = ckpt.get('run_name', None)
    if isinstance(saved_cfg, dict) and saved_cfg:
        saved_scenario = saved_cfg.get('scenario', 'unknown')
        saved_simple = bool(saved_cfg.get('simple_baseline', False))
        saved_fusion_tokens = bool(saved_cfg.get('fusion_tokens_sigreg', False))
        saved_frustum_slots = bool(saved_cfg.get('use_frustum_slots', False))
        saved_lidar_rope = bool(saved_cfg.get('lidar_rope_rgb', False))
        saved_variant = str(saved_cfg.get('fusion_tokens_variant', 'prune_after_first')).lower()
        model_name = type(model).__name__

        if saved_simple and model_name != 'ViTEncoder':
            raise ValueError(
                f"Checkpoint scenario '{saved_scenario}' is a simple baseline, but current model is {model_name}. "
                "Use a matching simple_* scenario for probe-only evaluation."
            )
        if saved_fusion_tokens and model_name != 'MMEncoderC_FusionTokens':
            raise ValueError(
                f"Checkpoint scenario '{saved_scenario}' uses MMEncoderC_FusionTokens, but current model is {model_name}. "
                "Use a matching fusion-tokens scenario for probe-only evaluation."
            )
        if saved_frustum_slots and model_name != 'MMEncoderC_FrustumSlots':
            raise ValueError(
                f"Checkpoint scenario '{saved_scenario}' uses MMEncoderC_FrustumSlots, but current model is {model_name}."
            )
        if saved_lidar_rope and model_name != 'MMEncoderC_LiDARRoPE':
            raise ValueError(
                f"Checkpoint scenario '{saved_scenario}' uses MMEncoderC_LiDARRoPE, but current model is {model_name}."
            )
        if model_name == 'MMEncoderC_FusionTokens':
            current_variant = str(getattr(model, 'attention_mode', 'prune_after_first')).lower()
            if saved_variant != current_variant:
                raise ValueError(
                    f"Fusion-token checkpoint variant mismatch: checkpoint uses '{saved_variant}' but current scenario builds '{current_variant}'. "
                    "Use the matching fusion_tokens_* scenario for probe-only evaluation."
                )
        print(f"  ✓ Loaded encoder from scenario={saved_scenario}, run={saved_run or 'unknown'}")
    return ckpt


def extract_patch_tokens(net, cam_views_probe, modality2_probe, arch, simple_baseline=False, simple_modality='rgb', batch_chunk_size=0):
    """Extract patch embeddings (B, N_patches, vit_dim) from the camera encoder.
    
    Uses forward_with_patches when available. Returns None for unsupported architectures.
    
    Args:
        net: The encoder model
        cam_views_probe: Camera views for the probe view (B, 1, 3, H, W) or dict
        modality2_probe: Second modality for the probe view
        arch: Architecture string
        
    Returns:
        cam_patch_tokens: (B, N_patches, vit_dim) or None
    """
    # All architectures now support forward_with_patches (except A)
    if arch == 'A':
        return None
    
    # Unwrap dict inputs: forward_with_patches needs plain tensors (B, V, C, H, W)
    # For validation, cam_views is often dict {'global': (B,1,C,H,W), 'local': empty}
    def _unwrap(x):
        if isinstance(x, dict):
            g = x.get('global')
            if g is not None and g.numel() > 0:
                return g  # (B, V, C, H, W) — typically V=1 for validation
            # Fallback to local if global is empty
            l = x.get('local')
            if l is not None and l.numel() > 0:
                return l
        return x  # Already a tensor
    
    cam_in = _unwrap(cam_views_probe)
    mod2_in = _unwrap(modality2_probe)
    
    try:
        if simple_baseline:
            # Simple baseline: single modality ViT
            if simple_modality in ('lidar', 'thermal'):
                _, _, (cam_patches, _) = maybe_chunked_forward(net.forward_with_patches, batch_chunk_size, mod2_in)
            else:
                _, _, (cam_patches, _) = maybe_chunked_forward(net.forward_with_patches, batch_chunk_size, cam_in)
            return cam_patches
        elif arch == 'D':
            # RGBD single input — concatenate depth if cam is still 3-ch
            if cam_in.shape[2] < 4 and mod2_in is not None:
                cam_in = torch.cat([cam_in, mod2_in], dim=2)
            _, _, (rgb_patches, _) = maybe_chunked_forward(net.forward_with_patches, batch_chunk_size, cam_in)
            return rgb_patches  # (B*1, N_patches, vit_dim)
        elif arch in ('B', 'C'):
            # Shared/fused trunk: forward_with_patches(cam, range)
            _, _, (cam_patches, _) = maybe_chunked_forward(net.forward_with_patches, batch_chunk_size, cam_in, mod2_in)
            return cam_patches  # (B*1, N_patches, vit_dim)
        elif arch in ('E', 'F'):
            # Separate encoders: forward_with_patches(cam, depth)
            _, _, (cam_patches, _) = maybe_chunked_forward(net.forward_with_patches, batch_chunk_size, cam_in, mod2_in)
            return cam_patches  # (B*1, N_patches, vit_dim)
    except Exception as e:
        print(f"⚠️  extract_patch_tokens failed: {e}")
        return None
    return None


def get_flir_probe_domains(enable_dual: bool = False):
    domains = [('rgb', '')]
    if enable_dual:
        domains.append(('thermal', '_thermal'))
    return domains


def normalize_flir_probe_label_modes(raw_modes=None, enable_dual: bool = False):
    valid_modes = ('rgb', 'thermal', 'consensus', 'union', 'weighted_union')
    if raw_modes is None:
        modes = ['rgb']
        if enable_dual:
            modes.append('thermal')
        return modes

    if isinstance(raw_modes, str):
        candidate_modes = [part.strip().lower() for part in raw_modes.split(',') if part.strip()]
    else:
        candidate_modes = [str(part).strip().lower() for part in raw_modes if str(part).strip()]

    if not candidate_modes:
        candidate_modes = ['rgb']
    if 'all' in candidate_modes:
        candidate_modes = list(valid_modes)

    deduped_modes = []
    for mode in candidate_modes:
        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported FLIR probe label mode {mode!r}. Valid options: {', '.join(valid_modes)}"
            )
        if mode not in deduped_modes:
            deduped_modes.append(mode)
    return deduped_modes


def get_flir_detection_targets(labels, device, domain: str = 'rgb'):
    suffix = '' if domain == 'rgb' else f'_{domain}'
    return {
        'gt_boxes_2d': labels.get(f'gt_boxes_2d{suffix}', labels.get('gt_boxes_2d')).to(device).float(),
        'gt_classes_2d': labels.get(f'gt_classes_2d{suffix}', labels.get('gt_classes_2d')).to(device).long(),
        'gt_mask_2d': labels.get(f'gt_mask_2d{suffix}', labels.get('gt_mask_2d')).to(device).float(),
    }


def get_flir_occupancy_targets(labels, device, domain: str = 'rgb'):
    suffix = '' if domain == 'rgb' else f'_{domain}'
    y_occ_hr = labels.get(f'grid_occupancy_hr{suffix}', labels.get('grid_occupancy_hr')).to(device).float()
    y_occ_hr_classes = labels.get(f'grid_occupancy_hr_classes{suffix}', labels.get('grid_occupancy_hr_classes')).to(device).float()
    return torch.cat(
        [
            reshape_flat_spatial_label(y_occ_hr),
            reshape_multichannel_flat_spatial_label(y_occ_hr_classes),
        ],
        dim=1,
    )


def get_flir_box_seg_targets(labels, device, domain: str = 'rgb'):
    suffix = '' if domain == 'rgb' else f'_{domain}'
    return {
        'has_box_seg_map': labels.get(f'has_box_seg_map{suffix}', labels.get('has_box_seg_map')).to(device),
        'box_seg_map': labels.get(f'box_seg_map{suffix}', labels.get('box_seg_map')).to(device).long(),
    }


def flir_probe_key(base_name: str, domain: str) -> str:
    return base_name if domain == 'rgb' else f'{base_name}_{domain}'


def get_input_stats(views):
    """Helper to get batch size and total views from potentially dictionary inputs."""
    if isinstance(views, dict):
        # Assuming 'global' key always exists and dominates B
        B = views['global'].shape[0]
        V = views['global'].shape[1]
        if 'local' in views and views['local'].numel() > 0:
            V += views['local'].shape[1]
        return B, V
    return views.shape[0], views.shape[1]


def to_device(x, device, non_blocking=True):
    """Helper to move potential dictionary inputs to device."""
    if isinstance(x, dict):
        return {k: v.to(device, non_blocking=non_blocking) for k, v in x.items()}
    return x.to(device, non_blocking=non_blocking)


def batch_size_of_tree(x):
    """Return batch size for a tensor or dict of tensors."""
    if isinstance(x, dict):
        for key in ('probe', 'global', 'local'):
            value = x.get(key)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return value.shape[0]
        raise ValueError("Cannot infer batch size from empty dict input")
    if isinstance(x, torch.Tensor):
        return x.shape[0]
    raise TypeError(f"Unsupported batched input type: {type(x)!r}")


def slice_batch_tree(x, start: int, end: int):
    """Slice the leading batch dimension for tensors or dicts of tensors."""
    if isinstance(x, dict):
        return {
            k: (v[start:end] if isinstance(v, torch.Tensor) and v.numel() > 0 else v)
            for k, v in x.items()
        }
    if isinstance(x, torch.Tensor):
        return x[start:end]
    return x


def concat_chunk_outputs(outputs):
    """Concatenate chunked outputs from tensors or nested tuples/lists."""
    first = outputs[0]
    if isinstance(first, torch.Tensor):
        if first.dim() == 0:
            return torch.stack(outputs, dim=0)

        concat_dim = None
        for dim_idx in range(first.dim()):
            dim_sizes = [out.shape[dim_idx] for out in outputs]
            if len(set(dim_sizes)) > 1:
                other_dims_match = all(
                    out.shape[:dim_idx] + out.shape[dim_idx + 1:] ==
                    first.shape[:dim_idx] + first.shape[dim_idx + 1:]
                    for out in outputs[1:]
                )
                if other_dims_match:
                    concat_dim = dim_idx
                    break

        if concat_dim is None:
            concat_dim = 0
        return torch.cat(outputs, dim=concat_dim)
    if first is None:
        return None
    if isinstance(first, tuple):
        return tuple(concat_chunk_outputs([out[i] for out in outputs]) for i in range(len(first)))
    if isinstance(first, list):
        return [concat_chunk_outputs([out[i] for out in outputs]) for i in range(len(first))]
    raise TypeError(f"Unsupported chunk output type: {type(first)!r}")


def maybe_chunked_forward(forward_fn, batch_chunk_size: int, *batched_args, **kwargs):
    """Run a forward in smaller batch chunks and concatenate outputs."""
    batch_chunk_size = int(batch_chunk_size or 0)
    if batch_chunk_size <= 0:
        return forward_fn(*batched_args, **kwargs)

    batch_sizes = [batch_size_of_tree(arg) for arg in batched_args if isinstance(arg, (dict, torch.Tensor))]
    if not batch_sizes:
        return forward_fn(*batched_args, **kwargs)
    batch_size = batch_sizes[0]
    if batch_size <= batch_chunk_size:
        return forward_fn(*batched_args, **kwargs)

    outputs = []
    for start in range(0, batch_size, batch_chunk_size):
        end = min(batch_size, start + batch_chunk_size)
        sliced_args = [slice_batch_tree(arg, start, end) if isinstance(arg, (dict, torch.Tensor)) else arg for arg in batched_args]
        outputs.append(forward_fn(*sliced_args, **kwargs))
    return concat_chunk_outputs(outputs)


def zeros_like_tree(x):
    """Create a zeroed copy for tensors or nested dicts of tensors."""
    if isinstance(x, dict):
        return {k: zeros_like_tree(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    return x


def _unwrap_global_views(x):
    """Return global views from dict inputs, otherwise pass tensors through."""
    if isinstance(x, dict):
        g = x.get('global')
        if g is not None and isinstance(g, torch.Tensor) and g.numel() > 0:
            return g
        l = x.get('local')
        if l is not None and isinstance(l, torch.Tensor) and l.numel() > 0:
            return l
    return x


def _unwrap_probe_or_global_views(x):
    """Prefer explicit probe views, otherwise fall back to global/local tensors."""
    if isinstance(x, dict):
        p = x.get('probe')
        if p is not None and isinstance(p, torch.Tensor) and p.numel() > 0:
            return p
    return _unwrap_global_views(x)


def extract_rgb_lidar_patch_pair(net, cam_views, modality2, arch, simple_baseline=False, batch_chunk_size=0):
    """Extract paired RGB/LiDAR patch-like tokens for similarity probing.

    Returns:
        (rgb_tokens, lidar_tokens) or (None, None) when unavailable.
    """
    if simple_baseline or arch == 'A':
        return None, None

    cam_in = _unwrap_global_views(cam_views)
    mod2_in = _unwrap_global_views(modality2)

    if not isinstance(cam_in, torch.Tensor):
        return None, None

    try:
        if arch == 'D':
            if isinstance(mod2_in, torch.Tensor) and cam_in.shape[2] < 4:
                cam_in = torch.cat([cam_in, mod2_in], dim=2)
            _, _, (rgb_patches, lidar_patches) = maybe_chunked_forward(net.forward_with_patches, batch_chunk_size, cam_in)
            return rgb_patches, lidar_patches

        if arch == 'C' and hasattr(net, 'forward_with_fusion_tokens') and isinstance(mod2_in, torch.Tensor):
            _, _, rgb_only_tokens = maybe_chunked_forward(
                net.forward_with_fusion_tokens, batch_chunk_size, cam_in, zeros_like_tree(mod2_in)
            )
            _, _, lidar_only_tokens = maybe_chunked_forward(
                net.forward_with_fusion_tokens, batch_chunk_size, zeros_like_tree(cam_in), mod2_in
            )
            return rgb_only_tokens, lidar_only_tokens

        if hasattr(net, 'forward_with_patches') and isinstance(mod2_in, torch.Tensor):
            _, _, (rgb_patches, lidar_patches) = maybe_chunked_forward(net.forward_with_patches, batch_chunk_size, cam_in, mod2_in)
            return rgb_patches, lidar_patches
    except Exception as e:
        print(f"⚠️  rgb-lidar patch probe extraction failed: {e}")

    return None, None


def compute_patch_mse_safe(rgb_patch_tokens, lidar_patch_tokens):
    """Return a scalar patch MSE when token layouts match, else None.

    Some encoders, notably official ImageBind, expose different token grids per
    modality. In that case a direct patch-wise MSE is not meaningful, so we skip
    the metric instead of broadcasting incompatible shapes.
    """
    if rgb_patch_tokens is None or lidar_patch_tokens is None:
        return None
    if rgb_patch_tokens.shape != lidar_patch_tokens.shape:
        return None
    return F.mse_loss(rgb_patch_tokens.float(), lidar_patch_tokens.float())


def estimate_encoder_macs_per_call(net, cam_views, modality2, arch, simple_baseline=False, simple_modality='rgb'):
    """Estimate encoder MACs for one forward-like call using THOP (if available)."""
    global THOP_AVAILABLE, thop_profile
    if not THOP_AVAILABLE:
        try:
            install_and_import("thop")
            from thop import profile as _thop_profile
            thop_profile = _thop_profile
            THOP_AVAILABLE = True
            print("📏 Installed THOP for encoder MAC/FLOP tracking")
        except Exception as e:
            print(f"⚠️  THOP unavailable for encoder MAC/FLOP tracking: {e}")
            return None

    def _cleanup_thop_artifacts(model):
        for module in model.modules():
            try:
                hooks_to_remove = []
                for hook_id, hook_fn in list(module._forward_hooks.items()):
                    mod_name = getattr(hook_fn, "__module__", "")
                    if isinstance(mod_name, str) and mod_name.startswith("thop"):
                        hooks_to_remove.append(hook_id)
                for hook_id in hooks_to_remove:
                    module._forward_hooks.pop(hook_id, None)
            except Exception:
                pass

            for attr_name in ("total_ops", "total_params"):
                try:
                    if hasattr(module, attr_name):
                        delattr(module, attr_name)
                except Exception:
                    pass

    def _make_profile_model(live_model):
        model_copy = copy.deepcopy(live_model).eval()
        for method_name in ("forward", "forward_with_patches", "forward_with_fusion_tokens", "forward_with_depth"):
            try:
                if method_name in model_copy.__dict__:
                    delattr(model_copy, method_name)
            except Exception:
                pass
        return model_copy

    cam_in = _unwrap_global_views(cam_views)
    mod2_in = _unwrap_global_views(modality2)

    try:
        if simple_baseline:
            model_input = mod2_in if simple_modality in ('lidar', 'thermal') else cam_in
            if not isinstance(model_input, torch.Tensor):
                return None
            sample = model_input[:1, :1].contiguous()

            class _ProfileSimple(nn.Module):
                def __init__(self, wrapped):
                    super().__init__()
                    self.wrapped = wrapped

                def forward(self, x):
                    return self.wrapped(x)

            net_profile = _make_profile_model(net)
            profile_module = _ProfileSimple(net_profile)
            try:
                with torch.no_grad():
                    macs, _ = thop_profile(profile_module, inputs=(sample,), verbose=False)
            finally:
                _cleanup_thop_artifacts(profile_module)
                _cleanup_thop_artifacts(net_profile)
            return float(macs)

        if arch == 'D':
            if not isinstance(cam_in, torch.Tensor):
                return None
            if isinstance(mod2_in, torch.Tensor) and cam_in.shape[2] < 4:
                cam_in = torch.cat([cam_in, mod2_in], dim=2)
            cam_sample = cam_in[:1, :1].contiguous()

            class _ProfileArchD(nn.Module):
                def __init__(self, wrapped):
                    super().__init__()
                    self.wrapped = wrapped

                def forward(self, cam_x):
                    return self.wrapped(cam_x)

            net_profile = _make_profile_model(net)
            profile_module = _ProfileArchD(net_profile)
            try:
                with torch.no_grad():
                    macs, _ = thop_profile(profile_module, inputs=(cam_sample,), verbose=False)
            finally:
                _cleanup_thop_artifacts(profile_module)
                _cleanup_thop_artifacts(net_profile)
            return float(macs)

        if not isinstance(cam_in, torch.Tensor) or not isinstance(mod2_in, torch.Tensor):
            return None

        cam_sample = cam_in[:1, :1].contiguous()
        mod2_sample = mod2_in[:1, :1].contiguous()

        class _ProfileMM(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, cam_x, mod_x):
                return self.wrapped(cam_x, mod_x)

        net_profile = _make_profile_model(net)
        profile_module = _ProfileMM(net_profile)
        try:
            with torch.no_grad():
                macs, _ = thop_profile(profile_module, inputs=(cam_sample, mod2_sample), verbose=False)
        finally:
            _cleanup_thop_artifacts(profile_module)
            _cleanup_thop_artifacts(net_profile)
        return float(macs)
    except Exception as e:
        _cleanup_thop_artifacts(net)
        print(f"⚠️  Encoder MAC estimation failed: {e}")
        return None


class ViTEncoder(nn.Module):
    """Simple Vision Transformer encoder with projection head (LeJEPA paper style).
    
    This is a vanilla ViT for single-modality baselines.
    """
    def __init__(self, proj_dim=128, img_size=224, in_channels=3, vit_size='small', pretrained_source=None):
        super().__init__()
        from src.encoder import get_vit_config
        vit_cfg = get_vit_config(vit_size)
        embed_dim = vit_cfg['embed_dim']
        self.in_channels = in_channels
        self.n_patches = (img_size // 16) ** 2
        self.backbone = timm.create_model(
            vit_cfg['model_name'],
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=img_size,
            in_chans=in_channels,
            dynamic_img_size=True,
        )
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

        if pretrained_source is not None:
            source = str(pretrained_source).lower()
            if source != 'dinov3':
                raise ValueError(f"Unsupported pretrained_source={pretrained_source!r} for ViTEncoder")
            initialize_module_from_dinov3(self, vit_size=vit_size)

    def forward(self, x):
        if isinstance(x, dict):
             # Process separately due to potential resolution diffs (Global 224, Local 96)
             parts_emb = []
             parts_proj = []
             total_V = 0
             
             # Global
             if 'global' in x and x['global'].numel() > 0:
                 g = x['global'] # (B, Vg, C, H, W)
                 B, Vg = g.shape[:2]
                 emb_g = self.backbone(g.flatten(0, 1))
                 proj_g = self.proj(emb_g).reshape(B, Vg, -1).transpose(0, 1) # (Vg, B, D)
                 parts_emb.append(emb_g)
                 parts_proj.append(proj_g)
                 
             # Local
             if 'local' in x and x['local'].numel() > 0:
                 l = x['local']
                 B, Vl = l.shape[:2]
                 emb_l = self.backbone(l.flatten(0, 1))
                 proj_l = self.proj(emb_l).reshape(B, Vl, -1).transpose(0, 1) # (Vl, B, D)
                 parts_emb.append(emb_l)
                 parts_proj.append(proj_l)
             
             # Concat embeddings (B*V, D) and projections (V, B, D)
             emb = torch.cat(parts_emb, dim=0)
             proj = torch.cat(parts_proj, dim=0)
             
             return emb, proj

        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)

    def forward_with_patches(self, x):
        """Forward pass returning patch embeddings alongside CLS embeddings.
        
        Returns:
            emb: CLS embeddings (B*V, embed_dim)
            proj: projected embeddings (V, B, proj_dim)
            (cam_patches, None): patch tokens (B*V, N_patches, vit_dim), None for second modality
        """
        if isinstance(x, dict):
            raise NotImplementedError("forward_with_patches doesn't support dict inputs yet")
        
        N, V = x.shape[:2]
        flat = x.flatten(0, 1)  # (B*V, C, H, W)
        
        # Manual forward through backbone components
        patch_tokens = self.backbone.patch_embed(flat)
        if patch_tokens.dim() == 4:
            # (B, H, W, D) -> (B, N, D)
            patch_tokens = patch_tokens.flatten(1, 2)
        
        cls_token = self.backbone.cls_token.expand(flat.shape[0], -1, -1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        
        # Positional embedding (with interpolation if needed)
        pos_embed = self.backbone.pos_embed
        if pos_embed.shape[1] != tokens.shape[1]:
            pos_embed_cls = pos_embed[:, :1]
            pos_embed_patches = pos_embed[:, 1:]
            N_orig = pos_embed_patches.shape[1]
            H_orig = int(N_orig**0.5)
            N_curr = tokens.shape[1] - 1
            H_curr = int(N_curr**0.5)
            pe = pos_embed_patches.reshape(1, H_orig, H_orig, -1).permute(0, 3, 1, 2)
            pe = torch.nn.functional.interpolate(pe, size=(H_curr, H_curr), mode='bicubic', align_corners=False)
            pe = pe.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = torch.cat([pos_embed_cls, pe], dim=1)
        
        tokens = tokens + pos_embed
        tokens = self.backbone.pos_drop(tokens)
        tokens = self.backbone.blocks(tokens)
        tokens = self.backbone.norm(tokens)
        
        cls_emb = self.backbone.head(tokens[:, 0])  # (B*V, embed_dim)
        patch_emb = tokens[:, 1:]  # (B*V, N_patches, vit_dim)
        
        proj = self.proj(cls_emb).reshape(N, V, -1).transpose(0, 1)
        
        return cls_emb, proj, (patch_emb, None)


def infer_square_patch_grid(num_tokens: int) -> int:
    grid = int(round(num_tokens ** 0.5))
    if grid * grid != num_tokens:
        raise ValueError(f"Expected square patch grid, got {num_tokens} tokens")
    return grid


def infer_encoder_patch_grid(model, default: int = 14) -> int:
    n_patches = getattr(model, 'n_patches', None)
    if isinstance(n_patches, int) and n_patches > 0:
        grid = int(round(n_patches ** 0.5))
        if grid * grid == n_patches:
            return grid

    input_size = getattr(model, 'input_size', None)
    patch_size = getattr(model, 'patch_size', None)
    if isinstance(input_size, int) and isinstance(patch_size, int) and patch_size > 0:
        return input_size // patch_size

    backbone = getattr(model, 'backbone', None)
    patch_embed = getattr(backbone, 'patch_embed', None)
    grid_size = getattr(patch_embed, 'grid_size', None)
    if isinstance(grid_size, tuple) and len(grid_size) == 2 and grid_size[0] == grid_size[1]:
        return int(grid_size[0])

    patch_size = getattr(patch_embed, 'patch_size', None)
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    if isinstance(input_size, int) and isinstance(patch_size, int) and patch_size > 0:
        return input_size // patch_size

    return default


def reshape_patch_scalar_map(values: torch.Tensor) -> torch.Tensor:
    grid = infer_square_patch_grid(values.shape[1])
    return values.squeeze(-1).reshape(values.shape[0], 1, grid, grid)


def reshape_flat_spatial_label(values: torch.Tensor) -> torch.Tensor:
    side = infer_square_patch_grid(values.shape[1])
    return values.reshape(values.shape[0], 1, side, side)


def reshape_multichannel_flat_spatial_label(values: torch.Tensor) -> torch.Tensor:
    side = infer_square_patch_grid(values.shape[-1])
    return values.reshape(values.shape[0], values.shape[1], side, side)


def resize_spatial_label_like(
    target: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.shape[-2:] == reference.shape[-2:]:
        return target, mask

    target = F.interpolate(
        target,
        size=reference.shape[-2:],
        mode='bilinear',
        align_corners=False,
    )
    mask = F.interpolate(mask, size=reference.shape[-2:], mode='nearest')
    return target, mask


class SIGReg(torch.nn.Module):
    """SIGReg objective - the core component of LeJEPA."""
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def load_wandb_key(project_root):
    """Load WandB API key from .wandb_key file."""
    key_file = Path(project_root) / ".wandb_key"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            os.environ["WANDB_API_KEY"] = key
            return True
    return False


def find_max_batch_size(model, arch, V, img_size=224, device="cuda", start=16, min_bs=4, max_cap=256, simple_mode=False, simple_channels=3):
    """Find optimal batch size for MM-LeJEPA."""
    if device != "cuda":
        return start
    
    model.train()
    dummy_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=True)
    
    bs = start
    last_working = 0
    
    print(f"Finding max batch size for arch={arch}, simple_mode={simple_mode}...")
    
    while bs <= max_cap:
        try:
            dummy_opt.zero_grad()
            
            if simple_mode:
                # Simple baseline: single modality input
                x = torch.randn(bs, V, simple_channels, img_size, img_size, device=device)
                with autocast(enabled=True):
                    _, proj = model(x)
                    loss = proj.mean()
                del x
            elif arch == "D":
                rgbd = torch.randn(bs, V, 4, img_size, img_size, device=device)
                with autocast(enabled=True):
                    _, proj = model(rgbd)
                    loss = proj.mean()
                del rgbd
            elif arch in ("B", "C"):
                cam = torch.randn(bs, V, 3, img_size, img_size, device=device)
                rng = torch.randn(bs, V, 5, img_size, img_size, device=device)
                with autocast(enabled=True):
                    _, proj = model(cam, rng)
                    loss = proj.mean()
                del cam, rng
            else:
                cam = torch.randn(bs, V, 3, img_size, img_size, device=device)
                pts = torch.randn(bs, 8192, 5, device=device)
                with autocast(enabled=True):
                    _, (cam_proj, lidar_proj) = model(cam, pts)
                    loss = cam_proj.mean() + lidar_proj.mean()
                del cam, pts
            
            scaler.scale(loss).backward()
            scaler.step(dummy_opt)
            scaler.update()
            
            last_working = bs
            print(f"  Batch size {bs} OK")
            
            del loss
            torch.cuda.empty_cache()
            
            bs *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch size {bs} OOM")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    del dummy_opt, scaler
    torch.cuda.empty_cache()
    
    final_bs = int(last_working * 0.9) if last_working > 0 else min_bs
    print(f"Using batch size {final_bs}")
    return max(min_bs, final_bs)


def get_spatial_label(labels, k, device):
    """Extract the probe-view label from a possibly stacked label tensor.

    If the tensor is stacked across views (B, V, ...), the *last* view
    (the clean probe view) is returned.  Otherwise it is returned as-is.
    """
    # Keys whose dim-1 is a feature dimension (not a view index)
    _FEATURE_KEYS = {
        "depth_grid", "depth_grid_mask",
        "depth_grid_hr", "depth_grid_mask_hr",
        "grid_occupancy", "grid_occupancy_car", "grid_occupancy_ped", "grid_occupancy_hr",
    }
    lbl = labels[k]
    if isinstance(lbl, torch.Tensor):
        # 3D Tensor: (B, V, D) — always slice last view
        if lbl.dim() == 3:
            return lbl[:, -1].to(device)
        # 2D Tensor: (B, V) scalar labels → slice, OR (B, D) feature → keep as-is
        if lbl.dim() == 2 and k not in _FEATURE_KEYS:
            return lbl[:, -1].to(device)
    # Fallback for Global labels (B, ...) or non-stacked items
    return lbl.to(device)


def count_invalid_grads(params):
    """Count non-finite gradient tensors and values for a parameter list."""
    invalid_tensors = 0
    invalid_values = 0
    for p in params:
        if p.grad is None:
            continue
        finite_mask = torch.isfinite(p.grad)
        if not torch.all(finite_mask):
            invalid_tensors += 1
            invalid_values += (~finite_mask).sum().item()
    return invalid_tensors, invalid_values


def evaluate_modality_dropout(net, probes, test_loader, arch, device):
    """
    Evaluate model robustness with modality dropout.
    
    Tests three conditions:
    1. Both modalities present (baseline)
    2. Camera only (LiDAR/range zeroed out)
    3. LiDAR/Range only (camera zeroed out)
    
    Returns dict with accuracy metrics for each condition.
    
    Note: For Option B (separate passes), zeroing one modality doesn't affect
    the other's embeddings since they don't interact within the forward pass.
    For Option C (true fusion), zeroing will affect outputs due to cross-attention.
    """
    net.eval()
    probes.eval()
    
    results = {
        "both": {"correct_cam": 0, "correct_scene": 0, "total": 0},
        "camera_only": {"correct_cam": 0, "correct_scene": 0, "total": 0},
        "lidar_only": {"correct_cam": 0, "correct_scene": 0, "total": 0},
    }
    
    # Helper recursive zeros_like
    def zeros_like_tree(data):
        if isinstance(data, dict):
             return {k: zeros_like_tree(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
             return torch.zeros_like(data)
        return data

    with torch.inference_mode():
        for cam_views, modality2, labels in test_loader:
            cam_views = to_device(cam_views, device)
            modality2 = to_device(modality2, device)
            B, _ = get_input_stats(cam_views)
            
            y_scene = labels["scene"].to(device)
            y_cam = labels["camera"].to(device)
            
            with autocast(enabled=(device == "cuda")):
                # 1. Both modalities
                if arch == "D":
                    # For RGBD, cam_views contains the full RGBD
                    emb_both, _ = net(cam_views)
                    cam_emb_both = emb_both
                elif arch in ("A", "E", "F"):
                    (cam_emb_both, _), _ = net(cam_views, modality2)
                else:  # B or C
                    emb_both, _ = net(cam_views, modality2)
                    _, V = get_input_stats(cam_views)
                    cam_emb_both = emb_both[:B*V]  # First B*V are camera embeddings

                # Reshape and select clean full view (last view)
                B_curr, V_curr = get_input_stats(cam_views)
                cam_emb_both = cam_emb_both.view(B_curr, V_curr, -1)[:, -1, :]
                
                pred_scene = probes["scene"](cam_emb_both).argmax(1)
                pred_cam = probes["camera"](cam_emb_both).argmax(1)
                results["both"]["correct_scene"] += (pred_scene == y_scene).sum().item()
                results["both"]["correct_cam"] += (pred_cam == y_cam).sum().item()
                results["both"]["total"] += B
                
                # 2. Camera only (zero out LiDAR/range/depth)
                if arch == "D":
                    # Zero out depth channel (channel 3)
                    cam_only = cam_views.clone() if isinstance(cam_views, torch.Tensor) else {k: v.clone() for k, v in cam_views.items()}
                    
                    if isinstance(cam_only, dict):
                        for k in cam_only:
                            v = cam_only[k]
                            if isinstance(v, torch.Tensor) and v.dim() >= 4 and v.shape[-3] > 3:
                                if v.dim() == 5: v[:, :, 3, :, :] = 0
                                else: v[:, 3, :, :] = 0
                    else:
                        if cam_only.dim() == 5: cam_only[:, :, 3, :, :] = 0
                        else: cam_only[:, 3, :, :] = 0
                        
                    emb_cam, _ = net(cam_only)
                    cam_emb_cam = emb_cam
                elif arch in ("A", "E", "F"):
                    modality2_zero = zeros_like_tree(modality2)
                    (cam_emb_cam, _), _ = net(cam_views, modality2_zero)
                else:  # B or C
                    modality2_zero = zeros_like_tree(modality2)
                    emb_cam, _ = net(cam_views, modality2_zero)
                    _, V = get_input_stats(cam_views)
                    cam_emb_cam = emb_cam[:B*V]

                cam_emb_cam = cam_emb_cam.view(B_curr, V_curr, -1)[:, -1, :]
                
                pred_scene = probes["scene"](cam_emb_cam).argmax(1)
                pred_cam = probes["camera"](cam_emb_cam).argmax(1)
                results["camera_only"]["correct_scene"] += (pred_scene == y_scene).sum().item()
                results["camera_only"]["correct_cam"] += (pred_cam == y_cam).sum().item()
                results["camera_only"]["total"] += B
                
                # 3. LiDAR/Range/Depth only (zero out camera/RGB)
                if arch == "D":
                    # Zero out RGB channels (channels 0-2)
                    depth_only = cam_views.clone() if isinstance(cam_views, torch.Tensor) else {k: v.clone() for k, v in cam_views.items()}
                    
                    if isinstance(depth_only, dict):
                        for k in depth_only:
                             v = depth_only[k]
                             if isinstance(v, torch.Tensor) and v.dim() >= 4 and v.shape[-3] >= 3:
                                 if v.dim() == 5: v[:, :, :3, :, :] = 0
                                 else: v[:, :3, :, :] = 0
                    else:
                        if depth_only.dim() == 5: depth_only[:, :, :3, :, :] = 0
                        else: depth_only[:, :3, :, :] = 0
                        
                    emb_depth, _ = net(depth_only)
                    lidar_emb = emb_depth
                    # Reshape for Arch D (depth only)
                    lidar_emb = lidar_emb.view(B_curr, V_curr, -1)[:, -1, :]
                elif arch in ("A", "E", "F"):
                    cam_zero = zeros_like_tree(cam_views)
                    (_, lidar_emb), _ = net(cam_zero, modality2)
                    if arch in ("E", "F"):
                         lidar_emb = lidar_emb.view(B_curr, V_curr, -1)[:, -1, :]
                    # For Arch A, do nothing (it's already B, D for the single point cloud).
                else:  # B or C
                    cam_zero = zeros_like_tree(cam_views)
                    emb_lidar, _ = net(cam_zero, modality2)
                    _, V = get_input_stats(cam_views)
                    # In B/C, output is cat([cam_emb, lidar_emb]). 
                    # cam_emb is first B*V (zeroed input). lidar_emb is second B*V.
                    lidar_emb = emb_lidar[B*V:] 
                    lidar_emb = lidar_emb.view(B_curr, V_curr, -1)[:, -1, :]
                
                pred_scene = probes["scene"](lidar_emb).argmax(1)
                pred_cam = probes["camera"](lidar_emb).argmax(1)
                results["lidar_only"]["correct_scene"] += (pred_scene == y_scene).sum().item()
                results["lidar_only"]["correct_cam"] += (pred_cam == y_cam).sum().item()
                results["lidar_only"]["total"] += B
    
    # Compute accuracies
    metrics = {}
    for cond in ["both", "camera_only", "lidar_only"]:
        total = results[cond]["total"]
        if total > 0:
            metrics[f"{cond}/acc_scene"] = results[cond]["correct_scene"] / total
            metrics[f"{cond}/acc_camera"] = results[cond]["correct_cam"] / total
    
    return metrics


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    """Main training loop for MM-LeJEPA with enhanced probes."""
    
    # Config defaults
    arch = getattr(cfg, 'arch', 'B')
    lidar_mode = getattr(cfg, 'lidar_mode', 'auto')  # range, depth, points, or auto
    modality_dropout = getattr(cfg, 'modality_dropout', 0.0)
    vit_size = str(getattr(cfg, 'vit_size', 'small')).lower()  # 'small', 'base', 'large'
    num_global_views = int(getattr(cfg, 'V', 2))
    global_crops_scale = getattr(cfg, 'global_crops_scale', (0.4, 1.0))
    local_crops_scale = getattr(cfg, 'local_crops_scale', (0.05, 0.4))
    local_crops_number = getattr(cfg, 'local_crops_number', 4)
    det_seg_label_mode = getattr(cfg, 'det_seg_label_mode', 'both')
    seg_filter_mode = getattr(cfg, 'seg_filter_mode', 'none')  # 'none', 'camera_seg', 'lidar_seg', 'any_seg'
    num_workers = getattr(cfg, 'num_workers', -1)  # -1 for smart auto-detection
    no_fusion = getattr(cfg, 'no_fusion', False)  # If True, zeroes out LiDAR input
    rgb_zero = getattr(cfg, 'rgb_zero', False)  # If True, zeroes out RGB input (LiDAR-only)
    freeze_encoder = getattr(cfg, 'freeze_encoder', False)  # If True, trains only probes
    pretrained_encoder_path = getattr(cfg, 'pretrained_encoder_path', None)
    probe_only_training = bool(getattr(cfg, 'probe_only_training', False))
    turbo = getattr(cfg, 'turbo', False)
    val_freq = getattr(cfg, 'val_freq', 0)
    probe_train_freq = getattr(cfg, 'probe_train_freq', 1)  # Train patch probes every N batches (1=every, 2=every other)
    enable_patch_probes = getattr(cfg, 'enable_patch_probes', True)
    estimate_encoder_train_compute = bool(getattr(cfg, 'estimate_encoder_train_compute', False))
    estimate_warmup_batches = max(0, int(getattr(cfg, 'estimate_warmup_batches', 1)))
    estimate_measure_batches = max(1, int(getattr(cfg, 'estimate_measure_batches', 1)))
    disable_probe_training = bool(getattr(cfg, 'disable_probe_training', False))
    encoder_only_mode = bool(getattr(cfg, 'encoder_only_mode', False))
    patch_bbox2d_loss_weight = getattr(cfg, 'patch_bbox2d_loss_weight', 1.0)
    patch_bbox2d_min_box_grid_cells = float(getattr(cfg, 'patch_bbox2d_min_box_grid_cells', 0.0))
    patch_bbox2d_probe_type = str(getattr(cfg, 'patch_bbox2d_probe_type', 'centernet')).lower()
    patch_bbox2d_upsample_factor = int(getattr(cfg, 'patch_bbox2d_upsample_factor', 2))
    patch_bbox3d_loss_weight = getattr(cfg, 'patch_bbox3d_loss_weight', 0.03)
    patch_spatial_bbox3d_loss_weight = getattr(cfg, 'patch_spatial_bbox3d_loss_weight', 0.05)
    patch_seg_loss_weight = getattr(cfg, 'patch_seg_loss_weight', 0.005)
    patch_panoptic_seg_loss_weight = getattr(cfg, 'patch_panoptic_seg_loss_weight', 0.005)
    patch_old_depth_loss_weight = getattr(cfg, 'patch_old_depth_loss_weight', 0.0025)
    patch_depth_map_loss_weight = getattr(cfg, 'patch_depth_map_loss_weight', 0.0025)
    patch_occupancy_map_loss_weight = float(getattr(cfg, 'patch_occupancy_map_loss_weight', 0.0))
    patch_box_seg_loss_weight = float(getattr(cfg, 'patch_box_seg_loss_weight', 0.0))
    patch_occupancy_map_upsample_factor = int(getattr(cfg, 'patch_occupancy_map_upsample_factor', 2))
    flir_dual_label_probes = bool(getattr(cfg, 'flir_dual_label_probes', False))
    flir_probe_label_modes_cfg = getattr(cfg, 'flir_probe_label_modes', None)
    occupancy_grid_size = int(getattr(cfg, 'occupancy_grid_size', 28))
    legacy_linear_probe_setup = getattr(cfg, 'legacy_linear_probe_setup', False)
    probe_train_sensor_drop = getattr(cfg, 'probe_train_sensor_drop', False)
    probe_eval_rgb_only = getattr(cfg, 'probe_eval_rgb_only', True)
    probe_eval_lidar_only = getattr(cfg, 'probe_eval_lidar_only', True)
    probe_forward_chunk_size = max(0, int(getattr(cfg, 'probe_forward_chunk_size', 0)))
    encoder_forward_chunk_size = max(0, int(getattr(cfg, 'encoder_forward_chunk_size', 0)))
    fusion_aux_train_freq = max(1, int(getattr(cfg, 'fusion_aux_train_freq', 1)))
    flir_loader_max_workers = max(0, int(getattr(cfg, 'flir_loader_max_workers', 8)))
    flir_loader_prefetch_factor = max(1, int(getattr(cfg, 'flir_loader_prefetch_factor', 1)))
    flir_loader_persistent_workers = bool(getattr(cfg, 'flir_loader_persistent_workers', False))
    flir_loader_pin_memory = bool(getattr(cfg, 'flir_loader_pin_memory', False))
    probe_lr = float(getattr(cfg, 'probe_lr', 1e-3))
    patch_probe_lr = float(getattr(cfg, 'patch_probe_lr', probe_lr))
    probe_weight_decay = float(getattr(cfg, 'probe_weight_decay', 1e-7))
    patch_probe_weight_decay = float(getattr(cfg, 'patch_probe_weight_decay', probe_weight_decay))
    track_patch_mse = getattr(cfg, 'track_patch_mse', True)
    patch_mse_probe_freq = max(1, int(getattr(cfg, 'patch_mse_probe_freq', 1)))
    track_encoder_compute = getattr(cfg, 'track_encoder_compute', True)
    flop_profile_freq = max(1, int(getattr(cfg, 'flop_profile_freq', 50)))  # Measure full FLOPs every N batches
    probe_view_mode = str(getattr(cfg, 'probe_view_mode', 'clean_last')).lower()  # clean_last | random_global
    if probe_view_mode not in {'clean_last', 'random_global'}:
        print(f"⚠️  Unknown probe_view_mode='{probe_view_mode}', falling back to 'clean_last'")
        probe_view_mode = 'clean_last'
    if patch_bbox2d_probe_type not in {'centernet', 'slot', 'both'}:
        raise ValueError(f"patch_bbox2d_probe_type must be 'centernet', 'slot', or 'both', got {patch_bbox2d_probe_type}")
    flir_probe_label_modes = normalize_flir_probe_label_modes(
        flir_probe_label_modes_cfg,
        enable_dual=flir_dual_label_probes,
    )
    
    # Novel multi-modal loss options (all default False for backward compatibility)
    cross_modal_sigreg = getattr(cfg, 'cross_modal_sigreg', False)  # Idea 1: SigReg across modalities
    modality_invariance = getattr(cfg, 'modality_invariance', False)  # Idea 2: Invariance to dropout
    gradient_balance = getattr(cfg, 'gradient_balance', False)  # Idea 3: Balance gradients per modality
    separate_projections = getattr(cfg, 'separate_projections', False)  # Idea 4: Separate then concat
    direct_alignment = getattr(cfg, 'direct_alignment', False)  # Idea 5: MSE between modalities (JEPA-style)
    shared_trunk_contrastive = getattr(cfg, 'shared_trunk_contrastive', False)  # Idea 6: Shared trunk + pairwise alignment
    shared_trunk_separate_sigreg = getattr(cfg, 'shared_trunk_separate_sigreg', False) # Idea 7: Shared, pairwise, separate SigReg
    patch_alignment = getattr(cfg, 'patch_alignment', False)  # Idea 8: Align patch embeddings (not just CLS)
    partial_dim_alignment = getattr(cfg, 'partial_dim_alignment', False)  # Idea 9: Reduced LiDAR dim, partial RGB alignment
    partial_dim_ratio = getattr(cfg, 'partial_dim_ratio', 1/3)  # Ratio of RGB dims to align with LiDAR (1/3 = one third)
    configured_dataset_name = str(getattr(cfg, 'dataset', 'nuscenes')).lower()
    input_img_size = int(getattr(cfg, 'flir_img_size', 224)) if configured_dataset_name == 'flir' else 224
    input_local_img_size = int(getattr(cfg, 'flir_local_img_size', 96)) if configured_dataset_name == 'flir' else 96
    probe_img_size = int(getattr(cfg, 'probe_img_size', input_img_size))
    probe_train_img_size = int(getattr(cfg, 'probe_train_img_size', probe_img_size))
    flir_resize_mode = str(getattr(cfg, 'flir_resize_mode', 'center_crop')).lower()
    if configured_dataset_name == 'flir' and flir_resize_mode not in {'center_crop', 'letterbox'}:
        raise ValueError(f"flir_resize_mode must be 'center_crop' or 'letterbox', got {flir_resize_mode}")
    if pretrained_encoder_path and probe_only_training:
        freeze_encoder = True
    
    # NEW scenarios (2026-02-04)
    lidar_sigreg = getattr(cfg, 'lidar_sigreg', False)  # Apply SIGReg specifically to LiDAR embeddings
    patch_direct_alignment = getattr(cfg, 'patch_direct_alignment', False)  # Patch alignment + CLS direct alignment
    patch_sigreg = getattr(cfg, 'patch_sigreg', False)  # Apply SIGReg on patch embeddings to prevent collapse
    single_cls_masked_sigreg = getattr(cfg, 'single_cls_masked_sigreg', False)
    
    # NEW scenarios (2026-02-14): Combined patch embedding scenarios
    patch_align_masked_sigreg = getattr(cfg, 'patch_align_masked_sigreg', False)
    patch_align_patch_sigreg = getattr(cfg, 'patch_align_patch_sigreg', False)
    fusion_tokens_sigreg = getattr(cfg, 'fusion_tokens_sigreg', False)
    fusion_triplet_alignment = getattr(cfg, 'fusion_triplet_alignment', False)
    fusion_tokens_variant = str(getattr(cfg, 'fusion_tokens_variant', 'prune_after_first')).lower()
    fusion_start_layer = int(getattr(cfg, 'fusion_start_layer', 0) or 0)
    fusion_joint_sigreg_only = getattr(cfg, 'fusion_joint_sigreg_only', False)
    fusion_skip_aux_sigreg = getattr(cfg, 'fusion_skip_aux_sigreg', False)
    # encoder_forward_chunk_size left at 0 (no chunking) unless explicitly set
    lidar_rope_rgb = getattr(cfg, 'lidar_rope_rgb', False)
    patch_align_weight = getattr(cfg, 'patch_align_weight', 0.1)
    patch_sigreg_weight_new = getattr(cfg, 'patch_sigreg_weight', 0.05)
    
    # Novel Regularizers (Beyond SIGReg) - GMM, Sinkhorn, Spectral
    gmm_regularizer = getattr(cfg, 'gmm_regularizer', False)
    sinkhorn_regularizer = getattr(cfg, 'sinkhorn_regularizer', False)
    spectral_regularizer = getattr(cfg, 'spectral_regularizer', False)
    replace_sigreg = getattr(cfg, 'replace_sigreg', False)  # If True, novel regularizer replaces SIGReg entirely
    
    # Novel regularizer hyperparameters
    gmm_num_prototypes = getattr(cfg, 'gmm_num_prototypes', 10)
    gmm_temperature = getattr(cfg, 'gmm_temperature', 0.1)
    sinkhorn_epsilon = getattr(cfg, 'sinkhorn_epsilon', 0.1)
    sinkhorn_iters = getattr(cfg, 'sinkhorn_iters', 5)
    spectral_mode = getattr(cfg, 'spectral_mode', 'isotropy')  # 'isotropy', 'barrier', or 'hybrid'
    novel_reg_weight = getattr(cfg, 'novel_reg_weight', 0.5)  # Weight for novel regularizer loss
    
    # ============================================
    # BASELINE SCENARIOS (VICReg, InfoNCE, DINOv3, ImageBind, MultiMAE, Late/Early Fusion)
    # ============================================
    use_vicreg = getattr(cfg, 'use_vicreg', False)
    use_infonce = getattr(cfg, 'use_infonce', False)
    use_adasignce = getattr(cfg, 'use_adasignce', False)
    use_signce = getattr(cfg, 'use_signce', False)
    use_dinov3_frozen = getattr(cfg, 'use_dinov3_frozen', getattr(cfg, 'use_dinov2_frozen', False))
    use_dinov3_pretrained = getattr(cfg, 'use_dinov3_pretrained', False)
    use_dinov3_scratch = getattr(cfg, 'use_dinov3_scratch', False)
    use_imagebind = getattr(cfg, 'use_imagebind', False)
    imagebind_use_clean_probe_views = bool(getattr(cfg, 'imagebind_use_clean_probe_views', True))
    imagebind_temperature = float(getattr(cfg, 'imagebind_temperature', 0.07))
    use_multimae = getattr(cfg, 'use_multimae', False)
    use_multimae_exact = getattr(cfg, 'use_multimae_exact', False)
    use_multimae_exact_mt = getattr(cfg, 'use_multimae_exact_mt', False)
    use_multimae_exact_segplus = getattr(cfg, 'use_multimae_exact_segplus', False)
    return_multimae_view_labels = bool(use_multimae_exact_mt or use_multimae_exact_segplus)
    use_late_fusion = getattr(cfg, 'use_late_fusion', False)
    late_fusion_patch_sigreg = getattr(cfg, 'late_fusion_patch_sigreg', False)
    use_early_fusion = getattr(cfg, 'use_early_fusion', False)
    use_whitened_infonce = getattr(cfg, 'use_whitened_infonce', False)
    use_eigennce = getattr(cfg, 'use_eigennce', False)
    use_fusion_dino = getattr(cfg, 'use_fusion_dino', False)
    use_frustum_slots = getattr(cfg, 'use_frustum_slots', False)
    use_rgb_dino = getattr(cfg, 'use_rgb_dino', False)
    short_budget_dino = bool(getattr(cfg, 'short_budget_dino', False))
    short_budget_dino_aggressive = bool(getattr(cfg, 'short_budget_dino_aggressive', False))
    dino_use_sinkhorn_teacher_requested = bool(getattr(cfg, 'dino_use_sinkhorn_teacher', False))
    dino_use_koleo = bool(getattr(cfg, 'dino_use_koleo', False))
    dino_use_ibot = bool(getattr(cfg, 'dino_use_ibot', False))
    dino_use_pre_head_features = bool(getattr(cfg, 'dino_use_pre_head_features', False))
    dino_use_param_groups = bool(getattr(cfg, 'dino_use_param_groups', False))
    dino_grad_clip_norm = float(getattr(cfg, 'dino_grad_clip_norm', 3.0))
    dino_use_official_augs = bool(getattr(cfg, 'dino_use_official_augs', True))
    dino_global_ignore_diagonal = bool(getattr(cfg, 'dino_global_ignore_diagonal', True))

    dino_norm_last_layer = bool(getattr(cfg, 'dino_norm_last_layer', True))
    dino_student_temp = float(getattr(cfg, 'dino_student_temp', 0.1))
    dino_teacher_temp = float(getattr(cfg, 'dino_teacher_temp', 0.07))
    dino_teacher_temp_warmup_start = float(getattr(cfg, 'dino_teacher_temp_warmup_start', 0.04))
    dino_teacher_temp_warmup_epochs = int(getattr(cfg, 'dino_teacher_temp_warmup_epochs', 30))
    dino_teacher_momentum = float(getattr(cfg, 'dino_teacher_momentum', 0.996))
    dino_short_budget_default_warmup = max(1, min(4, int(getattr(cfg, 'epochs', 10)) // 2))

    if use_fusion_dino and short_budget_dino:
        local_crops_number = min(local_crops_number, int(getattr(cfg, 'short_budget_dino_local_crops', 2)))
        dino_norm_last_layer = bool(getattr(cfg, 'short_budget_dino_norm_last_layer', False))
        dino_teacher_momentum = float(getattr(cfg, 'short_budget_dino_teacher_momentum', 0.99))
        dino_teacher_temp_warmup_epochs = min(
            dino_teacher_temp_warmup_epochs,
            int(getattr(cfg, 'short_budget_dino_warmup_epochs', dino_short_budget_default_warmup))
        )
        if not short_budget_dino_aggressive:
            print(
                "🎯 SHORT-BUDGET DINO MODE: "
                f"local_crops={local_crops_number}, "
                f"warmup_epochs={dino_teacher_temp_warmup_epochs}, "
                f"teacher_momentum={dino_teacher_momentum:.4f}, "
                f"norm_last_layer={dino_norm_last_layer}"
            )

    if use_fusion_dino and short_budget_dino_aggressive:
        local_crops_number = int(getattr(cfg, 'short_budget_dino_aggressive_local_crops', 8))
        dino_teacher_temp_warmup_epochs = int(getattr(cfg, 'short_budget_dino_aggressive_warmup_epochs', 1))
        print(
            "🎯 AGGRESSIVE SHORT-BUDGET DINO MODE: "
            f"local_crops={local_crops_number}, "
            f"warmup_epochs={dino_teacher_temp_warmup_epochs}, "
            f"teacher_momentum={dino_teacher_momentum:.4f}, "
            f"norm_last_layer={dino_norm_last_layer}"
        )

    if use_imagebind:
        # ImageBind-style training here uses one clean paired RGB/depth view per
        # sample with a symmetric cross-modal InfoNCE objective.
        local_crops_number = 0
        track_patch_mse = False
        if imagebind_use_clean_probe_views:
            probe_view_mode = 'clean_last'
        print(
            "🎯 IMAGEBIND BASELINE MODE: trainable dual encoders, no local crops, "
            f"clean_probe_views={imagebind_use_clean_probe_views}"
        )

    if use_multimae_exact or use_multimae_exact_mt or use_multimae_exact_segplus:
        if local_crops_number != 0:
            print("🎯 MULTIMAE EXACT MODE: disabling local crops because the exact adapterized decoder expects a fixed global patch grid")
        local_crops_number = 0

    if use_fusion_dino and dino_use_pre_head_features:
        print("🎯 DINO PRE-HEAD FEATURES MODE: using raw DINO backbone CLS features for the aggressive RGB baseline")

    pretrained_backbone = bool(getattr(cfg, 'pretrained_backbone', False))
    pretrained_backbone_source = str(getattr(cfg, 'pretrained_backbone_source', 'dinov3')).lower()
    pretrained_backbone_param_group = bool(getattr(cfg, 'pretrained_backbone_param_group', False))
    pretrained_trunk_lr = float(getattr(cfg, 'pretrained_trunk_lr', cfg.lr))
    encoder_aux_lr = float(getattr(cfg, 'encoder_aux_lr', cfg.lr))
    pretrained_trunk_weight_decay = float(getattr(cfg, 'pretrained_trunk_weight_decay', 5e-2))
    encoder_aux_weight_decay = float(getattr(cfg, 'encoder_aux_weight_decay', 5e-2))
    pretrained_trunk_warmup_freeze_epochs = int(getattr(cfg, 'pretrained_trunk_warmup_freeze_epochs', 0))
    
    # Simple baseline mode (vanilla ViT like LeJEPA paper)
    simple_baseline = getattr(cfg, 'simple_baseline', False)
    simple_modality = getattr(cfg, 'simple_modality', 'rgb')  # 'rgb', 'lidar', or 'thermal'
    
    # Debug modes:
    # - debug: Full debug with real data, 1 batch (slow - tests dataloader + model)
    # - debug_model: Fast debug with random dummy data (fast - tests model/losses only)
    debug = getattr(cfg, 'debug', False)
    debug_model = getattr(cfg, 'debug_model', False)
    
    # Get epochs, defaulting to value from config or 10
    epochs = getattr(cfg, 'epochs', 10)

    if estimate_encoder_train_compute:
        disable_probe_training = True
        track_encoder_compute = True
        enable_patch_probes = False
        track_patch_mse = False
        probe_train_sensor_drop = False
        probe_eval_rgb_only = False
        probe_eval_lidar_only = False
        fusion_aux_train_freq = 1
        val_freq = 0
        print(
            "📐 ENCODER COMPUTE ESTIMATE MODE: "
            f"warmup_batches={estimate_warmup_batches}, "
            f"measure_batches={estimate_measure_batches}, "
            f"target_epochs={epochs}"
        )

    if encoder_only_mode:
        disable_probe_training = True
        enable_patch_probes = False
        track_patch_mse = False
        probe_train_sensor_drop = False
        probe_eval_rgb_only = False
        probe_eval_lidar_only = False
        val_freq = 0
        print(
            "⚡ ENCODER-ONLY MODE: disabling probe training/validation, patch probes, "
            "and dataset-side probe/label work"
        )

    if fusion_tokens_sigreg and not fusion_skip_aux_sigreg and fusion_aux_train_freq > 1:
        print(
            "⚡ FUSION AUX THROTTLE: "
            f"running RGB-only/LiDAR-only auxiliary SigReg every {fusion_aux_train_freq} steps "
            "with loss rescaling to preserve expected weight"
        )
    
    if debug_model:
        print("🚀 FAST DEBUG MODE: Using random dummy data (no dataloader)")
        epochs = 1
        debug = True  # Also enable regular debug flags
    elif debug:
        print("🐞 DEBUG MODE ENABLED: Running 1 epoch, 1 batch per loop")
        epochs = 1
    
    # Load WandB key
    if not debug_model:
        project_root = Path(cfg.dataroot).parent
        load_wandb_key(project_root)
    
    # Initialize logging
    run_name = None
    use_wandb = WANDB_AVAILABLE and getattr(cfg, 'wandb', True) and not debug_model
    trainer_started_wandb_run = False
    if use_wandb:
        if getattr(wandb, 'run', None) is None:
            wandb.init(project="MM-LeJEPA-nuScenes", config=dict(cfg))
            trainer_started_wandb_run = True
        run_obj = getattr(wandb, 'run', None)
        run_name = getattr(run_obj, 'name', None)
    
    if not run_name:
        from datetime import datetime
        run_name = f"mmlejepa_{arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'='*60}")
    if simple_baseline:
        print(f"SIMPLE BASELINE - Vanilla ViT-{vit_size.capitalize()}, Modality: {simple_modality}")
    else:
        print(f"MM-LeJEPA Training - Architecture: {arch}, LiDAR Mode: {lidar_mode}, ViT: {vit_size}")
    print(f"Baselines: No Fusion={no_fusion}, RGB Zero={rgb_zero}, Freeze Encoder={freeze_encoder}")
    print(f"{'='*60}")
    
    if turbo:
        print("🚀 TURBO MODE ENABLED")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Random seed - set to None for random, or a specific int for reproducibility
    seed = getattr(cfg, 'seed', None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"🎲 Random seed: {seed}")
    else:
        print("🎲 Random seed: None (non-deterministic)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Data splitting strategy
    split_strategy = getattr(cfg, 'split_strategy', 'random')
    official_val_mode = getattr(cfg, 'official_val_mode', 'auto')
    # Camera filter - restrict to specific cameras (e.g., ["CAM_FRONT"] for better object density)
    cameras = getattr(cfg, 'cameras', None)
    # Handle common shorthand for front camera only
    if getattr(cfg, 'front_camera_only', False):
        cameras = ["CAM_FRONT"]
        print("📷 FRONT CAMERA ONLY: Filtering to CAM_FRONT (highest object density)")
    elif cameras is not None:
        print(f"📷 Camera filter: {cameras}")
    # Limit number of validation batches for frequent validation
    val_batches_limit = getattr(cfg, 'val_batches_limit', None)
    # Optional legacy validation (intra-scene split)
    validate_legacy = getattr(cfg, 'validate_legacy', False)
    if estimate_encoder_train_compute:
        validate_legacy = False
    
    if split_strategy == 'scene_based':
        print("🛡️ ROBUST SPLITTING: Using scene-based split (70% Train / 10% Buffer / 20% Val)")
    else:
        print(f"ℹ️ SPLITTING: Using {split_strategy} strategy")
    
    # Create datasets (skip in debug_model mode for speed)
    if debug_model:
        print("🚀 FAST DEBUG: Skipping dataset loading...")
        dataset_name = getattr(cfg, 'dataset', 'nuscenes')
        # Create minimal dummy dataset info for model creation
        class DummyDataset:
            def __init__(self):
                self.num_scenes = 10
                self.num_cameras = 6
                self.num_locations = 4
            def __len__(self):
                return 100
        train_ds = DummyDataset()
        test_ds = DummyDataset()
        test_loader_legacy = None
        test_ds_legacy = None
        _collate_fn = mm_collate_fn
    else:
        # ── Dataset configuration ─────────────────────────────────
        dataset_name = getattr(cfg, 'dataset', 'nuscenes')
        lidar_aug_preset = getattr(cfg, 'lidar_aug_preset', 'none')
        lidar_aug_cfg = getattr(cfg, 'lidar_aug_cfg', None)
        copy_paste_preset = getattr(cfg, 'copy_paste_preset', 'none')
        copy_paste_cfg = getattr(cfg, 'copy_paste_cfg', None)
        gt_database_path = getattr(cfg, 'gt_database_path', None)

        def _maybe_tuple(val):
            if isinstance(val, (list, tuple)):
                return tuple(val)
            return val

        def _merge_aug_overrides(base_cfg, overrides):
            base = {} if base_cfg is None else dict(base_cfg)
            for k, v in overrides.items():
                if v is None:
                    continue
                base[k] = _maybe_tuple(v)
            return base or None

        # Allow sweep YAML to control augmentations via simple scalar keys
        # (e.g. lidar_aug_point_jitter_std=0.02) without needing to pass nested dicts.
        # These are merged into lidar_aug_cfg/copy_paste_cfg if present.
        lidar_override_map = {
            # Scene-level
            'lidar_aug_global_scaling_range': 'global_scaling_range',
            'lidar_aug_global_rotation_range': 'global_rotation_range',
            'lidar_aug_global_translation_std': 'global_translation_std',
            'lidar_aug_object_translation_std': 'object_translation_std',
            'lidar_aug_point_jitter_std': 'point_jitter_std',
            'lidar_aug_object_point_jitter_std': 'object_point_jitter_std',
            'lidar_aug_intensity_noise_std': 'intensity_noise_std',
            'lidar_aug_random_flip_x': 'random_flip_x',
            'lidar_aug_random_flip_y': 'random_flip_y',
            'lidar_aug_point_dropout_rate': 'point_dropout_rate',
            'lidar_aug_frustum_dropout_prob': 'frustum_dropout_prob',
            'lidar_aug_frustum_dropout_angle': 'frustum_dropout_angle',
            # Enable flags
            'lidar_aug_enable_global_scaling': 'enable_global_scaling',
            'lidar_aug_enable_global_rotation': 'enable_global_rotation',
            'lidar_aug_enable_global_translation': 'enable_global_translation',
            'lidar_aug_enable_object_translation': 'enable_object_translation',
            'lidar_aug_enable_point_jitter': 'enable_point_jitter',
            'lidar_aug_enable_intensity_noise': 'enable_intensity_noise',
        }
        cp_override_map = {
            'copy_paste_max_paste_objects': 'max_paste_objects',
            'copy_paste_paste_image': 'paste_image',
            'copy_paste_min_distance': 'min_distance',
            'copy_paste_max_distance': 'max_distance',
            'copy_paste_collision_threshold': 'collision_threshold',
            'copy_paste_height_offset_std': 'height_offset_std',
            'copy_paste_require_in_view': 'require_in_view',
            'copy_paste_min_box_px': 'min_box_px',
            'copy_paste_min_paste_in_view': 'min_paste_in_view',
            'copy_paste_enable_scene_aware_placement': 'enable_scene_aware_placement',
            'copy_paste_front_clearance_m': 'front_clearance_m',
            'copy_paste_image_gate_mode': 'image_gate_mode',
            'copy_paste_image_gate_center_frac': 'image_gate_center_frac',
            'copy_paste_image_gate_min_box_px': 'image_gate_min_box_px',
            'copy_paste_simulate_lidar_occlusion': 'simulate_lidar_occlusion',
            'copy_paste_occlusion_H': 'occlusion_H',
            'copy_paste_occlusion_W': 'occlusion_W',
            'copy_paste_occlusion_fov_up': 'occlusion_fov_up',
            'copy_paste_occlusion_fov_down': 'occlusion_fov_down',
            'copy_paste_occlusion_bin_radius': 'occlusion_bin_radius',
            'copy_paste_occlusion_depth_margin_m': 'occlusion_depth_margin_m',
        }

        lidar_overrides = {}
        if str(lidar_aug_preset).lower() != 'none':
            for cfg_key, aug_key in lidar_override_map.items():
                try:
                    if cfg_key in cfg and cfg.get(cfg_key) is not None:
                        lidar_overrides[aug_key] = cfg.get(cfg_key)
                except Exception:
                    pass

        cp_overrides = {}
        if str(copy_paste_preset).lower() != 'none':
            for cfg_key, aug_key in cp_override_map.items():
                try:
                    if cfg_key in cfg and cfg.get(cfg_key) is not None:
                        cp_overrides[aug_key] = cfg.get(cfg_key)
                except Exception:
                    pass

        lidar_aug_cfg = _merge_aug_overrides(lidar_aug_cfg, lidar_overrides)
        copy_paste_cfg = _merge_aug_overrides(copy_paste_cfg, cp_overrides)

        aug_kwargs = dict(
            lidar_aug_preset=lidar_aug_preset,
            lidar_aug_cfg=lidar_aug_cfg,
            copy_paste_preset=copy_paste_preset,
            copy_paste_cfg=copy_paste_cfg,
            gt_database_path=gt_database_path,
        )
        dino_aug_mode = 'official' if use_fusion_dino and dino_use_official_augs else 'default'
        dataset_include_probe_view = not encoder_only_mode
        dataset_encoder_only_labels = bool(
            encoder_only_mode and not (use_multimae_exact or use_multimae_exact_mt or use_multimae_exact_segplus)
        )
        precomputed_labels_path = ''
        val_precomputed = ''
        if dataset_name in {'waymo', 'nuscenes'}:
            precomputed_labels_path = getattr(cfg, 'precomputed_labels_path', '')
            if not precomputed_labels_path:
                default_precomputed = Path(__file__).parent / "cache" / "det_seg_labels_v2"
                if default_precomputed.exists():
                    precomputed_labels_path = str(default_precomputed)

            if precomputed_labels_path and not Path(precomputed_labels_path).exists():
                print(f"⚠️  Precomputed labels path not found: {precomputed_labels_path}")
                precomputed_labels_path = ''

            if precomputed_labels_path:
                _val_dir = Path(precomputed_labels_path) / 'validation'
                if _val_dir.exists() and any(_val_dir.glob('shard_*.zip')):
                    val_precomputed = str(_val_dir)
                else:
                    val_precomputed = precomputed_labels_path

        if dataset_name == 'waymo':
            from src.waymo_dataset import WaymoDataset, waymo_collate_fn
            waymo_dataroot = getattr(cfg, 'waymo_dataroot', getattr(cfg, 'dataroot', ''))
            train_ds = WaymoDataset(
                waymo_dataroot,
                split="train",
                arch=arch,
                lidar_mode=lidar_mode,
                V=num_global_views,
                global_crops_scale=global_crops_scale,
                local_crops_scale=local_crops_scale,
                local_crops_number=local_crops_number,
                det_seg_label_mode=det_seg_label_mode,
                seg_filter_mode=seg_filter_mode,
                probe_img_size=probe_train_img_size,
                occupancy_grid_size=occupancy_grid_size,
                modality_dropout=modality_dropout,
                legacy_mode=legacy_linear_probe_setup,
                include_probe_view=dataset_include_probe_view,
                encoder_only_labels=dataset_encoder_only_labels,
                precomputed_labels_path=precomputed_labels_path,
                dino_aug_mode=dino_aug_mode,
                return_multimae_view_labels=return_multimae_view_labels,
                **aug_kwargs,
            )
            test_ds = WaymoDataset(
                waymo_dataroot,
                split="val",
                arch=arch,
                lidar_mode=lidar_mode,
                V=1,
                local_crops_number=0,
                det_seg_label_mode=det_seg_label_mode,
                seg_filter_mode=seg_filter_mode,
                probe_img_size=probe_img_size,
                occupancy_grid_size=occupancy_grid_size,
                modality_dropout=0.0,
                legacy_mode=legacy_linear_probe_setup,
                include_probe_view=dataset_include_probe_view,
                encoder_only_labels=dataset_encoder_only_labels,
                precomputed_labels_path=val_precomputed,
            )
            # Override collate function for Waymo
            _collate_fn = waymo_collate_fn
        elif dataset_name == 'flir':
            from src.flir_dataset import FlirAdasDataset, flir_collate_fn

            flir_dataroot = getattr(cfg, 'flir_dataroot', getattr(cfg, 'dataroot', ''))
            flir_train_split = getattr(cfg, 'flir_train_split', 'train')
            flir_val_split = getattr(cfg, 'flir_val_split', 'val')
            flir_label_source = getattr(cfg, 'flir_label_source', None)
            if flir_label_source is None:
                flir_label_source = 'thermal' if simple_baseline and simple_modality in ('lidar', 'thermal') else 'rgb'
            train_ds = FlirAdasDataset(
                flir_dataroot,
                split=flir_train_split,
                arch=arch,
                lidar_mode='depth',
                V=num_global_views,
                global_crops_scale=global_crops_scale,
                local_crops_scale=local_crops_scale,
                local_crops_number=local_crops_number,
                img_size=input_img_size,
                local_img_size=input_local_img_size,
                probe_img_size=probe_train_img_size,
                occupancy_grid_size=occupancy_grid_size,
                modality_dropout=modality_dropout,
                legacy_mode=legacy_linear_probe_setup,
                include_probe_view=dataset_include_probe_view,
                encoder_only_labels=dataset_encoder_only_labels,
                det_seg_label_mode='bbox_only',
                detection_label_source=flir_label_source,
                dino_aug_mode=dino_aug_mode,
                resize_mode=flir_resize_mode,
            )
            test_ds = FlirAdasDataset(
                flir_dataroot,
                split=flir_val_split,
                arch=arch,
                lidar_mode='depth',
                V=1,
                local_crops_number=0,
                img_size=input_img_size,
                local_img_size=input_local_img_size,
                probe_img_size=probe_img_size,
                occupancy_grid_size=occupancy_grid_size,
                modality_dropout=0.0,
                legacy_mode=legacy_linear_probe_setup,
                include_probe_view=dataset_include_probe_view,
                encoder_only_labels=dataset_encoder_only_labels,
                det_seg_label_mode='bbox_only',
                detection_label_source=flir_label_source,
                resize_mode=flir_resize_mode,
            )
            _collate_fn = flir_collate_fn
        else:
            train_ds = MMNuScenesDataset(
                cfg.dataroot, 
                split="train", 
                arch=arch,
                lidar_mode=lidar_mode,
                V=num_global_views,
                global_crops_scale=global_crops_scale,
                local_crops_scale=local_crops_scale,
                local_crops_number=local_crops_number,
                det_seg_label_mode=det_seg_label_mode,
                modality_dropout=modality_dropout,
                split_strategy=split_strategy,
                legacy_mode=legacy_linear_probe_setup,
                include_probe_view=dataset_include_probe_view,
                encoder_only_labels=dataset_encoder_only_labels,
                official_val_mode=official_val_mode,
                precomputed_labels_path=precomputed_labels_path,
                dino_aug_mode=dino_aug_mode,
                cameras=cameras,
                return_multimae_view_labels=return_multimae_view_labels,
                **aug_kwargs,
            )
            test_ds = MMNuScenesDataset(
                cfg.dataroot, 
                split="val", 
                arch=arch,
                lidar_mode=lidar_mode,
                V=1,
                local_crops_number=0,
                det_seg_label_mode=det_seg_label_mode,
                modality_dropout=0.0,
                split_strategy=split_strategy,
                legacy_mode=legacy_linear_probe_setup,
                include_probe_view=dataset_include_probe_view,
                encoder_only_labels=dataset_encoder_only_labels,
                official_val_mode=official_val_mode,
                precomputed_labels_path=val_precomputed,
                cameras=cameras,
            )
            _collate_fn = mm_collate_fn

        test_loader_legacy = None
        test_ds_legacy = None
        if validate_legacy and dataset_name == 'nuscenes':
            print("🏛️ LEGACY VALIDATION ENABLED: Also validating on 'scene_based' intra-scene split (last 20% of ALL scenes)")
            test_ds_legacy = MMNuScenesDataset(
                cfg.dataroot, 
                split="val", 
                arch=arch,
                lidar_mode=lidar_mode,
                V=1,
                local_crops_number=0,
                det_seg_label_mode=det_seg_label_mode,
                modality_dropout=0.0,
                split_strategy='scene_based', # Forced legacy strategy
                legacy_mode=legacy_linear_probe_setup,
                official_val_mode=official_val_mode,
                precomputed_labels_path=val_precomputed,
                cameras=cameras,
            )
            print(f"Legacy Validation samples: {len(test_ds_legacy)}")
        elif validate_legacy:
            print(f"🏛️ LEGACY VALIDATION SKIPPED: dataset='{dataset_name}' does not support MMNuScenesDataset legacy validation")
        
        print(f"Training samples: {len(train_ds)}")
        print(f"Validation samples: {len(test_ds)}")
        if probe_train_img_size != probe_img_size:
            print(
                f"Probe resolution split: train probe views use {probe_train_img_size}, validation probe views use {probe_img_size}"
            )

    if dataset_name == 'flir' and simple_baseline and simple_modality == 'lidar':
        simple_modality = 'thermal'
    
    # Check if aligned mode is enabled (passed from sweep agent)
    aligned_mode = getattr(cfg, 'aligned_mode', False)
    
    # Determine channels for second modality
    # 'depth' or aligned_mode means 1-channel, otherwise 5-channel (range images)
    if lidar_mode == "depth" or aligned_mode:
        modality2_channels = 1
    else:
        modality2_channels = 5
    
    # Create model - use simple ViT for simple baselines, MM encoder otherwise
    if simple_baseline:
        # Simple baseline: vanilla ViT (like LeJEPA paper)
        # For modality2-only runs (LiDAR/depth or thermal), use 1-channel input.
        in_channels = 1 if simple_modality in ('lidar', 'thermal') else 3
        simple_pretrained_source = pretrained_backbone_source if pretrained_backbone else None
        net = ViTEncoder(
            proj_dim=cfg.proj_dim, 
            img_size=input_img_size,
            in_channels=in_channels,
            vit_size=vit_size,
            pretrained_source=simple_pretrained_source,
        ).to(device)
        if simple_pretrained_source is not None:
            print(
                f"🎯 SIMPLE BASELINE: Using ViT with {in_channels}-channel input, "
                f"initialized from {simple_pretrained_source.upper()}"
            )
        else:
            print(f"🎯 SIMPLE BASELINE: Using vanilla ViT with {in_channels}-channel input")
    elif use_dinov3_frozen:
        # DINOv3 frozen encoder baseline (RGB only, probes only)
        net = DINOv3FrozenEncoder(
            proj_dim=cfg.proj_dim,
            img_size=input_img_size,
            vit_size=vit_size,
        ).to(device)
        freeze_encoder = True  # Only train probes
        print(f"🎯 DINOv3 FROZEN BASELINE: Frozen ViT-{vit_size}, train probes only")
    elif use_dinov3_pretrained:
        # Native DINOv3 backbone with pretrained weights, fine-tuned under the
        # default objective. This avoids the hybrid transfer path used by the
        # simple_rgb_dinov3 scenario.
        net = DINOv3FrozenEncoder(
            proj_dim=cfg.proj_dim,
            img_size=input_img_size,
            vit_size=vit_size,
            pretrained=True,
            freeze_backbone=False,
            allow_random_fallback=False,
        ).to(device)
        freeze_encoder = False
        print(f"🎯 DINOv3 PRETRAINED BASELINE: Trainable native DINOv3 ViT-{vit_size}")
    elif use_dinov3_scratch:
        # DINOv3 architecture baseline from scratch under the current training objective.
        net = DINOv3ScratchEncoder(
            proj_dim=cfg.proj_dim,
            img_size=input_img_size,
            vit_size=vit_size,
        ).to(device)
        if dino_use_pre_head_features:
            net.use_pre_head_features = True
            for param in net.head.parameters():
                param.requires_grad = False
        print(f"🎯 DINOv3 SCRATCH BASELINE: Trainable ViT-{vit_size}, random init")
    elif use_imagebind:
        # ImageBind baseline: upstream ImageBind architecture in either
        # scratch-sized mode or official pretrained-Huge mode.
        imagebind_allow_timm_fallback = getattr(cfg, 'imagebind_allow_timm_fallback', True)
        try:
            ensure_imagebind_available(install_all_envs=getattr(cfg, 'imagebind_install_all_envs', True))
        except Exception as e:
            if not imagebind_allow_timm_fallback:
                raise
            print(f"⚠️  ImageBind auto-install failed ({e}); falling back to timm approximation")
        net = ImageBindEncoder(
            proj_dim=cfg.proj_dim,
            img_size=input_img_size,
            vit_size=vit_size,
            use_official_weights=getattr(cfg, 'imagebind_use_official', True),
            allow_timm_fallback=imagebind_allow_timm_fallback,
            imagebind_ckpt_path=getattr(cfg, 'imagebind_ckpt_path', None),
            force_timm_backbone=getattr(cfg, 'imagebind_force_timm_backbone', False),
        ).to(device)
        if pretrained_backbone:
            if pretrained_backbone_source not in {'dinov3', 'timm'}:
                raise ValueError(f"Unsupported pretrained_backbone_source={pretrained_backbone_source!r}")
            if not hasattr(net, 'initialize_from_pretrained'):
                raise ValueError("ImageBind encoder does not support shared pretrained initialization")
            net.initialize_from_pretrained(source=pretrained_backbone_source, vit_size=vit_size)
        print(
            "🎯 IMAGEBIND BASELINE: upstream RGB/depth ImageBind pipeline with paired InfoNCE "
            f"(official_weights={getattr(cfg, 'imagebind_use_official', True)})"
        )
    elif use_multimae_exact or use_multimae_exact_mt or use_multimae_exact_segplus:
        net = MultiMAEExactEncoder(
            proj_dim=cfg.proj_dim,
            img_size=input_img_size,
            vit_size=vit_size,
            mask_ratio=float(getattr(cfg, 'multimae_exact_mask_ratio', 0.75)),
            decoder_depth=int(getattr(cfg, 'multimae_exact_decoder_depth', 2)),
            decoder_dim=int(getattr(cfg, 'multimae_exact_decoder_dim', 256)),
            depth_channels=modality2_channels,
            enable_semseg=(use_multimae_exact_mt or use_multimae_exact_segplus),
            enable_panoptic=use_multimae_exact_segplus,
            num_semseg_classes=NUM_SIMPLIFIED_SEG_CLASSES,
            semseg_aux_weight=float(getattr(cfg, 'multimae_exact_semseg_aux_weight', 1.0)),
            panoptic_aux_weight=float(getattr(cfg, 'multimae_exact_panoptic_aux_weight', 1.0)),
        ).to(device)
        if pretrained_backbone:
            if pretrained_backbone_source not in {'dinov3', 'timm'}:
                raise ValueError(f"Unsupported pretrained_backbone_source={pretrained_backbone_source!r}")
            if not hasattr(net, 'initialize_from_pretrained'):
                raise ValueError("MultiMAE exact encoder does not support shared pretrained initialization")
            net.initialize_from_pretrained(source=pretrained_backbone_source, vit_size=vit_size)
        if use_multimae_exact_segplus:
            variant_name = "segplus"
        elif use_multimae_exact_mt:
            variant_name = "multitask"
        else:
            variant_name = "self-supervised"
        print(f"🎯 MULTIMAE EXACT BASELINE: Official-style adapterized reconstruction ({variant_name})")
    elif use_multimae:
        # MultiMAE baseline: multi-modal masked autoencoder
        net = MultiMAEEncoder(
            proj_dim=cfg.proj_dim,
            img_size=input_img_size,
            vit_size=vit_size,
            mask_ratio=0.75,
            depth_channels=modality2_channels,
        ).to(device)
        print(f"🎯 MULTIMAE BASELINE: Multi-Modal Masked Autoencoder")
    elif getattr(cfg, 'use_mdm', False):
        # MDM baseline: Masked Depth Modeling (LingBot-Depth inspired)
        net = MaskedDepthModelEncoder(
            proj_dim=cfg.proj_dim,
            img_size=input_img_size,
            vit_size=vit_size,
            depth_mask_ratio=0.6,
            freeze_rgb_backbone=getattr(cfg, 'mdm_freeze_rgb', False),
            depth_channels=modality2_channels,
        ).to(device)
        print(f"🎯 MDM BASELINE: Masked Depth Modeling (LingBot-Depth inspired)")
    elif use_late_fusion:
        # Late Fusion baseline: separate encoders, concatenated at end
        net = LateFusionEncoder(
            proj_dim=cfg.proj_dim,
            img_size=input_img_size,
            vit_size=vit_size,
        ).to(device)
        print(f"🎯 LATE FUSION BASELINE: Separate ViTs, concatenated features")
    else:
        # Multi-modal encoder
        net = create_mm_encoder(
            arch=arch, 
            proj_dim=cfg.proj_dim, 
            img_size=input_img_size,
            second_modality_channels=modality2_channels,
            aligned_mode=aligned_mode,
            vit_size=vit_size,
        ).to(device)
        
        # Override encoder for special scenarios
        if fusion_tokens_sigreg:
            net = MMEncoderC_FusionTokens(
                proj_dim=cfg.proj_dim,
                img_size=input_img_size,
                range_channels=modality2_channels,
                aligned_mode=aligned_mode,
                vit_size=vit_size,
                attention_mode=fusion_tokens_variant,
                fusion_start_layer=fusion_start_layer,
            ).to(device)
            if pretrained_backbone:
                if pretrained_backbone_source == 'dinov3':
                    initialize_module_from_dinov3(net, vit_size=vit_size)
                elif pretrained_backbone_source == 'timm':
                    initialize_module_from_timm_vit(net, vit_size=vit_size)
                else:
                    raise ValueError(f"Unsupported pretrained_backbone_source={pretrained_backbone_source!r}")
                print(
                    f"🎯 FUSION TOKENS ENCODER: attention_mode={fusion_tokens_variant}, fusion_start_layer={fusion_start_layer}, "
                    f"initialized from {pretrained_backbone_source.upper()}"
                )
            else:
                print(
                    f"🎯 FUSION TOKENS ENCODER: attention_mode={fusion_tokens_variant}, "
                    f"fusion_start_layer={fusion_start_layer}"
                )
        elif use_frustum_slots:
            net = MMEncoderC_FrustumSlots(
                proj_dim=cfg.proj_dim,
                img_size=input_img_size,
                range_channels=modality2_channels,
                aligned_mode=aligned_mode,
                vit_size=vit_size,
                num_slots=int(getattr(cfg, 'frustum_num_slots', 8)),
                slot_layers=int(getattr(cfg, 'frustum_slot_layers', 3)),
            ).to(device)
            print("🎯 FRUSTUM SLOTS ENCODER: paired RGB-depth frusta pooled into learned slots")
        elif lidar_rope_rgb:
            net = MMEncoderC_LiDARRoPE(
                proj_dim=cfg.proj_dim,
                img_size=input_img_size,
                aligned_mode=aligned_mode,
                vit_size=vit_size,
            ).to(device)
            print(f"🎯 LIDAR RoPE ENCODER: Using depth-conditioned RoPE on RGB ViT")
        
        if aligned_mode:
            print(f"🎯 ALIGNED MODE: Using camera-aligned LiDAR representations")

    if pretrained_encoder_path:
        _load_pretrained_encoder_into_model(net, pretrained_encoder_path)
        print("🎯 PRETRAINED ENCODER MODE: Loaded checkpoint weights into current encoder")
        if probe_only_training:
            print("🎯 PROBE-ONLY TRAINING: Frozen encoder, train only probe heads")
    
    # Optional: torch.compile for 15-30% faster forward/backward (PyTorch 2.0+)
    teacher_net = None
    dino_student_head = None
    dino_teacher_head = None
    ibot_student_head = None
    ibot_teacher_head = None
    koleo_loss_fn = None
    ibot_patch_loss_fn = None
    if use_fusion_dino:
        teacher_net = copy.deepcopy(net).to(device)
        teacher_net.eval()
        for param in teacher_net.parameters():
            param.requires_grad = False
        dino_feature_dim = getattr(net, 'dino_dim', getattr(net, 'embed_dim', 512)) if dino_use_pre_head_features else getattr(net, 'embed_dim', getattr(net, 'dino_dim', 512))
        dino_student_head = DINOHead(
            in_dim=dino_feature_dim,
            out_dim=int(getattr(cfg, 'dino_out_dim', 8192)),
            hidden_dim=int(getattr(cfg, 'dino_hidden_dim', 2048)),
            bottleneck_dim=int(getattr(cfg, 'dino_bottleneck_dim', 256)),
            nlayers=int(getattr(cfg, 'dino_head_layers', 3)),
            norm_last_layer=dino_norm_last_layer,
        ).to(device)
        dino_teacher_head = DINOHead(
            in_dim=dino_feature_dim,
            out_dim=int(getattr(cfg, 'dino_out_dim', 8192)),
            hidden_dim=int(getattr(cfg, 'dino_hidden_dim', 2048)),
            bottleneck_dim=int(getattr(cfg, 'dino_bottleneck_dim', 256)),
            nlayers=int(getattr(cfg, 'dino_head_layers', 3)),
            norm_last_layer=dino_norm_last_layer,
        ).to(device)
        dino_teacher_head.load_state_dict(dino_student_head.state_dict())
        dino_teacher_head.eval()
        for param in dino_teacher_head.parameters():
            param.requires_grad = False
        if dino_use_ibot:
            ibot_in_dim = getattr(net, 'dino_dim', dino_feature_dim)
            ibot_student_head = DINOHead(
                in_dim=ibot_in_dim,
                out_dim=int(getattr(cfg, 'dino_ibot_out_dim', getattr(cfg, 'dino_out_dim', 8192))),
                hidden_dim=int(getattr(cfg, 'dino_hidden_dim', 2048)),
                bottleneck_dim=int(getattr(cfg, 'dino_bottleneck_dim', 256)),
                nlayers=int(getattr(cfg, 'dino_head_layers', 3)),
                norm_last_layer=dino_norm_last_layer,
            ).to(device)
            ibot_teacher_head = DINOHead(
                in_dim=ibot_in_dim,
                out_dim=int(getattr(cfg, 'dino_ibot_out_dim', getattr(cfg, 'dino_out_dim', 8192))),
                hidden_dim=int(getattr(cfg, 'dino_hidden_dim', 2048)),
                bottleneck_dim=int(getattr(cfg, 'dino_bottleneck_dim', 256)),
                nlayers=int(getattr(cfg, 'dino_head_layers', 3)),
                norm_last_layer=dino_norm_last_layer,
            ).to(device)
            ibot_teacher_head.load_state_dict(ibot_student_head.state_dict())
            ibot_teacher_head.eval()
            for param in ibot_teacher_head.parameters():
                param.requires_grad = False
            ibot_patch_loss_fn = IBOTPatchLoss(student_temp=dino_student_temp).to(device)
        if dino_use_koleo:
            koleo_loss_fn = KoLeoLoss().to(device)
        if use_rgb_dino:
            print(f"🎯 RGB DINO BASELINE: EMA teacher, momentum={dino_teacher_momentum:.4f}")
        else:
            print(f"🎯 DINO-STYLE FUSION BASELINE: EMA teacher, momentum={dino_teacher_momentum:.4f}")

    if turbo and hasattr(torch, 'compile') and not debug_model and not use_fusion_dino:
        try:
            net = torch.compile(net, mode='reduce-overhead')
            print("⚡ torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"⚠️ torch.compile failed, continuing without: {e}")
    elif turbo and use_fusion_dino:
        print("ℹ️  Skipping torch.compile for DINO-style fusion baseline to keep EMA teacher updates simple")

    encoder_compute_stats = {
        "batch_forward_calls": 0,
        "total_forward_calls": 0,
        "batch_forward_calls_excl_probes": 0,
        "total_forward_calls_excl_probes": 0,
        "macs_per_call": None,
        "flops_per_call": None,
        "macs_total": 0.0,
        "flops_total": 0.0,
        "profile_attempted": False,
        "profile_success": False,
    }

    # --- Timing & encoder-SSL FLOP tracking stats ---
    timing_stats = {
        "encoder_step_wall_time_excl_probes_total": 0.0,
        "probe_step_wall_time_total": 0.0,
        "encoder_ssl_flops_profiled_step": None,
        "encoder_ssl_flops_total_estimated": 0.0,
    }
    gpu_memory_stats = {
        "available": (device == "cuda" and torch.cuda.is_available()),
        "batch_peak_allocated_bytes_max": 0,
        "batch_peak_reserved_bytes_max": 0,
    }

    estimate_stats = None
    estimate_summary = None
    estimate_complete = False
    train_steps_per_epoch = None
    total_target_train_steps = None

    if track_encoder_compute:
        def _wrap_counted_method(method_name):
            method_ref = getattr(net, method_name, None)
            if method_ref is None or not callable(method_ref):
                return

            def _wrapped(*args, __method=method_ref, **kwargs):
                encoder_compute_stats["batch_forward_calls"] += 1
                encoder_compute_stats["total_forward_calls"] += 1
                return __method(*args, **kwargs)

            setattr(net, method_name, _wrapped)

        for _method_name in ("forward", "forward_with_patches", "forward_with_fusion_tokens", "forward_with_slot_tokens", "forward_with_depth"):
            _wrap_counted_method(_method_name)
    
    # Create linear probes - ENHANCED with more probes
    from src.encoder import get_vit_config
    vit_cfg = get_vit_config(vit_size)
    embed_dim = getattr(net, 'embed_dim', vit_cfg['embed_dim'])  # Prefer the actual model output dim when available
    if use_rgb_dino and dino_use_pre_head_features and hasattr(net, 'dino_dim'):
        embed_dim = net.dino_dim
    
    # Late fusion probes receive concatenated RGB+LiDAR embeddings (2*embed_dim)
    # so they can leverage both modalities for downstream tasks.
    use_concat_probe_embeddings = bool(use_late_fusion or use_imagebind)
    probe_dim = embed_dim * 2 if use_concat_probe_embeddings else embed_dim
    if use_late_fusion:
        print(f"\U0001f517 LATE FUSION PROBES: probe input dim = {probe_dim} (concat RGB + LiDAR embeddings)")
    elif use_imagebind:
        print(f"\U0001f517 IMAGEBIND PROBES: probe input dim = {probe_dim} (concat RGB + depth embeddings)")
    
    probes = nn.ModuleDict({
        # Classification probes
        "scene": nn.Sequential(nn.LayerNorm(probe_dim), nn.Linear(probe_dim, train_ds.num_scenes)),
        "camera": nn.Sequential(nn.LayerNorm(probe_dim), nn.Linear(probe_dim, train_ds.num_cameras)),
        "location": nn.Sequential(nn.LayerNorm(probe_dim), nn.Linear(probe_dim, train_ds.num_locations)),
        
        # Regression probes - object counting
        "num_cars": nn.Sequential(nn.LayerNorm(probe_dim), nn.Linear(probe_dim, 1)),
        "num_peds": nn.Sequential(nn.LayerNorm(probe_dim), nn.Linear(probe_dim, 1)),
        "num_objs": nn.Sequential(nn.LayerNorm(probe_dim), nn.Linear(probe_dim, 1)),
        
        # NEW: Depth estimation probe (predicts mean depth of objects)
        # Tests if camera features learned geometry from LiDAR
        "mean_depth": nn.Sequential(nn.LayerNorm(probe_dim), nn.Linear(probe_dim, 1)),
        
        # NEW: Dense depth probe (predicts 8x8 depth grid)
        # More challenging - requires spatial understanding
        "depth_grid": nn.Sequential(
            nn.LayerNorm(probe_dim), 
            nn.Linear(probe_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64)  # 8x8 grid
        ),
        
        # NEW: Grid occupancy probe (predicts 8x8 object presence grid)
        # Binary classification per cell - is there an object?
        "grid_occupancy": nn.Sequential(
            nn.LayerNorm(probe_dim),
            nn.Linear(probe_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # 8x8 grid
        ),
        
        # Class-specific occupancy probes
        "grid_occupancy_car": nn.Sequential(
            nn.LayerNorm(probe_dim),
            nn.Linear(probe_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # 8x8 grid for cars
        ),
        "grid_occupancy_ped": nn.Sequential(
            nn.LayerNorm(probe_dim),
            nn.Linear(probe_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # 8x8 grid for pedestrians
        ),
        
        # NEW: Distance distribution probe (near/mid/far object counts)
        # Predicts [near, mid, far] object percentages
        "distance_dist": nn.Sequential(
            nn.LayerNorm(probe_dim),
            nn.Linear(probe_dim, 3),
            nn.Softmax(dim=-1)
        ),
        
        # NEW: Cross-modal consistency probe
        # Takes concatenated cam+lidar embeddings, predicts if they match
        # For late fusion, cross_modal probe also uses probe_dim * 2 (= 4*embed_dim)
        # but that's overkill; for late fusion the cross_modal probe is less
        # meaningful since probes already see both modalities. Use probe_dim.
        "cross_modal": nn.Sequential(
            nn.LayerNorm(probe_dim * 2 if not use_concat_probe_embeddings else probe_dim),
            nn.Linear(probe_dim * 2 if not use_concat_probe_embeddings else probe_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ),
    }).to(device)
    
    # ── Patch-based probes (detection, segmentation, patch-token depth) ──
    # These consume patch embeddings (B, N_patches, vit_dim) from the ViT backbone.
    # Only available for architectures that expose patch tokens (B, D, E, F).
    vit_dim = vit_cfg['vit_dim']   # ViT hidden dim (before projection head), varies by vit_size
    patch_grid = infer_encoder_patch_grid(net, default=14)
    patch_probes_available = (arch in ('B', 'C', 'D', 'E', 'F') or simple_baseline) and not legacy_linear_probe_setup
    patch_probes_available = patch_probes_available and enable_patch_probes

    if use_frustum_slots and enable_patch_probes:
        print("⚙️  Frustum-slot patch probes use the fused 14x14 frustum token grid (patch-MSE remains disabled)")
    elif (fusion_tokens_sigreg or lidar_rope_rgb) and not enable_patch_probes:
        print("⚙️  Patch probes disabled for heavy custom Arch C scenario (memory-safe mode)")
    
    patch_det_num_classes = len(WAYMO_PATCH_DET_CLASSES) if dataset_name == 'waymo' else NUM_DETECTION_CLASSES

    if patch_probes_available:
        if dataset_name == 'flir':
            patch_probe_defs = {}
            patch_probe_descs = []
            for domain_name in flir_probe_label_modes:
                domain_suffix = "" if domain_name == 'rgb' else f"[{domain_name}]"
                if patch_bbox2d_probe_type in {'centernet', 'both'}:
                    probe_name = flir_probe_key("bbox2d_centernet", domain_name)
                    patch_probe_defs[probe_name] = SpatialBBox2DProbe(
                        vit_dim=vit_dim,
                        num_classes=len(FLIR_2D_DETECTION_CLASSES),
                        patch_grid=patch_grid,
                        upsample_factor=patch_bbox2d_upsample_factor,
                    )
                    patch_probe_descs.append(
                        f"{probe_name}{domain_suffix}({patch_grid * patch_bbox2d_upsample_factor}x"
                        f"{patch_grid * patch_bbox2d_upsample_factor})"
                    )
                if patch_bbox2d_probe_type in {'slot', 'both'}:
                    probe_name = flir_probe_key("bbox2d_slot", domain_name)
                    patch_probe_defs[probe_name] = BBox2DSlotProbe(
                        vit_dim=vit_dim,
                        num_classes=len(FLIR_2D_DETECTION_CLASSES),
                        patch_grid=patch_grid,
                    )
                    patch_probe_descs.append(
                        f"{probe_name}{domain_suffix}({patch_probe_defs[probe_name].max_objects} queries)"
                    )
                occ_name = flir_probe_key("occupancy_map", domain_name)
                patch_probe_defs[occ_name] = OccupancyMapProbe(
                    vit_dim=vit_dim,
                    patch_grid=patch_grid,
                    upsample_factor=patch_occupancy_map_upsample_factor,
                    output_channels=1 + len(FLIR_2D_DETECTION_CLASSES),
                )
                patch_probe_descs.append(
                    f"{occ_name}{domain_suffix}({patch_grid * patch_occupancy_map_upsample_factor}x"
                    f"{patch_grid * patch_occupancy_map_upsample_factor}, {1 + len(FLIR_2D_DETECTION_CLASSES)}ch)"
                )
                if patch_box_seg_loss_weight > 0.0:
                    box_seg_name = flir_probe_key("box_seg", domain_name)
                    patch_probe_defs[box_seg_name] = SemanticSegProbe(
                        vit_dim=vit_dim,
                        num_classes=len(FLIR_BOX_SEG_CLASSES),
                        patch_grid=patch_grid,
                        upsample_factor=patch_occupancy_map_upsample_factor,
                        target_size=(occupancy_grid_size, occupancy_grid_size),
                        ignore_index=None,
                        class_names=FLIR_BOX_SEG_CLASSES,
                        mean_exclude_indices=[0],
                    )
                    patch_probe_descs.append(
                        f"{box_seg_name}{domain_suffix}({occupancy_grid_size}x{occupancy_grid_size}, {len(FLIR_BOX_SEG_CLASSES)}cls)"
                    )
            patch_probes = nn.ModuleDict(patch_probe_defs).to(device)
            print(
                f"🔬 Patch-based probes enabled: "
                f"{', '.join(patch_probe_descs)} "
                f"for FLIR (arch={arch})"
            )
            det_metrics = None
            det_metrics_centernet = None
            det_metrics_2d = {}
            for domain_name in flir_probe_label_modes:
                if patch_bbox2d_probe_type in {'centernet', 'both'}:
                    det_metrics_2d[flir_probe_key("bbox2d_centernet", domain_name)] = DetectionMetrics2D(
                        class_names=FLIR_2D_DETECTION_CLASSES,
                        num_classes=len(FLIR_2D_DETECTION_CLASSES),
                    )
                if patch_bbox2d_probe_type in {'slot', 'both'}:
                    det_metrics_2d[flir_probe_key("bbox2d_slot", domain_name)] = DetectionMetrics2D(
                        class_names=FLIR_2D_DETECTION_CLASSES,
                        num_classes=len(FLIR_2D_DETECTION_CLASSES),
                    )
            if not det_metrics_2d:
                det_metrics_2d = None
            seg_metrics = None
            box_seg_metrics = None
            if patch_box_seg_loss_weight > 0.0:
                box_seg_metrics = {
                    flir_probe_key("box_seg", domain_name): SegmentationMetrics(
                        num_classes=len(FLIR_BOX_SEG_CLASSES),
                        ignore_index=None,
                        class_names=FLIR_BOX_SEG_CLASSES,
                        mean_exclude_indices=[0],
                    )
                    for domain_name in flir_probe_label_modes
                    if flir_probe_key("box_seg", domain_name) in patch_probe_defs
                }
                if not box_seg_metrics:
                    box_seg_metrics = None
            panoptic_seg_metrics = None
        else:
            enable_patch_occupancy_probe = dataset_name != 'nuscenes'
            patch_probe_modules = {
                # 1. DETR-style 3D bounding-box probe
                "bbox3d": BBox3DProbe(
                    vit_dim=vit_dim,
                    num_classes=patch_det_num_classes,
                    max_objects=50,
                    patch_grid=patch_grid,
                ),
                # 2. CenterNet-style dense 3D bbox probe (2× upsampled for finer heatmap)
                "spatial_bbox3d": SpatialBBox3DProbe(
                    vit_dim=vit_dim,
                    num_classes=patch_det_num_classes,
                    patch_grid=patch_grid,
                    upsample_factor=2,
                ),
                "seg": SemanticSegProbe(
                    vit_dim=vit_dim,
                    patch_grid=patch_grid,
                    target_size=(224, 224),
                ),
                "panoptic_seg": SemanticSegProbe(
                    vit_dim=vit_dim,
                    patch_grid=patch_grid,
                    target_size=(224, 224),
                ),
                "patch_depth_token": nn.Sequential(
                    nn.LayerNorm(vit_dim),
                    nn.Linear(vit_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, 1),
                ),
                "patch_depth_spatial": nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(16, 1, kernel_size=3, padding=1),
                ),
                "depth_map": DepthMapProbe(
                    vit_dim=vit_dim,
                    patch_grid=patch_grid,
                    upsample_factor=4,
                ),
            }
            if enable_patch_occupancy_probe:
                patch_probe_modules["occupancy_map"] = OccupancyMapProbe(
                    vit_dim=vit_dim,
                    patch_grid=patch_grid,
                    upsample_factor=patch_occupancy_map_upsample_factor,
                )
            patch_probes = nn.ModuleDict(patch_probe_modules).to(device)

            if dataset_name == 'waymo':
                det_metrics = NuScenesDetectionMetrics(
                    class_names=WAYMO_PATCH_DET_CLASSES,
                    num_classes=len(WAYMO_PATCH_DET_CLASSES),
                )
                det_metrics_centernet = NuScenesDetectionMetrics(
                    class_names=WAYMO_PATCH_DET_CLASSES,
                    num_classes=len(WAYMO_PATCH_DET_CLASSES),
                )
                print("  📊 Using dense Waymo patch classes: car(0), pedestrian(1), bicycle(2)")
            else:
                det_metrics = NuScenesDetectionMetrics()
                det_metrics_centernet = NuScenesDetectionMetrics()
            det_metrics_2d = None
            seg_metrics = SegmentationMetrics()
            box_seg_metrics = None
            panoptic_seg_metrics = SegmentationMetrics()
            patch_probe_summary = [
                "bbox3d",
                "spatial_bbox3d(28×28)",
                "seg",
                "panoptic_seg",
                "patch_depth(8×8)",
                "depth_map(56×56)",
            ]
            if enable_patch_occupancy_probe:
                patch_probe_summary.append(
                    f"occupancy_map({patch_grid * patch_occupancy_map_upsample_factor}×"
                    f"{patch_grid * patch_occupancy_map_upsample_factor})"
                )
            else:
                print("ℹ️  Patch occupancy probe disabled for NuScenes (grid_occupancy_hr labels unavailable)")
            print(f"🔬 Patch-based probes enabled: {', '.join(patch_probe_summary)} (arch={arch})")
    else:
        patch_probes = None
        det_metrics = None
        det_metrics_centernet = None
        det_metrics_2d = None
        seg_metrics = None
        box_seg_metrics = None
        panoptic_seg_metrics = None
        print(f"ℹ️  Patch-based probes skipped (arch={arch} does not expose patch tokens)")
    
    sigreg = SIGReg().to(device)
    
    # ============================================
    # BASELINE LOSS OBJECTS (VICReg, InfoNCE, AdaSigNCE, SigNCE)
    # ============================================
    vicreg_loss_fn = None
    infonce_loss_fn = None
    adasignce_loss_fn = None
    signce_loss_fn = None
    whitened_infonce_loss_fn = None
    eigennce_loss_fn = None
    dino_loss_fn = None
    if use_vicreg:
        vicreg_loss_fn = VICRegLoss(sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0).to(device)
        print("🎯 VICReg LOSS: Replacing SIGReg with VICReg (variance-invariance-covariance)")
    if use_infonce:
        infonce_loss_fn = InfoNCELoss(temperature=0.07).to(device)
        print("🎯 InfoNCE LOSS: Replacing SIGReg with InfoNCE (contrastive)")
    if use_whitened_infonce:
        whitened_infonce_loss_fn = WhitenedInfoNCELoss(temperature=0.07, whiten_eps=1e-4).to(device)
        print("🎯 WHITENED INFONCE LOSS: InfoNCE on batch-whitened projections")
    if use_eigennce:
        eigennce_loss_fn = EigenNCELoss(temperature=0.07, eig_power=0.5, eps=1e-4).to(device)
        print("🎯 EIGENNCE LOSS: InfoNCE with eigenspace spectrum reweighting")
    if use_adasignce:
        adasignce_loss_fn = AdaSigNCELoss(
            tau_base=0.07, tau_max=0.5, sigreg_weight=cfg.lamb,
            adaptive_tau=True,
        ).to(device)
        print("🎯 AdaSigNCE LOSS: Adaptive SigReg-guided InfoNCE (spectral-contrastive)")
    
    # Initialize novel regularizers (Beyond SIGReg)
    novel_regs = {}
    if gmm_regularizer:
        novel_regs['gmm'] = GMMRegularizer(
            embedding_dim=cfg.proj_dim,
            num_prototypes=gmm_num_prototypes,
            temperature=gmm_temperature
        ).to(device)
        print(f"🔮 GMM Regularizer: {gmm_num_prototypes} prototypes, temp={gmm_temperature}")
    
    if sinkhorn_regularizer:
        novel_regs['sinkhorn'] = SinkhornRegularizer(
            epsilon=sinkhorn_epsilon,
            n_iters=sinkhorn_iters
        ).to(device)
        print(f"🔮 Sinkhorn Regularizer: epsilon={sinkhorn_epsilon}, iters={sinkhorn_iters}")
    
    if spectral_regularizer:
        novel_regs['spectral'] = SpectralRegularizer(
            mode=spectral_mode
        ).to(device)
        print(f"🔮 Spectral Regularizer: mode={spectral_mode}")
    
    if replace_sigreg:
        print("⚠️  REPLACE_SIGREG MODE: Novel regularizer will replace SIGReg entirely!")
    
    # Determine batch size
    if cfg.bs == 0 or str(cfg.bs).lower() == 'auto':
        simple_channels = 1 if simple_baseline and simple_modality in ('lidar', 'thermal') else 3
        batch_size = find_max_batch_size(
            net, arch, num_global_views, device=device, 
            simple_mode=simple_baseline, simple_channels=simple_channels
        )
    else:
        batch_size = min(cfg.bs, len(train_ds))

    if use_imagebind:
        imagebind_max_bs = max(1, int(getattr(cfg, 'imagebind_max_bs', getattr(cfg, 'bs', batch_size))))
        if batch_size > imagebind_max_bs:
            print(f"⚙️  Capping ImageBind batch size from {batch_size} to {imagebind_max_bs} for memory safety")
            batch_size = imagebind_max_bs

    if use_fusion_dino:
        teacher_global_views = max(1, int(getattr(cfg, 'V', 2)))
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        effective_teacher_batch = batch_size * teacher_global_views * world_size
        dino_sinkhorn_min_teacher_batch = int(getattr(cfg, 'dino_sinkhorn_min_teacher_batch', 256))
        dino_out_dim = int(getattr(cfg, 'dino_out_dim', 8192))
        dino_sinkhorn_min_batch_to_out_dim_ratio = float(getattr(cfg, 'dino_sinkhorn_min_batch_to_out_dim_ratio', 0.125))
        sinkhorn_ratio_ok = effective_teacher_batch >= max(1.0, dino_sinkhorn_min_batch_to_out_dim_ratio * dino_out_dim)
        dino_use_sinkhorn_teacher = (
            dino_use_sinkhorn_teacher_requested
            and effective_teacher_batch >= dino_sinkhorn_min_teacher_batch
            and sinkhorn_ratio_ok
        )

        if dino_use_sinkhorn_teacher_requested and not dino_use_sinkhorn_teacher:
            print(
                "🎯 DINO LOSS: disabling Sinkhorn teacher at runtime because "
                f"effective_teacher_batch={effective_teacher_batch} < "
                f"dino_sinkhorn_min_teacher_batch={dino_sinkhorn_min_teacher_batch} "
                f"or because effective_teacher_batch / dino_out_dim = "
                f"{effective_teacher_batch / max(dino_out_dim, 1):.4f} < "
                f"dino_sinkhorn_min_batch_to_out_dim_ratio={dino_sinkhorn_min_batch_to_out_dim_ratio:.4f}"
            )
        elif dino_use_sinkhorn_teacher:
            print(
                "🎯 DINO LOSS: enabling Sinkhorn teacher at runtime with "
                f"effective_teacher_batch={effective_teacher_batch} and "
                f"effective_teacher_batch / dino_out_dim = {effective_teacher_batch / max(dino_out_dim, 1):.4f}"
            )

        dino_loss_fn = DINOLoss(
            student_temp=dino_student_temp,
            teacher_temp=dino_teacher_temp,
            teacher_temp_warmup_start=dino_teacher_temp_warmup_start,
            teacher_temp_warmup_epochs=dino_teacher_temp_warmup_epochs,
            center_momentum=float(getattr(cfg, 'dino_center_momentum', 0.9)),
            use_sinkhorn_teacher=dino_use_sinkhorn_teacher,
        ).to(device)
        if use_rgb_dino:
            if dino_use_sinkhorn_teacher or dino_use_koleo or dino_use_ibot:
                print("🎯 DINO LOSS: aggressive RGB DINO recipe with Sinkhorn teacher, KoLeo, and iBOT-style patch distillation")
            else:
                print("🎯 DINO LOSS: RGB-only teacher-on-global multi-crop self-distillation with centering")
        else:
            if dino_use_sinkhorn_teacher:
                print("🎯 DINO LOSS: Teacher-on-global multi-crop self-distillation with Sinkhorn teacher assignments")
            else:
                print("🎯 DINO LOSS: Teacher-on-global multi-crop self-distillation with centering")
    
    # DataLoaders
    if num_workers < 0:
        # Smart auto-detection respecting container limits
        try:
            # sched_getaffinity returns the set of CPUs the process is allowed to run on
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # Fallback for systems without sched_getaffinity (e.g. non-Linux)
            available_cpus = os.cpu_count() or 1
        
        # Use all available CPUs as requested by user
        # (Memory leak should be fixed by metadata filtering now)
        # CAP WORKERS TO 12 TO PREVENT RAM EXPLOSION (180GB+)
        MAX_WORKERS = 64
        if available_cpus > MAX_WORKERS:
             print(f"⚠️  Capping workers from {available_cpus} to {MAX_WORKERS} to save RAM")
             num_workers = MAX_WORKERS
        else:
             num_workers = available_cpus
        
        print(f"Auto-detected {available_cpus} CPUs, using {num_workers} workers")
    else:
        print(f"Using {num_workers} workers per dataloader (Manual override)")
    
    if debug:
        num_workers = 0
        print("🐞 DEBUG MODE: Forcing num_workers=0 for speed")

    loader_num_workers = num_workers
    train_pin_memory = True
    eval_pin_memory = True
    train_persistent_workers = (loader_num_workers > 0)
    eval_persistent_workers = (loader_num_workers > 0)
    train_prefetch_factor = (8 if turbo else 4) if loader_num_workers > 0 else None
    eval_prefetch_factor = None

    if configured_dataset_name == 'flir':
        capped_workers = min(loader_num_workers, flir_loader_max_workers)
        if capped_workers != loader_num_workers:
            print(
                f"⚠️  FLIR loader memory mode: capping workers from {loader_num_workers} to {capped_workers}"
            )
        loader_num_workers = capped_workers
        train_pin_memory = flir_loader_pin_memory
        eval_pin_memory = flir_loader_pin_memory
        train_persistent_workers = (loader_num_workers > 0) and flir_loader_persistent_workers
        eval_persistent_workers = (loader_num_workers > 0) and flir_loader_persistent_workers
        train_prefetch_factor = flir_loader_prefetch_factor if loader_num_workers > 0 else None
        eval_prefetch_factor = flir_loader_prefetch_factor if loader_num_workers > 0 else None
        print(
            "🧠 FLIR loader memory mode: "
            f"workers={loader_num_workers}, prefetch={train_prefetch_factor or 0}, "
            f"persistent={train_persistent_workers}, pin_memory={train_pin_memory}"
        )

    # DataLoaders (skip in debug_model mode)
    if debug_model:
        print("🚀 FAST DEBUG: Skipping dataloader creation...")
        train_loader = None
        test_loader = None
        test_loader_legacy = None
    else:
        # Use dynamic collate function (_collate_fn) to support both NuScenes and Waymo
        train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True, 
            num_workers=loader_num_workers, 
            pin_memory=train_pin_memory, 
            collate_fn=_collate_fn,
            persistent_workers=train_persistent_workers,
            prefetch_factor=train_prefetch_factor,
        )
        test_loader = DataLoader(
            test_ds, 
            batch_size=batch_size, 
            num_workers=loader_num_workers, 
            pin_memory=eval_pin_memory, 
            collate_fn=_collate_fn,
            shuffle=True, # Randomize validation batches for representative partial validation
            persistent_workers=eval_persistent_workers,
            prefetch_factor=eval_prefetch_factor,
        )
        
        # Initialize legacy loader if requested
        if validate_legacy and test_ds_legacy is not None:
            # Re-use worker count but maybe scale down if needed? 
            # Actually legacy validation is infrequent, so no persistent workers needed maybe?
            # Let's keep consistent.
            if test_loader_legacy is None: # Should be None from earlier definition check failure or just re-do
                 # Actually I defined it earlier but without batch_size. I need to redefine it here.
                 pass

            test_loader_legacy = DataLoader(
                test_ds_legacy, 
                batch_size=batch_size, 
                num_workers=loader_num_workers, 
                pin_memory=eval_pin_memory, 
                collate_fn=_collate_fn,
                shuffle=True, # Randomize for partial validation
                persistent_workers=eval_persistent_workers,
                prefetch_factor=eval_prefetch_factor,
            )

    train_steps_per_epoch = 3 if debug_model else len(train_loader)
    total_target_train_steps = train_steps_per_epoch * max(int(epochs), 1)
    if estimate_encoder_train_compute:
        if train_steps_per_epoch <= 0:
            raise RuntimeError("estimate_encoder_train_compute requires at least one train batch per epoch")
        available_estimate_steps = max(total_target_train_steps, 1)
        adjusted_warmup_batches = min(estimate_warmup_batches, max(available_estimate_steps - 1, 0))
        adjusted_measure_batches = min(
            estimate_measure_batches,
            max(available_estimate_steps - adjusted_warmup_batches, 1),
        )
        if (
            adjusted_warmup_batches != estimate_warmup_batches
            or adjusted_measure_batches != estimate_measure_batches
        ):
            print(
                "📐 Adjusting encoder estimate window to available steps: "
                f"warmup {estimate_warmup_batches}->{adjusted_warmup_batches}, "
                f"measure {estimate_measure_batches}->{adjusted_measure_batches}, "
                f"available_steps={available_estimate_steps}"
            )
        estimate_warmup_batches = adjusted_warmup_batches
        estimate_measure_batches = adjusted_measure_batches
        estimate_stats = {
            "warmup_batches_completed": 0,
            "measured_batches": 0,
            "measured_encoder_step_time_total_sec": 0.0,
            "measured_encoder_forward_calls_total": 0.0,
            "measured_encoder_macs_total": 0.0,
            "measured_encoder_forward_flops_total": 0.0,
            "profiled_step_flops": None,
            "measured_peak_allocated_bytes": [],
            "measured_peak_reserved_bytes": [],
        }
        print(
            "📐 Encoder estimate target: "
            f"steps_per_epoch={train_steps_per_epoch}, "
            f"target_total_steps={total_target_train_steps}, "
            f"probe_training={'off' if disable_probe_training else 'on'}"
        )
    
    # Optimizers (independent lanes to isolate probe-family updates)
    if freeze_encoder:
        print("❄️ FROZEN ENCODER MODE: Training probes only ❄️")
        for param in net.parameters():
            param.requires_grad = False

    encoder_params = [p for p in net.parameters() if p.requires_grad]
    if dino_student_head is not None:
        encoder_params.extend([p for p in dino_student_head.parameters() if p.requires_grad])
    if ibot_student_head is not None:
        encoder_params.extend([p for p in ibot_student_head.parameters() if p.requires_grad])

    def _name_matches_prefixes(name, prefixes):
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix):
                return True
        return False

    def _split_pretrained_encoder_param_groups():
        if freeze_encoder or not pretrained_backbone or not pretrained_backbone_param_group:
            return None, [], []

        trunk_prefixes = []
        if fusion_tokens_sigreg and isinstance(net, MMEncoderC_FusionTokens):
            trunk_prefixes = [
                'backbone.',
                'range_patch_embed.',
            ]
        elif use_imagebind and hasattr(net, 'rgb_encoder') and hasattr(net, 'depth_encoder'):
            trunk_prefixes = [
                'rgb_encoder.',
                'depth_encoder.',
            ]
        elif (use_multimae_exact or use_multimae_exact_mt or use_multimae_exact_segplus) and hasattr(net, 'model'):
            trunk_prefixes = [
                'model.encoder_blocks.',
                'model.encoder_norm.',
                'model.global_tokens',
                'model.input_adapters.rgb.patch_embed.',
                'model.input_adapters.depth.patch_embed.',
            ]
        else:
            return None, [], []

        trunk_params = []
        aux_params = []
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            if _name_matches_prefixes(name, trunk_prefixes):
                trunk_params.append(param)
            else:
                aux_params.append(param)

        if dino_student_head is not None:
            aux_params.extend([p for p in dino_student_head.parameters() if p.requires_grad])
        if ibot_student_head is not None:
            aux_params.extend([p for p in ibot_student_head.parameters() if p.requires_grad])

        if not trunk_params:
            return None, [], []

        param_groups = []
        if trunk_params:
            param_groups.append({
                'params': trunk_params,
                'lr': pretrained_trunk_lr,
                'weight_decay': pretrained_trunk_weight_decay,
                'group_name': 'pretrained_trunk',
            })
        if aux_params:
            param_groups.append({
                'params': aux_params,
                'lr': encoder_aux_lr,
                'weight_decay': encoder_aux_weight_decay,
                'group_name': 'encoder_aux',
            })
        return param_groups, trunk_params, aux_params

    encoder_param_groups, pretrained_trunk_params, encoder_aux_params = _split_pretrained_encoder_param_groups()
    if encoder_param_groups is not None:
        print(
            '🎯 PRETRAINED TRUNK PARAM GROUPS: '
            f'trunk_lr={pretrained_trunk_lr:g}, aux_lr={encoder_aux_lr:g}, '
            f'freeze_warmup_epochs={pretrained_trunk_warmup_freeze_epochs}'
        )
        print(
            '   '
            f'trunk_params={sum(p.numel() for p in pretrained_trunk_params):,}, '
            f'aux_params={sum(p.numel() for p in encoder_aux_params):,}'
        )

    def _set_param_trainable(params, trainable: bool):
        for param in params:
            param.requires_grad_(trainable)

    def _build_dino_param_groups(model, cls_head=None, patch_head=None):
        if not dino_use_param_groups or not hasattr(model, 'backbone'):
            return None

        n_blocks = len(getattr(model.backbone, 'blocks', []))
        layer_decay = float(getattr(cfg, 'dino_layer_decay', 0.9))
        patch_embed_lr_mult = float(getattr(cfg, 'dino_patch_embed_lr_mult', 1.0))
        dino_head_wd_multiplier = float(getattr(cfg, 'dino_head_wd_multiplier', 1.0))

        def _layer_decay_rate(name: str) -> float:
            layer_id = n_blocks + 1
            if name.startswith('backbone.'):
                if any(tok in name for tok in ('pos_embed', 'patch_embed', 'mask_token', 'cls_token', 'register_tokens', 'storage_tokens')):
                    layer_id = 0
                elif '.blocks.' in name:
                    layer_id = int(name[name.find('.blocks.'):].split('.')[2]) + 1
            return layer_decay ** (n_blocks + 1 - layer_id)

        params_with_names = [(f'net.{n}', p) for n, p in model.named_parameters() if p.requires_grad]
        if cls_head is not None:
            params_with_names.extend((f'dino_head.{n}', p) for n, p in cls_head.named_parameters() if p.requires_grad)
        if patch_head is not None:
            params_with_names.extend((f'ibot_head.{n}', p) for n, p in patch_head.named_parameters() if p.requires_grad)

        param_groups = []
        for name, param in params_with_names:
            model_name = name[4:] if name.startswith('net.') else name
            lr_multiplier = _layer_decay_rate(model_name)
            wd_multiplier = 1.0
            is_last_layer = False

            if 'dino_head' in name or 'ibot_head' in name:
                wd_multiplier = dino_head_wd_multiplier
            if 'last_layer' in name:
                is_last_layer = True
            if name.endswith('bias') or 'norm' in name or 'gamma' in name or 'pos_embed' in name or 'cls_token' in name or 'register_tokens' in name or 'storage_tokens' in name:
                wd_multiplier = 0.0
            if 'patch_embed' in name:
                lr_multiplier *= patch_embed_lr_mult

            param_groups.append({
                'params': [param],
                'lr_multiplier': lr_multiplier,
                'wd_multiplier': wd_multiplier,
                'is_last_layer': is_last_layer,
            })
        return param_groups

    def _dino_teacher_momentum(step_idx: int, total_step_count: int) -> float:
        if total_step_count <= 1:
            return dino_teacher_momentum
        progress = min(max(step_idx / (total_step_count - 1), 0.0), 1.0)
        return 1.0 - 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - dino_teacher_momentum)

    def _update_ema_teacher(student_model, teacher_model, momentum: float):
        with torch.no_grad():
            student_state = student_model.state_dict()
            teacher_state = teacher_model.state_dict()
            for name, teacher_value in teacher_state.items():
                student_value = student_state[name]
                if torch.is_floating_point(teacher_value):
                    teacher_value.mul_(momentum).add_(student_value.detach(), alpha=1.0 - momentum)
                else:
                    teacher_value.copy_(student_value)

    def _reshape_view_major_embeddings(flat_emb: torch.Tensor, input_views) -> torch.Tensor:
        if flat_emb.dim() != 2:
            raise ValueError(f"Expected flat embeddings with shape (B*V, D), got {tuple(flat_emb.shape)}")

        if isinstance(input_views, dict):
            parts = []
            offset = 0
            for key in ('global', 'local'):
                views = input_views.get(key)
                if views is None or views.numel() == 0:
                    continue
                batch_size, num_views = views.shape[:2]
                num_tokens = batch_size * num_views
                part = flat_emb[offset:offset + num_tokens]
                if part.shape[0] != num_tokens:
                    raise RuntimeError(f"Missing {key} DINO embeddings: expected {num_tokens}, got {part.shape[0]}")
                parts.append(part.reshape(batch_size, num_views, -1).transpose(0, 1))
                offset += num_tokens
            if offset != flat_emb.shape[0]:
                raise RuntimeError(
                    f"Unexpected extra DINO embeddings after view packing: used {offset}, have {flat_emb.shape[0]}"
                )
            return torch.cat(parts, dim=0)

        batch_size, num_views = input_views.shape[:2]
        return flat_emb.reshape(batch_size, num_views, -1).transpose(0, 1)

    def _cosine_schedule_value(start: float, peak: float, end: float, step_idx: int, total_step_count: int, warmup_step_count: int = 0) -> float:
        if total_step_count <= 1:
            return peak
        warmup_step_count = min(max(warmup_step_count, 0), total_step_count - 1)
        if warmup_step_count > 0 and step_idx < warmup_step_count:
            alpha = step_idx / max(warmup_step_count - 1, 1)
            return start + alpha * (peak - start)
        cosine_total = max(total_step_count - warmup_step_count, 1)
        cosine_step = min(max(step_idx - warmup_step_count, 0), cosine_total - 1)
        progress = cosine_step / max(cosine_total - 1, 1)
        return end + 0.5 * (peak - end) * (1.0 + math.cos(math.pi * progress))
    
    # Schedulers
    if debug_model:
        warmup_steps = 3  # Just 3 dummy batches
        total_steps = 3 * epochs
    else:
        warmup_steps = len(train_loader)
        total_steps = len(train_loader) * epochs
    warmup_steps = max(1, warmup_steps)
    cosine_steps = max(1, total_steps - warmup_steps)

    def _build_scheduler(optimizer):
        s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        s2 = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
        return SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

    dino_encoder_schedule = None
    dino_encoder_lr = float(getattr(cfg, 'dino_lr', cfg.lr))

    probe_lane_modules = {
        "scene": ["scene"],
        "camera": ["camera"],
        "location": ["location"],
        "num_cars": ["num_cars"],
        "num_peds": ["num_peds"],
        "num_objs": ["num_objs"],
        "mean_depth": ["mean_depth"],
        "depth_grid": ["depth_grid"],
        "grid_occupancy": ["grid_occupancy"],
        "grid_occupancy_car": ["grid_occupancy_car"],
        "grid_occupancy_ped": ["grid_occupancy_ped"],
        "cross_modal": ["cross_modal"],
    }
    patch_lane_modules = {
        "bbox2d": ["bbox2d"],
        "bbox2d_centernet": ["bbox2d_centernet"],
        "bbox2d_slot": ["bbox2d_slot"],
        "bbox3d": ["bbox3d"],
        "spatial_bbox3d": ["spatial_bbox3d"],
        "seg": ["seg"],
        "panoptic_seg": ["panoptic_seg"],
        "box_seg": ["box_seg"],
        "old_patch_depth": ["patch_depth_token", "patch_depth_spatial"],
        "depth_map": ["depth_map"],
        "occupancy_map": ["occupancy_map"],
    }
    # Add dynamic entries for domain-specific FLIR probes not covered by the
    # static lane mapping (e.g. box_seg_consensus, occupancy_map_thermal).
    if patch_probes is not None:
        for probe_name in patch_probes:
            if probe_name not in patch_lane_modules:
                patch_lane_modules[probe_name] = [probe_name]

    def _collect_lane_params(module_dict, module_names):
        lane_params = []
        for mod_name in module_names:
            if mod_name in module_dict:
                lane_params.extend([p for p in module_dict[mod_name].parameters() if p.requires_grad])
        return lane_params

    probe_params_by_lane = {
        lane_name: _collect_lane_params(probes, module_names)
        for lane_name, module_names in probe_lane_modules.items()
    }
    patch_params_by_lane = {
        lane_name: _collect_lane_params(patch_probes, module_names) if patch_probes is not None else []
        for lane_name, module_names in patch_lane_modules.items()
    }

    probe_params_legacy = [p for p in probes.parameters() if p.requires_grad]

    # Legacy variables (used by non-legacy path references, kept as None in legacy mode)
    opt_probe_legacy = None
    scheduler_probe_legacy = None
    scaler_probe_legacy = None
    # Combined legacy optimizer (used only in legacy mode)
    opt_combined_legacy = None
    scheduler_combined_legacy = None
    scaler_combined_legacy = None
    opt_encoder = None  # Will be set below for non-legacy path

    if legacy_linear_probe_setup:
        # ── LEGACY PATH: Single optimizer for encoder + probes ──────────
        # Replicates pre-Feb-5 behavior exactly:
        #   ONE AdamW with param groups, ONE GradScaler, ONE scheduler,
        #   ONE backward on (lejepa_loss + probe_loss), no patch probes.
        probe_opts = {}
        patch_opts = {}
        probe_schedulers = {}
        patch_schedulers = {}
        probe_scalers = {}
        patch_scalers = {}
        if freeze_encoder:
            # Frozen encoder: only probe params
            opt_combined_legacy = torch.optim.AdamW(
                probe_params_legacy, lr=1e-3, weight_decay=1e-7
            ) if probe_params_legacy else None
        else:
            # Joint encoder + probe training with param groups
            param_groups = []
            if encoder_params:
                param_groups.append({"params": encoder_params, "lr": cfg.lr, "weight_decay": 5e-2})
            if probe_params_legacy:
                param_groups.append({"params": probe_params_legacy, "lr": 1e-3, "weight_decay": 1e-7})
            opt_combined_legacy = torch.optim.AdamW(param_groups) if param_groups else None
        scheduler_combined_legacy = _build_scheduler(opt_combined_legacy) if opt_combined_legacy else None
        scaler_combined_legacy = GradScaler(enabled=(device == "cuda")) if opt_combined_legacy else None
        print("🔧 LEGACY MODE: Single optimizer for encoder + probes (Feb-5 behavior)")
    else:
        # ── MODERN PATH: Per-lane optimizers ────────────────────────────
        if encoder_param_groups is not None:
            opt_encoder = torch.optim.AdamW(encoder_param_groups) if encoder_param_groups else None
        elif dino_use_param_groups and use_fusion_dino and use_rgb_dino:
            dino_param_groups = _build_dino_param_groups(net, dino_student_head, ibot_student_head)
            opt_encoder = torch.optim.AdamW(
                dino_param_groups,
                lr=dino_encoder_lr,
                weight_decay=float(getattr(cfg, 'dino_weight_decay_start', 0.04)),
                betas=(float(getattr(cfg, 'dino_adamw_beta1', 0.9)), float(getattr(cfg, 'dino_adamw_beta2', 0.95))),
            ) if dino_param_groups else None
        else:
            opt_encoder = torch.optim.AdamW(
                encoder_params,
                lr=dino_encoder_lr if (use_fusion_dino and use_rgb_dino) else cfg.lr,
                weight_decay=5e-2,
            ) if encoder_params else None

        probe_opts = {
            lane_name: (
                torch.optim.AdamW(lane_params, lr=probe_lr, weight_decay=probe_weight_decay)
                if lane_params else None
            )
            for lane_name, lane_params in probe_params_by_lane.items()
        }
        patch_opts = {
            lane_name: (
                torch.optim.AdamW(lane_params, lr=patch_probe_lr, weight_decay=patch_probe_weight_decay)
                if lane_params else None
            )
            for lane_name, lane_params in patch_params_by_lane.items()
        }

        probe_schedulers = {
            lane_name: (_build_scheduler(opt_obj) if opt_obj is not None else None)
            for lane_name, opt_obj in probe_opts.items()
        }
        patch_schedulers = {
            lane_name: (_build_scheduler(opt_obj) if opt_obj is not None else None)
            for lane_name, opt_obj in patch_opts.items()
        }

        probe_scalers = {
            lane_name: GradScaler(enabled=(device == "cuda" and opt_obj is not None))
            for lane_name, opt_obj in probe_opts.items()
        }
        patch_scalers = {
            lane_name: GradScaler(enabled=(device == "cuda" and opt_obj is not None))
            for lane_name, opt_obj in patch_opts.items()
        }

    if dino_use_param_groups and use_fusion_dino and use_rgb_dino and opt_encoder is not None:
        lr_warmup_steps = steps_per_epoch = (3 if debug_model else len(train_loader))
        lr_warmup_steps *= int(getattr(cfg, 'dino_lr_warmup_epochs', 1))
        dino_encoder_schedule = {
            'lr_start': 0.0,
            'lr_peak': dino_encoder_lr,
            'lr_end': float(getattr(cfg, 'dino_lr_end', 1e-5)),
            'wd_start': float(getattr(cfg, 'dino_weight_decay_start', 0.04)),
            'wd_end': float(getattr(cfg, 'dino_weight_decay_end', 0.2)),
            'warmup_steps': lr_warmup_steps,
            'freeze_last_layer_steps': (3 if debug_model else len(train_loader)) * int(getattr(cfg, 'dino_freeze_last_layer_epochs', 1)),
        }
        scheduler_encoder = None
    else:
        scheduler_encoder = _build_scheduler(opt_encoder) if opt_encoder is not None else None
    scaler_encoder = GradScaler(enabled=(device == "cuda" and opt_encoder is not None))

    def _new_health_stats():
        return {
            "steps": 0,
            "backward": 0,
            "overflow": 0,
            "nonfinite_loss": 0,
            "invalid_grad_tensors": 0,
            "invalid_grad_values": 0,
        }

    def _accumulate_health(dst, src):
        for k in dst:
            if k in src:
                dst[k] += src[k]
            elif k == "steps":
                dst[k] += src.get("did_step", 0)
            elif k == "backward":
                dst[k] += src.get("did_backward", 0)

    optimizer_health_totals = {
        "encoder": _new_health_stats(),
        "probe": _new_health_stats(),
        "patch": _new_health_stats(),
    }
    optimizer_lane_totals = {
        **{f"probe/{lane_name}": _new_health_stats() for lane_name in probe_opts},
        **{f"patch/{lane_name}": _new_health_stats() for lane_name in patch_opts},
    }

    def _current_lr(scheduler_obj, optimizer_obj):
        if scheduler_obj is not None:
            return scheduler_obj.get_last_lr()[0]
        if optimizer_obj is not None:
            return max(group["lr"] for group in optimizer_obj.param_groups)
        return 0.0

    def _group_lr(optimizer_obj, group_name: str):
        if optimizer_obj is None:
            return 0.0
        for group in optimizer_obj.param_groups:
            if group.get('group_name') == group_name:
                return float(group['lr'])
        return 0.0

    def _apply_dino_encoder_schedule(step_idx: int):
        if dino_encoder_schedule is None or opt_encoder is None:
            return
        current_lr = _cosine_schedule_value(
            dino_encoder_schedule['lr_start'],
            dino_encoder_schedule['lr_peak'],
            dino_encoder_schedule['lr_end'],
            step_idx,
            total_steps,
            dino_encoder_schedule['warmup_steps'],
        )
        current_wd = _cosine_schedule_value(
            dino_encoder_schedule['wd_start'],
            dino_encoder_schedule['wd_start'],
            dino_encoder_schedule['wd_end'],
            step_idx,
            total_steps,
            0,
        )
        freeze_last_layer = step_idx < dino_encoder_schedule['freeze_last_layer_steps']
        for param_group in opt_encoder.param_groups:
            lr_multiplier = param_group.get('lr_multiplier', 1.0)
            wd_multiplier = param_group.get('wd_multiplier', 1.0)
            is_last_layer = param_group.get('is_last_layer', False)
            param_group['weight_decay'] = current_wd * wd_multiplier
            if freeze_last_layer and is_last_layer:
                param_group['lr'] = 0.0
            else:
                param_group['lr'] = current_lr * lr_multiplier

    def _mean_lr(scheduler_map, optimizer_map):
        lrs = [
            _current_lr(scheduler_map[lane_name], optimizer_map[lane_name])
            for lane_name in optimizer_map
            if optimizer_map[lane_name] is not None
        ]
        if not lrs:
            return 0.0
        return sum(lrs) / len(lrs)

    def _run_optimizer_step(loss_tensor, optimizer_obj, scheduler_obj, scaler_obj, params, post_backward_fn=None, max_grad_norm=None):
        step_stats = {
            "did_backward": 0,
            "did_step": 0,
            "overflow": 0,
            "nonfinite_loss": 0,
            "invalid_grad_tensors": 0,
            "invalid_grad_values": 0,
        }

        if optimizer_obj is None or loss_tensor.grad_fn is None:
            return step_stats
        if not torch.isfinite(loss_tensor):
            step_stats["nonfinite_loss"] = 1
            return step_stats

        optimizer_obj.zero_grad(set_to_none=True)
        scaler_enabled = scaler_obj is not None and scaler_obj.is_enabled()

        if scaler_enabled:
            scaler_obj.scale(loss_tensor).backward()
            if post_backward_fn is not None:
                post_backward_fn()

            # Check if any params received gradients; if not, skip scaler ops
            # (prevents "No inf checks" assertion when loss doesn't connect to optimizer params)
            has_grads = any(p.grad is not None for p in params if p.requires_grad)
            if has_grads:
                scaler_obj.unscale_(optimizer_obj)
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                invalid_tensors, invalid_values = count_invalid_grads(params)
                step_stats["invalid_grad_tensors"] = invalid_tensors
                step_stats["invalid_grad_values"] = invalid_values

                scale_before = scaler_obj.get_scale()
                scaler_obj.step(optimizer_obj)
                scaler_obj.update()
                scale_after = scaler_obj.get_scale()
                if scale_after < scale_before:
                    step_stats["overflow"] = 1
                else:
                    if scheduler_obj is not None:
                        scheduler_obj.step()
                    step_stats["did_step"] = 1
            # else: no grads produced — skip step entirely (no unscale/step/update needed)
        else:
            loss_tensor.backward()
            if post_backward_fn is not None:
                post_backward_fn()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            invalid_tensors, invalid_values = count_invalid_grads(params)
            step_stats["invalid_grad_tensors"] = invalid_tensors
            step_stats["invalid_grad_values"] = invalid_values
            if invalid_tensors == 0:
                optimizer_obj.step()
                if scheduler_obj is not None:
                    scheduler_obj.step()
                step_stats["did_step"] = 1

        step_stats["did_backward"] = 1
        return step_stats
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Hyperparams: arch={arch}, lamb={cfg.lamb}, V={num_global_views}, proj_dim={cfg.proj_dim}, bs={batch_size}")
    print(f"Probes: {list(probes.keys())}")
    if legacy_linear_probe_setup:
        print("🧪 A/B MODE: Using Feb-5 style linear probe optimization (single probe optimizer, no patch-probe training loss)")
    
    # For debug_model mode: create dummy data iterator
    if debug_model:
        print("🚀 Generating dummy batches (no dataloader)...")
        V_global = num_global_views  # Number of global crops
        V_local = local_crops_number  # Number of local crops
        
        # Determine correct lidar_mode for dummy data
        # aligned_mode=True means we use 1-channel depth
        if aligned_mode:
            actual_lidar_mode = 'depth'
        elif lidar_mode == 'auto':
            # When auto and not aligned, use range images (5-ch) for B/C/D
            if arch in ('B', 'C', 'D'):
                actual_lidar_mode = 'range'
            else:
                actual_lidar_mode = 'depth'
        else:
            actual_lidar_mode = lidar_mode
        
        print(f"  Using lidar_mode={actual_lidar_mode}, aligned_mode={aligned_mode}")
        
        def dummy_data_iterator(num_batches=3):
            """Generate dummy batches for fast model testing."""
            for i in range(num_batches):
                yield generate_dummy_batch(
                    batch_size=batch_size,
                    V_global=V_global + 1,  # +1 for probe view
                    V_local=V_local,
                    img_size=input_img_size,
                    local_size=input_local_img_size,
                    arch=arch,
                    device=device,
                    num_scenes=train_ds.num_scenes if hasattr(train_ds, 'num_scenes') else 10,
                    num_cameras=6,
                    num_locations=4,
                    lidar_mode=actual_lidar_mode,
                    aligned_mode=aligned_mode
                )
    
    acc_scene = 0.0 # Initialize to avoid UnboundLocalError
    for epoch in range(epochs):
        pretrained_trunk_frozen = False
        if pretrained_trunk_params and pretrained_trunk_warmup_freeze_epochs > 0 and not freeze_encoder:
            pretrained_trunk_frozen = epoch < pretrained_trunk_warmup_freeze_epochs
            _set_param_trainable(pretrained_trunk_params, not pretrained_trunk_frozen)

        if freeze_encoder:
            net.eval()
        else:
            net.train()
        if dino_student_head is not None:
            if freeze_encoder:
                dino_student_head.eval()
            else:
                dino_student_head.train()
        if teacher_net is not None:
            teacher_net.eval()
        if dino_teacher_head is not None:
            dino_teacher_head.eval()
        probes.train()
        if patch_probes is not None:
            patch_probes.train()

        if pretrained_trunk_params and not freeze_encoder:
            status = 'frozen' if pretrained_trunk_frozen else 'trainable'
            print(
                f"🎯 PRETRAINED TRUNK STATUS: {status} "
                f"(epoch {epoch + 1}/{epochs}, warmup_epochs={pretrained_trunk_warmup_freeze_epochs})"
            )
        
        epoch_losses = {"total": 0, "sigreg": 0, "inv": 0, "novel_reg": 0, "gmm": 0, "sinkhorn": 0, "spectral": 0}
        
        # Use dummy data or real dataloader
        if debug_model:
            data_iterator = dummy_data_iterator(num_batches=3)
            pbar = tqdm.tqdm(data_iterator, desc=f"Epoch {epoch+1}/{epochs} [DUMMY DATA]", total=3)
        else:
            pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", total=1 if debug else None)
        
        for batch_idx, (cam_views, modality2, labels) in enumerate(pbar):
            _batch_start_time = time.perf_counter()
            if gpu_memory_stats["available"]:
                torch.cuda.reset_peak_memory_stats()
            if track_encoder_compute:
                encoder_compute_stats["batch_forward_calls"] = 0
                encoder_compute_stats["batch_forward_calls_excl_probes"] = 0

            with autocast(enabled=(device == "cuda")):
                # Move to device handled below (dummy data already on device)
                
                if debug and batch_idx > 0 and not debug_model:
                    break
                    
                # Handle dictionary inputs (Global vs Local)
                # We need to extract the "Clean Full View" for probing.
                # In mm_dataset.py, we appended the full view to the 'global' list.
                # So the last element of 'global' is the probe view.
                
                cam_views_probe = None
                modality2_probe = None
                probe_label_view_idx = -1
                sampled_probe_idx = None
                
                cam_train = None
                mod2_train = None
                
                # --- Process Camera Views ---
                if isinstance(cam_views, dict):
                    p_views = cam_views.get('probe')
                    if disable_probe_training:
                        p_views = None  # skip GPU transfer — probes unused
                    else:
                        p_views = p_views.to(device, non_blocking=True) if isinstance(p_views, torch.Tensor) and p_views.numel() > 0 else None
                    if probe_only_training and probe_view_mode == 'clean_last' and p_views is not None:
                        cam_views_probe = p_views
                        probe_label_view_idx = -1
                        train_g = p_views
                        cam_train = p_views
                    else:
                        # 1. Global Views
                        g_views = cam_views['global'].to(device, non_blocking=True)
                    # Probe view selection from globals:
                    # - clean_last: use appended clean center-crop (last)
                    # - random_global: sample one of JEPA global crops [0..V-1]
                        if disable_probe_training:
                            probe_label_view_idx = None
                            train_g = g_views
                        elif probe_view_mode == 'clean_last' and p_views is not None:
                            cam_views_probe = p_views
                            probe_label_view_idx = -1
                            train_g = g_views
                        elif g_views.shape[1] > 0:
                            if probe_view_mode == 'random_global' and g_views.shape[1] > 1:
                                probe_global_count = g_views.shape[1] - 1
                                sampled_probe_idx = int(torch.randint(0, probe_global_count, (1,), device=g_views.device).item())
                                cam_views_probe = g_views[:, sampled_probe_idx:sampled_probe_idx + 1]
                                probe_label_view_idx = sampled_probe_idx
                            else:
                                cam_views_probe = g_views[:, -1:]
                                probe_label_view_idx = -1
                            train_g = g_views[:, :-1] if g_views.shape[1] > 1 else g_views
                        else:
                            train_g = g_views
                            
                        # 2. Local Views (all for training)
                        if cam_views['local'].numel() > 0:
                            l_views = cam_views['local'].to(device, non_blocking=True)
                            cam_train = {'global': train_g, 'local': l_views}
                        else:
                            cam_train = train_g
                else:
                    # Legacy tensor mode
                    c_views = cam_views.to(device, non_blocking=True)
                    if disable_probe_training:
                        cam_train = c_views
                    elif c_views.shape[1] > 0:
                        if probe_view_mode == 'random_global' and c_views.shape[1] > 1:
                            probe_global_count = c_views.shape[1] - 1
                            sampled_probe_idx = int(torch.randint(0, probe_global_count, (1,), device=c_views.device).item())
                            cam_views_probe = c_views[:, sampled_probe_idx:sampled_probe_idx + 1]
                            probe_label_view_idx = sampled_probe_idx
                        else:
                            cam_views_probe = c_views[:, -1:]
                            probe_label_view_idx = -1
                        cam_train = c_views[:, :-1] if c_views.shape[1] > 1 else c_views

                # --- Process Modality 2 ---
                if isinstance(modality2, dict):
                    p_views = modality2.get('probe')
                    if disable_probe_training:
                        p_views = None  # skip GPU transfer — probes unused
                    else:
                        p_views = p_views.to(device, non_blocking=True) if isinstance(p_views, torch.Tensor) and p_views.numel() > 0 else None
                    if probe_only_training and probe_view_mode == 'clean_last' and p_views is not None:
                        modality2_probe = p_views
                        train_g = p_views
                        mod2_train = p_views
                    else:
                        # 1. Global Views
                        g_views = modality2['global'].to(device, non_blocking=True)
                        # Split last view as probe
                        if disable_probe_training:
                            train_g = g_views
                        elif probe_view_mode == 'clean_last' and p_views is not None:
                            modality2_probe = p_views
                            train_g = g_views
                        elif g_views.shape[1] > 0:
                            if probe_view_mode == 'random_global' and sampled_probe_idx is not None and (g_views.shape[1] - 1) > sampled_probe_idx:
                                modality2_probe = g_views[:, sampled_probe_idx:sampled_probe_idx + 1]
                            else:
                                modality2_probe = g_views[:, -1:]
                            train_g = g_views[:, :-1] if g_views.shape[1] > 1 else g_views
                        else:
                            train_g = g_views
                            
                        # 2. Local
                        if modality2['local'].numel() > 0:
                            l_views = modality2['local'].to(device, non_blocking=True)
                            mod2_train = {'global': train_g, 'local': l_views}
                        else:
                            mod2_train = train_g
                        
                elif isinstance(modality2, torch.Tensor):
                    m_views = modality2.to(device, non_blocking=True)
                    # If points (N, 5) -> No view dim to slice?
                    # Check dim. Points usually (B, N, 5) -> dim=3.
                    # Views usually (B, V, ...) -> dim>=4.
                    if disable_probe_training:
                        mod2_train = m_views
                    elif m_views.dim() >= 4 and m_views.shape[1] > 1:
                        # It has views (depth/range mode)
                        if probe_view_mode == 'random_global' and sampled_probe_idx is not None and (m_views.shape[1] - 1) > sampled_probe_idx:
                            modality2_probe = m_views[:, sampled_probe_idx:sampled_probe_idx + 1]
                        else:
                            modality2_probe = m_views[:, -1:]
                        mod2_train = m_views[:, :-1]
                    elif isinstance(m_views, dict): # Should be caught above
                         pass # handled
                    else:
                        # Points mode: Single tensor shared for all views?
                        # Or aligned points dict?
                        # If simple tensor (B, N, 5), use as both train and probe
                        modality2_probe = m_views
                        mod2_train = m_views
                        
                # Update variable names for rest of loop
                cam_views = cam_train
                modality2 = mod2_train

                if probe_only_training:
                    if cam_views_probe is None:
                        raise RuntimeError("probe_only_training requires an explicit probe view in the batch")
                    cam_views = cam_views_probe
                    modality2 = modality2_probe

                imagebind_ssl_cam_views = None
                imagebind_ssl_modality2 = None
                if use_imagebind:
                    if imagebind_use_clean_probe_views and cam_views_probe is not None and modality2_probe is not None:
                        imagebind_ssl_cam_views = cam_views_probe
                        imagebind_ssl_modality2 = modality2_probe
                    else:
                        imagebind_ssl_cam_views = cam_views
                        imagebind_ssl_modality2 = modality2
                
                # Pre-process inputs based on no_fusion flag
                # Pre-process inputs based on no_fusion / rgb_zero flag
                if no_fusion:
                    if arch == "D":
                         # Zero out depth (separate modality2 before concat)
                         if isinstance(modality2, dict):
                             modality2 = {k: torch.zeros_like(v) for k, v in modality2.items()}
                         elif isinstance(modality2, torch.Tensor):
                             modality2 = torch.zeros_like(modality2)
                    elif arch in ("B", "C", "A", "E", "F"):
                         # Zero out second modality
                         modality2 = torch.zeros_like(modality2)
                
                if rgb_zero:
                     if arch == "D":
                         # Zero out RGB channels (will be zeroed before concat)
                         if isinstance(cam_views, dict):
                             cam_views = {k: torch.zeros_like(v) for k, v in cam_views.items()}
                         elif isinstance(cam_views, torch.Tensor):
                             cam_views = torch.zeros_like(cam_views)
                     elif arch in ("B", "C", "A", "E", "F"):
                         # Zero out camera views
                         cam_views = torch.zeros_like(cam_views)

                # ── Arch D: concatenate depth as 4th channel → RGBD ──
                # The dataset returns 3-ch RGB + 1-ch depth separately;
                # arch D expects a single 4-ch RGBD input.
                if arch == 'D':
                    def _concat_rgbd(rgb, depth):
                        """Concatenate 3-ch RGB + 1-ch depth → 4-ch RGBD."""
                        if rgb.shape[2] >= 4:
                            return rgb  # Already RGBD
                        return torch.cat([rgb, depth], dim=2)
                    if isinstance(cam_views, dict) and isinstance(modality2, dict):
                        cam_views = {
                            'global': _concat_rgbd(cam_views['global'], modality2['global']),
                            'local': _concat_rgbd(cam_views['local'], modality2['local']) if cam_views.get('local') is not None and cam_views['local'].numel() > 0 else cam_views.get('local', torch.empty(0)),
                        }
                    elif isinstance(cam_views, torch.Tensor) and isinstance(modality2, torch.Tensor):
                        cam_views = _concat_rgbd(cam_views, modality2)

                if track_encoder_compute and not encoder_compute_stats["profile_attempted"]:
                    total_calls_before = encoder_compute_stats["total_forward_calls"]
                    batch_calls_before = encoder_compute_stats["batch_forward_calls"]
                    macs_per_call = estimate_encoder_macs_per_call(
                        net,
                        cam_views,
                        modality2,
                        arch,
                        simple_baseline=simple_baseline,
                        simple_modality=simple_modality,
                    )
                    encoder_compute_stats["profile_attempted"] = True

                    calls_added = encoder_compute_stats["total_forward_calls"] - total_calls_before
                    if calls_added > 0:
                        encoder_compute_stats["total_forward_calls"] -= calls_added
                        encoder_compute_stats["batch_forward_calls"] = batch_calls_before

                    if macs_per_call is not None:
                        encoder_compute_stats["macs_per_call"] = macs_per_call
                        encoder_compute_stats["flops_per_call"] = 2.0 * macs_per_call
                        encoder_compute_stats["profile_success"] = True
                        print(f"📏 Encoder compute estimate: {macs_per_call/1e9:.3f} GMAC/call, {2.0*macs_per_call/1e9:.3f} GFLOP/call")

                # ── Encoder-only SSL FLOP profiling via FlopCounterMode ──
                # Profile only the encoder training path we want to compare:
                # encoder forward + self-supervised loss + encoder backward.
                # Probe forward/backward are excluded entirely.
                if estimate_encoder_train_compute:
                    estimate_profile_batch_idx = max(0, estimate_warmup_batches - 1)
                    _run_encoder_ssl_flop_profile = (
                        FLOP_COUNTER_AVAILABLE
                        and track_encoder_compute
                        and not legacy_linear_probe_setup
                        and not probe_only_training
                        and batch_idx == estimate_profile_batch_idx
                    )
                else:
                    _run_encoder_ssl_flop_profile = (
                        FLOP_COUNTER_AVAILABLE
                        and track_encoder_compute
                        and not legacy_linear_probe_setup
                        and not probe_only_training
                        and batch_idx % flop_profile_freq == 0
                    )
                _encoder_ssl_forward_flop_counter = None
                _encoder_ssl_backward_flop_counter = None
                _encoder_ssl_forward_flops = None
                if _run_encoder_ssl_flop_profile:
                    _encoder_ssl_forward_flop_counter = FlopCounterMode(display=False)
                    _encoder_ssl_forward_flop_counter.__enter__()

                # Forward pass (architecture-specific)
                objective_proj = None
                objective_cls_emb = None
                cached_probe_patch_tokens = None
                def _encoder_forward(callable_obj, *args, **kwargs):
                    if freeze_encoder:
                        with torch.no_grad():
                            if probe_only_training and probe_forward_chunk_size > 0:
                                return maybe_chunked_forward(callable_obj, probe_forward_chunk_size, *args, **kwargs)
                            return callable_obj(*args, **kwargs)
                    return callable_obj(*args, **kwargs)

                if probe_only_training:
                    sigreg_loss = torch.tensor(0.0, device=device)
                    inv_loss = torch.tensor(0.0, device=device)
                    novel_loss = torch.tensor(0.0, device=device)
                    novel_reg_loss = torch.tensor(0.0, device=device)
                    lejepa_loss = torch.tensor(0.0, device=device)
                    proj = torch.zeros(1, 1, int(cfg.proj_dim), device=device)

                    if simple_baseline:
                        if simple_modality in ('lidar', 'thermal'):
                            simple_input = modality2
                        else:
                            simple_input = cam_views

                        emb, _ = _encoder_forward(net, simple_input)
                        B, V = get_input_stats(simple_input)
                        cam_emb = emb
                        lidar_emb = emb
                    elif arch == "D":
                        emb, _ = _encoder_forward(net, cam_views)
                        B, _ = get_input_stats(cam_views)
                        _, V = get_input_stats(cam_views)
                        cam_emb = emb
                        lidar_emb = emb
                    elif arch in ("B", "C"):
                        if use_frustum_slots and arch == 'C' and hasattr(net, 'forward_with_slot_tokens'):
                            cls_emb_fused, _, _ = _encoder_forward(net.forward_with_slot_tokens, cam_views, modality2)
                            emb = torch.cat([cls_emb_fused, cls_emb_fused], dim=0)
                        elif fusion_tokens_sigreg and arch == 'C' and hasattr(net, 'forward_with_fusion_tokens'):
                            cls_emb_fused, _, cached_probe_patch_tokens = _encoder_forward(net.forward_with_fusion_tokens, cam_views, modality2)
                            emb = torch.cat([cls_emb_fused, cls_emb_fused], dim=0)
                        else:
                            exact_probe_aux = None
                            if use_multimae_exact_mt or use_multimae_exact_segplus:
                                exact_probe_aux = {}
                                exact_mt_waymo_panoptic = bool(use_multimae_exact_mt and not use_multimae_exact_segplus and dataset_name == 'waymo')
                                if exact_mt_waymo_panoptic:
                                    global_seg = labels.get('multimae_global_panoptic_seg_map', None)
                                    global_has_seg = labels.get('multimae_global_has_panoptic_seg_map', None)
                                elif use_multimae_exact_mt and not use_multimae_exact_segplus:
                                    global_seg = labels.get('multimae_global_panoptic_seg_map', labels.get('multimae_global_seg_map', None))
                                    global_has_seg = labels.get('multimae_global_has_panoptic_seg_map', labels.get('multimae_global_has_seg_map', None))
                                else:
                                    global_seg = labels.get('multimae_global_seg_map', None)
                                    global_has_seg = labels.get('multimae_global_has_seg_map', None)
                                if global_seg is not None and global_has_seg is not None:
                                    exact_probe_aux['global_seg_map'] = global_seg.to(device).long()
                                    exact_probe_aux['global_has_seg_map'] = global_has_seg.to(device) if hasattr(global_has_seg, 'to') else torch.as_tensor(global_has_seg, device=device)
                                if cam_views_probe is not None and modality2_probe is not None:
                                    exact_probe_aux['cam_views'] = cam_views_probe
                                    exact_probe_aux['range_views'] = modality2_probe
                                    if probe_label_view_idx is not None and probe_label_view_idx >= 0 and global_seg is not None and global_has_seg is not None:
                                        exact_probe_aux['seg_map'] = global_seg[:, probe_label_view_idx].to(device).long()
                                        exact_probe_aux['has_seg_map'] = global_has_seg[:, probe_label_view_idx].to(device) if hasattr(global_has_seg, 'to') else torch.as_tensor(global_has_seg[:, probe_label_view_idx], device=device)
                                    else:
                                        if exact_mt_waymo_panoptic:
                                            probe_seg = labels.get('multimae_probe_panoptic_seg_map', labels.get('panoptic_seg_map'))
                                            probe_has_seg = labels.get('multimae_probe_has_panoptic_seg_map', labels.get('has_panoptic_seg_map'))
                                        elif use_multimae_exact_mt and not use_multimae_exact_segplus:
                                            probe_seg = labels.get('multimae_probe_panoptic_seg_map', labels.get('panoptic_seg_map', labels.get('multimae_probe_seg_map', labels.get('seg_map'))))
                                            probe_has_seg = labels.get('multimae_probe_has_panoptic_seg_map', labels.get('has_panoptic_seg_map', labels.get('multimae_probe_has_seg_map', labels.get('has_seg_map'))))
                                        else:
                                            probe_seg = labels.get('multimae_probe_seg_map', labels.get('seg_map'))
                                            probe_has_seg = labels.get('multimae_probe_has_seg_map', labels.get('has_seg_map'))
                                        if probe_seg is not None and probe_has_seg is not None:
                                            exact_probe_aux['seg_map'] = probe_seg.to(device).long()
                                            exact_probe_aux['has_seg_map'] = probe_has_seg.to(device) if hasattr(probe_has_seg, 'to') else torch.as_tensor(probe_has_seg, device=device)
                                    if use_multimae_exact_segplus:
                                        global_panoptic = labels.get('multimae_global_panoptic_seg_map', None)
                                        global_has_panoptic = labels.get('multimae_global_has_panoptic_seg_map', None)
                                        if global_panoptic is not None and global_has_panoptic is not None:
                                            exact_probe_aux['global_panoptic_seg_map'] = global_panoptic.to(device).long()
                                            exact_probe_aux['global_has_panoptic_seg_map'] = global_has_panoptic.to(device) if hasattr(global_has_panoptic, 'to') else torch.as_tensor(global_has_panoptic, device=device)
                                        if probe_label_view_idx is not None and probe_label_view_idx >= 0 and global_panoptic is not None and global_has_panoptic is not None:
                                            exact_probe_aux['panoptic_seg_map'] = global_panoptic[:, probe_label_view_idx].to(device).long()
                                            exact_probe_aux['has_panoptic_seg_map'] = global_has_panoptic[:, probe_label_view_idx].to(device) if hasattr(global_has_panoptic, 'to') else torch.as_tensor(global_has_panoptic[:, probe_label_view_idx], device=device)
                                        else:
                                            probe_panoptic = labels.get('multimae_probe_panoptic_seg_map', labels.get('panoptic_seg_map'))
                                            probe_has_panoptic = labels.get('multimae_probe_has_panoptic_seg_map', labels.get('has_panoptic_seg_map'))
                                            if probe_panoptic is not None and probe_has_panoptic is not None:
                                                exact_probe_aux['panoptic_seg_map'] = probe_panoptic.to(device).long()
                                                exact_probe_aux['has_panoptic_seg_map'] = probe_has_panoptic.to(device) if hasattr(probe_has_panoptic, 'to') else torch.as_tensor(probe_has_panoptic, device=device)
                                if not exact_probe_aux:
                                    exact_probe_aux = None
                            emb, _ = _encoder_forward(net, cam_views, modality2, probe_aux=exact_probe_aux) if (use_multimae_exact or use_multimae_exact_mt or use_multimae_exact_segplus) else _encoder_forward(net, cam_views, modality2)

                        B, _ = get_input_stats(cam_views)
                        _, V = get_input_stats(cam_views)
                        cam_emb = emb[:B*V]
                        lidar_emb = emb[B*V:]
                    elif arch == "F":
                        (cam_emb, lidar_emb), _ = _encoder_forward(net, cam_views, modality2)
                        B, _ = get_input_stats(cam_views)
                        _, V = get_input_stats(cam_views)
                    elif arch in ("A", "E"):
                        (cam_emb, lidar_emb), _ = _encoder_forward(net, cam_views, modality2)
                        B, _ = get_input_stats(cam_views)
                        _, V = get_input_stats(cam_views)
                    else:
                        raise RuntimeError(f"Unsupported probe_only_training architecture: {arch}")
                elif simple_baseline:
                    # Simple baseline: single modality vanilla ViT
                    if simple_modality in ('lidar', 'thermal'):
                        # Use second modality only: aligned LiDAR/depth or FLIR thermal.
                        simple_input = modality2
                    else:
                        # Use RGB
                        simple_input = cam_views
                    
                    emb, proj = _encoder_forward(net, simple_input)
                    sigreg_loss = sigreg(proj)
                    B, V = get_input_stats(simple_input)
                    V_g = simple_input['global'].shape[1] if isinstance(simple_input, dict) else V
                    centers = proj[:V_g].mean(0)
                    inv_loss = (centers - proj).square().mean()
                    cam_emb = emb
                    lidar_emb = emb  # Same for simple
                elif arch == "D":
                    # Option D: RGBD single input (cam_views contains RGBD, modality2 is dummy)
                    emb, proj = _encoder_forward(net, cam_views)  # cam_views is actually RGBD for arch D
                    sigreg_loss = sigreg(proj)
                    B, _ = get_input_stats(cam_views)
                    _, V = get_input_stats(cam_views)
                    V_g = cam_views['global'].shape[1] if isinstance(cam_views, dict) else V
                    centers = proj[:V_g].mean(0)
                    inv_loss = (centers - proj).square().mean()
                    cam_emb = emb  # All embeddings are "camera" (RGBD)
                    lidar_emb = emb  # Same as cam_emb for D
                elif arch in ("B", "C"):
                    # Option B/C: Unified encoder (B=separate passes, C=true fusion)
                    fusion_tokens_joint = None
                    frustum_slots_joint = None
                    objective_proj = None
                    if use_frustum_slots and arch == 'C' and hasattr(net, 'forward_with_slot_tokens'):
                        cls_emb_fused, proj_fused, frustum_slots_joint = _encoder_forward(net.forward_with_slot_tokens, cam_views, modality2)
                        emb = torch.cat([cls_emb_fused, cls_emb_fused], dim=0)
                        proj = torch.cat([proj_fused, proj_fused], dim=0)
                        objective_proj = proj_fused
                        objective_cls_emb = cls_emb_fused
                    elif fusion_tokens_sigreg and arch == 'C' and hasattr(net, 'forward_with_fusion_tokens'):
                        cls_emb_fused, proj_fused, fusion_tokens_joint = _encoder_forward(net.forward_with_fusion_tokens, cam_views, modality2)
                        emb = torch.cat([cls_emb_fused, cls_emb_fused], dim=0)
                        proj = torch.cat([proj_fused, proj_fused], dim=0)
                        objective_proj = proj_fused
                        objective_cls_emb = cls_emb_fused
                    else:
                        exact_probe_aux = None
                        if use_multimae_exact_mt or use_multimae_exact_segplus:
                            exact_probe_aux = {}
                            exact_mt_waymo_panoptic = bool(use_multimae_exact_mt and not use_multimae_exact_segplus and dataset_name == 'waymo')
                            if exact_mt_waymo_panoptic:
                                global_seg = labels.get('multimae_global_panoptic_seg_map', None)
                                global_has_seg = labels.get('multimae_global_has_panoptic_seg_map', None)
                            elif use_multimae_exact_mt and not use_multimae_exact_segplus:
                                global_seg = labels.get('multimae_global_panoptic_seg_map', labels.get('multimae_global_seg_map', None))
                                global_has_seg = labels.get('multimae_global_has_panoptic_seg_map', labels.get('multimae_global_has_seg_map', None))
                            else:
                                global_seg = labels.get('multimae_global_seg_map', None)
                                global_has_seg = labels.get('multimae_global_has_seg_map', None)
                            if global_seg is not None and global_has_seg is not None:
                                exact_probe_aux['global_seg_map'] = global_seg.to(device).long()
                                exact_probe_aux['global_has_seg_map'] = global_has_seg.to(device) if hasattr(global_has_seg, 'to') else torch.as_tensor(global_has_seg, device=device)
                            if cam_views_probe is not None and modality2_probe is not None:
                                exact_probe_aux['cam_views'] = cam_views_probe
                                exact_probe_aux['range_views'] = modality2_probe
                                if probe_label_view_idx is not None and probe_label_view_idx >= 0 and global_seg is not None and global_has_seg is not None:
                                    exact_probe_aux['seg_map'] = global_seg[:, probe_label_view_idx].to(device).long()
                                    exact_probe_aux['has_seg_map'] = global_has_seg[:, probe_label_view_idx].to(device) if hasattr(global_has_seg, 'to') else torch.as_tensor(global_has_seg[:, probe_label_view_idx], device=device)
                                else:
                                    if exact_mt_waymo_panoptic:
                                        probe_seg = labels.get('multimae_probe_panoptic_seg_map', labels.get('panoptic_seg_map'))
                                        probe_has_seg = labels.get('multimae_probe_has_panoptic_seg_map', labels.get('has_panoptic_seg_map'))
                                    elif use_multimae_exact_mt and not use_multimae_exact_segplus:
                                        probe_seg = labels.get('multimae_probe_panoptic_seg_map', labels.get('panoptic_seg_map', labels.get('multimae_probe_seg_map', labels.get('seg_map'))))
                                        probe_has_seg = labels.get('multimae_probe_has_panoptic_seg_map', labels.get('has_panoptic_seg_map', labels.get('multimae_probe_has_seg_map', labels.get('has_seg_map'))))
                                    else:
                                        probe_seg = labels.get('multimae_probe_seg_map', labels.get('seg_map'))
                                        probe_has_seg = labels.get('multimae_probe_has_seg_map', labels.get('has_seg_map'))
                                    if probe_seg is not None and probe_has_seg is not None:
                                        exact_probe_aux['seg_map'] = probe_seg.to(device).long()
                                        exact_probe_aux['has_seg_map'] = probe_has_seg.to(device) if hasattr(probe_has_seg, 'to') else torch.as_tensor(probe_has_seg, device=device)
                                if use_multimae_exact_segplus:
                                    global_panoptic = labels.get('multimae_global_panoptic_seg_map', None)
                                    global_has_panoptic = labels.get('multimae_global_has_panoptic_seg_map', None)
                                    if global_panoptic is not None and global_has_panoptic is not None:
                                        exact_probe_aux['global_panoptic_seg_map'] = global_panoptic.to(device).long()
                                        exact_probe_aux['global_has_panoptic_seg_map'] = global_has_panoptic.to(device) if hasattr(global_has_panoptic, 'to') else torch.as_tensor(global_has_panoptic, device=device)
                                    if probe_label_view_idx is not None and probe_label_view_idx >= 0 and global_panoptic is not None and global_has_panoptic is not None:
                                        exact_probe_aux['panoptic_seg_map'] = global_panoptic[:, probe_label_view_idx].to(device).long()
                                        exact_probe_aux['has_panoptic_seg_map'] = global_has_panoptic[:, probe_label_view_idx].to(device) if hasattr(global_has_panoptic, 'to') else torch.as_tensor(global_has_panoptic[:, probe_label_view_idx], device=device)
                                    else:
                                        probe_panoptic = labels.get('multimae_probe_panoptic_seg_map', labels.get('panoptic_seg_map'))
                                        probe_has_panoptic = labels.get('multimae_probe_has_panoptic_seg_map', labels.get('has_panoptic_seg_map'))
                                        if probe_panoptic is not None and probe_has_panoptic is not None:
                                            exact_probe_aux['panoptic_seg_map'] = probe_panoptic.to(device).long()
                                            exact_probe_aux['has_panoptic_seg_map'] = probe_has_panoptic.to(device) if hasattr(probe_has_panoptic, 'to') else torch.as_tensor(probe_has_panoptic, device=device)
                            if not exact_probe_aux:
                                exact_probe_aux = None
                        emb, proj = _encoder_forward(net, cam_views, modality2, probe_aux=exact_probe_aux) if (use_multimae_exact or use_multimae_exact_mt or use_multimae_exact_segplus) else _encoder_forward(net, cam_views, modality2)
                        if use_rgb_dino:
                            objective_proj = proj[:proj.shape[0] // 2]
                            objective_cls_emb = emb[:emb.shape[0] // 2]
                        else:
                            objective_proj = proj
                    sigreg_loss = sigreg(proj)
                    B, _ = get_input_stats(cam_views)
                    _, V = get_input_stats(cam_views)
                    V_g = cam_views['global'].shape[1] if isinstance(cam_views, dict) else V
                    # In B/C, proj is cat([cam_proj, range_proj]) -> (2*V_total, B, D)
                    # For global center, we pull each modality to its own global center or a joint one?
                    # The user requested "centers are only calculated on the big crops".
                    # We'll use the first V_g views as target for everything (simple interpretation).
                    centers = proj[:V_g].mean(0)
                    inv_loss = (centers - proj).square().mean()
                    cam_emb = emb[:B*V]
                    lidar_emb = emb[B*V:]  # Range embeddings
                elif arch == "F":
                    # Option F: Separate encoders + separate projectors
                    # Returns 3 projections: cam_proj, depth_proj, cross_proj
                    (cam_emb, lidar_emb), (cam_proj, lidar_proj, cross_proj) = _encoder_forward(net, cam_views, modality2)
                    
                    # Baseline: ONLY intra-modal SIGReg (no cross-modal in baseline)
                    sigreg_cam = sigreg(cam_proj)       # RGB views aligned with RGB views
                    sigreg_lidar = sigreg(lidar_proj)   # Depth views aligned with depth views
                    sigreg_loss = sigreg_cam + sigreg_lidar
                    
                    # cross_proj is available for cross_modal_sigreg novel loss
                    V_g = cam_views['global'].shape[1] if isinstance(cam_views, dict) else cam_proj.shape[0]
                    centers_c = cam_proj[:V_g].mean(0)
                    centers_l = lidar_proj[:V_g].mean(0)
                    inv_loss = (centers_c - cam_proj).square().mean() + (centers_l - lidar_proj).square().mean()
                    B, _ = get_input_stats(cam_views)
                    _, V = get_input_stats(cam_views)
                elif arch in ("A", "E"):
                    # Option A/E: Separate encoders (A=ViT+PointMLP, E=ViT+ViT for aligned depth)
                    (cam_emb, lidar_emb), (cam_proj, lidar_proj) = _encoder_forward(net, cam_views, modality2)
                    sigreg_cam = sigreg(cam_proj)
                    sigreg_lidar = sigreg(lidar_proj)
                    sigreg_loss = sigreg_cam + sigreg_lidar
                    
                    V_g = cam_views['global'].shape[1] if isinstance(cam_views, dict) else cam_proj.shape[0]
                    centers_c = cam_proj[:V_g].mean(0)
                    # For A, lidar_proj is often (1, B, D). V_g for lidar might be 1.
                    V_gl = lidar_proj.shape[0] if lidar_proj.shape[0] < V_g else V_g
                    centers_l = lidar_proj[:V_gl].mean(0)
                    
                    inv_loss = (centers_c - cam_proj).square().mean() + (centers_l - lidar_proj).square().mean()
                    B, _ = get_input_stats(cam_views)
                    _, V = get_input_stats(cam_views)

                train_rgb_lidar_patch_mse = None
                objective_proj = objective_proj if 'objective_proj' in locals() and objective_proj is not None else proj
                if objective_cls_emb is None and arch in ("B", "C"):
                    objective_cls_emb = emb[:B*V]
                should_probe_patch_mse = (
                    track_patch_mse
                    and (batch_idx % patch_mse_probe_freq == 0)
                    and (not probe_only_training)
                    and (not simple_baseline)
                    and arch in ('B', 'C', 'D', 'E', 'F')
                    and (not use_frustum_slots)
                    and (not use_dinov3_frozen)
                    and (not use_dinov3_pretrained)
                    and (not use_dinov3_scratch)
                )
                if should_probe_patch_mse:
                    with torch.no_grad():
                        rgb_patch_tokens, lidar_patch_tokens = extract_rgb_lidar_patch_pair(
                            net,
                            cam_views,
                            modality2,
                            arch,
                            simple_baseline=simple_baseline,
                            batch_chunk_size=probe_forward_chunk_size,
                        )
                        train_rgb_lidar_patch_mse = compute_patch_mse_safe(rgb_patch_tokens, lidar_patch_tokens)
                
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                run_encoder_ssl_objective = not probe_only_training
                
                # ============================================
                # BASELINE LOSS REPLACEMENTS (VICReg, InfoNCE, MultiMAE)
                # ============================================
                if run_encoder_ssl_objective and use_vicreg and vicreg_loss_fn is not None:
                    # Replace SIGReg + invariance with VICReg
                    vicreg_total = vicreg_loss_fn(objective_proj)
                    lejepa_loss = vicreg_total
                    if batch_idx % 100 == 0 and use_wandb:
                        components = vicreg_loss_fn.forward_components(objective_proj)
                        wandb.log({
                            'vicreg_invariance': components['invariance'].item(),
                            'vicreg_variance': components['variance'].item(),
                            'vicreg_covariance': components['covariance'].item(),
                            'vicreg_total': components['total'].item(),
                        })
                elif run_encoder_ssl_objective and use_infonce and infonce_loss_fn is not None:
                    # Replace SIGReg + invariance with InfoNCE
                    infonce_total = infonce_loss_fn(objective_proj)
                    lejepa_loss = infonce_total
                    if batch_idx % 100 == 0 and use_wandb:
                        wandb.log({'infonce_loss': infonce_total.item()})
                elif run_encoder_ssl_objective and use_whitened_infonce and whitened_infonce_loss_fn is not None:
                    whitened_total = whitened_infonce_loss_fn(objective_proj)
                    lejepa_loss = whitened_total
                    if batch_idx % 100 == 0 and use_wandb:
                        components = whitened_infonce_loss_fn.forward_components(objective_proj)
                        wandb.log({
                            'whitened_infonce_loss': components['whitened_infonce'].item(),
                            'whitening_offdiag': components['whitening_offdiag'].item(),
                        })
                elif run_encoder_ssl_objective and use_eigennce and eigennce_loss_fn is not None:
                    eigennce_total = eigennce_loss_fn(objective_proj)
                    lejepa_loss = eigennce_total
                    if batch_idx % 100 == 0 and use_wandb:
                        components = eigennce_loss_fn.forward_components(objective_proj)
                        wandb.log({
                            'eigennce_loss': components['eigennce'].item(),
                            'eig_top': components['eig_top'].item(),
                            'eig_bottom': components['eig_bottom'].item(),
                            'eig_condition': components['eig_condition'].item(),
                        })
                elif run_encoder_ssl_objective and use_adasignce and adasignce_loss_fn is not None:
                    # AdaSigNCE: Adaptive spectral-contrastive learning
                    adasignce_total = adasignce_loss_fn(objective_proj)
                    lejepa_loss = adasignce_total
                    if batch_idx % 100 == 0 and use_wandb:
                        components = adasignce_loss_fn.forward_components(objective_proj)
                        wandb.log({
                            'adasignce_infonce': components['infonce'].item(),
                            'adasignce_sigreg_stat': components['sigreg_stat'].item(),
                            'adasignce_tau': components['adaptive_tau'].item(),
                            'adasignce_sigreg_weight': components['adaptive_sigreg_weight'].item(),
                            'adasignce_total': components['total'].item(),
                        })
                elif run_encoder_ssl_objective and use_signce and signce_loss_fn is not None:
                    # SigNCE: Fixed-tau weighted sum (ablation for AdaSigNCE)
                    signce_total = signce_loss_fn(objective_proj)
                    lejepa_loss = signce_total
                    if batch_idx % 100 == 0 and use_wandb:
                        components = signce_loss_fn.forward_components(objective_proj)
                        wandb.log({
                            'signce_infonce': components['infonce'].item(),
                            'signce_sigreg_stat': components['sigreg_stat'].item(),
                            'signce_tau': components['adaptive_tau'].item(),
                            'signce_total': components['total'].item(),
                        })
                elif run_encoder_ssl_objective and use_fusion_dino and dino_loss_fn is not None and teacher_net is not None:
                    if objective_cls_emb is None or dino_student_head is None or dino_teacher_head is None:
                        raise RuntimeError("DINO fusion baseline requires explicit CLS embeddings and DINO heads")

                    student_cls_views = _reshape_view_major_embeddings(objective_cls_emb, cam_views)
                    student_logits = dino_student_head(student_cls_views.reshape(-1, student_cls_views.shape[-1]))
                    student_logits = student_logits.reshape(student_cls_views.shape[0], student_cls_views.shape[1], -1)
                    teacher_temp = dino_loss_fn.get_teacher_temp(epoch, epochs)
                    teacher_cam = cam_views['global'] if isinstance(cam_views, dict) else cam_views[:, :V_g]
                    teacher_modality2 = modality2['global'] if isinstance(modality2, dict) else modality2[:, :V_g]

                    with torch.no_grad():
                        if use_frustum_slots and arch == 'C' and hasattr(teacher_net, 'forward_with_slot_tokens'):
                            teacher_cls, _, _ = teacher_net.forward_with_slot_tokens(teacher_cam, teacher_modality2)
                        elif hasattr(teacher_net, 'forward_with_fusion_tokens'):
                            teacher_cls, _, _ = teacher_net.forward_with_fusion_tokens(teacher_cam, teacher_modality2)
                        else:
                            teacher_emb, _ = teacher_net(teacher_cam, teacher_modality2)
                            teacher_cls = teacher_emb[:B * V_g]
                        teacher_logits = dino_teacher_head(teacher_cls).reshape(B, V_g, -1).transpose(0, 1)

                    teacher_probs = dino_loss_fn.get_teacher_probs(
                        teacher_logits,
                        teacher_global_views=V_g,
                        teacher_temp=teacher_temp,
                    )
                    student_global_logits = student_logits[:V_g]
                    student_local_logits = student_logits[V_g:]
                    dino_global = dino_loss_fn(
                        student_global_logits,
                        teacher_probs,
                        ignore_diagonal=dino_global_ignore_diagonal,
                    )
                    dino_local = dino_loss_fn(student_local_logits, teacher_probs) if student_local_logits.shape[0] > 0 else torch.tensor(0.0, device=device)

                    dino_global_terms = V_g * max(V_g - (1 if dino_global_ignore_diagonal else 0), 0)
                    dino_local_terms = V_g * student_local_logits.shape[0]
                    dino_total_terms = max(dino_global_terms + dino_local_terms, 1)
                    dino_global_scale = dino_global_terms / dino_total_terms if dino_global_terms > 0 else 0.0
                    dino_local_scale = dino_local_terms / dino_total_terms if dino_local_terms > 0 else 0.0
                    dino_total = dino_global_scale * dino_global + dino_local_scale * dino_local
                    lejepa_loss = dino_total

                    if dino_use_koleo and koleo_loss_fn is not None:
                        student_global_cls = student_cls_views[:V_g].reshape(-1, student_cls_views.shape[-1])
                        koleo_total = koleo_loss_fn(student_global_cls)
                        lejepa_loss = lejepa_loss + float(getattr(cfg, 'dino_koleo_weight', 0.1)) * koleo_total
                    else:
                        koleo_total = torch.tensor(0.0, device=device)

                    if dino_use_ibot and ibot_patch_loss_fn is not None and ibot_student_head is not None and ibot_teacher_head is not None and use_rgb_dino:
                        mask_ratio = float(getattr(cfg, 'dino_ibot_mask_ratio', 0.3))
                        # ---- True masked iBOT (DINOv2-style) ----
                        # Student sees mask tokens at masked positions;
                        # teacher sees the full unmasked input.
                        if hasattr(net, 'forward_masked_ibot'):
                            student_cls_masked, student_patch_masked, bool_mask = net.forward_masked_ibot(
                                teacher_cam, mask_ratio=mask_ratio,
                            )
                            # Teacher: full unmasked forward
                            with torch.no_grad():
                                teacher_cls_full, teacher_patch_full = teacher_net._encode(
                                    teacher_net._maybe_resize_input(teacher_cam.flatten(0, 1))
                                )
                            # Project through iBOT heads at masked positions
                            student_patch_logits = ibot_student_head(student_patch_masked[bool_mask])
                            with torch.no_grad():
                                teacher_patch_logits = ibot_teacher_head(teacher_patch_full[bool_mask])
                            ibot_total = ibot_patch_loss_fn.forward_masked(
                                student_patch_logits,
                                teacher_patch_logits,
                                teacher_temp=teacher_temp,
                            )
                        elif hasattr(net, 'forward_with_patches'):
                            # Fallback: old-style post-hoc masking
                            student_patch_emb, _, (student_patch_tokens, _) = net.forward_with_patches({'global': teacher_cam})
                            with torch.no_grad():
                                teacher_patch_emb, _, (teacher_patch_tokens, _) = teacher_net.forward_with_patches({'global': teacher_cam})
                            student_patch_tokens = student_patch_tokens.reshape(B * V_g, student_patch_tokens.shape[1], -1)
                            teacher_patch_tokens = teacher_patch_tokens.reshape(B * V_g, teacher_patch_tokens.shape[1], -1)
                            n_patches = student_patch_tokens.shape[1]
                            n_mask = max(1, int(round(mask_ratio * n_patches)))
                            patch_mask = torch.zeros(B * V_g, n_patches, dtype=torch.bool, device=student_patch_tokens.device)
                            rand_scores = torch.rand(B * V_g, n_patches, device=student_patch_tokens.device)
                            mask_idx = rand_scores.topk(k=n_mask, dim=1, largest=False).indices
                            patch_mask.scatter_(1, mask_idx, True)
                            student_patch_logits = ibot_student_head(student_patch_tokens[patch_mask])
                            with torch.no_grad():
                                teacher_patch_logits = ibot_teacher_head(teacher_patch_tokens[patch_mask])
                            ibot_total = ibot_patch_loss_fn.forward_masked(
                                student_patch_logits,
                                teacher_patch_logits,
                                teacher_temp=teacher_temp,
                            )
                        else:
                            ibot_total = torch.tensor(0.0, device=device)
                        lejepa_loss = lejepa_loss + float(getattr(cfg, 'dino_ibot_weight', 1.0)) * ibot_total
                    else:
                        ibot_total = torch.tensor(0.0, device=device)

                    if batch_idx % 100 == 0 and use_wandb:
                        steps_per_epoch = 3 if debug_model else len(train_loader)
                        dino_momentum_now = _dino_teacher_momentum(epoch * steps_per_epoch + batch_idx, total_steps)
                        with torch.no_grad():
                            eps = 1e-12
                            entropy_norm = math.log(max(teacher_probs.shape[-1], 2))
                            teacher_entropy = -(teacher_probs.clamp_min(eps) * teacher_probs.clamp_min(eps).log()).sum(dim=-1).mean()
                            teacher_max_prob = teacher_probs.max(dim=-1).values.mean()

                            student_global_probs = F.softmax(student_global_logits.float() / dino_student_temp, dim=-1)
                            student_global_entropy = -(student_global_probs.clamp_min(eps) * student_global_probs.clamp_min(eps).log()).sum(dim=-1).mean()
                            student_global_max_prob = student_global_probs.max(dim=-1).values.mean()

                            if student_local_logits.shape[0] > 0:
                                student_local_probs = F.softmax(student_local_logits.float() / dino_student_temp, dim=-1)
                                student_local_entropy = -(student_local_probs.clamp_min(eps) * student_local_probs.clamp_min(eps).log()).sum(dim=-1).mean()
                                student_local_max_prob = student_local_probs.max(dim=-1).values.mean()
                            else:
                                student_local_entropy = torch.tensor(0.0, device=device)
                                student_local_max_prob = torch.tensor(0.0, device=device)

                        wandb.log({
                            'dino_loss': dino_total.item(),
                            'dino_global_loss': dino_global.item(),
                            'dino_local_loss': dino_local.item(),
                            'dino_global_scale': dino_global_scale,
                            'dino_local_scale': dino_local_scale,
                            'dino_global_terms': dino_global_terms,
                            'dino_local_terms': dino_local_terms,
                            'dino_local_crops_active': int(student_local_logits.shape[0]),
                            'dino_teacher_entropy': teacher_entropy.item(),
                            'dino_teacher_entropy_norm': teacher_entropy.item() / max(entropy_norm, 1e-12),
                            'dino_teacher_max_prob': teacher_max_prob.item(),
                            'dino_student_global_entropy': student_global_entropy.item(),
                            'dino_student_global_entropy_norm': student_global_entropy.item() / max(entropy_norm, 1e-12),
                            'dino_student_global_max_prob': student_global_max_prob.item(),
                            'dino_student_local_entropy': student_local_entropy.item(),
                            'dino_student_local_entropy_norm': student_local_entropy.item() / max(entropy_norm, 1e-12),
                            'dino_student_local_max_prob': student_local_max_prob.item(),
                            'dino_koleo_loss': koleo_total.item(),
                            'dino_ibot_loss': ibot_total.item(),
                            'dino_teacher_momentum': dino_momentum_now,
                            'dino_teacher_temp': teacher_temp,
                        })
                
                if run_encoder_ssl_objective and (use_multimae or use_multimae_exact or use_multimae_exact_mt or use_multimae_exact_segplus) and hasattr(net, 'get_recon_loss'):
                    # MultiMAE learns purely from reconstruction — NO SigReg/invariance
                    recon_loss = net.get_recon_loss()
                    if isinstance(recon_loss, torch.Tensor) and recon_loss.requires_grad:
                        lejepa_loss = recon_loss  # Replace entirely — reconstruction is the only loss
                        if batch_idx % 100 == 0 and use_wandb:
                            metric_name = 'multimae_exact_recon_loss' if (use_multimae_exact or use_multimae_exact_mt or use_multimae_exact_segplus) else 'multimae_recon_loss'
                            wandb.log({metric_name: recon_loss.item()})
                
                if run_encoder_ssl_objective and getattr(cfg, 'use_mdm', False) and hasattr(net, 'get_recon_loss'):
                    # MDM (LingBot-Depth style) learns from depth reconstruction
                    recon_loss = net.get_recon_loss()
                    if isinstance(recon_loss, torch.Tensor) and recon_loss.requires_grad:
                        lejepa_loss = recon_loss
                        if batch_idx % 100 == 0 and use_wandb:
                            wandb.log({'mdm_depth_recon_loss': recon_loss.item()})
                
                if run_encoder_ssl_objective and use_imagebind:
                    if imagebind_ssl_cam_views is None or imagebind_ssl_modality2 is None:
                        raise RuntimeError("ImageBind training requires paired RGB and depth views")

                    _encoder_forward(net, imagebind_ssl_cam_views, imagebind_ssl_modality2)
                    if not hasattr(net, '_last_proj_rgb') or not hasattr(net, '_last_proj_depth'):
                        raise RuntimeError("ImageBind encoder did not expose modality-specific projections")

                    pr = net._last_proj_rgb
                    pd = net._last_proj_depth
                    pr_flat = F.normalize(pr.reshape(-1, pr.shape[-1]), dim=-1)
                    pd_flat = F.normalize(pd.reshape(-1, pd.shape[-1]), dim=-1)
                    sim = pr_flat @ pd_flat.T / imagebind_temperature
                    ib_labels = torch.arange(sim.shape[0], device=sim.device)
                    lejepa_loss = (F.cross_entropy(sim, ib_labels) + F.cross_entropy(sim.T, ib_labels)) / 2
                    if batch_idx % 100 == 0 and use_wandb:
                        wandb.log({
                            'imagebind_contrastive_loss': lejepa_loss.item(),
                            'imagebind_temperature': imagebind_temperature,
                            'imagebind_clean_probe_views': 1 if imagebind_use_clean_probe_views else 0,
                        })
                
                if run_encoder_ssl_objective and use_late_fusion and late_fusion_patch_sigreg:
                    # Apply SigReg on per-modality patch embeddings to prevent collapse
                    if hasattr(net, '_last_proj_rgb') and hasattr(net, '_last_proj_depth'):
                        _lr = net._last_proj_rgb
                        _ld = net._last_proj_depth
                        patch_sig_rgb = sigreg(_lr)
                        patch_sig_depth = sigreg(_ld)
                        lejepa_loss = lejepa_loss + (patch_sig_rgb + patch_sig_depth) * cfg.lamb * 0.5
                        if batch_idx % 100 == 0 and use_wandb:
                            wandb.log({
                                'late_fusion_patch_sigreg_rgb': patch_sig_rgb.item(),
                                'late_fusion_patch_sigreg_depth': patch_sig_depth.item(),
                            })
                
                # ============================================
                # NOVEL MULTI-MODAL LOSSES
                # ============================================
                novel_loss = torch.tensor(0.0, device=device)
                
                # Idea 1: Cross-Modal SigReg (A, B, C, E, F)
                # FIXED: Interleave modalities along batch dimension so SIGReg sees them as ONE distribution
                if run_encoder_ssl_objective and cross_modal_sigreg and arch in ('A', 'B', 'C', 'E', 'F'):
                    if arch == 'F':
                        # F: cam_proj is (V, B, D), lidar_proj is (V, B, D)
                        # Interleave along batch: (V, 2*B, D) - cam and lidar as same distribution
                        cross_proj_interleaved = torch.cat([cam_proj, lidar_proj], dim=1)  # (V, 2*B, D)
                        cross_modal_loss = sigreg(cross_proj_interleaved)
                        
                    elif arch in ('A', 'E'):
                        # A: cam_proj is (V, B, D), lidar_proj is (1, B, D)
                        # E: cam_proj is (V, B, D), lidar_proj is (V, B, D)
                        if arch == 'A':
                            # Expand lidar to match V views
                            lidar_proj_exp = lidar_proj.expand(cam_proj.shape[0], -1, -1)
                        else:
                            lidar_proj_exp = lidar_proj
                        
                        cross_proj_interleaved = torch.cat([cam_proj, lidar_proj_exp], dim=1)  # (V, 2*B, D)
                        cross_modal_loss = sigreg(cross_proj_interleaved)
                        
                    else:  # B/C
                        # cam_emb is (B*V, D), lidar_emb is (B*V, D)
                        cam_proj_bc = net.proj(cam_emb).reshape(B, V, -1).transpose(0, 1)  # (V, B, D)
                        lidar_proj_bc = net.proj(lidar_emb).reshape(B, V, -1).transpose(0, 1)  # (V, B, D)
                        
                        cross_proj_interleaved = torch.cat([cam_proj_bc, lidar_proj_bc], dim=1)  # (V, 2*B, D)
                        cross_modal_loss = sigreg(cross_proj_interleaved)
                    
                    # Only enforce distribution matching
                    novel_loss = novel_loss + cross_modal_loss * 0.5
                
                # Idea 2: Modality Dropout + Invariance (ALL archs)
                if run_encoder_ssl_objective and modality_invariance:
                    with torch.no_grad():
                        # Get embeddings with each modality zeroed
                        if arch == 'D':
                            # Zero depth channel
                            cam_views_no_depth = cam_views.clone()
                            cam_views_no_depth[:, :, 3:] = 0
                            emb_cam_only, _ = net(cam_views_no_depth)
                            emb_lidar_only = emb  # Can't isolate LiDAR in D
                        elif arch in ('B', 'C'):
                            # B/C return [cam; range] - we need to slice
                            B_val, V_val = get_input_stats(cam_views)
                            total_views = B_val * V_val
                            
                            # Camera only pass
                            full_emb_cam, _ = net(cam_views, torch.zeros_like(modality2))
                            emb_cam_only = full_emb_cam[:total_views]
                            
                            # LiDAR only pass
                            full_emb_lidar, _ = net(torch.zeros_like(cam_views), modality2)
                            emb_lidar_only = full_emb_lidar[total_views:]
                        else:  # A, E, F
                            cam_emb_only = net.cam_encoder(cam_views.flatten(0, 1))
                            if arch == 'A':
                                lidar_emb_only = net.lidar_encoder(modality2)
                            else: # E, F
                                # E/F use depth_encoder. modality2 is aligned depth (B, V, 1, H, W)
                                depth_flat = modality2.flatten(0, 1)
                                lidar_emb_only = net.depth_encoder(depth_flat)
                            emb_cam_only, emb_lidar_only = cam_emb_only, lidar_emb_only
                    
                    # Invariance: each modality should approximate full
                    if arch != 'D':
                        inv_cam = (cam_emb - emb_cam_only).square().mean()
                        inv_lidar = (lidar_emb - emb_lidar_only).square().mean()
                        modal_inv_loss = inv_cam + inv_lidar
                    else:
                        inv_cam = (emb - emb_cam_only).square().mean()
                        modal_inv_loss = inv_cam
                    novel_loss = novel_loss + modal_inv_loss * 0.1
                
                # Idea 4: Modality-Specific Projections (A, B, C only)
                # Note: This requires separate projection heads, which we compute on-the-fly
                if run_encoder_ssl_objective and separate_projections and arch in ('A', 'B', 'C'):
                    # Create modality-specific projections
                    proj_dim = cfg.proj_dim
                    if not hasattr(net, '_sep_proj_cam'):
                        net._sep_proj_cam = nn.Linear(cam_emb.shape[-1], proj_dim // 2).to(device)
                        net._sep_proj_lidar = nn.Linear(lidar_emb.shape[-1], proj_dim // 2).to(device)
                    
                    sep_cam = net._sep_proj_cam(cam_emb)
                    sep_lidar = net._sep_proj_lidar(lidar_emb.repeat(V, 1) if arch == 'A' else lidar_emb)
                    combined = torch.cat([sep_cam, sep_lidar], dim=-1)  # (B*V, proj_dim)
                    
                    # Apply SigReg on combined
                    combined_views = combined.reshape(B, V, -1).transpose(0, 1)  # (V, B, proj_dim)
                    sep_sigreg_loss = sigreg(combined_views)
                    novel_loss = novel_loss + sep_sigreg_loss * 0.5
                
                # Idea 5: Direct Alignment (MSE between normalized projections)
                # JEPA-style: Forces camera and LiDAR to produce similar representations
                # SIGReg on each branch prevents collapse, so MSE alignment is safe
                # IMPORTANT: We align per-crop, not per-mean. Each cam_crop[i] aligns with lidar_crop[i]
                if run_encoder_ssl_objective and direct_alignment and arch in ('A', 'B', 'E', 'F'):
                    if arch == 'F':
                        # F: cam_proj is (V, B, proj_dim), lidar_proj is (V, B, proj_dim)
                        # Align per-crop: cam_proj[v] ↔ lidar_proj[v] for each view v
                        cam_proj_flat = cam_proj.transpose(0, 1).reshape(B * V, -1)  # (B*V, proj_dim)
                        lidar_proj_flat = lidar_proj.transpose(0, 1).reshape(B * V, -1)  # (B*V, proj_dim)
                    elif arch == 'E':
                        # E: cam_proj is (V, B, proj_dim), lidar_proj is (V, B, proj_dim)
                        cam_proj_flat = cam_proj.transpose(0, 1).reshape(B * V, -1)  # (B*V, proj_dim)
                        lidar_proj_flat = lidar_proj.transpose(0, 1).reshape(B * V, -1)  # (B*V, proj_dim)
                    elif arch == 'A':
                        # A: cam_proj is (V, B, proj_dim), lidar_proj is (1, B, proj_dim)
                        # LiDAR is global (single point cloud), so repeat for each view
                        cam_proj_flat = cam_proj.transpose(0, 1).reshape(B * V, -1)  # (B*V, proj_dim)
                        lidar_proj_flat = lidar_proj.squeeze(0).unsqueeze(1).expand(-1, V, -1).reshape(B * V, -1)  # (B*V, proj_dim)
                    else:  # B
                        # B: cam_emb is (B*V,), lidar_emb is (B*V,) - already flat
                        cam_proj_flat = net.proj(cam_emb)  # (B*V, proj_dim)
                        lidar_proj_flat = net.proj(lidar_emb)  # (B*V, proj_dim)
                    
                    # L2 normalize before MSE (cosine-like alignment)
                    cam_norm = F.normalize(cam_proj_flat, dim=-1)
                    lidar_norm = F.normalize(lidar_proj_flat, dim=-1)
                    
                    # Per-crop MSE: each crop pair is aligned individually
                    # MSE on normalized vectors = 2 * (1 - cosine_similarity)
                    alignment_loss = (cam_norm - lidar_norm).square().mean()
                    novel_loss = novel_loss + alignment_loss # * 0.5
                
                # Idea 6: Shared Trunk Contrastive (B only)
                # Same Encoder + Pairwise Embedding Alignment + SigReg on Overal Space
                # Why distinct from direct_alignment? 
                # 1. SigReg is applied on ALL embeddings together (done by default for B above: proj is cat([cam, range]))
                # 2. Alignment is on embeddings (or projections), forcing same-crop agreement.
                if run_encoder_ssl_objective and shared_trunk_contrastive and arch == 'B':
                    # For B: proj is (2*V, B, D) stacked [cam_proj; range_proj]
                    # We want to align cam_proj[v, b] with range_proj[v, b]
                    
                    # Split projections by View dimension
                    cam_proj_views = proj[:V]    # (V, B, D)
                    range_proj_views = proj[V:]  # (V, B, D)
                    
                    # Flatten to (B*V, D) for pairwise comparison
                    # Transpose (V, B) -> (B, V) -> Flatten
                    cam_proj_flat = cam_proj_views.transpose(0, 1).reshape(-1, cam_proj_views.shape[-1])
                    range_proj_flat = range_proj_views.transpose(0, 1).reshape(-1, range_proj_views.shape[-1])
                    
                    # Maximize similarity / Minimize distance
                    cam_norm = F.normalize(cam_proj_flat, dim=-1)

                    range_norm = F.normalize(range_proj_flat, dim=-1)
                    
                    # Pairwise alignment loss
                    align_loss = (cam_norm - range_norm).square().mean()
                    novel_loss = novel_loss + align_loss * 1.0  # Strong alignment
                
                    # Pairwise alignment loss
                    align_loss = (cam_norm - range_norm).square().mean()
                    novel_loss = novel_loss + align_loss * 1.0  # Strong alignment
                
                # Idea 7: Shared Trunk Separate SigReg (B only)
                # Like Idea 6, but we DON'T want SigReg on the combined space.
                # We want SigReg(cam) + SigReg(range).
                # NOTE: The default code above calculated `sigreg_loss = sigreg(proj)` where proj is CAT([cam, range]).
                # We need to *subtract* that and replace it with separate terms.
                if run_encoder_ssl_objective and shared_trunk_separate_sigreg and arch == 'B':
                     # 1. Cancel out the "Combined SigReg" (which enforces they live in same manifold)
                     # We do this by modifying lejepa_loss or just tracking the difference?
                     # Ideally we just re-calculate the components.
                     
                     # Get separate projections
                     # proj is (2V, B, D)
                     cam_proj_views = proj[:V]    # (V, B, D)
                     range_proj_views = proj[V:]  # (V, B, D)
                     
                     # Calculate SEPARATE SigReg
                     sigreg_cam = sigreg(cam_proj_views)

                     sigreg_range = sigreg(range_proj_views)
                     
                     # Replace the default sigreg_loss in the total sum
                     # lejepa_loss = sigreg_loss * lamb + inv * (1-lamb) + novel
                     # We subtract the "combined" part and add the "separate" part
                     
                     # New SigReg component
                     new_sigreg_loss = sigreg_cam + sigreg_range
                     
                     # Adjustment term: - (old_sigreg * lambda) + (new_sigreg * lambda)
                     # But we must act on 'lejepa_loss' variable
                     sigreg_diff = (new_sigreg_loss - sigreg_loss) * cfg.lamb
                     novel_loss = novel_loss + sigreg_diff
                     
                     # 2. Pairwise Alignment (Same as Idea 6)
                     # Flatten to (B*V, D)
                     cam_proj_flat = cam_proj_views.transpose(0, 1).reshape(-1, cam_proj_views.shape[-1])
                     range_proj_flat = range_proj_views.transpose(0, 1).reshape(-1, range_proj_views.shape[-1])
                     
                     cam_norm = F.normalize(cam_proj_flat, dim=-1)
                     range_norm = F.normalize(range_proj_flat, dim=-1)
                     align_loss = (cam_norm - range_norm).square().mean()
                     
                     novel_loss = novel_loss + align_loss * 1.0

                # Idea 8: Patch-Level Alignment (B, D, E, F)
                # Instead of only aligning CLS tokens, also align patch embeddings
                # This provides a much richer supervision signal for cross-modal learning
                if run_encoder_ssl_objective and patch_alignment and arch in ('B', 'D', 'E', 'F'):
                    # Get patch embeddings using forward_with_patches
                    if arch == 'D':
                        # For arch D, cam_views is already RGBD (B, V, 4, H, W)
                        # forward_with_patches will create RGB-only and depth-only passes
                        # Handle dict input by extracting global views
                        if isinstance(cam_views, dict):
                            rgbd_tensor = cam_views['global']
                        else:
                            rgbd_tensor = cam_views
                        _, _, (cam_patches, lidar_patches) = net.forward_with_patches(rgbd_tensor)
                    else:
                        # For B, E, F: separate cam_views and modality2
                        if isinstance(cam_views, dict):
                            cam_views_tensor = cam_views['global']
                            lidar_views_tensor = modality2['global']
                        else:
                            cam_views_tensor = cam_views
                            lidar_views_tensor = modality2
                        
                        _, _, (cam_patches, lidar_patches) = net.forward_with_patches(
                            cam_views_tensor, lidar_views_tensor
                        )
                    
                    # cam_patches, lidar_patches: (B*V, N_patches, vit_dim)
                    # Align corresponding patches: patch[i] in cam should align with patch[i] in lidar
                    # (since they see the same spatial location when depth is camera-aligned)
                    
                    # L2 normalize patches
                    cam_patches_norm = F.normalize(cam_patches, dim=-1)  # (B*V, N, D)
                    lidar_patches_norm = F.normalize(lidar_patches, dim=-1)  # (B*V, N, D)
                    
                    # Per-patch MSE alignment (mean over batch, views, patches)
                    patch_align_loss = (cam_patches_norm - lidar_patches_norm).square().mean()
                    
                    # Weight: patches are N times more signals than CLS, so downweight
                    patch_weight = 0.1  # Can be tuned
                    novel_loss = novel_loss + patch_align_loss * patch_weight
                    
                    # NEW: Apply SIGReg on patch embeddings to prevent patch collapse
                    # Without this, patches might collapse to trivial representations
                    if patch_sigreg:
                        # Reshape patches for SIGReg: treat each patch position as a "view"
                        # cam_patches: (B*V, N_patches, D) -> (N_patches, B*V, D)
                        cam_patches_views = cam_patches.transpose(0, 1)  # (N, B*V, D)
                        lidar_patches_views = lidar_patches.transpose(0, 1)  # (N, B*V, D)
                        
                        # Apply SIGReg to each modality's patches
                        patch_sigreg_cam = sigreg(cam_patches_views)
                        patch_sigreg_lidar = sigreg(lidar_patches_views)
                        patch_sigreg_loss = (patch_sigreg_cam + patch_sigreg_lidar) * 0.5
                        
                        novel_loss = novel_loss + patch_sigreg_loss * cfg.lamb
                        if batch_idx % 100 == 0 and use_wandb:
                            wandb.log({
                                'patch_sigreg_cam': patch_sigreg_cam.item(),
                                'patch_sigreg_lidar': patch_sigreg_lidar.item(),
                            })

                    # For E and F: Also add CLS token alignment (direct alignment)
                    # This ensures the CLS tokens are aligned like in direct_alignment scenario
                    if arch in ('E', 'F'):
                        # cam_proj and lidar_proj are already computed above
                        # They are (V, B, proj_dim) shaped
                        cam_proj_flat = cam_proj.transpose(0, 1).reshape(B * V, -1)  # (B*V, proj_dim)
                        lidar_proj_flat = lidar_proj.transpose(0, 1).reshape(B * V, -1)  # (B*V, proj_dim)
                        
                        cam_cls_norm = F.normalize(cam_proj_flat, dim=-1)
                        lidar_cls_norm = F.normalize(lidar_proj_flat, dim=-1)
                        
                        cls_align_loss = (cam_cls_norm - lidar_cls_norm).square().mean()
                        novel_loss = novel_loss + cls_align_loss  # Full weight like direct_alignment

                # single_cls_masked_sigreg (Architecture C): Triple-pass fused training
                # 1. Joint Pass (Done in main loop) -> sigma_joint in lejepa_loss
                # 2. RGB-Only Pass -> SIGReg on shared latent
                # 3. LiDAR-Only Pass -> SIGReg on shared latent
                if run_encoder_ssl_objective and single_cls_masked_sigreg and arch == 'C':
                    # Helper: create a zeroed-out copy (handles dict or tensor)
                    def _zeros_like_views(views):
                        if isinstance(views, dict):
                            return {k: torch.zeros_like(v) for k, v in views.items()}
                        return torch.zeros_like(views)
                    
                    # RGB-Only Pass: zero out LiDAR
                    _, proj_rgb = net(cam_views, _zeros_like_views(modality2))
                    loss_sigreg_rgb = sigreg(proj_rgb)
                    
                    # LiDAR-Only Pass: zero out RGB
                    _, proj_lidar = net(_zeros_like_views(cam_views), modality2)
                    loss_sigreg_lidar = sigreg(proj_lidar)
                    
                    # Combined novel sigreg loss
                    # Weight them by lamb to follow the user's strategy
                    masked_sigreg_loss = (loss_sigreg_rgb + loss_sigreg_lidar) * 0.5
                    novel_loss = novel_loss + masked_sigreg_loss * cfg.lamb
                    
                    if batch_idx % 100 == 0 and use_wandb:
                         wandb.log({
                             'masked_sigreg_rgb': loss_sigreg_rgb.item(),
                             'masked_sigreg_lidar': loss_sigreg_lidar.item(),
                             'masked_sigreg_total': masked_sigreg_loss.item()
                         })
                
                # ============================================
                # PATCH ALIGNMENT + MASKED SIGREG (Scenarios 1 & 2)
                # ============================================
                # Extends single_cls_masked_sigreg: also aligns same-position patch embeddings
                if run_encoder_ssl_objective and (patch_align_masked_sigreg or patch_align_patch_sigreg) and arch == 'C':
                    # Use GLOBAL views only for patch-level losses.
                    # Local crops have different spatial resolution (e.g., 96x96 => 36 patches)
                    # than global crops (224x224 => 196 patches), so concatenating both causes
                    # patch-token shape mismatch.
                    if isinstance(cam_views, dict):
                        cam_views_patch = cam_views['global']
                        modality2_patch = modality2['global']
                    else:
                        cam_views_patch = cam_views
                        modality2_patch = modality2

                    # Get patch embeddings from joint forward pass
                    _, _, (cam_patch_out, range_patch_out) = net.forward_with_patches(
                        cam_views_patch, modality2_patch
                    )
                    
                    # Cosine-normalized MSE alignment: pull same-position patches together
                    cam_patches_norm = F.normalize(cam_patch_out, dim=-1)
                    range_patches_norm = F.normalize(range_patch_out, dim=-1)
                    patch_align_loss = (cam_patches_norm - range_patches_norm).square().mean()
                    
                    novel_loss = novel_loss + patch_align_loss * patch_align_weight
                    
                    # Scenario 2: Also apply SigReg to individual patch embeddings
                    if patch_align_patch_sigreg:
                        # Treat patches as views: (N_patches, B*V, vit_dim)
                        cam_patch_views = cam_patch_out.transpose(0, 1)  # (N_patches, B*V, D)
                        range_patch_views = range_patch_out.transpose(0, 1)
                        
                        # Sample a subset of patches for efficiency (SigReg on 196 patches is expensive)
                        n_sample = min(16, cam_patch_views.shape[0])
                        indices = torch.randperm(cam_patch_views.shape[0])[:n_sample]
                        
                        patch_sigreg_loss = 0
                        for idx in indices:
                            patch_sigreg_loss += sigreg(cam_patch_views[idx:idx+1])
                            patch_sigreg_loss += sigreg(range_patch_views[idx:idx+1])
                        patch_sigreg_loss = patch_sigreg_loss / (2 * n_sample)
                        
                        novel_loss = novel_loss + patch_sigreg_loss * patch_sigreg_weight_new * cfg.lamb
                    
                    if batch_idx % 100 == 0 and use_wandb:
                        log_dict = {
                            'patch_align_loss': patch_align_loss.item(),
                        }
                        if patch_align_patch_sigreg:
                            log_dict['patch_sigreg_loss_new'] = patch_sigreg_loss.item()
                        wandb.log(log_dict)
                
                # ============================================
                # FUSION TOKENS + MASKED SIGREG (Scenario 3)
                # ============================================
                if run_encoder_ssl_objective and fusion_tokens_sigreg and arch == 'C' and not fusion_skip_aux_sigreg:
                    # Joint pass is already computed in the main forward/sigreg_loss path.
                    # Reuse it to avoid an extra high-memory forward pass.
                    loss_sigreg_joint = sigreg_loss
                    current_step = epoch * (3 if debug_model else len(train_loader)) + batch_idx

                    def _fusion_aux_forward(callable_obj, *args, **kwargs):
                        return maybe_chunked_forward(callable_obj, encoder_forward_chunk_size, *args, **kwargs)

                    fusion_patch_align_loss = None
                    if fusion_triplet_alignment:
                        if fusion_tokens_joint is None:
                            _, _, fusion_tokens_joint = _fusion_aux_forward(net.forward_with_fusion_tokens, cam_views, modality2)

                        _, _, fusion_tokens_rgb_only = _fusion_aux_forward(
                            net.forward_with_fusion_tokens,
                            cam_views, zeros_like_tree(modality2)
                        )
                        _, _, fusion_tokens_lidar_only = _fusion_aux_forward(
                            net.forward_with_fusion_tokens,
                            zeros_like_tree(cam_views), modality2
                        )

                        fusion_tokens_joint_norm = F.normalize(fusion_tokens_joint, dim=-1)
                        fusion_tokens_rgb_norm = F.normalize(fusion_tokens_rgb_only, dim=-1)
                        fusion_tokens_lidar_norm = F.normalize(fusion_tokens_lidar_only, dim=-1)

                        # Decompose triplet loss for granular logging
                        fusion_align_j_rgb = (fusion_tokens_joint_norm - fusion_tokens_rgb_norm).square().mean()
                        fusion_align_j_lidar = (fusion_tokens_joint_norm - fusion_tokens_lidar_norm).square().mean()
                        fusion_align_rgb_lidar = (fusion_tokens_rgb_norm - fusion_tokens_lidar_norm).square().mean()

                        fusion_patch_align_loss = (fusion_align_j_rgb + fusion_align_j_lidar + fusion_align_rgb_lidar) / 3.0
                        novel_loss = novel_loss + fusion_patch_align_loss * patch_align_weight
                    
                    if fusion_joint_sigreg_only:
                        loss_sigreg_rgb_ft = torch.tensor(0.0, device=device)
                        loss_sigreg_lidar_ft = torch.tensor(0.0, device=device)
                        fusion_masked_loss = torch.tensor(0.0, device=device)
                        fusion_sigreg_total = loss_sigreg_joint
                        fusion_aux_applied = False
                        fusion_aux_scale = 1.0
                    else:
                        fusion_aux_applied = (current_step % fusion_aux_train_freq == 0)
                        fusion_aux_scale = float(fusion_aux_train_freq) if fusion_aux_applied else 0.0

                        if fusion_aux_applied:
                            # RGB-Only Pass: zero out LiDAR
                            _, proj_rgb_ft = _fusion_aux_forward(
                                net,
                                cam_views, zeros_like_tree(modality2)
                            )
                            loss_sigreg_rgb_ft = sigreg(proj_rgb_ft)

                            # LiDAR-Only Pass: zero out RGB
                            _, proj_lidar_ft = _fusion_aux_forward(
                                net,
                                zeros_like_tree(cam_views), modality2
                            )
                            loss_sigreg_lidar_ft = sigreg(proj_lidar_ft)
                            fusion_masked_loss = (loss_sigreg_rgb_ft + loss_sigreg_lidar_ft) * 0.5
                        else:
                            loss_sigreg_rgb_ft = torch.tensor(0.0, device=device)
                            loss_sigreg_lidar_ft = torch.tensor(0.0, device=device)
                            fusion_masked_loss = torch.tensor(0.0, device=device)

                        # Keep the cheap joint contribution every step and throttle only the
                        # expensive modality-specific auxiliary passes.
                        fusion_sigreg_total = loss_sigreg_joint / 3.0
                        if fusion_aux_applied:
                            fusion_sigreg_total = fusion_sigreg_total + (
                                (loss_sigreg_rgb_ft + loss_sigreg_lidar_ft) / 3.0
                            ) * fusion_aux_scale
                    novel_loss = novel_loss + fusion_sigreg_total * cfg.lamb
                    
                    if batch_idx % 100 == 0 and use_wandb:
                        log_dict = {
                            'fusion_sigreg_joint': loss_sigreg_joint.item(),
                            'fusion_sigreg_rgb': loss_sigreg_rgb_ft.item(),
                            'fusion_sigreg_lidar': loss_sigreg_lidar_ft.item(),
                            'fusion_masked_total': fusion_masked_loss.item(),
                            'fusion_sigreg_total': fusion_sigreg_total.item(),
                            'fusion_aux_applied': float(fusion_aux_applied),
                            'fusion_aux_train_freq': float(fusion_aux_train_freq),
                            'fusion_aux_scale': float(fusion_aux_scale),
                        }
                        if fusion_joint_sigreg_only:
                            log_dict['fusion_joint_sigreg_only'] = 1
                        if fusion_patch_align_loss is not None:
                            log_dict.update({
                                'fusion_patch_align_loss': fusion_patch_align_loss.item(),
                                'fusion_align_joint_rgb': fusion_align_j_rgb.item(),
                                'fusion_align_joint_lidar': fusion_align_j_lidar.item(),
                                'fusion_align_rgb_lidar': fusion_align_rgb_lidar.item(),
                            })
                        wandb.log(log_dict)
                
                # ============================================
                # LIDAR RoPE ON RGB VIT (Scenario 4)
                # ============================================
                if run_encoder_ssl_objective and lidar_rope_rgb and arch == 'C':
                    # Get both depth-conditioned and depth-free embeddings
                    emb_with_depth, emb_without_depth, proj_rope = net.forward_with_depth(
                        cam_views, modality2
                    )
                    
                    # SigReg on depth-conditioned projections
                    loss_sigreg_rope = sigreg(proj_rope)
                    
                    # Contrastive loss: encourage depth-conditioned embeddings to be different
                    # from depth-free ones when real depth is present
                    emb_with_norm = F.normalize(emb_with_depth, dim=-1)
                    emb_without_norm = F.normalize(emb_without_depth, dim=-1)
                    
                    # Cosine similarity should be LOW (embeddings should differ when depth helps)
                    cos_sim = (emb_with_norm * emb_without_norm).sum(dim=-1)  # (B*V,)
                    # Hinge loss: push similarity below 0.5
                    contrastive_loss = F.relu(cos_sim - 0.5).mean()
                    
                    novel_loss = novel_loss + loss_sigreg_rope * cfg.lamb + contrastive_loss * 0.1
                    
                    if batch_idx % 100 == 0 and use_wandb:
                        wandb.log({
                            'rope_sigreg': loss_sigreg_rope.item(),
                            'rope_contrastive': contrastive_loss.item(),
                            'rope_cos_similarity': cos_sim.mean().item(),
                        })

                # Idea 9: Partial Dimension Alignment (E, F, B)
                # RGB has richer information, so we keep part of its embedding space "free"
                # LiDAR uses a reduced embedding dimension, aligned only with part of RGB
                # This allows RGB to encode RGB-specific information in the unaligned dimensions
                # For arch B: Since it's a shared trunk, we slice the depth branch output
                if run_encoder_ssl_objective and partial_dim_alignment and arch in ('E', 'F', 'B'):
                    # Get embeddings based on architecture
                    if arch == 'B':
                        # For arch B: Run forward with patches to get separate embeddings
                        cam_input = cam_views['global'] if isinstance(cam_views, dict) else cam_views
                        range_input = modality2['global'] if isinstance(modality2, dict) else modality2
                        
                        # Forward pass for each modality separately
                        # net is MMEncoderB which has forward_patches method
                        with torch.no_grad():
                            cam_emb_pd = net.forward_single(cam_input, 'cam')  # (B*V, embed_dim)
                            lidar_emb_pd = net.forward_single(range_input, 'range')  # (B*V, embed_dim)
                    elif arch == 'E':
                        (cam_emb_pd, lidar_emb_pd), _ = net(
                            cam_views['global'] if isinstance(cam_views, dict) else cam_views,
                            modality2['global'] if isinstance(modality2, dict) else modality2
                        )
                    else:  # F
                        (cam_emb_pd, lidar_emb_pd), _ = net(
                            cam_views['global'] if isinstance(cam_views, dict) else cam_views,
                            modality2['global'] if isinstance(modality2, dict) else modality2
                        )
                    
                    # cam_emb_pd: (B*V, embed_dim), lidar_emb_pd: (B*V, embed_dim)
                    embed_dim = cam_emb_pd.shape[-1]
                    aligned_dim = int(embed_dim * partial_dim_ratio)  # e.g., 256 out of 512
                    
                    # Split RGB embedding: aligned part + free part
                    cam_aligned = cam_emb_pd[:, :aligned_dim]  # (B*V, aligned_dim)
                    # cam_free = cam_emb_pd[:, aligned_dim:]  # Not used in loss, just conceptually free
                    
                    # Create projection for LiDAR to reduced dimension (lazy init)
                    if not hasattr(net, '_partial_lidar_proj'):
                        net._partial_lidar_proj = nn.Linear(embed_dim, aligned_dim).to(cam_emb_pd.device)
                        # Initialize to extract first aligned_dim dimensions initially
                        with torch.no_grad():
                            net._partial_lidar_proj.weight.copy_(
                                torch.eye(aligned_dim, embed_dim, device=cam_emb_pd.device)
                            )
                            net._partial_lidar_proj.bias.zero_()
                    
                    # Project LiDAR to aligned dimension
                    lidar_aligned = net._partial_lidar_proj(lidar_emb_pd)  # (B*V, aligned_dim)
                    
                    # Align the overlapping dimensions
                    cam_aligned_norm = F.normalize(cam_aligned, dim=-1)
                    lidar_aligned_norm = F.normalize(lidar_aligned, dim=-1)
                    
                    partial_align_loss = (cam_aligned_norm - lidar_aligned_norm).square().mean()
                    novel_loss = novel_loss + partial_align_loss * 0.5
                    
                    # Also add full CLS token alignment (direct alignment style)
                    # This ensures CLS tokens are aligned like in direct_alignment scenario
                    if arch in ('E', 'F'):
                        # cam_proj and lidar_proj are already computed above: (V, B, proj_dim)
                        cam_proj_flat = cam_proj.transpose(0, 1).reshape(B * V, -1)  # (B*V, proj_dim)
                        lidar_proj_flat = lidar_proj.transpose(0, 1).reshape(B * V, -1)  # (B*V, proj_dim)
                        
                        cam_cls_norm = F.normalize(cam_proj_flat, dim=-1)
                        lidar_cls_norm = F.normalize(lidar_proj_flat, dim=-1)
                        
                        cls_align_loss = (cam_cls_norm - lidar_cls_norm).square().mean()
                        novel_loss = novel_loss + cls_align_loss  # Full weight like direct_alignment
                    elif arch == 'B':
                        # For arch B, we already have cam_emb_pd and lidar_emb_pd as CLS embeddings
                        # Just align them directly (they're already the full embeddings)
                        cam_cls_norm = F.normalize(cam_emb_pd, dim=-1)
                        lidar_cls_norm = F.normalize(lidar_emb_pd, dim=-1)
                        
                        cls_align_loss = (cam_cls_norm - lidar_cls_norm).square().mean()
                        novel_loss = novel_loss + cls_align_loss  # Full weight like direct_alignment
                    
                    # NEW: LiDAR SIGReg - Apply SIGReg specifically to LiDAR embeddings
                    # This forces the LiDAR encoder to maximize information in its embedding space
                    if lidar_sigreg:
                        # Apply SIGReg to LiDAR embeddings only (not joint)
                        # lidar_emb_pd is (B*V_global, embed_dim) where V_global is from global views only
                        # We need to infer V_global from the input shape
                        total_samples = lidar_emb_pd.shape[0]
                        V_global = total_samples // B  # Infer V from actual batch
                        lidar_emb_views = lidar_emb_pd.reshape(B, V_global, -1).transpose(0, 1)  # (V, B, D)
                        lidar_sigreg_loss = sigreg(lidar_emb_views)
                        novel_loss = novel_loss + lidar_sigreg_loss * cfg.lamb
                        if batch_idx % 100 == 0 and use_wandb:
                            wandb.log({'lidar_sigreg_loss': lidar_sigreg_loss.item()})
                
                # NEW: Patch + Direct Alignment - Both patch-level AND CLS token alignment
                # Combines spatial patch correspondence with global CLS token alignment
                if run_encoder_ssl_objective and patch_direct_alignment and arch in ('B', 'D'):
                    # Direct alignment on CLS tokens (in addition to patch alignment above)
                    if arch == 'B':
                        # Get CLS embeddings from net
                        cam_input = cam_views['global'] if isinstance(cam_views, dict) else cam_views
                        range_input = modality2['global'] if isinstance(modality2, dict) else modality2
                        
                        # Use forward_single to get CLS embeddings
                        cam_cls = net.forward_single(cam_input, 'cam')  # (B*V, embed_dim)
                        lidar_cls = net.forward_single(range_input, 'range')  # (B*V, embed_dim)
                    else:  # arch D
                        # For RGBD, cam_views already contains depth, use the forward result
                        # proj already computed: (V, B, proj_dim)
                        cam_cls = proj.transpose(0, 1).reshape(B * V, -1)  # Use existing proj
                        lidar_cls = cam_cls  # Same for D (single encoder)
                    
                    # Align CLS tokens
                    if arch == 'B':
                        cam_cls_norm = F.normalize(cam_cls, dim=-1)
                        lidar_cls_norm = F.normalize(lidar_cls, dim=-1)
                        patch_direct_cls_loss = (cam_cls_norm - lidar_cls_norm).square().mean()
                        novel_loss = novel_loss + patch_direct_cls_loss
                        if batch_idx % 100 == 0 and use_wandb:
                            wandb.log({'patch_direct_cls_loss': patch_direct_cls_loss.item()})

                # ============================================
                # NOVEL REGULARIZERS (Beyond SIGReg)
                # GMM, Sinkhorn, Spectral regularization
                # ============================================
                novel_reg_loss = torch.tensor(0.0, device=device)
                
                if run_encoder_ssl_objective and novel_regs:
                    # Get the projection tensor for regularization
                    # Handle different architectures:
                    # - B, C, D, simple: have single 'proj' variable
                    # - A, E, F: have separate cam_proj and lidar_proj
                    
                    if arch in ('A', 'E', 'F'):
                        # Concatenate camera and lidar projections for joint regularization
                        # Both are (V, B, proj_dim) -> cat along batch dim: (V, 2*B, proj_dim)
                        if arch == 'A':
                            # A: lidar_proj might be (1, B, D), expand to match cam_proj views
                            lidar_proj_exp = lidar_proj.expand(cam_proj.shape[0], -1, -1)
                        else:
                            lidar_proj_exp = lidar_proj
                        proj_for_reg = torch.cat([cam_proj, lidar_proj_exp], dim=1)  # (V, 2*B, D)
                    else:
                        # B, C, D, simple: use the existing proj variable
                        proj_for_reg = proj
                    
                    if 'gmm' in novel_regs:
                        gmm_loss = novel_regs['gmm'](proj_for_reg)
                        novel_reg_loss = novel_reg_loss + gmm_loss * novel_reg_weight
                        if batch_idx % 100 == 0:
                            # Log prototype usage statistics
                            with torch.no_grad():
                                assignments = novel_regs['gmm'].get_assignments(proj_for_reg)
                                unique_protos = len(assignments.unique())
                            if use_wandb:
                                wandb.log({
                                    'gmm_loss': gmm_loss.item(),
                                    'gmm_unique_prototypes': unique_protos,
                                })
                    
                    if 'sinkhorn' in novel_regs:
                        sinkhorn_loss = novel_regs['sinkhorn'](proj_for_reg)
                        novel_reg_loss = novel_reg_loss + sinkhorn_loss * novel_reg_weight
                        if batch_idx % 100 == 0 and use_wandb:
                            wandb.log({'sinkhorn_loss': sinkhorn_loss.item()})
                    
                    if 'spectral' in novel_regs:
                        spectral_loss = novel_regs['spectral'](proj_for_reg)
                        novel_reg_loss = novel_reg_loss + spectral_loss * novel_reg_weight
                        if batch_idx % 100 == 0:
                            # Log spectrum statistics
                            with torch.no_grad():
                                spectrum = novel_regs['spectral'].get_spectrum(proj_for_reg)
                                spectrum_ratio = (spectrum[0] / (spectrum[-1] + 1e-6)).item()
                            if use_wandb:
                                wandb.log({
                                    'spectral_loss': spectral_loss.item(),
                                    'spectral_condition_ratio': spectrum_ratio,
                                })
                
                # In "replace_sigreg" mode, novel regularizer replaces SIGReg entirely
                if run_encoder_ssl_objective and replace_sigreg and novel_reg_loss.item() > 0:
                    # Replace sigreg_loss component with novel regularizer
                    # lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
                    # We want: lejepa_loss = novel_reg_loss * lamb + inv_loss * (1 - lamb)
                    sigreg_replacement = (novel_reg_loss - sigreg_loss) * cfg.lamb
                    novel_loss = novel_loss + sigreg_replacement
                else:
                    # Add novel regularizer as additional loss
                    novel_loss = novel_loss + novel_reg_loss

                # Add novel losses to total
                lejepa_loss = lejepa_loss + novel_loss
                
                _encoder_fwd_loss_end = time.perf_counter()
                if _encoder_ssl_forward_flop_counter is not None:
                    _encoder_ssl_forward_flop_counter.__exit__(None, None, None)
                    _encoder_ssl_forward_flops = _encoder_ssl_forward_flop_counter.get_total_flops()
                if track_encoder_compute:
                    encoder_compute_stats["batch_forward_calls_excl_probes"] = encoder_compute_stats["batch_forward_calls"]
                    encoder_compute_stats["total_forward_calls_excl_probes"] += encoder_compute_stats["batch_forward_calls_excl_probes"]
                # Probe losses (using camera embeddings)
                # ============================================
                # PROBE LOSSES (probe_view_mode controls clean vs random global view)
                # ============================================
                
                probe_loss = torch.tensor(0.0, device=device)
                probe_loss_terms = {}
                probe_scene = torch.tensor(0.0, device=device)
                probe_cam = torch.tensor(0.0, device=device)
                probe_loc = torch.tensor(0.0, device=device)
                probe_cars = torch.tensor(0.0, device=device)
                probe_peds = torch.tensor(0.0, device=device)
                probe_objs = torch.tensor(0.0, device=device)
                probe_depth = torch.tensor(0.0, device=device)
                probe_depth_grid = torch.tensor(0.0, device=device)
                probe_grid_occ = torch.tensor(0.0, device=device)
                probe_grid_occ_car = torch.tensor(0.0, device=device)
                probe_grid_occ_ped = torch.tensor(0.0, device=device)
                probe_cross = torch.tensor(0.0, device=device)
                if (not disable_probe_training) and batch_idx % probe_train_freq == 0:
                    emb_probe_c_rgb_only = None
                    emb_probe_c_lidar_only = None
                    emb_probe_l_rgb_only = None
                    emb_probe_l_lidar_only = None

                    # 1. Forward pass for probe view (Linear Probing on clean features)
                    # Use no_grad for detached features (Linear Probing standard)
                    with torch.no_grad():
                        if probe_only_training:
                            if simple_baseline:
                                emb_probe_c = cam_emb
                                emb_probe_l = None
                            elif arch == "D":
                                emb_probe_c = cam_emb
                                emb_probe_l = None
                            elif arch in ("A", "E", "F"):
                                emb_probe_c = cam_emb
                                emb_probe_l = lidar_emb
                            elif arch in ("B", "C"):
                                emb_probe_c = cam_emb
                                emb_probe_l = lidar_emb
                                if use_concat_probe_embeddings:
                                    emb_probe_c = torch.cat([emb_probe_c, emb_probe_l], dim=-1)
                                    emb_probe_l = None
                            else:
                                raise RuntimeError(f"Unsupported probe_only_training architecture: {arch}")
                        else:
                            # Arch D probe: concatenate depth into cam_views_probe
                            _cam_probe = cam_views_probe
                            if arch == 'D' and cam_views_probe is not None and modality2_probe is not None:
                                if isinstance(_cam_probe, dict) and isinstance(modality2_probe, dict):
                                    _cam_probe = {k: torch.cat([_cam_probe[k], modality2_probe[k]], dim=2) if _cam_probe[k].shape[2] < 4 else _cam_probe[k] for k in _cam_probe if _cam_probe[k].numel() > 0}
                                elif isinstance(_cam_probe, torch.Tensor) and isinstance(modality2_probe, torch.Tensor) and _cam_probe.shape[2] < 4:
                                    _cam_probe = torch.cat([_cam_probe, modality2_probe], dim=2)
                            if simple_baseline:
                                probe_in = modality2_probe if simple_modality in ('lidar', 'thermal') else cam_views_probe
                                p_emb, _ = maybe_chunked_forward(net, probe_forward_chunk_size, probe_in)
                                emb_probe_c = p_emb
                                emb_probe_l = None
                            elif arch == "D":
                                p_emb, _ = maybe_chunked_forward(net, probe_forward_chunk_size, _cam_probe)
                                emb_probe_c = p_emb
                                emb_probe_l = None
                            elif arch in ("A", "E", "F"):
                                (p_c, p_l), _ = maybe_chunked_forward(net, probe_forward_chunk_size, cam_views_probe, modality2_probe)
                                emb_probe_c = p_c
                                emb_probe_l = p_l
                            elif arch in ("B", "C"):
                                if fusion_tokens_sigreg and arch == 'C' and hasattr(net, 'forward_with_fusion_tokens'):
                                    p_emb, _, _ = maybe_chunked_forward(
                                        net.forward_with_fusion_tokens,
                                        probe_forward_chunk_size,
                                        cam_views_probe,
                                        modality2_probe,
                                    )
                                    emb_probe_c = p_emb
                                    emb_probe_l = p_emb
                                else:
                                    p_emb, _ = maybe_chunked_forward(net, probe_forward_chunk_size, cam_views_probe, modality2_probe)
                                    # B/C returns [cam; lid] concatenated.
                                    # input (B, 1, ..) -> output (2*B, D)
                                    B_dim = p_emb.shape[0] // 2
                                    emb_probe_c = p_emb[:B_dim]
                                    emb_probe_l = p_emb[B_dim:]

                                # Some baselines probe on the concatenated multimodal embedding.
                                if use_concat_probe_embeddings:
                                    emb_probe_c = torch.cat([emb_probe_c, emb_probe_l], dim=-1)
                                    emb_probe_l = None

                                if probe_train_sensor_drop and fusion_tokens_sigreg and arch == 'C':
                                    p_emb_rgb_only, _, _ = maybe_chunked_forward(
                                        net.forward_with_fusion_tokens,
                                        probe_forward_chunk_size,
                                        cam_views_probe,
                                        zeros_like_tree(modality2_probe),
                                    )
                                    p_emb_lidar_only, _, _ = maybe_chunked_forward(
                                        net.forward_with_fusion_tokens,
                                        probe_forward_chunk_size,
                                        zeros_like_tree(cam_views_probe),
                                        modality2_probe,
                                    )

                                    emb_probe_c_rgb_only = p_emb_rgb_only
                                    emb_probe_l_rgb_only = p_emb_rgb_only
                                    emb_probe_c_lidar_only = p_emb_lidar_only
                                    emb_probe_l_lidar_only = p_emb_lidar_only
                            else:
                                raise RuntimeError(f"Unsupported probe architecture: {arch}")
                            
                    # 2. Compute Probe Losses
                    
                    # Global Labels (Shape B) - No slicing needed
                    y_scene = labels["scene"].to(device)
                    y_cam = labels["camera"].to(device)
                    y_loc = labels["location"].to(device)
                    
                    def _compute_probe_losses(emb_c, emb_l):
                        probe_scene_local = F.cross_entropy(probes["scene"](emb_c), y_scene)
                        probe_cam_local = F.cross_entropy(probes["camera"](emb_c), y_cam)
                        probe_loc_local = F.cross_entropy(probes["location"](emb_c), y_loc)

                        probe_cars_local = F.mse_loss(probes["num_cars"](emb_c).squeeze(-1), y_cars)
                        probe_peds_local = F.mse_loss(probes["num_peds"](emb_c).squeeze(-1), y_peds)
                        probe_objs_local = F.mse_loss(probes["num_objs"](emb_c).squeeze(-1), y_objs)
                        probe_depth_local = F.mse_loss(probes["mean_depth"](emb_c).squeeze(-1), y_depth)

                        pred_depth_grid_local = probes["depth_grid"](emb_c)
                        depth_grid_diff_local = (pred_depth_grid_local - y_depth_grid).square() * y_depth_grid_mask
                        valid_counts_local = y_depth_grid_mask.sum(dim=1).clamp(min=1)
                        probe_depth_grid_local = (depth_grid_diff_local.sum(dim=1) / valid_counts_local).mean()

                        probe_grid_occ_local = F.binary_cross_entropy_with_logits(
                            probes["grid_occupancy"](emb_c), y_grid_occ
                        )
                        probe_grid_occ_car_local = F.binary_cross_entropy_with_logits(
                            probes["grid_occupancy_car"](emb_c), y_grid_occ_car
                        )
                        probe_grid_occ_ped_local = F.binary_cross_entropy_with_logits(
                            probes["grid_occupancy_ped"](emb_c), y_grid_occ_ped
                        )

                        if emb_l is not None:
                            cross_input_local = torch.cat([emb_c, emb_l], dim=-1)
                            cross_pred_local = probes["cross_modal"](cross_input_local).squeeze(-1)
                            cross_target_local = torch.ones_like(cross_pred_local)
                            probe_cross_local = F.binary_cross_entropy_with_logits(cross_pred_local, cross_target_local)
                        else:
                            probe_cross_local = torch.tensor(0.0, device=device)

                        return {
                            "scene": probe_scene_local,
                            "camera": probe_cam_local,
                            "location": probe_loc_local,
                            "num_cars": probe_cars_local,
                            "num_peds": probe_peds_local,
                            "num_objs": probe_objs_local,
                            "mean_depth": probe_depth_local,
                            "depth_grid": probe_depth_grid_local,
                            "grid_occupancy": probe_grid_occ_local,
                            "grid_occupancy_car": probe_grid_occ_car_local,
                            "grid_occupancy_ped": probe_grid_occ_ped_local,
                            "cross_modal": probe_cross_local,
                        }
    
                    def get_probe_label(k):
                        _feature_keys = {
                            "depth_grid", "depth_grid_mask",
                            "depth_grid_hr", "depth_grid_mask_hr",
                            "grid_occupancy", "grid_occupancy_car", "grid_occupancy_ped", "grid_occupancy_hr",
                        }
                        lbl = labels[k]
                        if isinstance(lbl, torch.Tensor):
                            if lbl.dim() == 3:
                                idx = probe_label_view_idx if probe_label_view_idx >= 0 else -1
                                return lbl[:, idx].to(device)
                            if lbl.dim() == 2 and k not in _feature_keys:
                                idx = probe_label_view_idx if probe_label_view_idx >= 0 else -1
                                return lbl[:, idx].to(device)
                        return lbl.to(device)

                    y_cars = get_probe_label("num_cars").float()
                    y_peds = get_probe_label("num_pedestrians").float()
                    y_objs = get_probe_label("num_objects").float()
                    y_depth = get_probe_label("mean_depth").float()
                    
                    # Dense Labels (Shape B, V_total, 64) -> Slice Last
                    y_depth_grid = get_probe_label("depth_grid")
                    y_depth_grid_mask = get_probe_label("depth_grid_mask")
                    y_grid_occ = get_probe_label("grid_occupancy")
                    y_grid_occ_car = get_probe_label("grid_occupancy_car")
                    y_grid_occ_ped = get_probe_label("grid_occupancy_ped")

                    probe_losses_joint = _compute_probe_losses(emb_probe_c, emb_probe_l)
                    if (
                        probe_train_sensor_drop
                        and fusion_tokens_sigreg
                        and arch == 'C'
                        and emb_probe_c_rgb_only is not None
                        and emb_probe_c_lidar_only is not None
                    ):
                        probe_losses_rgb_only = _compute_probe_losses(emb_probe_c_rgb_only, emb_probe_l_rgb_only)
                        probe_losses_lidar_only = _compute_probe_losses(emb_probe_c_lidar_only, emb_probe_l_lidar_only)
                        probe_losses_raw = {
                            k: (probe_losses_joint[k] + probe_losses_rgb_only[k] + probe_losses_lidar_only[k]) / 3.0
                            for k in probe_losses_joint
                        }
                    else:
                        probe_losses_raw = probe_losses_joint

                    probe_scene = probe_losses_raw["scene"]
                    probe_cam = probe_losses_raw["camera"]
                    probe_loc = probe_losses_raw["location"]
                    probe_cars = probe_losses_raw["num_cars"]
                    probe_peds = probe_losses_raw["num_peds"]
                    probe_objs = probe_losses_raw["num_objs"]
                    probe_depth = probe_losses_raw["mean_depth"]
                    probe_depth_grid = probe_losses_raw["depth_grid"]
                    probe_grid_occ = probe_losses_raw["grid_occupancy"]
                    probe_grid_occ_car = probe_losses_raw["grid_occupancy_car"]
                    probe_grid_occ_ped = probe_losses_raw["grid_occupancy_ped"]
                    probe_cross = probe_losses_raw["cross_modal"]
                    
                    probe_loss_terms = {
                        "scene": probe_scene / 3.0,
                        "camera": probe_cam / 3.0,
                        "location": probe_loc / 3.0,
                        "num_cars": (0.01 / 5.0) * probe_cars,
                        "num_peds": (0.01 / 5.0) * probe_peds,
                        "num_objs": (0.01 / 5.0) * probe_objs,
                        "mean_depth": (0.01 / 5.0) * probe_depth,
                        "depth_grid": (0.01 / 5.0) * probe_depth_grid,
                        "grid_occupancy": (0.01 / 3.0) * probe_grid_occ,
                        "grid_occupancy_car": (0.01 / 3.0) * probe_grid_occ_car,
                        "grid_occupancy_ped": (0.01 / 3.0) * probe_grid_occ_ped,
                        "cross_modal": 0.1 * probe_cross,
                    }
                    probe_loss = sum(probe_loss_terms.values())
                
                # ── Patch-based probe losses ─────────────────────────
                patch_probe_loss = torch.tensor(0.0, device=device)
                patch_probe_loss_terms = {}
                if (
                    (not disable_probe_training)
                    and (not legacy_linear_probe_setup)
                    and patch_probes is not None
                    and (batch_idx % probe_train_freq == 0)
                ):
                    with torch.no_grad():
                        if cached_probe_patch_tokens is not None:
                            cam_patch_tokens = cached_probe_patch_tokens
                        else:
                            cam_patch_tokens = extract_patch_tokens(
                                net, cam_views_probe, modality2_probe, arch,
                                simple_baseline=simple_baseline,
                                simple_modality=simple_modality if simple_baseline else 'rgb',
                                batch_chunk_size=probe_forward_chunk_size,
                            )
                    
                    if cam_patch_tokens is not None:
                        if dataset_name == 'flir':
                            patch_probe_loss_terms = {}
                            for flir_domain in flir_probe_label_modes:
                                det_targets_2d = get_flir_detection_targets(labels, device, flir_domain)
                                flir_bbox2d_probe_keys = []
                                for base_probe_key in ("bbox2d_centernet", "bbox2d_slot", "bbox2d"):
                                    probe_key = flir_probe_key(base_probe_key, flir_domain)
                                    if probe_key in patch_probes:
                                        flir_bbox2d_probe_keys.append(probe_key)

                                for probe_key in flir_bbox2d_probe_keys:
                                    bbox2d_pred = patch_probes[probe_key](cam_patch_tokens)
                                    if probe_key.startswith("bbox2d_centernet") or (probe_key == "bbox2d" and patch_bbox2d_probe_type == 'centernet'):
                                        cn_grid = bbox2d_pred['heatmap'].shape[-1]
                                        bbox2d_targets = generate_centernet_targets_2d(
                                            gt_boxes_2d=det_targets_2d['gt_boxes_2d'],
                                            gt_classes=det_targets_2d['gt_classes_2d'],
                                            gt_mask=det_targets_2d['gt_mask_2d'],
                                            grid_h=cn_grid,
                                            grid_w=cn_grid,
                                            num_classes=patch_probes[probe_key].num_classes,
                                            min_box_grid_cells=patch_bbox2d_min_box_grid_cells,
                                        )
                                    else:
                                        bbox2d_targets = det_targets_2d
                                    bbox2d_losses = patch_probes[probe_key].compute_loss(bbox2d_pred, bbox2d_targets)
                                    patch_probe_loss_terms[probe_key] = patch_bbox2d_loss_weight * bbox2d_losses['loss_total']

                                occ_key = flir_probe_key("occupancy_map", flir_domain)
                                if occ_key in patch_probes:
                                    y_occ_targets = get_flir_occupancy_targets(labels, device, flir_domain)
                                    occ_pred = patch_probes[occ_key](cam_patch_tokens)
                                    occ_losses = patch_probes[occ_key].compute_loss(occ_pred, y_occ_targets)
                                    patch_probe_loss_terms[occ_key] = (
                                        patch_occupancy_map_loss_weight * occ_losses['loss_occ']
                                    )

                                box_seg_key = flir_probe_key("box_seg", flir_domain)
                                if box_seg_key in patch_probes and labels.get("has_box_seg_map", None) is not None:
                                    box_seg_targets = get_flir_box_seg_targets(labels, device, flir_domain)
                                    if box_seg_targets["has_box_seg_map"].any():
                                        box_seg_logits = patch_probes[box_seg_key](cam_patch_tokens)
                                        box_seg_losses = patch_probes[box_seg_key].compute_loss(box_seg_logits, box_seg_targets["box_seg_map"])
                                        patch_probe_loss_terms[box_seg_key] = (
                                            patch_box_seg_loss_weight * box_seg_losses['loss_seg']
                                        )
                            patch_probe_loss = sum(patch_probe_loss_terms.values())
                        else:
                            # Detection / segmentation labels (per-sample, not per-view)
                            y_gt_classes = labels["gt_classes"].to(device).long()
                            y_gt_centers = labels["gt_centers"].to(device).float()
                            y_gt_sizes   = labels["gt_sizes"].to(device).float()
                            y_gt_orient  = labels["gt_orientations"].to(device).float()
                            y_gt_mask    = labels["gt_mask"].to(device).float()
                            y_gt_ctr2d   = labels["gt_centers_2d"].to(device).float()
                            y_seg_map    = labels["seg_map"].to(device).long()

                            det_targets = {
                                'gt_classes': y_gt_classes,
                                'gt_centers': y_gt_centers,
                                'gt_sizes': y_gt_sizes,
                                'gt_orientations': y_gt_orient,
                                'gt_mask': y_gt_mask,
                            }

                            if dataset_name == 'waymo':
                                det_targets['gt_classes'], det_targets['gt_mask'] = remap_waymo_patch_gt_classes(
                                    det_targets['gt_classes'], det_targets['gt_mask'], device,
                                )

                            bbox3d_pred = patch_probes["bbox3d"](cam_patch_tokens)
                            bbox3d_losses = patch_probes["bbox3d"].compute_loss(bbox3d_pred, det_targets)

                            # 2. SpatialBBox3D (CenterNet-style) probe
                            spatial_pred = patch_probes["spatial_bbox3d"](cam_patch_tokens)
                            cn_grid = spatial_pred['heatmap'].shape[-1]
                            centernet_targets = generate_centernet_targets(
                                gt_centers_2d=y_gt_ctr2d,
                                gt_classes=det_targets['gt_classes'].float(),
                                gt_sizes=y_gt_sizes,
                                gt_centers_3d=y_gt_centers,
                                gt_orientations=y_gt_orient,
                                gt_mask=det_targets['gt_mask'],
                                num_classes=patch_probes["spatial_bbox3d"].num_classes,
                                grid_h=cn_grid, grid_w=cn_grid,
                            )
                            spatial_losses = patch_probes["spatial_bbox3d"].compute_loss(spatial_pred, centernet_targets)

                            # 3. Semantic segmentation probe (LiDAR-projected labels)
                            # Only compute loss when seg_map has valid labels
                            has_seg_map = labels.get("has_seg_map", None)
                            if has_seg_map is not None and has_seg_map.any():
                                # Compute loss only for samples with seg labels
                                seg_logits = patch_probes["seg"](cam_patch_tokens)
                                seg_losses = patch_probes["seg"].compute_loss(seg_logits, y_seg_map)
                            else:
                                seg_losses = {'loss_seg': torch.tensor(0.0, device=device, requires_grad=True)}

                            # 3b. Panoptic segmentation probe (camera panoptic labels)
                            # Only compute loss when panoptic_seg_map has valid labels
                            has_panoptic = labels.get("has_panoptic_seg_map", None)
                            if has_panoptic is not None and has_panoptic.any() and "panoptic_seg_map" in labels:
                                y_panoptic_seg_map = labels["panoptic_seg_map"].to(device).long()
                                panoptic_seg_logits = patch_probes["panoptic_seg"](cam_patch_tokens)
                                panoptic_seg_losses = patch_probes["panoptic_seg"].compute_loss(panoptic_seg_logits, y_panoptic_seg_map)
                            else:
                                panoptic_seg_losses = {'loss_seg': torch.tensor(0.0, device=device, requires_grad=True)}

                            # 4a. Old patch-token depth probe (MLP + spatial conv → 8×8)
                            B_pt = cam_patch_tokens.shape[0]
                            patch_depth_pred = patch_probes["patch_depth_token"](cam_patch_tokens)  # (B, 196, 1)
                            patch_depth_2d = reshape_patch_scalar_map(patch_depth_pred)
                            patch_depth_2d = patch_depth_2d + patch_probes["patch_depth_spatial"](patch_depth_2d)
                            patch_depth_8x8 = F.adaptive_avg_pool2d(patch_depth_2d, (8, 8)).view(B_pt, 64)
                            y_dg = get_probe_label("depth_grid")
                            y_dg_mask = get_probe_label("depth_grid_mask")
                            pd_diff = (patch_depth_8x8 - y_dg).square() * y_dg_mask
                            pd_valid = y_dg_mask.sum(dim=1).clamp(min=1)
                            old_patch_depth_loss = (pd_diff.sum(dim=1) / pd_valid).mean()

                            # 4b. Dense depth map probe (PixelShuffle, seg-style → 56×56)
                            depth_pred = patch_probes["depth_map"](cam_patch_tokens)
                            y_dg_hr = get_probe_label("depth_grid_hr")      # (B, 3136)
                            y_dg_mask_hr = get_probe_label("depth_grid_mask_hr")  # (B, 3136)
                            y_dg_2d = reshape_flat_spatial_label(y_dg_hr)
                            y_mask_2d = reshape_flat_spatial_label(y_dg_mask_hr)
                            depth_losses = patch_probes["depth_map"].compute_loss(depth_pred, y_dg_2d, y_mask_2d)
                            new_depth_map_loss = depth_losses['loss_depth']

                            if "occupancy_map" in patch_probes and "grid_occupancy_hr" in labels:
                                occ_pred = patch_probes["occupancy_map"](cam_patch_tokens)
                                y_occ_hr = get_probe_label("grid_occupancy_hr")
                                y_occ_2d = reshape_flat_spatial_label(y_occ_hr)
                                occ_losses = patch_probes["occupancy_map"].compute_loss(occ_pred, y_occ_2d)
                            else:
                                occ_losses = {'loss_occ': torch.tensor(0.0, device=device, requires_grad=True)}

                            patch_probe_loss_terms = {
                                "bbox3d": patch_bbox3d_loss_weight * bbox3d_losses['loss_total'],
                                "spatial_bbox3d": patch_spatial_bbox3d_loss_weight * spatial_losses['loss_total'],
                                "seg": patch_seg_loss_weight * seg_losses['loss_seg'],
                                "panoptic_seg": patch_panoptic_seg_loss_weight * panoptic_seg_losses['loss_seg'],
                                "old_patch_depth": patch_old_depth_loss_weight * old_patch_depth_loss,
                                "depth_map": patch_depth_map_loss_weight * new_depth_map_loss,
                                "occupancy_map": patch_occupancy_map_loss_weight * occ_losses['loss_occ'],
                            }
                            patch_probe_loss = sum(patch_probe_loss_terms.values())
                
                loss = lejepa_loss + probe_loss + patch_probe_loss

            _probe_fwd_end = _encoder_fwd_loss_end if disable_probe_training else time.perf_counter()
            # ── Backward/step pass ─────────────────────────────────────
            def _apply_gradient_balance():
                # Idea 3: Gradient Balance (A, B only)
                if not (gradient_balance and arch in ('A', 'B')):
                    return
                if arch == 'A':
                    cam_params = list(net.cam_encoder.parameters())
                    lidar_params = list(net.lidar_encoder.parameters())
                else:  # B - shared but we can balance patch embed gradients
                    cam_params = [net.patch_embed_cam.weight] if hasattr(net, 'patch_embed_cam') else []
                    lidar_params = [net.patch_embed_lidar.weight] if hasattr(net, 'patch_embed_lidar') else []

                if cam_params and lidar_params:
                    cam_grad_norm = sum(p.grad.norm() for p in cam_params if p.grad is not None) / max(len(cam_params), 1)
                    lidar_grad_norm = sum(p.grad.norm() for p in lidar_params if p.grad is not None) / max(len(lidar_params), 1)
                    if lidar_grad_norm > 0:
                        scale_factor = (cam_grad_norm / lidar_grad_norm).clamp(0.1, 10.0)
                        for p in lidar_params:
                            if p.grad is not None:
                                p.grad *= scale_factor

            probe_step_stats = _new_health_stats()
            probe_lane_step_stats = {}
            patch_step_stats = _new_health_stats()
            patch_lane_step_stats = {}

            if legacy_linear_probe_setup:
                # ── LEGACY: Single backward + single optimizer step ────
                # Replicates pre-Feb-5 behavior: one backward on combined
                # loss, one scaler, one scheduler for encoder + probes.
                combined_loss = lejepa_loss + probe_loss  # no patch_probe_loss in legacy
                combined_step_stats = _run_optimizer_step(
                    combined_loss,
                    opt_combined_legacy,
                    scheduler_combined_legacy,
                    scaler_combined_legacy,
                    encoder_params + probe_params_legacy,
                    post_backward_fn=_apply_gradient_balance,
                )
                encoder_step_stats = combined_step_stats
                _accumulate_health(probe_step_stats, combined_step_stats)
                _encoder_bwd_end = time.perf_counter()
                _probe_bwd_end = _encoder_bwd_end  # Can't separate in legacy mode
            else:
                # ── MODERN: Independent optimizer lanes ────────────────
                if dino_encoder_schedule is not None:
                    encoder_schedule_step = epoch * (3 if debug_model else len(train_loader)) + batch_idx
                    _apply_dino_encoder_schedule(encoder_schedule_step)
                if _run_encoder_ssl_flop_profile:
                    _encoder_ssl_backward_flop_counter = FlopCounterMode(display=False)
                    _encoder_ssl_backward_flop_counter.__enter__()
                encoder_step_stats = _run_optimizer_step(
                    lejepa_loss,
                    opt_encoder,
                    scheduler_encoder,
                    scaler_encoder,
                    encoder_params,
                    post_backward_fn=_apply_gradient_balance,
                    max_grad_norm=dino_grad_clip_norm if use_fusion_dino else None,
                )
                _encoder_bwd_end = time.perf_counter()
                if _encoder_ssl_backward_flop_counter is not None:
                    _encoder_ssl_backward_flop_counter.__exit__(None, None, None)
                    _encoder_ssl_backward_flops = _encoder_ssl_backward_flop_counter.get_total_flops()
                    timing_stats["encoder_ssl_flops_profiled_step"] = (
                        (_encoder_ssl_forward_flops or 0.0) + _encoder_ssl_backward_flops
                    )
                    if batch_idx == 0:
                        print(
                            f"📏 Encoder SSL FLOPs/profiled step: "
                            f"{timing_stats['encoder_ssl_flops_profiled_step']/1e9:.3f} GFLOP"
                        )
                for lane_name, lane_loss in probe_loss_terms.items():
                    lane_stats = _run_optimizer_step(
                        lane_loss,
                        probe_opts.get(lane_name),
                        probe_schedulers.get(lane_name),
                        probe_scalers.get(lane_name),
                        probe_params_by_lane.get(lane_name, []),
                    )
                    probe_lane_step_stats[lane_name] = lane_stats
                    _accumulate_health(probe_step_stats, lane_stats)
                    _accumulate_health(optimizer_lane_totals[f"probe/{lane_name}"], lane_stats)

                for lane_name, lane_loss in patch_probe_loss_terms.items():
                    lane_stats = _run_optimizer_step(
                        lane_loss,
                        patch_opts.get(lane_name),
                        patch_schedulers.get(lane_name),
                        patch_scalers.get(lane_name),
                        patch_params_by_lane.get(lane_name, []),
                    )
                    patch_lane_step_stats[lane_name] = lane_stats
                    _accumulate_health(patch_step_stats, lane_stats)
                    if f"patch/{lane_name}" not in optimizer_lane_totals:
                        optimizer_lane_totals[f"patch/{lane_name}"] = _new_health_stats()
                    _accumulate_health(optimizer_lane_totals[f"patch/{lane_name}"], lane_stats)
                _probe_bwd_end = time.perf_counter()

            _accumulate_health(optimizer_health_totals["encoder"], encoder_step_stats)
            _accumulate_health(optimizer_health_totals["probe"], probe_step_stats)
            _accumulate_health(optimizer_health_totals["patch"], patch_step_stats)

            if teacher_net is not None and encoder_step_stats["did_step"]:
                steps_per_epoch = 3 if debug_model else len(train_loader)
                global_step = epoch * steps_per_epoch + batch_idx
                dino_momentum_now = _dino_teacher_momentum(global_step, total_steps)
                _update_ema_teacher(net, teacher_net, dino_momentum_now)
                if dino_student_head is not None and dino_teacher_head is not None:
                    _update_ema_teacher(dino_student_head, dino_teacher_head, dino_momentum_now)
                if ibot_student_head is not None and ibot_teacher_head is not None:
                    _update_ema_teacher(ibot_student_head, ibot_teacher_head, dino_momentum_now)

            epoch_losses["total"] += loss.item()
            epoch_losses["sigreg"] += sigreg_loss.item()
            epoch_losses["inv"] += inv_loss.item()
            epoch_losses["novel_reg"] += novel_reg_loss.item() if isinstance(novel_reg_loss, torch.Tensor) else novel_reg_loss
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sigreg": f"{sigreg_loss.item():.4f}",
                "nreg": f"{novel_reg_loss.item() if isinstance(novel_reg_loss, torch.Tensor) else 0:.4f}" if novel_regs else ""
            })

            encoder_calls_batch = encoder_compute_stats["batch_forward_calls_excl_probes"] if track_encoder_compute else 0
            encoder_macs_batch = None
            encoder_flops_batch = None
            if track_encoder_compute and encoder_compute_stats["macs_per_call"] is not None:
                encoder_macs_batch = encoder_compute_stats["macs_per_call"] * encoder_calls_batch
                encoder_flops_batch = encoder_compute_stats["flops_per_call"] * encoder_calls_batch
                encoder_compute_stats["macs_total"] += encoder_macs_batch
                encoder_compute_stats["flops_total"] += encoder_flops_batch

            # ── Timing & encoder-SSL FLOP accumulation ────────────────
            # This is model-side wall time excluding probe work and excluding
            # dataloader wait/validation. It is intentionally named narrowly.
            _batch_encoder_step_wall_time_excl_probes = None
            _batch_probe_step_wall_time = None
            if not legacy_linear_probe_setup:
                _batch_encoder_step_wall_time_excl_probes = (
                    (_encoder_fwd_loss_end - _batch_start_time) + (_encoder_bwd_end - _probe_fwd_end)
                )
                _batch_probe_step_wall_time = (
                    (_probe_fwd_end - _encoder_fwd_loss_end) + (_probe_bwd_end - _encoder_bwd_end)
                )
                timing_stats["encoder_step_wall_time_excl_probes_total"] += _batch_encoder_step_wall_time_excl_probes
                timing_stats["probe_step_wall_time_total"] += _batch_probe_step_wall_time

            if timing_stats["encoder_ssl_flops_profiled_step"] is not None and not legacy_linear_probe_setup:
                timing_stats["encoder_ssl_flops_total_estimated"] += timing_stats["encoder_ssl_flops_profiled_step"]

            batch_peak_allocated_bytes = None
            batch_peak_reserved_bytes = None
            if gpu_memory_stats["available"]:
                torch.cuda.synchronize()
                batch_peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
                batch_peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
                gpu_memory_stats["batch_peak_allocated_bytes_max"] = max(
                    gpu_memory_stats["batch_peak_allocated_bytes_max"],
                    batch_peak_allocated_bytes,
                )
                gpu_memory_stats["batch_peak_reserved_bytes_max"] = max(
                    gpu_memory_stats["batch_peak_reserved_bytes_max"],
                    batch_peak_reserved_bytes,
                )

            if estimate_encoder_train_compute:
                if (
                    timing_stats["encoder_ssl_flops_profiled_step"] is not None
                    and estimate_stats["profiled_step_flops"] is None
                ):
                    estimate_stats["profiled_step_flops"] = timing_stats["encoder_ssl_flops_profiled_step"]

                if batch_idx < estimate_warmup_batches:
                    estimate_stats["warmup_batches_completed"] += 1
                else:
                    estimate_stats["measured_batches"] += 1
                    if _batch_encoder_step_wall_time_excl_probes is not None:
                        estimate_stats["measured_encoder_step_time_total_sec"] += _batch_encoder_step_wall_time_excl_probes
                    estimate_stats["measured_encoder_forward_calls_total"] += encoder_calls_batch
                    if encoder_macs_batch is not None:
                        estimate_stats["measured_encoder_macs_total"] += encoder_macs_batch
                    if encoder_flops_batch is not None:
                        estimate_stats["measured_encoder_forward_flops_total"] += encoder_flops_batch
                    if batch_peak_allocated_bytes is not None:
                        estimate_stats["measured_peak_allocated_bytes"].append(batch_peak_allocated_bytes)
                        estimate_stats["peak_allocated_bytes_max"] = max(
                            estimate_stats.get("peak_allocated_bytes_max", 0),
                            batch_peak_allocated_bytes,
                        )
                    if batch_peak_reserved_bytes is not None:
                        estimate_stats["measured_peak_reserved_bytes"].append(batch_peak_reserved_bytes)
                        estimate_stats["peak_reserved_bytes_max"] = max(
                            estimate_stats.get("peak_reserved_bytes_max", 0),
                            batch_peak_reserved_bytes,
                        )

                estimate_complete = estimate_stats["measured_batches"] >= estimate_measure_batches

            if use_wandb:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/inv": inv_loss.item(),
                    "train/probe_scene": probe_scene.item(),
                    "train/probe_camera": probe_cam.item(),
                    "train/probe_location": probe_loc.item(),
                    "train/probe_cars": probe_cars.item(),
                    "train/probe_peds": probe_peds.item(),
                    "train/probe_objs": probe_objs.item(),
                    "train/probe_depth": probe_depth.item(),
                    "train/probe_cross_modal": probe_cross.item(),
                    "train/lr_encoder": _current_lr(scheduler_combined_legacy, opt_combined_legacy) if legacy_linear_probe_setup else _current_lr(scheduler_encoder, opt_encoder),
                    "train/lr_encoder_trunk": 0.0 if legacy_linear_probe_setup else _group_lr(opt_encoder, 'pretrained_trunk'),
                    "train/lr_encoder_aux": 0.0 if legacy_linear_probe_setup else _group_lr(opt_encoder, 'encoder_aux'),
                    "train/lr_probe": _current_lr(scheduler_combined_legacy, opt_combined_legacy) if legacy_linear_probe_setup else _mean_lr(probe_schedulers, probe_opts),
                    "train/lr_patch": 0.0 if legacy_linear_probe_setup else _mean_lr(patch_schedulers, patch_opts),
                    "train/opt_step_encoder": encoder_step_stats["did_step"],
                    "train/opt_step_probe": probe_step_stats["steps"],
                    "train/opt_step_patch": patch_step_stats["steps"],
                    "train/overflow_encoder": encoder_step_stats["overflow"],
                    "train/overflow_probe": probe_step_stats["overflow"],
                    "train/overflow_patch": patch_step_stats["overflow"],
                    "train/nonfinite_loss_encoder": encoder_step_stats["nonfinite_loss"],
                    "train/nonfinite_loss_probe": probe_step_stats["nonfinite_loss"],
                    "train/nonfinite_loss_patch": patch_step_stats["nonfinite_loss"],
                    "train/invalid_grad_tensors_encoder": encoder_step_stats["invalid_grad_tensors"],
                    "train/invalid_grad_tensors_probe": probe_step_stats["invalid_grad_tensors"],
                    "train/invalid_grad_tensors_patch": patch_step_stats["invalid_grad_tensors"],
                    "train/invalid_grad_values_encoder": encoder_step_stats["invalid_grad_values"],
                    "train/invalid_grad_values_probe": probe_step_stats["invalid_grad_values"],
                    "train/invalid_grad_values_patch": patch_step_stats["invalid_grad_values"],
                    "train/overflow_total_encoder": optimizer_health_totals["encoder"]["overflow"],
                    "train/overflow_total_probe": optimizer_health_totals["probe"]["overflow"],
                    "train/overflow_total_patch": optimizer_health_totals["patch"]["overflow"],
                    "train/nonfinite_loss_total_encoder": optimizer_health_totals["encoder"]["nonfinite_loss"],
                    "train/nonfinite_loss_total_probe": optimizer_health_totals["probe"]["nonfinite_loss"],
                    "train/nonfinite_loss_total_patch": optimizer_health_totals["patch"]["nonfinite_loss"],
                    "train/invalid_grad_total_tensors_encoder": optimizer_health_totals["encoder"]["invalid_grad_tensors"],
                    "train/invalid_grad_total_tensors_probe": optimizer_health_totals["probe"]["invalid_grad_tensors"],
                    "train/invalid_grad_total_tensors_patch": optimizer_health_totals["patch"]["invalid_grad_tensors"],
                    "train/invalid_grad_total_values_encoder": optimizer_health_totals["encoder"]["invalid_grad_values"],
                    "train/invalid_grad_total_values_probe": optimizer_health_totals["probe"]["invalid_grad_values"],
                    "train/invalid_grad_total_values_patch": optimizer_health_totals["patch"]["invalid_grad_values"],
                    "train/patch_probes_enabled": 1 if patch_probes is not None else 0,
                    "train/imagebind_memory_safe_mode": 1 if use_imagebind else 0,
                }

                if train_rgb_lidar_patch_mse is not None:
                    log_dict["train/rgb_lidar_patch_mse"] = train_rgb_lidar_patch_mse.item()

                if track_encoder_compute:
                    log_dict["train/encoder_forward_calls"] = encoder_calls_batch
                    log_dict["train/encoder_forward_calls_total"] = encoder_compute_stats["total_forward_calls_excl_probes"]
                    log_dict["train/model_forward_calls_total_including_probes"] = encoder_compute_stats["total_forward_calls"]
                    log_dict["train/encoder_compute_profile_attempted"] = 1 if encoder_compute_stats["profile_attempted"] else 0
                    log_dict["train/encoder_compute_profile_success"] = 1 if encoder_compute_stats["profile_success"] else 0
                    if encoder_macs_batch is not None and encoder_flops_batch is not None:
                        log_dict["train/encoder_macs_batch"] = encoder_macs_batch
                        log_dict["train/encoder_flops_batch"] = encoder_flops_batch
                        log_dict["train/encoder_macs_total"] = encoder_compute_stats["macs_total"]
                        log_dict["train/encoder_flops_total"] = encoder_compute_stats["flops_total"]
                        log_dict["train/encoder_macs_per_call"] = encoder_compute_stats["macs_per_call"]
                        log_dict["train/encoder_flops_per_call"] = encoder_compute_stats["flops_per_call"]
                    else:
                        log_dict["train/encoder_macs_batch"] = 0.0
                        log_dict["train/encoder_flops_batch"] = 0.0
                        log_dict["train/encoder_macs_total"] = 0.0
                        log_dict["train/encoder_flops_total"] = 0.0
                        log_dict["train/encoder_macs_per_call"] = 0.0
                        log_dict["train/encoder_flops_per_call"] = 0.0

                if _batch_encoder_step_wall_time_excl_probes is not None:
                    log_dict["train/encoder_step_wall_time_excl_probes"] = _batch_encoder_step_wall_time_excl_probes
                    log_dict["train/encoder_step_wall_time_excl_probes_total"] = timing_stats["encoder_step_wall_time_excl_probes_total"]
                    log_dict["train/probe_step_wall_time"] = _batch_probe_step_wall_time
                    log_dict["train/probe_step_wall_time_total"] = timing_stats["probe_step_wall_time_total"]
                if timing_stats["encoder_ssl_flops_profiled_step"] is not None:
                    log_dict["train/encoder_ssl_flops_profiled_step"] = timing_stats["encoder_ssl_flops_profiled_step"]
                    log_dict["train/encoder_ssl_flops_total_estimated"] = timing_stats["encoder_ssl_flops_total_estimated"]
                if batch_peak_allocated_bytes is not None and batch_peak_reserved_bytes is not None:
                    log_dict["train/gpu_peak_vram_allocated_bytes"] = batch_peak_allocated_bytes
                    log_dict["train/gpu_peak_vram_reserved_bytes"] = batch_peak_reserved_bytes
                    log_dict["train/gpu_peak_vram_allocated_gb"] = batch_peak_allocated_bytes / (1024 ** 3)
                    log_dict["train/gpu_peak_vram_reserved_gb"] = batch_peak_reserved_bytes / (1024 ** 3)
                    log_dict["train/gpu_peak_vram_allocated_bytes_max"] = gpu_memory_stats["batch_peak_allocated_bytes_max"]
                    log_dict["train/gpu_peak_vram_reserved_bytes_max"] = gpu_memory_stats["batch_peak_reserved_bytes_max"]
                    log_dict["train/gpu_peak_vram_allocated_gb_max"] = gpu_memory_stats["batch_peak_allocated_bytes_max"] / (1024 ** 3)
                    log_dict["train/gpu_peak_vram_reserved_gb_max"] = gpu_memory_stats["batch_peak_reserved_bytes_max"] / (1024 ** 3)

                for lane_name, lane_stats in probe_lane_step_stats.items():
                    log_dict[f"train/probe_lane_step/{lane_name}"] = lane_stats["did_step"]
                    log_dict[f"train/probe_lane_overflow/{lane_name}"] = lane_stats["overflow"]
                    log_dict[f"train/probe_lane_invalid_grad_tensors/{lane_name}"] = lane_stats["invalid_grad_tensors"]
                    log_dict[f"train/probe_lane_overflow_total/{lane_name}"] = optimizer_lane_totals[f"probe/{lane_name}"]["overflow"]

                for lane_name, lane_stats in patch_lane_step_stats.items():
                    log_dict[f"train/patch_lane_step/{lane_name}"] = lane_stats["did_step"]
                    log_dict[f"train/patch_lane_overflow/{lane_name}"] = lane_stats["overflow"]
                    log_dict[f"train/patch_lane_invalid_grad_tensors/{lane_name}"] = lane_stats["invalid_grad_tensors"]
                    log_dict[f"train/patch_lane_overflow_total/{lane_name}"] = optimizer_lane_totals[f"patch/{lane_name}"]["overflow"]

                # Log per-lane patch probe losses for monitoring
                for lane_name, lane_loss in patch_probe_loss_terms.items():
                    if isinstance(lane_loss, torch.Tensor) and lane_loss.grad_fn is not None:
                        log_dict[f"train/patch_loss/{lane_name}"] = lane_loss.item()

                if legacy_linear_probe_setup and opt_combined_legacy is not None:
                    log_dict["train/lr"] = _current_lr(scheduler_combined_legacy, opt_combined_legacy)
                elif opt_encoder is not None:
                    log_dict["train/lr"] = _current_lr(scheduler_encoder, opt_encoder)
                elif any(opt_obj is not None for opt_obj in probe_opts.values()):
                    log_dict["train/lr"] = _mean_lr(probe_schedulers, probe_opts)
                elif any(opt_obj is not None for opt_obj in patch_opts.values()):
                    log_dict["train/lr"] = _mean_lr(patch_schedulers, patch_opts)
                # Log novel losses if enabled
                if cross_modal_sigreg or modality_invariance or separate_projections:
                    log_dict["train/novel_loss"] = novel_loss.item()
                
                # Log novel regularizer losses
                if novel_regs:
                    log_dict["train/novel_reg_loss"] = novel_reg_loss.item() if isinstance(novel_reg_loss, torch.Tensor) else novel_reg_loss
                    if 'gmm' in novel_regs:
                        log_dict["train/gmm_reg"] = 1  # Flag that GMM is active
                    if 'sinkhorn' in novel_regs:
                        log_dict["train/sinkhorn_reg"] = 1
                    if 'spectral' in novel_regs:
                        log_dict["train/spectral_reg"] = 1

                if patch_probes is None:
                    log_dict.update({
                        "train/patch_probe_metrics_skipped": 1,
                        "train/patch_lane_step/bbox3d": 0,
                        "train/patch_lane_step/bbox2d": 0,
                        "train/patch_lane_step/spatial_bbox3d": 0,
                        "train/patch_lane_step/seg": 0,
                        "train/patch_lane_step/box_seg": 0,
                        "train/patch_lane_step/panoptic_seg": 0,
                        "train/patch_lane_step/old_patch_depth": 0,
                        "train/patch_lane_step/depth_map": 0,
                    })

                if encoder_only_mode or estimate_encoder_train_compute:
                    global_step = epoch * (3 if debug_model else len(train_loader)) + batch_idx
                    encoder_batch_size = batch_size_of_tree(cam_views)
                    log_dict.update({
                        "encoder_only/active": 1 if encoder_only_mode else 0,
                        "encoder_only/estimate_active": 1 if estimate_encoder_train_compute else 0,
                        "encoder_only/global_step": global_step,
                        "encoder_only/epoch": epoch,
                        "encoder_only/batch_idx": batch_idx,
                        "encoder_only/batch_size": encoder_batch_size,
                        "encoder_only/loss": lejepa_loss.item(),
                        "encoder_only/sigreg": sigreg_loss.item(),
                        "encoder_only/inv": inv_loss.item(),
                        "encoder_only/novel_loss": novel_loss.item() if isinstance(novel_loss, torch.Tensor) else float(novel_loss),
                        "encoder_only/probes_disabled": 1 if disable_probe_training else 0,
                        "encoder_only/patch_probes_enabled": 1 if patch_probes is not None else 0,
                        "encoder_only/encoder_forward_calls": encoder_calls_batch,
                    })
                    if _batch_encoder_step_wall_time_excl_probes is not None and _batch_encoder_step_wall_time_excl_probes > 0:
                        log_dict["encoder_only/step_wall_time_sec"] = _batch_encoder_step_wall_time_excl_probes
                        log_dict["encoder_only/samples_per_sec"] = encoder_batch_size / _batch_encoder_step_wall_time_excl_probes
                        log_dict["encoder_only/steps_per_sec"] = 1.0 / _batch_encoder_step_wall_time_excl_probes
                    if batch_peak_allocated_bytes is not None:
                        log_dict["encoder_only/gpu_peak_vram_allocated_gb"] = batch_peak_allocated_bytes / (1024 ** 3)
                    if batch_peak_reserved_bytes is not None:
                        log_dict["encoder_only/gpu_peak_vram_reserved_gb"] = batch_peak_reserved_bytes / (1024 ** 3)
                    if track_encoder_compute and encoder_macs_batch is not None and encoder_flops_batch is not None:
                        log_dict["encoder_only/macs_batch"] = encoder_macs_batch
                        log_dict["encoder_only/flops_batch"] = encoder_flops_batch
                    if estimate_encoder_train_compute and estimate_stats is not None:
                        measured_progress = min(estimate_stats["measured_batches"] / max(estimate_measure_batches, 1), 1.0)
                        log_dict["encoder_only/estimate_measured_batches"] = estimate_stats["measured_batches"]
                        log_dict["encoder_only/estimate_progress"] = measured_progress
                
                wandb.log(log_dict)

            if estimate_complete:
                print(
                    "📐 Encoder compute estimate window complete: "
                    f"warmup={estimate_stats['warmup_batches_completed']}, "
                    f"measured={estimate_stats['measured_batches']}"
                )
                break

            # Periodic Validation (skip in debug_model mode)
            if val_freq > 0 and (batch_idx + 1) % val_freq == 0 and not debug_model:
                # 1. Official Validation (limited batches)
                validate(net, probes, test_loader, device, epoch, cfg, run_name, simple_baseline, simple_modality, debug=False, log_prefix="val", limit_batches=val_batches_limit, total_epochs=epochs, patch_probes=patch_probes, det_metrics=det_metrics, det_metrics_centernet=det_metrics_centernet, det_metrics_2d=det_metrics_2d, seg_metrics=seg_metrics, box_seg_metrics=box_seg_metrics, panoptic_seg_metrics=panoptic_seg_metrics, modality_eval="both")
                if not simple_baseline:
                    if probe_eval_rgb_only:
                        validate(net, probes, test_loader, device, epoch, cfg, run_name, simple_baseline, simple_modality, debug=False, log_prefix="rgb_only", limit_batches=val_batches_limit, total_epochs=epochs, patch_probes=patch_probes, det_metrics=det_metrics, det_metrics_centernet=det_metrics_centernet, det_metrics_2d=det_metrics_2d, seg_metrics=seg_metrics, box_seg_metrics=box_seg_metrics, panoptic_seg_metrics=panoptic_seg_metrics, modality_eval="rgb_only")
                    if probe_eval_lidar_only:
                        validate(net, probes, test_loader, device, epoch, cfg, run_name, simple_baseline, simple_modality, debug=False, log_prefix="lidar_only", limit_batches=val_batches_limit, total_epochs=epochs, patch_probes=patch_probes, det_metrics=det_metrics, det_metrics_centernet=det_metrics_centernet, det_metrics_2d=det_metrics_2d, seg_metrics=seg_metrics, box_seg_metrics=box_seg_metrics, panoptic_seg_metrics=panoptic_seg_metrics, modality_eval="lidar_only")

                # Legacy test/ validation removed — no longer needed

                net.train()
                probes.train()
                if patch_probes is not None:
                    patch_probes.train()
        

        if estimate_encoder_train_compute:
            if estimate_complete:
                break
            continue

        # Standard validation at end of epoch (Full, no limits) - skip in debug_model and encoder_only modes
        if not debug_model and not encoder_only_mode:
            acc_scene = validate(net, probes, test_loader, device, epoch, cfg, run_name, simple_baseline, simple_modality, debug=debug, log_prefix="val", total_epochs=epochs, patch_probes=patch_probes, det_metrics=det_metrics, det_metrics_centernet=det_metrics_centernet, det_metrics_2d=det_metrics_2d, seg_metrics=seg_metrics, box_seg_metrics=box_seg_metrics, panoptic_seg_metrics=panoptic_seg_metrics, modality_eval="both")
            if not simple_baseline:
                if probe_eval_rgb_only:
                    validate(net, probes, test_loader, device, epoch, cfg, run_name, simple_baseline, simple_modality, debug=debug, log_prefix="rgb_only", total_epochs=epochs, patch_probes=patch_probes, det_metrics=det_metrics, det_metrics_centernet=det_metrics_centernet, det_metrics_2d=det_metrics_2d, seg_metrics=seg_metrics, box_seg_metrics=box_seg_metrics, panoptic_seg_metrics=panoptic_seg_metrics, modality_eval="rgb_only")
                if probe_eval_lidar_only:
                    validate(net, probes, test_loader, device, epoch, cfg, run_name, simple_baseline, simple_modality, debug=debug, log_prefix="lidar_only", total_epochs=epochs, patch_probes=patch_probes, det_metrics=det_metrics, det_metrics_centernet=det_metrics_centernet, det_metrics_2d=det_metrics_2d, seg_metrics=seg_metrics, box_seg_metrics=box_seg_metrics, panoptic_seg_metrics=panoptic_seg_metrics, modality_eval="lidar_only")
            
            # Legacy test/ validation removed — no longer needed
        else:
            if encoder_only_mode and not debug_model:
                print("⚡ Skipping end-of-epoch validation in encoder-only mode")
            else:
                print("🚀 Skipping validation in debug_model mode")
            acc_scene = 0.0

    if estimate_encoder_train_compute:
        if estimate_stats is None or estimate_stats["measured_batches"] <= 0:
            raise RuntimeError("Encoder compute estimate mode finished without any measured batches")

        avg_encoder_step_wall_time = (
            estimate_stats["measured_encoder_step_time_total_sec"] / estimate_stats["measured_batches"]
        )
        avg_encoder_forward_calls = (
            estimate_stats["measured_encoder_forward_calls_total"] / estimate_stats["measured_batches"]
        )
        avg_encoder_macs_batch = (
            estimate_stats["measured_encoder_macs_total"] / estimate_stats["measured_batches"]
            if estimate_stats["measured_encoder_macs_total"] > 0
            else 0.0
        )
        avg_encoder_forward_flops_batch = (
            estimate_stats["measured_encoder_forward_flops_total"] / estimate_stats["measured_batches"]
            if estimate_stats["measured_encoder_forward_flops_total"] > 0
            else 0.0
        )
        estimated_total_encoder_train_time_sec = avg_encoder_step_wall_time * total_target_train_steps
        profiled_step_flops = estimate_stats["profiled_step_flops"]
        estimated_total_encoder_ssl_flops = (
            profiled_step_flops * total_target_train_steps
            if profiled_step_flops is not None
            else 0.0
        )
        allocated_peaks = estimate_stats.get("measured_peak_allocated_bytes", [])
        reserved_peaks = estimate_stats.get("measured_peak_reserved_bytes", [])

        def _summarize_memory_samples(samples):
            if not samples:
                return {}
            samples_np = np.asarray(samples, dtype=np.float64)
            return {
                "avg": float(samples_np.mean()),
                "median": float(np.median(samples_np)),
                "p90": float(np.percentile(samples_np, 90)),
                "min": float(samples_np.min()),
                "max": float(samples_np.max()),
            }

        allocated_summary = _summarize_memory_samples(allocated_peaks)
        reserved_summary = _summarize_memory_samples(reserved_peaks)

        estimate_summary = {
            "estimate/active": 1,
            "estimate/warmup_batches": estimate_warmup_batches,
            "estimate/measured_batches": estimate_stats["measured_batches"],
            "estimate/train_steps_per_epoch": train_steps_per_epoch,
            "estimate/target_epochs": int(epochs),
            "estimate/target_train_steps": total_target_train_steps,
            "estimate/avg_encoder_step_wall_time_excl_probes_sec": avg_encoder_step_wall_time,
            "estimate/estimated_total_encoder_train_time_sec": estimated_total_encoder_train_time_sec,
            "estimate/estimated_total_encoder_train_time_min": estimated_total_encoder_train_time_sec / 60.0,
            "estimate/avg_encoder_forward_calls_per_step": avg_encoder_forward_calls,
            "estimate/avg_encoder_macs_batch": avg_encoder_macs_batch,
            "estimate/avg_encoder_forward_flops_batch": avg_encoder_forward_flops_batch,
            "estimate/estimated_total_encoder_macs": avg_encoder_macs_batch * total_target_train_steps,
            "estimate/estimated_total_encoder_forward_flops": avg_encoder_forward_flops_batch * total_target_train_steps,
            "estimate/profiled_encoder_ssl_flops_step": profiled_step_flops or 0.0,
            "estimate/profiled_encoder_ssl_gflops_step": (profiled_step_flops or 0.0) / 1e9,
            "estimate/estimated_total_encoder_ssl_flops": estimated_total_encoder_ssl_flops,
            "estimate/estimated_total_encoder_ssl_tflops": estimated_total_encoder_ssl_flops / 1e12,
        }
        if gpu_memory_stats["available"]:
            estimate_summary["estimate/peak_gpu_vram_allocated_bytes"] = estimate_stats.get("peak_allocated_bytes_max", 0)
            estimate_summary["estimate/peak_gpu_vram_reserved_bytes"] = estimate_stats.get("peak_reserved_bytes_max", 0)
            estimate_summary["estimate/peak_gpu_vram_allocated_gb"] = estimate_stats.get("peak_allocated_bytes_max", 0) / (1024 ** 3)
            estimate_summary["estimate/peak_gpu_vram_reserved_gb"] = estimate_stats.get("peak_reserved_bytes_max", 0) / (1024 ** 3)
            if allocated_summary:
                estimate_summary["estimate/avg_gpu_vram_allocated_bytes"] = allocated_summary["avg"]
                estimate_summary["estimate/median_gpu_vram_allocated_bytes"] = allocated_summary["median"]
                estimate_summary["estimate/p90_gpu_vram_allocated_bytes"] = allocated_summary["p90"]
                estimate_summary["estimate/min_gpu_vram_allocated_bytes"] = allocated_summary["min"]
                estimate_summary["estimate/avg_gpu_vram_allocated_gb"] = allocated_summary["avg"] / (1024 ** 3)
                estimate_summary["estimate/median_gpu_vram_allocated_gb"] = allocated_summary["median"] / (1024 ** 3)
                estimate_summary["estimate/p90_gpu_vram_allocated_gb"] = allocated_summary["p90"] / (1024 ** 3)
                estimate_summary["estimate/min_gpu_vram_allocated_gb"] = allocated_summary["min"] / (1024 ** 3)
            if reserved_summary:
                estimate_summary["estimate/avg_gpu_vram_reserved_bytes"] = reserved_summary["avg"]
                estimate_summary["estimate/median_gpu_vram_reserved_bytes"] = reserved_summary["median"]
                estimate_summary["estimate/p90_gpu_vram_reserved_bytes"] = reserved_summary["p90"]
                estimate_summary["estimate/min_gpu_vram_reserved_bytes"] = reserved_summary["min"]
                estimate_summary["estimate/avg_gpu_vram_reserved_gb"] = reserved_summary["avg"] / (1024 ** 3)
                estimate_summary["estimate/median_gpu_vram_reserved_gb"] = reserved_summary["median"] / (1024 ** 3)
                estimate_summary["estimate/p90_gpu_vram_reserved_gb"] = reserved_summary["p90"] / (1024 ** 3)
                estimate_summary["estimate/min_gpu_vram_reserved_gb"] = reserved_summary["min"] / (1024 ** 3)
        if use_wandb:
            wandb.log(estimate_summary)
        print("📐 Encoder compute estimate summary:")
        print(f"   Avg encoder step wall time excl probes: {avg_encoder_step_wall_time:.4f} sec")
        print(f"   Estimated total encoder train time:     {estimated_total_encoder_train_time_sec/60.0:.2f} min")
        if profiled_step_flops is not None:
            print(f"   Profiled encoder SSL FLOPs/step:       {profiled_step_flops/1e9:.3f} GFLOP")
            print(f"   Estimated total encoder SSL FLOPs:     {estimated_total_encoder_ssl_flops/1e12:.3f} TFLOP")
        if gpu_memory_stats["available"]:
            print(f"   Peak GPU VRAM allocated:               {estimate_stats.get('peak_allocated_bytes_max', 0) / (1024 ** 3):.2f} GiB")
            print(f"   Peak GPU VRAM reserved:                {estimate_stats.get('peak_reserved_bytes_max', 0) / (1024 ** 3):.2f} GiB")
            if allocated_summary:
                print(f"   Avg measured GPU VRAM allocated:       {allocated_summary['avg'] / (1024 ** 3):.2f} GiB")
                print(f"   Median measured GPU VRAM allocated:    {allocated_summary['median'] / (1024 ** 3):.2f} GiB")
                print(f"   P90 measured GPU VRAM allocated:       {allocated_summary['p90'] / (1024 ** 3):.2f} GiB")
            if reserved_summary:
                print(f"   Avg measured GPU VRAM reserved:        {reserved_summary['avg'] / (1024 ** 3):.2f} GiB")
                print(f"   Median measured GPU VRAM reserved:     {reserved_summary['median'] / (1024 ** 3):.2f} GiB")
                print(f"   P90 measured GPU VRAM reserved:        {reserved_summary['p90'] / (1024 ** 3):.2f} GiB")

    if WANDB_AVAILABLE:
        # Log training compute summary metrics when the active run exposes a summary dict.
        run_obj = getattr(wandb, 'run', None)
        run_summary = getattr(run_obj, 'summary', None)
        if run_summary is not None:
            run_summary["mode/encoder_only"] = 1 if encoder_only_mode else 0
            run_summary["mode/estimate_encoder_train_compute"] = 1 if estimate_encoder_train_compute else 0
            run_summary["compute/encoder_step_wall_time_excl_probes_total_sec"] = timing_stats["encoder_step_wall_time_excl_probes_total"]
            run_summary["compute/probe_step_wall_time_total_sec"] = timing_stats["probe_step_wall_time_total"]
            run_summary["compute/encoder_step_wall_time_excl_probes_total_min"] = timing_stats["encoder_step_wall_time_excl_probes_total"] / 60.0
            run_summary["compute/probe_step_wall_time_total_min"] = timing_stats["probe_step_wall_time_total"] / 60.0
            if timing_stats["encoder_ssl_flops_profiled_step"] is not None:
                run_summary["compute/encoder_ssl_flops_profiled_step"] = timing_stats["encoder_ssl_flops_profiled_step"]
                run_summary["compute/encoder_ssl_gflops_profiled_step"] = timing_stats["encoder_ssl_flops_profiled_step"] / 1e9
                run_summary["compute/encoder_ssl_flops_total_estimated"] = timing_stats["encoder_ssl_flops_total_estimated"]
                run_summary["compute/encoder_ssl_tflops_total_estimated"] = timing_stats["encoder_ssl_flops_total_estimated"] / 1e12
            if track_encoder_compute and encoder_compute_stats["macs_per_call"] is not None:
                run_summary["compute/encoder_macs_per_call"] = encoder_compute_stats["macs_per_call"]
                run_summary["compute/encoder_macs_total"] = encoder_compute_stats["macs_total"]
                run_summary["compute/encoder_flops_total"] = encoder_compute_stats["flops_total"]
            if gpu_memory_stats["available"]:
                run_summary["compute/gpu_peak_vram_allocated_bytes_max"] = gpu_memory_stats["batch_peak_allocated_bytes_max"]
                run_summary["compute/gpu_peak_vram_reserved_bytes_max"] = gpu_memory_stats["batch_peak_reserved_bytes_max"]
                run_summary["compute/gpu_peak_vram_allocated_gb_max"] = gpu_memory_stats["batch_peak_allocated_bytes_max"] / (1024 ** 3)
                run_summary["compute/gpu_peak_vram_reserved_gb_max"] = gpu_memory_stats["batch_peak_reserved_bytes_max"] / (1024 ** 3)
            if estimate_summary is not None:
                for metric_name, metric_value in estimate_summary.items():
                    run_summary[metric_name] = metric_value
        print(f"📏 Training compute summary:")
        print(f"   Encoder step wall time excl probes: {timing_stats['encoder_step_wall_time_excl_probes_total']/60:.2f} min")
        print(f"   Probe step wall time total:        {timing_stats['probe_step_wall_time_total']/60:.2f} min")
        if timing_stats["encoder_ssl_flops_profiled_step"] is not None:
            print(f"   Encoder SSL FLOPs/profiled step:   {timing_stats['encoder_ssl_flops_profiled_step']/1e9:.3f} GFLOP")
            print(f"   Encoder SSL FLOPs total est.:      {timing_stats['encoder_ssl_flops_total_estimated']/1e12:.3f} TFLOP")
        if gpu_memory_stats["available"]:
            print(f"   Peak GPU VRAM allocated:           {gpu_memory_stats['batch_peak_allocated_bytes_max'] / (1024 ** 3):.2f} GiB")
            print(f"   Peak GPU VRAM reserved:            {gpu_memory_stats['batch_peak_reserved_bytes_max'] / (1024 ** 3):.2f} GiB")
        if trainer_started_wandb_run:
            wandb.finish()



    # Save model
    if not debug_model and run_name is not None:
        save_dir = PROJECT_DIR / "saved_models" / run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "encoder": net.state_dict(),
            "probes": probes.state_dict(),
            "config": dict(cfg),
            "arch": arch,
            "run_name": run_name,
        }
        if patch_probes is not None:
            save_dict["patch_probes"] = patch_probes.state_dict()
        final_name = f"mmlejepa_{arch}_lamb{cfg.lamb}_final.pt"
        save_path = save_dir / final_name
        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")
        # Latest symlink
        latest_path = save_dir / "latest.pt"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(final_name)
    
    # Explicit Cleanup to prevent Memory Leaks and File Descriptor Leaks in Sweeps
    # NOTE: persistent_workers=True keeps worker processes alive with open file handles.
    # We MUST explicitly shutdown workers before deleting loaders to prevent "Too many open files" errors.
    print("Cleaning up training resources...")
    
    def _shutdown_loader(loader, name="loader"):
        """Explicitly shutdown DataLoader workers to release file descriptors."""
        if loader is None:
            return
        try:
            # Access internal iterator and shutdown workers
            if hasattr(loader, '_iterator') and loader._iterator is not None:
                if hasattr(loader._iterator, '_shutdown_workers'):
                    loader._iterator._shutdown_workers()
                    print(f"  ✓ Shutdown workers for {name}")
        except Exception as e:
            print(f"  ⚠ Could not shutdown {name} workers: {e}")
    
    _shutdown_loader(train_loader, "train_loader")
    _shutdown_loader(test_loader, "test_loader")
    _shutdown_loader(test_loader_legacy, "test_loader_legacy")
    
    # Now safe to delete
    if train_loader is not None:
        del train_loader
    if test_loader is not None:
        del test_loader
    if test_loader_legacy is not None:
        del test_loader_legacy
    
    # Delete datasets to release any file handles they may hold
    if 'train_ds' in dir():
        del train_ds
    if 'test_ds' in dir():
        del test_ds
    if 'test_ds_legacy' in dir():
        del test_ds_legacy
        
    del net
    del probes
    if patch_probes is not None:
        del patch_probes
    if opt_encoder is not None:
        del opt_encoder
    if opt_combined_legacy is not None:
        del opt_combined_legacy
    if opt_probe_legacy is not None:
        del opt_probe_legacy
    for opt_obj in probe_opts.values():
        if opt_obj is not None:
            del opt_obj
    for opt_obj in patch_opts.values():
        if opt_obj is not None:
            del opt_obj
    if scaler_combined_legacy is not None:
        del scaler_combined_legacy
    if scaler_encoder is not None:
        del scaler_encoder
    if scaler_probe_legacy is not None:
        del scaler_probe_legacy
    del probe_scalers
    del patch_scalers
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Force close any leaked file descriptors (nuclear option for sweeps)
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"  File descriptor limits: soft={soft}, hard={hard}")
    except:
        pass
    
    print("  ✓ Cleanup complete")
    
    return acc_scene # Return a metric


def validate(net, probes, test_loader, device, epoch, cfg, run_name, simple_baseline=False, simple_modality='rgb', debug=False, log_prefix="test", limit_batches=None, total_epochs=1, patch_probes=None, det_metrics=None, det_metrics_centernet=None, det_metrics_2d=None, seg_metrics=None, box_seg_metrics=None, panoptic_seg_metrics=None, modality_eval="both"):
    """
    Run validation loop and log results.
    Args:
        limit_batches: If set, only validate on this many batches (for speed)
        patch_probes: Optional ModuleDict of patch-token probes (bbox3d, spatial_bbox3d, seg, patch_depth_grid)
        det_metrics: Optional NuScenesDetectionMetrics tracker (DETR probe)
        det_metrics_centernet: Optional NuScenesDetectionMetrics tracker (CenterNet probe)
        det_metrics_2d: Optional DetectionMetrics2D tracker (FLIR 2D probe)
        seg_metrics: Optional SegmentationMetrics tracker
        box_seg_metrics: Optional SegmentationMetrics tracker for FLIR dense box labels
        cfg.grid_iou_empty_policy: How to score empty-union occupancy samples for
            occupancy IoU metrics only:
            val/grid_iou, old_grid_count/grid_iou_car, old_grid_count/grid_iou_ped.
            'negative' -> IoU=0, 'neutral' -> ignored, 'positive' -> IoU=1.
    """
    net.eval()
    probes.eval()
    if patch_probes is not None:
        patch_probes.eval()

    if bool(getattr(cfg, 'encoder_only_mode', False)):
        print(f"⚡ Skipping validate({log_prefix}) because encoder_only_mode is enabled")
        return 0.0
    
    use_wandb = WANDB_AVAILABLE and getattr(cfg, 'wandb', True)
    arch = getattr(cfg, 'arch', 'B')
    dataset_name = getattr(cfg, 'dataset', 'nuscenes')
    fusion_tokens_sigreg = getattr(cfg, 'fusion_tokens_sigreg', False)
    use_late_fusion = getattr(cfg, 'use_late_fusion', False)
    use_frustum_slots = getattr(cfg, 'use_frustum_slots', False)
    use_dinov3_frozen = getattr(cfg, 'use_dinov3_frozen', getattr(cfg, 'use_dinov2_frozen', False))
    use_dinov3_pretrained = getattr(cfg, 'use_dinov3_pretrained', False)
    use_dinov3_scratch = getattr(cfg, 'use_dinov3_scratch', False)
    use_imagebind = getattr(cfg, 'use_imagebind', False)
    use_concat_probe_embeddings = bool(use_late_fusion or use_imagebind)
    probe_forward_chunk_size = max(0, int(getattr(cfg, 'probe_forward_chunk_size', 0)))
    flir_dual_label_probes = bool(getattr(cfg, 'flir_dual_label_probes', False))
    valid_empty_policies = {"negative": 0, "neutral": 1, "positive": 2}
    grid_iou_empty_policy = str(getattr(cfg, 'grid_iou_empty_policy', 'positive')).lower()
    if grid_iou_empty_policy not in valid_empty_policies:
        print(f"⚠️  Unknown grid_iou_empty_policy='{grid_iou_empty_policy}', falling back to 'positive'")
        grid_iou_empty_policy = "positive"

    modality_eval = str(modality_eval).lower()
    if modality_eval not in {"both", "rgb_only", "lidar_only"}:
        print(f"⚠️  Unknown modality_eval='{modality_eval}', falling back to 'both'")
        modality_eval = "both"

    # Ensure each validate() invocation starts from a clean metric state.
    # This prevents cross-pass leakage when val/rgb_only/lidar_only are run
    # sequentially with shared metric objects.
    if det_metrics is not None:
        det_metrics.reset()
    if det_metrics_centernet is not None:
        det_metrics_centernet.reset()
    if isinstance(det_metrics_2d, dict):
        for metric in det_metrics_2d.values():
            metric.reset()
    elif det_metrics_2d is not None:
        det_metrics_2d.reset()
    if seg_metrics is not None:
        seg_metrics.reset()
    if isinstance(box_seg_metrics, dict):
        for metric in box_seg_metrics.values():
            metric.reset()
    elif box_seg_metrics is not None:
        box_seg_metrics.reset()

    def _make_compare_metric(metric_obj):
        if metric_obj is None:
            return None
        compare_plane = 'xy' if getattr(metric_obj, 'matching_plane', 'xz') != 'xy' else 'xz'
        return NuScenesDetectionMetrics(
            class_names=getattr(metric_obj, 'class_names', None),
            num_classes=getattr(metric_obj, 'num_classes', None),
            matching_plane=compare_plane,
        )

    det_metrics_compare = _make_compare_metric(det_metrics)
    det_metrics_centernet_compare = _make_compare_metric(det_metrics_centernet)

    def _accumulate_occupancy_iou(intersection, union):
        iou_raw = intersection / union.clamp(min=1.0)
        if grid_iou_empty_policy == "positive":
            iou = torch.where(union == 0, torch.ones_like(iou_raw), iou_raw)
            valid = torch.ones_like(union)
        elif grid_iou_empty_policy == "neutral":
            valid = (union > 0).float()
            iou = iou_raw * valid
        else:  # negative
            iou = iou_raw
            valid = torch.ones_like(union)
        return iou.sum().item(), valid.sum().item()
    
    if debug:
        print("  Running single-batch validation...")
    elif limit_batches:
        print(f"  Running validation ({log_prefix}) on {limit_batches} batches...")
        
    correct_scene, correct_cam, correct_loc = 0, 0, 0
    mae_cars, mae_peds, mae_objs, mae_depth, mae_depth_grid, depth_grid_iou = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    occupancy_acc, occupancy_iou = 0.0, 0.0
    occupancy_iou_car, occupancy_iou_ped = 0.0, 0.0
    occupancy_iou_count, occupancy_iou_car_count, occupancy_iou_ped_count = 0.0, 0.0, 0.0
    mae_patch_depth_grid, patch_depth_grid_iou = 0.0, 0.0
    mae_depth_map_hr, depth_map_hr_iou = 0.0, 0.0
    occupancy_map_acc, occupancy_map_iou = 0.0, 0.0
    occupancy_map_iou_count = 0.0
    flir_patch_occupancy_acc = {}
    flir_patch_occupancy_iou = {}
    flir_patch_occupancy_iou_count = {}
    if dataset_name == 'flir':
        flir_probe_label_modes_cfg = getattr(cfg, 'flir_probe_label_modes', None)
        flir_probe_label_modes = normalize_flir_probe_label_modes(
            flir_probe_label_modes_cfg,
            enable_dual=flir_dual_label_probes,
        )
        for _flir_domain in flir_probe_label_modes:
            flir_patch_occupancy_acc[_flir_domain] = {"any": 0.0}
            flir_patch_occupancy_iou[_flir_domain] = {"any": 0.0}
            flir_patch_occupancy_iou_count[_flir_domain] = {"any": 0.0}
            for _flir_cls_name in FLIR_2D_DETECTION_CLASSES:
                flir_patch_occupancy_acc[_flir_domain][_flir_cls_name] = 0.0
                flir_patch_occupancy_iou[_flir_domain][_flir_cls_name] = 0.0
                flir_patch_occupancy_iou_count[_flir_domain][_flir_cls_name] = 0.0
    rgb_lidar_patch_mse_sum, rgb_lidar_patch_mse_count = 0.0, 0
    spatial_loss_accum = {}  # CenterNet spatial probe losses
    spatial_loss_counts = 0
    
    counts = 0
    
    with torch.inference_mode():
        for i, (cam_views, modality2, labels) in enumerate(test_loader):
            if debug and i > 0:
                break
            if limit_batches is not None and i >= limit_batches:
                break
            
            # Helper to move dict/tensor
            cam_views = to_device(cam_views, device)
            modality2 = to_device(modality2, device)
            
            with autocast(enabled=(device == "cuda")):
                cam_eval = cam_views
                modality2_eval = modality2

                # ── Arch D: concatenate depth as 4th channel → RGBD ──
                # The dataset returns 3-ch RGB + 1-ch depth separately;
                # arch D expects a single 4-ch RGBD input.
                if arch == 'D':
                    def _concat_rgbd_val(rgb, depth):
                        """Concatenate 3-ch RGB + 1-ch depth → 4-ch RGBD."""
                        if rgb.shape[2] >= 4:
                            return rgb  # Already RGBD
                        return torch.cat([rgb, depth], dim=2)
                    if isinstance(cam_eval, dict) and isinstance(modality2_eval, dict):
                        cam_eval = {
                            k: _concat_rgbd_val(cam_eval[k], modality2_eval[k])
                            if isinstance(cam_eval.get(k), torch.Tensor) and cam_eval[k].numel() > 0
                               and isinstance(modality2_eval.get(k), torch.Tensor) and modality2_eval[k].numel() > 0
                            else cam_eval.get(k, torch.empty(0))
                            for k in cam_eval
                        }
                    elif isinstance(cam_eval, torch.Tensor) and isinstance(modality2_eval, torch.Tensor):
                        cam_eval = _concat_rgbd_val(cam_eval, modality2_eval)

                if modality_eval == "rgb_only":
                    if arch == "D":
                        cam_eval = cam_views.clone() if isinstance(cam_views, torch.Tensor) else {k: v.clone() for k, v in cam_views.items()}
                        if isinstance(cam_eval, dict):
                            for k in cam_eval:
                                v = cam_eval[k]
                                if isinstance(v, torch.Tensor) and v.dim() >= 4 and v.shape[-3] > 3:
                                    if v.dim() == 5:
                                        v[:, :, 3, :, :] = 0
                                    else:
                                        v[:, 3, :, :] = 0
                        else:
                            if cam_eval.dim() == 5:
                                cam_eval[:, :, 3, :, :] = 0
                            else:
                                cam_eval[:, 3, :, :] = 0
                    else:
                        modality2_eval = zeros_like_tree(modality2)
                elif modality_eval == "lidar_only":
                    if arch == "D":
                        cam_eval = cam_views.clone() if isinstance(cam_views, torch.Tensor) else {k: v.clone() for k, v in cam_views.items()}
                        if isinstance(cam_eval, dict):
                            for k in cam_eval:
                                v = cam_eval[k]
                                if isinstance(v, torch.Tensor) and v.dim() >= 4 and v.shape[-3] >= 3:
                                    if v.dim() == 5:
                                        v[:, :, :3, :, :] = 0
                                    else:
                                        v[:, :3, :, :] = 0
                        else:
                            if cam_eval.dim() == 5:
                                cam_eval[:, :, :3, :, :] = 0
                            else:
                                cam_eval[:, :3, :, :] = 0
                    else:
                        cam_eval = zeros_like_tree(cam_views)

                cam_eval_probe = _unwrap_probe_or_global_views(cam_eval)
                modality2_eval_probe = _unwrap_probe_or_global_views(modality2_eval)

                cached_eval_patch_tokens = None

                if simple_baseline:
                    # Simple baseline: single modality
                    if simple_modality in ('lidar', 'thermal'):
                        simple_input = modality2_eval_probe
                    else:
                        simple_input = cam_eval_probe
                    emb, _ = maybe_chunked_forward(net, probe_forward_chunk_size, simple_input)
                    cam_emb = emb
                elif arch == "D":
                    emb, _ = maybe_chunked_forward(net, probe_forward_chunk_size, cam_eval_probe)  # RGBD single input
                    cam_emb = emb
                elif arch in ("B", "C"):
                    # Check for shared trunk specific modes
                    shared_trunk_contrastive = getattr(cfg, 'shared_trunk_contrastive', False)
                    shared_trunk_separate_sigreg = getattr(cfg, 'shared_trunk_separate_sigreg', False)
                    
                    if modality_eval == "both" and (shared_trunk_contrastive or shared_trunk_separate_sigreg) and arch == "B":
                        # Explicitly verify RGB-only performance
                        emb, _ = maybe_chunked_forward(net.forward_camera_only, probe_forward_chunk_size, cam_eval_probe)
                    elif fusion_tokens_sigreg and arch == 'C' and hasattr(net, 'forward_with_fusion_tokens'):
                        emb, _, cached_eval_patch_tokens = maybe_chunked_forward(
                            net.forward_with_fusion_tokens,
                            probe_forward_chunk_size,
                            cam_eval_probe,
                            modality2_eval_probe,
                        )
                    else:
                        emb, _ = maybe_chunked_forward(net, probe_forward_chunk_size, cam_eval_probe, modality2_eval_probe)
                    
                    B, _ = get_input_stats(cam_eval_probe)
                    _, V = get_input_stats(cam_eval_probe)
                    if use_late_fusion:
                        # Late fusion eval: concatenate RGB + LiDAR embeddings
                        # so probes see both modalities (matching training)
                        if fusion_tokens_sigreg and arch == 'C':
                            cam_emb = torch.cat([emb, emb], dim=-1)
                        else:
                            cam_emb = torch.cat([emb[:B*V], emb[B*V:]], dim=-1)
                    elif use_concat_probe_embeddings:
                        # Concat-based baselines such as ImageBind train probes on
                        # the joined RGB+depth representation, so evaluation must
                        # preserve the same input contract.
                        cam_emb = torch.cat([emb[:B*V], emb[B*V:]], dim=-1)
                    elif modality_eval == "lidar_only":
                        cam_emb = emb if (fusion_tokens_sigreg and arch == 'C') else emb[B*V:]
                    else:
                        cam_emb = emb if (fusion_tokens_sigreg and arch == 'C') else emb[:B*V] # First half is camera
                elif arch in ("A", "E", "F"):
                    (cam_emb_eval, lidar_emb_eval), _ = maybe_chunked_forward(net, probe_forward_chunk_size, cam_eval_probe, modality2_eval_probe)
                    cam_emb = lidar_emb_eval if modality_eval == "lidar_only" else cam_emb_eval
                
                # --- Probe Evaluation on Clean Full View (Last View) ---
                B, V_val = get_input_stats(cam_eval_probe)

                if (
                    getattr(cfg, 'track_patch_mse', True)
                    and modality_eval == "both"
                    and (not simple_baseline)
                    and arch in ('B', 'C', 'D', 'E', 'F')
                    and (not use_frustum_slots)
                    and (not use_dinov3_frozen)
                    and (not use_dinov3_pretrained)
                    and (not use_dinov3_scratch)
                ):
                    rgb_patch_tokens, lidar_patch_tokens = extract_rgb_lidar_patch_pair(
                        net,
                        cam_eval_probe,
                        modality2_eval_probe,
                        arch,
                        simple_baseline=simple_baseline,
                        batch_chunk_size=probe_forward_chunk_size,
                    )
                    patch_mse_val = compute_patch_mse_safe(rgb_patch_tokens, lidar_patch_tokens)
                    if patch_mse_val is not None:
                        patch_mse_val = patch_mse_val.item()
                        rgb_lidar_patch_mse_sum += patch_mse_val * B
                        rgb_lidar_patch_mse_count += B
                
                if V_val > 1:
                    if cam_emb.dim() == 2 and cam_emb.shape[0] == B * V_val:
                        cam_emb_reshaped = cam_emb.view(B, V_val, -1)
                        cam_emb_probe = cam_emb_reshaped[:, -1, :]
                    else:
                        cam_emb_probe = cam_emb
                else:
                    cam_emb_probe = cam_emb

                y_scene = labels["scene"].to(device)
                y_cam = labels["camera"].to(device)
                y_loc = labels["location"].to(device)
                


                y_cars = get_spatial_label(labels, "num_cars", device).float()
                y_peds = get_spatial_label(labels, "num_pedestrians", device).float()
                y_objs = get_spatial_label(labels, "num_objects", device).float()
                y_depth = get_spatial_label(labels, "mean_depth", device).float()
                y_depth_grid = get_spatial_label(labels, "depth_grid", device)
                
                correct_scene += (probes["scene"](cam_emb_probe).argmax(1) == y_scene).sum().item()
                correct_cam += (probes["camera"](cam_emb_probe).argmax(1) == y_cam).sum().item()
                correct_loc += (probes["location"](cam_emb_probe).argmax(1) == y_loc).sum().item()
                mae_cars += (probes["num_cars"](cam_emb_probe).squeeze(-1) - y_cars).abs().sum().item()
                mae_peds += (probes["num_peds"](cam_emb_probe).squeeze(-1) - y_peds).abs().sum().item()
                mae_objs += (probes["num_objs"](cam_emb_probe).squeeze(-1) - y_objs).abs().sum().item()
                mae_depth += (probes["mean_depth"](cam_emb_probe).squeeze(-1) - y_depth).abs().sum().item()
                
                # Depth grid evaluation (64 values per sample)
                pred_depth_grid = probes["depth_grid"](cam_emb_probe) # (B, 64)
                mask = get_spatial_label(labels, "depth_grid_mask", device) # (B, 64)
                
                diff = (pred_depth_grid - y_depth_grid).abs() * mask
                valid_counts = mask.sum(dim=1).clamp(min=1.0)
                mae_depth_grid += (diff.sum(dim=1) / valid_counts).sum().item()
                
                # Grid IoU: % of cells where prediction is within threshold of target
                threshold = 8.0  # 8 meters tolerance (labels now in actual meters)
                correct_cells = ((diff < threshold) & (mask > 0)).float()
                depth_grid_iou += (correct_cells.sum(dim=1) / valid_counts).sum().item()
                
                # Grid Occupancy evaluation (8x8 object detection)
                y_occ = get_spatial_label(labels, "grid_occupancy", device)  # 64-dim binary
                pred_occ = torch.sigmoid(probes["grid_occupancy"](cam_emb_probe))  # 64-dim
                
                pred_binary = (pred_occ > 0.5).float()
                occupancy_acc += (pred_binary == y_occ).float().mean(dim=1).sum().item()
                
                intersection = (pred_binary * y_occ).sum(dim=1)
                union = ((pred_binary + y_occ) > 0).float().sum(dim=1)
                iou_sum, iou_count = _accumulate_occupancy_iou(intersection, union)
                occupancy_iou += iou_sum
                occupancy_iou_count += iou_count
                
                y_occ_car = get_spatial_label(labels, "grid_occupancy_car", device)
                y_occ_ped = get_spatial_label(labels, "grid_occupancy_ped", device)
                
                pred_occ_car = torch.sigmoid(probes["grid_occupancy_car"](cam_emb_probe))
                pred_occ_ped = torch.sigmoid(probes["grid_occupancy_ped"](cam_emb_probe))
                
                pred_binary_car = (pred_occ_car > 0.5).float()
                pred_binary_ped = (pred_occ_ped > 0.5).float()
                
                inter_car = (pred_binary_car * y_occ_car).sum(dim=1)
                union_car = ((pred_binary_car + y_occ_car) > 0).float().sum(dim=1)
                iou_car_sum, iou_car_count = _accumulate_occupancy_iou(inter_car, union_car)
                occupancy_iou_car += iou_car_sum
                occupancy_iou_car_count += iou_car_count
                
                inter_ped = (pred_binary_ped * y_occ_ped).sum(dim=1)
                union_ped = ((pred_binary_ped + y_occ_ped) > 0).float().sum(dim=1)
                iou_ped_sum, iou_ped_count = _accumulate_occupancy_iou(inter_ped, union_ped)
                occupancy_iou_ped += iou_ped_sum
                occupancy_iou_ped_count += iou_ped_count
                
                # --- Patch-token probe evaluation (summary prefixes only) ---
                if patch_probes is not None and log_prefix in {"val", "rgb_only", "lidar_only", "val/rgb_only", "val/lidar_only"}:
                    # Extract patch tokens from encoder
                    if cached_eval_patch_tokens is not None:
                        patch_tokens = cached_eval_patch_tokens
                    else:
                        patch_tokens = extract_patch_tokens(
                            net, cam_eval_probe, modality2_eval_probe, arch,
                            simple_baseline=simple_baseline, simple_modality=simple_modality,
                            batch_chunk_size=probe_forward_chunk_size,
                        )
                    if patch_tokens is not None:
                        # If multi-view, take last view's patches
                        B_val, V_val_p = get_input_stats(cam_eval_probe)
                        if V_val_p > 1 and patch_tokens.shape[0] == B_val * V_val_p:
                            patch_tokens = patch_tokens.view(B_val, V_val_p, patch_tokens.shape[1], patch_tokens.shape[2])[:, -1]

                        if dataset_name == 'flir':
                            for flir_domain in flir_probe_label_modes:
                                det_targets_2d = get_flir_detection_targets(labels, device, flir_domain)
                                flir_bbox2d_probe_keys = []
                                for base_probe_key in ("bbox2d_centernet", "bbox2d_slot", "bbox2d"):
                                    probe_key = flir_probe_key(base_probe_key, flir_domain)
                                    if probe_key in patch_probes:
                                        flir_bbox2d_probe_keys.append(probe_key)
                                for probe_key in flir_bbox2d_probe_keys:
                                    bbox2d_preds = patch_probes[probe_key](patch_tokens)
                                    metric_obj = det_metrics_2d.get(probe_key) if isinstance(det_metrics_2d, dict) else det_metrics_2d
                                    if metric_obj is not None:
                                        try:
                                            det2d_boxes = patch_probes[probe_key].decode_to_boxes(bbox2d_preds)
                                            metric_obj.update(det2d_boxes, det_targets_2d)
                                        except Exception as e:
                                            print(f"⚠️  {probe_key} decode failed: {e}")

                                occ_key = flir_probe_key("occupancy_map", flir_domain)
                                if occ_key in patch_probes and 'grid_occupancy_hr' in labels and 'grid_occupancy_hr_classes' in labels:
                                    occ_pred = patch_probes[occ_key](patch_tokens)
                                    y_occ_targets = get_flir_occupancy_targets(labels, device, flir_domain)
                                    if y_occ_targets.shape[-2:] != occ_pred.shape[-2:]:
                                        y_occ_targets = F.interpolate(y_occ_targets, size=occ_pred.shape[-2:], mode='nearest')

                                    pred_occ_hr = (torch.sigmoid(occ_pred) > 0.5).float()
                                    flir_occ_names = ["any", *FLIR_2D_DETECTION_CLASSES]
                                    for occ_idx, occ_name in enumerate(flir_occ_names):
                                        pred_occ_single = pred_occ_hr[:, occ_idx:occ_idx + 1]
                                        y_occ_single = y_occ_targets[:, occ_idx:occ_idx + 1]
                                        flir_patch_occupancy_acc[flir_domain][occ_name] += (
                                            (pred_occ_single == y_occ_single).float().mean(dim=(1, 2, 3)).sum().item()
                                        )
                                        inter_cls = (pred_occ_single * y_occ_single).sum(dim=(1, 2, 3))
                                        union_cls = ((pred_occ_single + y_occ_single) > 0).float().sum(dim=(1, 2, 3))
                                        cls_iou_sum, cls_iou_count = _accumulate_occupancy_iou(inter_cls, union_cls)
                                        flir_patch_occupancy_iou[flir_domain][occ_name] += cls_iou_sum
                                        flir_patch_occupancy_iou_count[flir_domain][occ_name] += cls_iou_count

                                box_seg_key = flir_probe_key("box_seg", flir_domain)
                                metric_obj = box_seg_metrics.get(box_seg_key) if isinstance(box_seg_metrics, dict) else box_seg_metrics
                                if box_seg_key in patch_probes and metric_obj is not None and labels.get('has_box_seg_map', None) is not None:
                                    box_seg_targets = get_flir_box_seg_targets(labels, device, flir_domain)
                                    has_box_seg = box_seg_targets['has_box_seg_map']
                                    if has_box_seg.any():
                                        box_seg_logits = patch_probes[box_seg_key](patch_tokens)
                                        valid_idx = has_box_seg.nonzero(as_tuple=True)[0]
                                        if len(valid_idx) > 0:
                                            metric_obj.update(box_seg_logits[valid_idx], box_seg_targets['box_seg_map'][valid_idx])
                        else:
                            det_targets = {
                                'gt_classes': labels.get('gt_classes', torch.zeros(B_val, 50, dtype=torch.long)).to(device),
                                'gt_centers': labels.get('gt_centers', torch.zeros(B_val, 50, 3)).to(device),
                                'gt_centers_2d': labels.get('gt_centers_2d', torch.zeros(B_val, 50, 2)).to(device),
                                'gt_sizes': labels.get('gt_sizes', torch.zeros(B_val, 50, 3)).to(device),
                                'gt_orientations': labels.get('gt_orientations', torch.zeros(B_val, 50, 2)).to(device),
                                'gt_mask': labels.get('gt_mask', torch.zeros(B_val, 50, dtype=torch.float32)).to(device),
                            }
                            if dataset_name == 'waymo':
                                det_targets['gt_classes'], det_targets['gt_mask'] = remap_waymo_patch_gt_classes(
                                    det_targets['gt_classes'], det_targets['gt_mask'], device,
                                )
                            
                            # 1) BBox3D probe (DETR-style)
                            bbox3d_preds = patch_probes["bbox3d"](patch_tokens)
                            if det_metrics is not None:
                                det_metrics.update(bbox3d_preds, det_targets)
                            if det_metrics_compare is not None:
                                det_metrics_compare.update(bbox3d_preds, det_targets)
                            
                            # 2) SpatialBBox3D probe (CenterNet-style) 
                            spatial_preds = patch_probes["spatial_bbox3d"](patch_tokens)
                            # Decode heatmap peaks to boxes and feed to centernet metrics
                            if det_metrics_centernet is not None:
                                try:
                                    cn_boxes = patch_probes["spatial_bbox3d"].decode_to_boxes(spatial_preds)

                                    # Unnormalize CenterNet XY (image-normalized [0,1])
                                    # back to camera-frame meters before AP matching.
                                    # Pinhole camera model: X/Z = a_x * u_norm + b_x
                                    # where a_x = cw/fx, b_x = (cj - cx)/fx.
                                    # Fit a_x, b_x from GT pairs (robust with >=2 points).
                                    if cn_boxes.get('centers') is not None and det_targets is not None:
                                        pred_centers = cn_boxes['centers'].clone()
                                        gt_mask_cn = det_targets['gt_mask'] > 0.5
                                        gt_uv = det_targets['gt_centers_2d']
                                        gt_cam = det_targets['gt_centers']

                                        B_map = pred_centers.shape[0]
                                        for b_map in range(B_map):
                                            valid = gt_mask_cn[b_map]
                                            n_valid = int(valid.sum().item())
                                            uv_pred = pred_centers[b_map, :, :2]
                                            z_pred = pred_centers[b_map, :, 2].clamp(min=1.0)

                                            if n_valid >= 2:
                                                uv_gt = gt_uv[b_map][valid].float()
                                                cam_gt = gt_cam[b_map][valid].float()
                                                z_gt = cam_gt[:, 2].clamp(min=1.0)
                                                xoz = cam_gt[:, 0] / z_gt
                                                yoz = cam_gt[:, 1] / z_gt
                                                ones = torch.ones(n_valid, 1, device=uv_gt.device)
                                                A_u = torch.cat([uv_gt[:, 0:1], ones], dim=1)
                                                theta_x = torch.linalg.lstsq(A_u, xoz.unsqueeze(1)).solution
                                                A_v = torch.cat([uv_gt[:, 1:2], ones], dim=1)
                                                theta_y = torch.linalg.lstsq(A_v, yoz.unsqueeze(1)).solution
                                                pred_centers[b_map, :, 0] = (uv_pred[:, 0] * theta_x[0, 0] + theta_x[1, 0]) * z_pred
                                                pred_centers[b_map, :, 1] = (uv_pred[:, 1] * theta_y[0, 0] + theta_y[1, 0]) * z_pred
                                            elif n_valid == 1:
                                                uv_gt = gt_uv[b_map][valid][0].float()
                                                cam_gt = gt_cam[b_map][valid][0].float()
                                                z_gt = cam_gt[2].clamp(min=1.0)
                                                a_default = 0.71
                                                b_x = cam_gt[0] / z_gt - a_default * uv_gt[0]
                                                b_y = cam_gt[1] / z_gt - a_default * uv_gt[1]
                                                pred_centers[b_map, :, 0] = (uv_pred[:, 0] * a_default + b_x) * z_pred
                                                pred_centers[b_map, :, 1] = (uv_pred[:, 1] * a_default + b_y) * z_pred
                                            else:
                                                pred_centers[b_map, :, 0] = (uv_pred[:, 0] * 0.71 - 0.37) * z_pred
                                                pred_centers[b_map, :, 1] = (uv_pred[:, 1] * 0.71 - 0.39) * z_pred

                                        cn_boxes['centers'] = pred_centers

                                    det_metrics_centernet.update(cn_boxes, det_targets)
                                    if det_metrics_centernet_compare is not None:
                                        det_metrics_centernet_compare.update(cn_boxes, det_targets)
                                except Exception as e:
                                    print(f"⚠️  centernet decode failed: {e}")
                            try:
                                cn_grid_v = spatial_preds['heatmap'].shape[-1]
                                ct_targets = generate_centernet_targets(
                                    gt_centers_2d=det_targets['gt_classes'].new_zeros(B_val, 50, 2).float()
                                        if labels.get('gt_centers_2d') is None
                                        else labels['gt_centers_2d'].to(device).float(),
                                    gt_classes=det_targets['gt_classes'].float(),
                                    gt_sizes=det_targets['gt_sizes'],
                                    gt_centers_3d=det_targets['gt_centers'],
                                    gt_orientations=det_targets['gt_orientations'],
                                    gt_mask=det_targets['gt_mask'],
                                    num_classes=patch_probes["spatial_bbox3d"].num_classes,
                                    grid_h=cn_grid_v, grid_w=cn_grid_v,
                                )
                                spatial_losses = patch_probes["spatial_bbox3d"].compute_loss(spatial_preds, ct_targets)
                            except Exception as e:
                                print(f"⚠️  spatial probe loss failed: {e}")
                                spatial_losses = {}
                            
                            if spatial_losses:
                                for k, v in spatial_losses.items():
                                    if isinstance(v, torch.Tensor):
                                        spatial_loss_accum[k] = spatial_loss_accum.get(k, 0.0) + v.item()
                                spatial_loss_counts += 1
                            
                            seg_logits = patch_probes["seg"](patch_tokens)
                            if seg_metrics is not None:
                                seg_gt = labels.get('seg_map', None)
                                has_seg = labels.get('has_seg_map', None)
                                if seg_gt is not None:
                                    seg_gt = seg_gt.to(device)
                                    if seg_gt.dim() == 4:
                                        seg_gt = seg_gt[:, -1]
                                    if seg_gt.dim() == 3 and seg_gt.shape[0] == B_val:
                                        if has_seg is not None:
                                            valid_idx = has_seg.nonzero(as_tuple=True)[0]
                                            if len(valid_idx) > 0:
                                                seg_metrics.update(seg_logits[valid_idx], seg_gt[valid_idx])
                                        else:
                                            seg_metrics.update(seg_logits, seg_gt)
                            
                            if "panoptic_seg" in patch_probes:
                                panoptic_seg_logits = patch_probes["panoptic_seg"](patch_tokens)
                                if panoptic_seg_metrics is not None:
                                    panoptic_seg_gt = labels.get('panoptic_seg_map', None)
                                    has_panoptic = labels.get('has_panoptic_seg_map', None)
                                    if panoptic_seg_gt is not None:
                                        panoptic_seg_gt = panoptic_seg_gt.to(device)
                                        if panoptic_seg_gt.dim() == 4:
                                            panoptic_seg_gt = panoptic_seg_gt[:, -1]
                                        if panoptic_seg_gt.dim() == 3 and panoptic_seg_gt.shape[0] == B_val:
                                            if has_panoptic is not None:
                                                valid_idx = has_panoptic.nonzero(as_tuple=True)[0]
                                                if len(valid_idx) > 0:
                                                    panoptic_seg_metrics.update(panoptic_seg_logits[valid_idx], panoptic_seg_gt[valid_idx])
                                            else:
                                                panoptic_seg_metrics.update(panoptic_seg_logits, panoptic_seg_gt)
                            
                            pred_patch_depth = patch_probes["patch_depth_token"](patch_tokens)
                            pred_pd = reshape_patch_scalar_map(pred_patch_depth)
                            pred_pd = pred_pd + patch_probes["patch_depth_spatial"](pred_pd)
                            pred_pd_8x8 = F.adaptive_avg_pool2d(pred_pd, 8).squeeze(1).view(B_val, 64)
                            pd_diff_old = (pred_pd_8x8 - y_depth_grid).abs() * mask
                            mae_patch_depth_grid += (pd_diff_old.sum(dim=1) / valid_counts).sum().item()
                            pd_correct_old = ((pd_diff_old < threshold) & (mask > 0)).float()
                            patch_depth_grid_iou += (pd_correct_old.sum(dim=1) / valid_counts).sum().item()

                            depth_pred = patch_probes["depth_map"](patch_tokens)
                            y_dg_hr = get_spatial_label(labels, "depth_grid_hr", device)
                            y_dg_mask_hr = get_spatial_label(labels, "depth_grid_mask_hr", device)
                            y_dg_2d = reshape_flat_spatial_label(y_dg_hr)
                            y_mask_2d = reshape_flat_spatial_label(y_dg_mask_hr)
                            y_dg_2d, y_mask_2d = resize_spatial_label_like(y_dg_2d, y_mask_2d, depth_pred)
                            pd_diff_hr = (depth_pred - y_dg_2d).abs() * y_mask_2d
                            pd_valid_hr = y_mask_2d.sum(dim=(1, 2, 3)).clamp(min=1)
                            mae_depth_map_hr += (pd_diff_hr.sum(dim=(1, 2, 3)) / pd_valid_hr).sum().item()
                            pd_correct_hr = ((pd_diff_hr < threshold) & (y_mask_2d > 0)).float()
                            depth_map_hr_iou += (pd_correct_hr.sum(dim=(1, 2, 3)) / pd_valid_hr).sum().item()

                            if "occupancy_map" in patch_probes and "grid_occupancy_hr" in labels:
                                occ_pred = patch_probes["occupancy_map"](patch_tokens)
                                y_occ_hr = get_spatial_label(labels, "grid_occupancy_hr", device)
                                y_occ_2d = reshape_flat_spatial_label(y_occ_hr)
                                if y_occ_2d.shape[-2:] != occ_pred.shape[-2:]:
                                    y_occ_2d = F.interpolate(y_occ_2d, size=occ_pred.shape[-2:], mode='nearest')
                                pred_occ_hr = (torch.sigmoid(occ_pred) > 0.5).float()
                                occupancy_map_acc += (pred_occ_hr == y_occ_2d).float().mean(dim=(1, 2, 3)).sum().item()
                                inter_hr = (pred_occ_hr * y_occ_2d).sum(dim=(1, 2, 3))
                                union_hr = ((pred_occ_hr + y_occ_2d) > 0).float().sum(dim=(1, 2, 3))
                                iou_hr_sum, iou_hr_count = _accumulate_occupancy_iou(inter_hr, union_hr)
                                occupancy_map_iou += iou_hr_sum
                                occupancy_map_iou_count += iou_hr_count
            
            counts += B


    # Average metrics
    if counts == 0: counts = 1
    
    acc_scene = correct_scene / counts
    acc_cam = correct_cam / counts
    acc_loc = correct_loc / counts
    mae_cars /= counts
    mae_peds /= counts
    mae_objs /= counts
    mae_depth /= counts
    mae_depth_grid /= counts
    depth_grid_iou /= counts
    occupancy_acc /= counts
    if occupancy_iou_count > 0:
        grid_iou = occupancy_iou / occupancy_iou_count
    else:
        grid_iou = 0.0
    if occupancy_iou_car_count > 0:
        grid_iou_car = occupancy_iou_car / occupancy_iou_car_count
    else:
        grid_iou_car = 0.0
    if occupancy_iou_ped_count > 0:
        grid_iou_ped = occupancy_iou_ped / occupancy_iou_ped_count
    else:
        grid_iou_ped = 0.0
    mae_patch_depth_grid /= counts
    patch_depth_grid_iou /= counts
    mae_depth_map_hr /= counts
    depth_map_hr_iou /= counts
    occupancy_map_acc /= counts
    occupancy_map_iou /= max(occupancy_map_iou_count, 1.0)
    if dataset_name == 'flir' and flir_patch_occupancy_iou_count.get("rgb", {}).get("any", 0) > 0:
        occupancy_map_acc = flir_patch_occupancy_acc["rgb"]["any"] / counts
        occupancy_map_iou = flir_patch_occupancy_iou["rgb"]["any"] / flir_patch_occupancy_iou_count["rgb"]["any"]
    rgb_lidar_patch_mse = (
        rgb_lidar_patch_mse_sum / rgb_lidar_patch_mse_count
        if rgb_lidar_patch_mse_count > 0 else None
    )
    
    # Compute patch probe aggregate metrics
    patch_probe_metrics = {}
    probe_summary_prefixes = {"val", "rgb_only", "lidar_only", "val/rgb_only", "val/lidar_only"}
    if patch_probes is not None and log_prefix in probe_summary_prefixes:
        if isinstance(det_metrics_2d, dict):
            for metric_name, metric_obj in det_metrics_2d.items():
                suffix = metric_name.replace("bbox2d_", "")
                try:
                    det2d_results = metric_obj.compute()
                    if log_prefix == "val":
                        patch_probe_metrics.update({f"det2d/{suffix}/{k}": v for k, v in det2d_results.items()})
                    patch_probe_metrics.update({f"{log_prefix}/det2d_{suffix}_{k}": v for k, v in det2d_results.items()})
                except Exception as e:
                    print(f"⚠️  {metric_name} compute failed: {e}")
                metric_obj.reset()
        elif det_metrics_2d is not None:
            try:
                det2d_results = det_metrics_2d.compute()
                if log_prefix == "val":
                    patch_probe_metrics.update({f"det2d/{k}": v for k, v in det2d_results.items()})
                patch_probe_metrics.update({f"{log_prefix}/det2d_{k}": v for k, v in det2d_results.items()})
            except Exception as e:
                print(f"⚠️  det_metrics_2d compute failed: {e}")
            det_metrics_2d.reset()

        detr_mAP = 0.0
        cn_mAP = 0.0
        detr_mADE = 0.0
        cn_mADE = 0.0

        def _store_match_metrics(prefix, results_dict):
            plane = results_dict.get('matching_plane', 'xz')
            if 'mAP' in results_dict:
                patch_probe_metrics[f"{log_prefix}/{prefix}_mAP_{plane}_match"] = results_dict['mAP']
            if 'mADE' in results_dict:
                patch_probe_metrics[f"{log_prefix}/{prefix}_mADE_{plane}_match"] = results_dict['mADE']
        
        # DETR-style detection metrics
        if det_metrics is not None:
            try:
                det_results = det_metrics.compute()
                det_results['matching_plane'] = getattr(det_metrics, 'matching_plane', 'xz')
                if log_prefix == "val":
                    patch_probe_metrics.update({f"det/detr/{k}": v for k, v in det_results.items()})
                detr_mAP = det_results.get('mAP', 0.0)
                detr_mADE = det_results.get('mADE', 0.0)
                _store_match_metrics('detr', det_results)
            except Exception as e:
                print(f"\u26a0\ufe0f  det_metrics (DETR) compute failed: {e}")
            det_metrics.reset()

        if det_metrics_compare is not None:
            try:
                det_compare_results = det_metrics_compare.compute()
                det_compare_results['matching_plane'] = getattr(det_metrics_compare, 'matching_plane', 'xy')
                if log_prefix == "val":
                    patch_probe_metrics.update({f"det/detr/{det_compare_results['matching_plane']}_match/{k}": v for k, v in det_compare_results.items() if k != 'matching_plane'})
                _store_match_metrics('detr', det_compare_results)
            except Exception as e:
                print(f"\u26a0\ufe0f  det_metrics (DETR compare) compute failed: {e}")
            det_metrics_compare.reset()
        
        # CenterNet-style detection metrics
        if det_metrics_centernet is not None:
            try:
                cn_results = det_metrics_centernet.compute()
                cn_results['matching_plane'] = getattr(det_metrics_centernet, 'matching_plane', 'xz')
                if log_prefix == "val":
                    patch_probe_metrics.update({f"det/centernet/{k}": v for k, v in cn_results.items()})
                cn_mAP = cn_results.get('mAP', 0.0)
                cn_mADE = cn_results.get('mADE', 0.0)
                _store_match_metrics('centernet', cn_results)
            except Exception as e:
                print(f"\u26a0\ufe0f  det_metrics (CenterNet) compute failed: {e}")
            det_metrics_centernet.reset()

        if det_metrics_centernet_compare is not None:
            try:
                cn_compare_results = det_metrics_centernet_compare.compute()
                cn_compare_results['matching_plane'] = getattr(det_metrics_centernet_compare, 'matching_plane', 'xy')
                if log_prefix == "val":
                    patch_probe_metrics.update({f"det/centernet/{cn_compare_results['matching_plane']}_match/{k}": v for k, v in cn_compare_results.items() if k != 'matching_plane'})
                _store_match_metrics('centernet', cn_compare_results)
            except Exception as e:
                print(f"\u26a0\ufe0f  det_metrics (CenterNet compare) compute failed: {e}")
            det_metrics_centernet_compare.reset()
        
        # CenterNet spatial losses
        if spatial_loss_counts > 0:
            for k, v in spatial_loss_accum.items():
                if log_prefix == "val":
                    patch_probe_metrics[f"det/centernet/{k}"] = v / spatial_loss_counts
        
        # Detection summaries
        patch_probe_metrics[f"{log_prefix}/detr_mAP"] = detr_mAP
        patch_probe_metrics[f"{log_prefix}/detr_mADE"] = detr_mADE
        patch_probe_metrics[f"{log_prefix}/centernet_mAP"] = cn_mAP
        patch_probe_metrics[f"{log_prefix}/centernet_mADE"] = cn_mADE
        # For FLIR: alias det2d centernet metric under centernet_mAP so wandb shows it
        if dataset_name == 'flir' and cn_mAP == 0 and isinstance(det_metrics_2d, dict):
            for metric_name, metric_obj in det_metrics_2d.items():
                if 'centernet' in metric_name and 'thermal' not in metric_name:
                    det2d_mAP50 = patch_probe_metrics.get(f'{log_prefix}/det2d_{metric_name.replace("bbox2d_", "")}_mAP50', 0)
                    if det2d_mAP50 > 0:
                        patch_probe_metrics[f"{log_prefix}/centernet_mAP"] = det2d_mAP50
                    break
        
        # Segmentation metrics
        if seg_metrics is not None:
            try:
                seg_results = seg_metrics.compute()
                if log_prefix == "val":
                    patch_probe_metrics.update({f"seg/{k}": v for k, v in seg_results.items()})
                if 'mIoU' in seg_results:
                    patch_probe_metrics[f"{log_prefix}/seg_mIoU"] = seg_results['mIoU']
            except Exception as e:
                print(f"\u26a0\ufe0f  seg_metrics.compute() failed: {e}")
            seg_metrics.reset()

        # Panoptic segmentation metrics
        if isinstance(box_seg_metrics, dict):
            for metric_name, metric_obj in box_seg_metrics.items():
                metric_suffix = metric_name.replace("box_seg", "").strip("_")
                metric_prefix = "box_seg" if not metric_suffix else f"box_seg/{metric_suffix}"
                metric_log_name = "box_seg" if not metric_suffix else f"box_seg_{metric_suffix}"
                try:
                    box_seg_results = metric_obj.compute()
                    if log_prefix == "val":
                        patch_probe_metrics.update({f"{metric_prefix}/{k}": v for k, v in box_seg_results.items()})
                    if 'mIoU' in box_seg_results:
                        patch_probe_metrics[f"{log_prefix}/{metric_log_name}_mIoU"] = box_seg_results['mIoU']
                except Exception as e:
                    print(f"⚠️  {metric_name}.compute() failed: {e}")
                metric_obj.reset()
        elif box_seg_metrics is not None:
            try:
                box_seg_results = box_seg_metrics.compute()
                if log_prefix == "val":
                    patch_probe_metrics.update({f"box_seg/{k}": v for k, v in box_seg_results.items()})
                if 'mIoU' in box_seg_results:
                    patch_probe_metrics[f"{log_prefix}/box_seg_mIoU"] = box_seg_results['mIoU']
            except Exception as e:
                print(f"⚠️  box_seg_metrics.compute() failed: {e}")
            box_seg_metrics.reset()

        if panoptic_seg_metrics is not None:
            try:
                panoptic_results = panoptic_seg_metrics.compute()
                if log_prefix == "val":
                    patch_probe_metrics.update({f"panoptic_seg/{k}": v for k, v in panoptic_results.items()})
                if 'mIoU' in panoptic_results:
                    patch_probe_metrics[f"{log_prefix}/panoptic_seg_mIoU"] = panoptic_results['mIoU']
            except Exception as e:
                print(f"\u26a0\ufe0f  panoptic_seg_metrics.compute() failed: {e}")
            panoptic_seg_metrics.reset()

        patch_probe_metrics[f"{log_prefix}/patch_depth_grid_mae"] = mae_patch_depth_grid
        patch_probe_metrics[f"{log_prefix}/patch_depth_grid_iou"] = patch_depth_grid_iou
        patch_probe_metrics[f"{log_prefix}/depth_map_hr_mae"] = mae_depth_map_hr
        patch_probe_metrics[f"{log_prefix}/depth_map_hr_iou"] = depth_map_hr_iou
        patch_probe_metrics[f"{log_prefix}/occupancy_map_acc"] = occupancy_map_acc
        patch_probe_metrics[f"{log_prefix}/occupancy_map_iou"] = occupancy_map_iou
        if dataset_name == 'flir':
            for flir_domain, occ_counts in flir_patch_occupancy_iou_count.items():
                for occ_name, occ_count in occ_counts.items():
                    if occ_count <= 0:
                        continue
                    occ_iou = flir_patch_occupancy_iou[flir_domain][occ_name] / occ_count
                    occ_acc = flir_patch_occupancy_acc[flir_domain][occ_name] / counts
                    domain_suffix = "" if flir_domain == 'rgb' else f"_{flir_domain}"
                    patch_probe_metrics[f"{log_prefix}/patch_grid{domain_suffix}_iou_{occ_name}"] = occ_iou
                    patch_probe_metrics[f"{log_prefix}/patch_grid{domain_suffix}_acc_{occ_name}"] = occ_acc
                    if flir_domain == 'rgb':
                        patch_probe_metrics[f"{log_prefix}/patch_grid_iou_{occ_name}"] = occ_iou
                        patch_probe_metrics[f"{log_prefix}/patch_grid_acc_{occ_name}"] = occ_acc
                    if log_prefix == "val":
                        count_prefix = "patch_grid_count" if flir_domain == 'rgb' else f"patch_grid_count/{flir_domain}"
                        patch_probe_metrics[f"{count_prefix}/grid_iou_{occ_name}"] = occ_iou
                        patch_probe_metrics[f"{count_prefix}/acc_{occ_name}"] = occ_acc
    
    print(f"{log_prefix} {epoch+1}: scene={acc_scene:.3f}, cam={acc_cam:.3f}, car_iou={grid_iou_car:.3f}")
    
    if patch_probe_metrics:
        seg_mIoU = patch_probe_metrics.get(f'{log_prefix}/seg_mIoU', 0)
        box_seg_mIoU = patch_probe_metrics.get(f'{log_prefix}/box_seg_mIoU', 0)
        panoptic_seg_mIoU = patch_probe_metrics.get(f'{log_prefix}/panoptic_seg_mIoU', 0)
        d_mAP = patch_probe_metrics.get(f'{log_prefix}/detr_mAP', 0)
        c_mAP = patch_probe_metrics.get(f'{log_prefix}/centernet_mAP', 0)
        # For FLIR: use det2d metrics (centernet_mAP is always 0 since 3D CenterNet is disabled)
        if dataset_name == 'flir' and c_mAP == 0:
            c_mAP = patch_probe_metrics.get(f'{log_prefix}/det2d_centernet_mAP50', 0)
        d_xy = patch_probe_metrics.get(f'{log_prefix}/detr_mAP_xy_match', float('nan'))
        d_xz = patch_probe_metrics.get(f'{log_prefix}/detr_mAP_xz_match', float('nan'))
        c_xy = patch_probe_metrics.get(f'{log_prefix}/centernet_mAP_xy_match', float('nan'))
        c_xz = patch_probe_metrics.get(f'{log_prefix}/centernet_mAP_xz_match', float('nan'))
        occ_map_iou = patch_probe_metrics.get(f'{log_prefix}/occupancy_map_iou', 0)
        print(f"  Patch probes: detr_mAP={d_mAP:.4f}, cn_mAP={c_mAP:.4f}, seg_mIoU={seg_mIoU:.4f}, box_seg_mIoU={box_seg_mIoU:.4f}, old_depth_iou={patch_depth_grid_iou:.3f}, hr_depth_iou={depth_map_hr_iou:.3f}, occ_map_iou={occ_map_iou:.3f}")
        if not math.isnan(d_xy) or not math.isnan(c_xy):
            print(f"  Match planes: detr(XY={d_xy:.4f}, XZ={d_xz:.4f}) | centernet(XY={c_xy:.4f}, XZ={c_xz:.4f})")

    if rgb_lidar_patch_mse is not None:
        print(f"  {log_prefix} rgb_lidar_patch_mse={rgb_lidar_patch_mse:.6f}")
    
    if use_wandb:
        log_dict = {
            f"{log_prefix}/acc_scene": acc_scene,
            f"{log_prefix}/acc_camera": acc_cam,
            f"{log_prefix}/acc_location": acc_loc,
            f"{log_prefix}/mae_objects": mae_objs,
            f"{log_prefix}/mae_depth": mae_depth,
            f"{log_prefix}/mae_depth_grid": mae_depth_grid,
            f"{log_prefix}/depth_grid_iou": depth_grid_iou,
            f"{log_prefix}/occupancy_acc": occupancy_acc,
            f"{log_prefix}/grid_iou": grid_iou,
            f"{log_prefix}/grid_iou_empty_policy": valid_empty_policies[grid_iou_empty_policy],
            f"{log_prefix}/epoch": epoch,
            f"{log_prefix}/patch_probes_enabled": 1 if patch_probes is not None else 0,
            f"{log_prefix}/imagebind_memory_safe_mode": 1 if use_imagebind else 0,
        }
        if log_prefix == "val":
            # Old grid-count metrics → separate section
            log_dict.update({
                "old_grid_count/grid_iou_car": grid_iou_car,
                "old_grid_count/grid_iou_ped": grid_iou_ped,
                "old_grid_count/mae_cars": mae_cars,
                "old_grid_count/mae_peds": mae_peds,
            })
        if rgb_lidar_patch_mse is not None:
            log_dict[f"{log_prefix}/rgb_lidar_patch_mse"] = rgb_lidar_patch_mse
        elif patch_probes is None:
            log_dict[f"{log_prefix}/patch_probe_metrics_skipped"] = 1
        log_dict.update(patch_probe_metrics)

        if patch_probes is None:
            log_dict.update({
                f"{log_prefix}/detr_mAP": 0.0,
                f"{log_prefix}/detr_mADE": 0.0,
                f"{log_prefix}/centernet_mAP": 0.0,
                f"{log_prefix}/centernet_mADE": 0.0,
                f"{log_prefix}/seg_mIoU": 0.0,
                f"{log_prefix}/box_seg_mIoU": 0.0,
                f"{log_prefix}/panoptic_seg_mIoU": 0.0,
                f"{log_prefix}/patch_depth_grid_mae": 0.0,
                f"{log_prefix}/patch_depth_grid_iou": 0.0,
                f"{log_prefix}/depth_map_hr_mae": 0.0,
                f"{log_prefix}/depth_map_hr_iou": 0.0,
                f"{log_prefix}/occupancy_map_acc": 0.0,
                f"{log_prefix}/occupancy_map_iou": 0.0,
            })
        wandb.log(log_dict)
    
    return acc_scene


if __name__ == "__main__":
    main()
