"""
detection_integration.py  (v2)

Thin glue between the training loop (train.py)
and the detection/segmentation probes.

Key changes from v1
--------------------
* Probes consume **patch embeddings** (B, N_patches, vit_dim) not CLS tokens.
* Labels are **precomputed** (loaded from cache, not computed at train time).
* Only `MMEncoderB` currently exposes `_forward_features_with_patches()`.
  Other architectures fall back to repeating the CLS token.

Usage in train.py
------------------
    from src.detection_integration import (
        create_det_seg_probes,
        extract_patch_embeddings,
        compute_det_seg_loss,
        load_det_seg_labels_for_batch,
    )

    # After model creation:
    det_seg = create_det_seg_probes(vit_dim=384).to(device)

    # In forward (inside torch.no_grad()):
    patch_embs = extract_patch_embeddings(net, arch, cam_views_probe)  # (B,196,384)

    # Load labels (from precomputed cache):
    det_labels = load_det_seg_labels_for_batch(pairs, cache_dir, has_shards)

    # Compute loss:
    losses = compute_det_seg_loss(det_seg, patch_embs, det_labels)
    loss = loss + losses['loss_total']
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path

from src.detection_probes import DetectionSegmentationProbes, generate_centernet_targets
from src.detection_labels import load_det_seg_labels, CACHE_DIR_NAME


# ═══════════════════════════════════════════════════════════════════════════════
# PROBE CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_det_seg_probes(
    vit_dim: int = 384,
    max_objects: int = 50,
    patch_grid: int = 14,
    seg_upsample: int = 4,
    target_size: Tuple[int, int] = (224, 224),
    bbox_style: str = 'detr',
) -> DetectionSegmentationProbes:
    """Create the combined detection + segmentation probe module.

    Args:
        bbox_style: 'detr' (Hungarian matching) or 'centernet' (dense heatmap).
    """
    return DetectionSegmentationProbes(
        vit_dim=vit_dim,
        max_objects=max_objects,
        patch_grid=patch_grid,
        seg_upsample=seg_upsample,
        target_size=target_size,
        bbox_style=bbox_style,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_patch_embeddings(
    net: nn.Module,
    arch: str,
    cam_probe_view: torch.Tensor,
    range_probe_view: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Extract patch-level embeddings from the encoder for the probe view.

    Currently only MMEncoderB has _forward_features_with_patches().
    Other architectures fall back to a tiled CLS embedding.

    Args:
        net:             The encoder module.
        arch:            Architecture letter ("A"-"F").
        cam_probe_view:  (B, 1, 3, H, W) or (B, 3, H, W) camera probe view.
        range_probe_view: Optional lidar/range probe view (not used for patches).

    Returns:
        patch_embs: (B, N_patches, vit_dim) — spatial patch features.
    """
    # Normalise to (B, C, H, W) if needed
    if cam_probe_view.dim() == 5:
        cam_probe_view = cam_probe_view[:, 0]  # take single view

    if arch == "B" and hasattr(net, "_forward_features_with_patches"):
        _cls, patches = net._forward_features_with_patches(cam_probe_view, is_range=False)
        return patches  # (B, N, vit_dim)

    # Fallback for other archs: use CLS token tiled to N patches
    # (detects vit_dim from model)
    if arch == "D" and hasattr(net, "backbone"):
        # D's backbone is standard ViT with 4-channel input; run patch path manually
        x = net.backbone.patch_embed(cam_probe_view)
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        cls_t = net.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_t, x], dim=1) + net.backbone.pos_embed
        x = net.backbone.pos_drop(x)
        x = net.backbone.blocks(x)
        x = net.backbone.norm(x)
        return x[:, 1:]  # patch tokens

    # Generic fallback: run normal forward and tile CLS
    if hasattr(net, "forward"):
        if arch in ("A", "E", "F") and range_probe_view is not None:
            emb = net(cam_probe_view.unsqueeze(1), range_probe_view)[0]
            if isinstance(emb, tuple):
                emb = emb[0]  # camera embedding
        else:
            emb = net(cam_probe_view.unsqueeze(1))[0] if cam_probe_view.dim() == 4 else net(cam_probe_view)[0]
            if isinstance(emb, tuple):
                emb = emb[0]

    # emb: (B, embed_dim)  →  tile to (B, 196, embed_dim)
    # This is a degenerate fallback; the probes will still train but
    # lose spatial resolution.  Prefer arch B for proper patch probes.
    N_patches = 14 * 14
    return emb.unsqueeze(1).expand(-1, N_patches, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL LOADING (from precomputed cache)
# ═══════════════════════════════════════════════════════════════════════════════

def load_det_seg_labels_for_batch(
    pairs: List[Dict],
    cache_dir: Optional[Path] = None,
    has_shards: bool = True,
    device: torch.device = torch.device("cpu"),
    max_objects: int = 50,
    seg_size: int = 224,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load precomputed detection/seg labels for a batch of pairs.

    Args:
        pairs:      List of pair dicts (each has 'sample_token', 'camera_name').
        cache_dir:  Path to cache/det_seg_labels_v2/ directory.
        has_shards: Whether shard zips exist.
        device:     Target device.
        max_objects: Expected max objects (for empty fallback shape).
        seg_size:   Expected seg map H=W.

    Returns:
        Batched dict {gt_classes, gt_centers, …, seg_map} as tensors on device,
        or None if no precomputed labels were found for the batch.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache" / CACHE_DIR_NAME

    batch = []
    for pair in pairs:
        lbl = load_det_seg_labels(
            sample_token=pair["sample_token"],
            cam_name=pair["camera_name"],
            cache_dir=cache_dir,
            has_shards=has_shards,
        )
        if lbl is None:
            # Empty fallback
            lbl = {
                "gt_classes": np.full(max_objects, -1, dtype=np.int64),
                "gt_centers": np.zeros((max_objects, 3), dtype=np.float32),
                "gt_sizes": np.zeros((max_objects, 3), dtype=np.float32),
                "gt_orientations": np.zeros((max_objects, 2), dtype=np.float32),
                "gt_mask": np.zeros(max_objects, dtype=np.float32),
                "gt_centers_2d": np.zeros((max_objects, 2), dtype=np.float32),
                "seg_map": np.zeros((seg_size, seg_size), dtype=np.int64),
            }
        batch.append(lbl)

    # Stack into tensors
    out = {}
    for key in ["gt_classes", "gt_centers", "gt_sizes", "gt_orientations",
                "gt_mask", "gt_centers_2d", "seg_map"]:
        arrs = [b[key] for b in batch if key in b]
        if arrs:
            out[key] = torch.from_numpy(np.stack(arrs)).to(device)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_det_seg_loss(
    probes: DetectionSegmentationProbes,
    patch_embs: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    weight_bbox: float = 0.01,
    weight_seg: float = 0.01,
    update_metrics: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Full forward + loss for detection and segmentation probes.

    Args:
        probes:      DetectionSegmentationProbes module.
        patch_embs:  (B, N, vit_dim)  patch embeddings (detached from encoder).
        labels:      Dict with gt_classes, gt_centers, gt_sizes, gt_orientations,
                     gt_mask, seg_map from load_det_seg_labels_for_batch().
        weight_bbox: Scalar on bbox loss.
        weight_seg:  Scalar on seg loss.
        update_metrics: If True, accumulate into probe metric trackers.

    Returns:
        Dict with individual + combined loss tensors.
    """
    device = patch_embs.device

    bbox_out, seg_out = probes(patch_embs)

    # Build bbox targets depending on probe style
    if hasattr(probes, 'bbox_style') and probes.bbox_style == 'centernet':
        # CenterNet: generate dense heatmap targets on the fly
        if "gt_centers_2d" in labels:
            bbox_targets = generate_centernet_targets(
                gt_centers_2d=labels["gt_centers_2d"],
                gt_classes=labels["gt_classes"],
                gt_sizes=labels["gt_sizes"],
                gt_centers_3d=labels["gt_centers"],
                gt_orientations=labels["gt_orientations"],
                gt_mask=labels["gt_mask"],
                num_classes=probes.bbox_probe.num_classes,
            )
        else:
            bbox_targets = None
    else:
        # DETR-style: pass raw GT arrays
        bbox_targets = {
            "gt_classes": labels["gt_classes"],
            "gt_centers": labels["gt_centers"],
            "gt_sizes": labels["gt_sizes"],
            "gt_orientations": labels["gt_orientations"],
            "gt_mask": labels["gt_mask"],
        }
    seg_targets = labels.get("seg_map")

    losses = probes.compute_losses(bbox_out, seg_out, bbox_targets, seg_targets)

    # Weighted total
    bbox_total = losses.get("bbox_loss_total", torch.tensor(0.0, device=device))
    seg_total = losses.get("seg_loss_seg", torch.tensor(0.0, device=device))
    losses["loss_total"] = weight_bbox * bbox_total + weight_seg * seg_total

    if update_metrics:
        probes.update_metrics(bbox_out, seg_out, bbox_targets, seg_targets)

    return losses


# ═══════════════════════════════════════════════════════════════════════════════
# WANDB LOGGING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def log_det_seg_metrics(
    probes: DetectionSegmentationProbes,
    prefix: str = "val",
    wandb_obj=None,
) -> Dict[str, float]:
    """
    Compute metrics, log to WandB, reset, and return metrics dict.
    """
    metrics = probes.compute_metrics()
    probes.reset_metrics()

    if wandb_obj is not None:
        wandb_obj.log({f"{prefix}/{k}": v for k, v in metrics.items()})

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP SNIPPET
# ═══════════════════════════════════════════════════════════════════════════════

INTEGRATION_SNIPPET = """
# ============================================================================
# INTEGRATION SNIPPET  (add to train.py)
# ============================================================================

# ── Imports (top of file) ────────────────────────────────────────────────────
from src.detection_integration import (
    create_det_seg_probes,
    extract_patch_embeddings,
    compute_det_seg_loss,
    load_det_seg_labels_for_batch,
    log_det_seg_metrics,
)
from src.detection_labels import CACHE_DIR_NAME
from pathlib import Path

# ── After creating standard probes (~line 760) ──────────────────────────────
# bbox_style='detr' (default, Hungarian matching) or 'centernet' (dense heatmap)
det_seg_probes = create_det_seg_probes(vit_dim=384, bbox_style='centernet').to(device)
det_seg_cache = Path(__file__).parent / "cache" / CACHE_DIR_NAME
det_seg_has_shards = any(det_seg_cache.glob("shard_*.zip")) if det_seg_cache.exists() else False
det_seg_opt = torch.optim.Adam(det_seg_probes.parameters(), lr=1e-3)  # separate optimizer

# ── Inside training loop (after standard probe losses, ~line 1712) ──────────
# Extract patch embeddings for probe view (no grad through encoder)
with torch.no_grad():
    patch_embs_probe = extract_patch_embeddings(net, arch, cam_views_probe)

# Load precomputed labels (list of pair dicts for this batch)
batch_pairs = [dataset._get_pair(idx) for idx in batch_indices]  # adapt to your batch tracking
det_labels = load_det_seg_labels_for_batch(batch_pairs, det_seg_cache, det_seg_has_shards, device)

if det_labels is not None:
    det_losses = compute_det_seg_loss(det_seg_probes, patch_embs_probe, det_labels)
    loss = loss + det_losses['loss_total']

# ── Validation (after epoch) ────────────────────────────────────────────────
det_metrics = log_det_seg_metrics(det_seg_probes, prefix="val", wandb_obj=wandb)
print(f"  Det mAP={det_metrics.get('bbox_mAP',0):.4f}, Seg mIoU={det_metrics.get('seg_mIoU',0):.4f}")
"""

if __name__ == "__main__":
    print("detection_integration v2 — integration snippet:")
    print(INTEGRATION_SNIPPET)

    # Quick smoke test — DETR style
    probes = create_det_seg_probes(vit_dim=384, bbox_style='detr')
    print(f"[detr] Params: {sum(p.numel() for p in probes.parameters()):,}")
    dummy = torch.randn(2, 196, 384)
    bbox, seg = probes(dummy)
    print(f"[detr] bbox keys: {list(bbox.keys())}")
    print(f"[detr] seg shape: {seg.shape}")

    # Quick smoke test — CenterNet style
    probes_cn = create_det_seg_probes(vit_dim=384, bbox_style='centernet')
    print(f"[centernet] Params: {sum(p.numel() for p in probes_cn.parameters()):,}")
    bbox_cn, seg_cn = probes_cn(dummy)
    print(f"[centernet] bbox keys: {list(bbox_cn.keys())}")
    print(f"[centernet] seg shape: {seg_cn.shape}")
    print("✅ OK")
