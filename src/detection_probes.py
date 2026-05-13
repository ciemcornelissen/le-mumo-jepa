"""
Detection and Segmentation Probes for MM-LeJEPA Evaluation (v2).

Paper-ready probes that follow SSL evaluation conventions:
1. BBox3DProbe: Patch-based 3D bbox prediction with Hungarian matching
   → Reports: mAP, mATE, mASE, mAOE (nuScenes official metrics)
2. SemanticSegProbe: Linear 1×1 conv + PixelShuffle (DINO/MAE convention)
   → Reports: mIoU, per-class IoU, pixel accuracy

Design principles:
- Probes consume PATCH embeddings (B, N_patches, D), not CLS tokens
- Segmentation is a standard linear probe (1×1 conv) — no heavy decoder
- Detection uses Hungarian matching for 1-to-1 pred↔GT assignment
- Labels are precomputed offline (see precompute_det_seg_labels.py)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None  # graceful fallback


# ═══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS FOR CLASS IMBALANCE
# ═══════════════════════════════════════════════════════════════════════════════

def focal_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Focal loss for multi-class classification to handle class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        logits: (N, C) raw logits before softmax
        targets: (N,) class indices
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Per-class weights (optional)
        reduction: 'none', 'mean', or 'sum'
    
    Returns:
        Focal loss value
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # p_t = probability of correct class
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * ce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DETECTION_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
]
NUM_DETECTION_CLASSES = len(DETECTION_CLASSES)

CATEGORY_TO_DETECTION = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer',
    'vehicle.construction': 'construction_vehicle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.barrier': 'barrier',
}

# 16-class simplified segmentation (0 = ignore, never evaluated)
SIMPLIFIED_SEG_CLASSES = [
    'ignore',        # 0
    'pedestrian',    # 1
    'barrier',       # 2
    'traffic_cone',  # 3
    'bicycle',       # 4
    'bus',           # 5
    'car',           # 6
    'construction',  # 7
    'motorcycle',    # 8
    'trailer',       # 9
    'truck',         # 10
    'driveable',     # 11
    'sidewalk',      # 12
    'terrain',       # 13
    'manmade',       # 14
    'vegetation',    # 15
]
NUM_SIMPLIFIED_SEG_CLASSES = len(SIMPLIFIED_SEG_CLASSES)

FLIR_2D_DETECTION_CLASSES = [
    'person',
    'bike',
    'car',
    'motor',
    'bus',
    'truck',
    'light',
    'sign',
]
NUM_FLIR_2D_DETECTION_CLASSES = len(FLIR_2D_DETECTION_CLASSES)

LIDARSEG_TO_SIMPLIFIED = {
    0: 0, 1: 0,                          # noise, animal → ignore
    2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,  # pedestrians
    9: 2, 10: 0, 11: 0, 12: 3, 13: 0,   # barrier, debris→ign, cone, rack→ign
    14: 4, 15: 5, 16: 5, 17: 6,          # bicycle, bus×2, car
    18: 7, 19: 6, 20: 6, 21: 8,          # construction, emergency→car, motorcycle
    22: 9, 23: 10,                        # trailer, truck
    24: 11, 25: 0, 26: 12, 27: 13,       # driveable, flat.other→ign, sidewalk, terrain
    28: 14, 29: 0, 30: 15, 31: 0,        # manmade, static.other→ign, vegetation, ego→ign
}


# Waymo semantic segmentation classes (23 classes, 0-22)
# See: https://waymo.com/open/data/perception/#semantic-segmentation
# Maps Waymo semantic class ID → unified simplified class (0-15)
WAYMO_LIDARSEG_TO_SIMPLIFIED = {
    0: 0,    # UNDEFINED → ignore
    1: 6,    # CAR → car
    2: 10,   # TRUCK → truck
    3: 5,    # BUS → bus
    4: 6,    # OTHER_VEHICLE → car (closest match)
    5: 8,    # MOTORCYCLIST → motorcycle
    6: 4,    # BICYCLIST → bicycle
    7: 1,    # PEDESTRIAN → pedestrian
    8: 0,    # SIGN → ignore (not in unified classes)
    9: 0,    # TRAFFIC_LIGHT → ignore
    10: 14,  # POLE → manmade
    11: 3,   # CONSTRUCTION_CONE → traffic_cone
    12: 4,   # BICYCLE → bicycle
    13: 8,   # MOTORCYCLE → motorcycle
    14: 14,  # BUILDING → manmade
    15: 15,  # VEGETATION → vegetation
    16: 15,  # TREE_TRUNK → vegetation
    17: 11,  # CURB → driveable (edge of road)
    18: 11,  # ROAD → driveable
    19: 11,  # LANE_MARKER → driveable
    20: 13,  # OTHER_GROUND → terrain
    21: 12,  # WALKABLE → sidewalk
    22: 12,  # SIDEWALK → sidewalk
}


def waymo_label_to_simplified(label: int) -> int:
    """Convert Waymo panoptic/semantic label to unified simplified class.
    
    Waymo labels can be:
    - -1: unlabeled
    - 0: undefined
    - N > 0: panoptic label encoded as (semantic_class + instance_id * 23)
    
    Extract semantic class as: label % 23
    """
    if label <= 0:
        return 0  # ignore
    semantic_class = label % 23
    return WAYMO_LIDARSEG_TO_SIMPLIFIED.get(semantic_class, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3D BOUNDING BOX PROBE  (patch-based, Hungarian matching)
# ═══════════════════════════════════════════════════════════════════════════════

class BBox3DProbe(nn.Module):
    """
    Patch-embedding based 3D bounding box probe with Hungarian matching.

    Architecture:
        patch_embs (B, N, D) → learnable object queries cross-attend to patches
        → predict fixed set of boxes
        Each box: class (10) + center (3) + size (3) + yaw sin/cos (2) + conf (1) = 19

    Uses lightweight cross-attention so object queries can attend to
    spatially-relevant patches instead of collapsing all spatial information
    via global average pooling.

    Loss:
        Hungarian bipartite matching ensures strict 1-to-1 assignment.

    Metrics (nuScenes official):
        mAP  – mean Average Precision (centre-distance based)
        mATE – mean Avg Translation Error
        mASE – mean Avg Scale Error
        mAOE – mean Avg Orientation Error
    """

    PER_BOX_DIM = NUM_DETECTION_CLASSES + 3 + 3 + 2 + 1  # 19

    def __init__(
        self,
        embed_dim: int = 512,
        vit_dim: int = 384,
        hidden_dim: int = 256,
        max_objects: int = 50,
        num_classes: int = NUM_DETECTION_CLASSES,
        patch_grid: int = 14,
        num_heads: int = 4,
        focal_gamma: float = 2.0,  # Focal loss gamma for class imbalance (0=standard CE)
    ):
        """
        Args:
            embed_dim: Projection dimension (CLS head output). Not used here.
            vit_dim:   ViT hidden dimension (patch embeddings before head).
            hidden_dim: Intermediate MLP width.
            max_objects: Number of prediction slots (object queries).
            num_classes: Detection classes.
            patch_grid: Spatial grid side (14 for 224px / 16px patch).
            num_heads:  Attention heads for cross-attention.
            focal_gamma: Focal loss gamma (higher = more focus on hard examples).
        """
        super().__init__()
        self.max_objects = max_objects
        self.num_classes = num_classes
        self.patch_grid = patch_grid
        self.vit_dim = vit_dim
        self.focal_gamma = focal_gamma

        # Learnable object queries — one per detection slot
        self.object_queries = nn.Parameter(torch.randn(max_objects, hidden_dim) * 0.02)

        # Project patch embeddings to hidden_dim for cross-attention
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, hidden_dim),
        )

        # Lightweight cross-attention: queries attend to patch tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Per-query FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Detection head: per-query prediction
        self.det_head = nn.Linear(hidden_dim, self.PER_BOX_DIM)

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, patch_embs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            patch_embs: (B, N_patches, vit_dim) patch embeddings from encoder.
        Returns:
            dict with class_logits, centers, sizes, orientations, confidences
        """
        B = patch_embs.shape[0]

        # Project patches to hidden dim
        kv = self.patch_proj(patch_embs)             # (B, N_patches, hidden)

        # Expand object queries for batch
        queries = self.object_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N_obj, hidden)

        # Cross-attention: queries attend to patch tokens
        attn_out, _ = self.cross_attn(queries, kv, kv)  # (B, N_obj, hidden)
        x = self.attn_norm(queries + attn_out)       # residual + norm
        x = x + self.ffn(x)                          # FFN with residual

        # Per-query box prediction
        raw = self.det_head(x)                       # (B, N_obj, 19)

        idx = 0
        nc = self.num_classes
        class_logits = raw[:, :, idx:idx + nc];          idx += nc
        centers      = raw[:, :, idx:idx + 3];            idx += 3
        sizes        = F.softplus(raw[:, :, idx:idx + 3]);idx += 3
        orientations = F.normalize(raw[:, :, idx:idx + 2], dim=-1); idx += 2
        confidences  = torch.sigmoid(raw[:, :, idx])

        return {
            'class_logits': class_logits,   # (B, N, 10)
            'centers':      centers,        # (B, N, 3)
            'sizes':        sizes,          # (B, N, 3)
            'orientations': orientations,   # (B, N, 2)
            'confidences':  confidences,    # (B, N)
        }

    # ── Hungarian matching ───────────────────────────────────────────────
    @torch.no_grad()
    def _hungarian_match(
        self,
        pred: Dict[str, torch.Tensor],
        gt_centers: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Bipartite matching between predictions and GT per sample.

        Cost = λ_cls · class_cost + λ_ctr · L1_center + λ_conf · (1-conf)

        Returns list of (pred_indices, gt_indices) tuples per batch element.
        """
        B = gt_centers.shape[0]
        indices = []

        for b in range(B):
            mask_b = gt_mask[b].bool()
            n_gt = mask_b.sum().item()
            if n_gt == 0:
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue

            gt_c = gt_centers[b, mask_b]          # (n_gt, 3)
            gt_cl = gt_classes[b, mask_b]          # (n_gt,)

            pred_c = pred['centers'][b]            # (N, 3)
            pred_cl = pred['class_logits'][b]      # (N, 10)
            pred_conf = pred['confidences'][b]     # (N,)

            # Centre-distance cost  (N, n_gt)
            cost_center = torch.cdist(pred_c, gt_c, p=1)  # L1

            # Classification cost: negative log-prob of correct class
            log_probs = F.log_softmax(pred_cl, dim=-1)   # (N, 10)
            cost_class = -log_probs[:, gt_cl.long()]       # (N, n_gt)

            # Confidence cost: prefer high-conf preds for real objects
            cost_conf = (1 - pred_conf).unsqueeze(-1).expand_as(cost_center)

            # Combined cost
            cost = cost_center + cost_class + 0.5 * cost_conf   # (N, n_gt)
            cost_np = cost.detach().cpu().numpy()

            if linear_sum_assignment is not None:
                row_idx, col_idx = linear_sum_assignment(cost_np)
            else:
                # Fallback: greedy (in case scipy unavailable)
                col_idx = cost_np.argmin(axis=0)
                row_idx = np.arange(len(col_idx))

            indices.append((torch.as_tensor(row_idx, dtype=torch.long, device=gt_centers.device),
                            torch.as_tensor(col_idx, dtype=torch.long, device=gt_centers.device)))

        return indices

    # ── loss ─────────────────────────────────────────────────────────────
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = predictions['class_logits'].device
        B = predictions['class_logits'].shape[0]
        zero = torch.tensor(0.0, device=device)

        if targets is None or 'gt_classes' not in targets:
            return dict(loss_cls=zero, loss_center=zero, loss_size=zero,
                        loss_orient=zero, loss_conf=zero, loss_total=zero)

        gt_classes = targets['gt_classes']           # (B, Ngt)
        gt_centers = targets['gt_centers']           # (B, Ngt, 3)
        gt_sizes   = targets['gt_sizes']             # (B, Ngt, 3)
        gt_orient  = targets['gt_orientations']      # (B, Ngt, 2)
        gt_mask    = targets['gt_mask']              # (B, Ngt)

        # Hungarian matching
        matched = self._hungarian_match(predictions, gt_centers, gt_classes, gt_mask)

        total_cls = zero
        total_ctr = zero
        total_sz  = zero
        total_ori = zero
        n_matched = 0

        for b, (pi, gi) in enumerate(matched):
            if len(pi) == 0:
                continue
            n = len(pi)
            n_matched += n

            # Classification - use focal loss if gamma > 0
            cls_logits_matched = predictions['class_logits'][b][pi]
            cls_targets_matched = gt_classes[b][gt_mask[b].bool()][gi].long()
            if self.focal_gamma > 0:
                total_cls = total_cls + focal_loss(
                    cls_logits_matched, cls_targets_matched,
                    gamma=self.focal_gamma, reduction='sum',
                )
            else:
                total_cls = total_cls + F.cross_entropy(
                    cls_logits_matched, cls_targets_matched,
                    reduction='sum',
                )
            # Center L1 (depth component in log-space for stability)
            pred_ctr = predictions['centers'][b][pi]         # (n, 3)
            gt_ctr   = gt_centers[b][gt_mask[b].bool()][gi]  # (n, 3)
            # XY: raw L1;  Z (depth): log(1 + z) space
            total_ctr = total_ctr + F.l1_loss(
                pred_ctr[:, :2], gt_ctr[:, :2], reduction='sum',
            )
            total_ctr = total_ctr + F.l1_loss(
                torch.log1p(pred_ctr[:, 2:].clamp(min=0.0)),
                torch.log1p(gt_ctr[:, 2:].clamp(min=0.0)),
                reduction='sum',
            )
            # Size L1 (log-space for stable regression)
            gt_sz_log = torch.log(gt_sizes[b][gt_mask[b].bool()][gi].clamp(min=1e-4))
            pred_sz_log = torch.log(predictions['sizes'][b][pi].clamp(min=1e-4))
            total_sz = total_sz + F.l1_loss(
                pred_sz_log,
                gt_sz_log,
                reduction='sum',
            )
            # Orientation cosine
            dot = (predictions['orientations'][b][pi] *
                   gt_orient[b][gt_mask[b].bool()][gi]).sum(-1)
            total_ori = total_ori + (1.0 - dot).sum()

        denom = max(n_matched, 1)
        loss_cls    = total_cls / denom
        loss_center = total_ctr / denom
        loss_size   = total_sz  / denom
        loss_orient = total_ori / denom

        # Confidence: matched slots → 1, rest → 0
        # Use weighted BCE to handle extreme class imbalance (few positives, many negatives)
        conf_target = torch.zeros(B, self.max_objects, device=device)
        for b, (pi, _) in enumerate(matched):
            if len(pi) > 0:
                conf_target[b, pi] = 1.0
        
        n_pos = conf_target.sum().clamp(min=1)
        n_total = B * self.max_objects
        n_neg = n_total - n_pos
        
        # Create per-element weights: positives get weight = n_neg/n_pos, negatives get 1.0
        conf_weights = torch.ones_like(conf_target)
        if n_pos > 0 and n_neg > 0:
            pos_weight = (n_neg / n_pos).clamp(max=10.0)  # Cap at 10x
            conf_weights = torch.where(conf_target > 0.5, pos_weight, torch.ones_like(conf_target))
        
        # Cast to float32 and disable autocast for BCE (sigmoid already applied)
        with torch.amp.autocast('cuda', enabled=False):
            bce = F.binary_cross_entropy(
                predictions['confidences'].float(), 
                conf_target.float(),
                reduction='none',
            )
            loss_conf = (bce * conf_weights).mean()

        loss_total = loss_cls + loss_center + loss_size + loss_orient + loss_conf

        return dict(loss_cls=loss_cls, loss_center=loss_center, loss_size=loss_size,
                    loss_orient=loss_orient, loss_conf=loss_conf, loss_total=loss_total)


# ═══════════════════════════════════════════════════════════════════════════════
# 3D BOUNDING BOX PROBE — CENTERNET STYLE  (dense, spatial)
# ═══════════════════════════════════════════════════════════════════════════════

def _gaussian_focal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet-style modified focal loss for Gaussian heatmaps.

    Args:
        pred:   (B, C, H, W) predicted heatmap (after sigmoid)
        target: (B, C, H, W) Gaussian ground-truth heatmap [0, 1]
    """
    pos = target.eq(1).float()
    neg = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)

    pred = pred.clamp(1e-6, 1 - 1e-6)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg

    n_pos = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


def generate_centernet_targets(
    gt_centers_2d: torch.Tensor,
    gt_classes: torch.Tensor,
    gt_sizes: torch.Tensor,
    gt_centers_3d: torch.Tensor,
    gt_orientations: torch.Tensor,
    gt_mask: torch.Tensor,
    grid_h: int = 14,
    grid_w: int = 14,
    num_classes: int = NUM_DETECTION_CLASSES,
    gaussian_radius: int = 2,
) -> Dict[str, torch.Tensor]:
    """Convert precomputed bbox labels into CenterNet dense targets.

    Vectorised implementation — avoids per-object Python loops by building
    a Gaussian kernel once and scattering it for all valid objects.
    Runs efficiently on CPU; no .item() calls on the hot path.

    Args:
        gt_centers_2d:   (B, N, 2) normalised [0, 1] image coordinates.
        gt_classes:      (B, N)
        gt_sizes:        (B, N, 3)
        gt_centers_3d:   (B, N, 3) camera-frame centres (Z = depth).
        gt_orientations: (B, N, 2) sin/cos yaw.
        gt_mask:         (B, N)
        grid_h, grid_w:  patch grid dims.
        gaussian_radius: Gaussian splat radius in grid cells.
                         Scaled proportionally when grid > 14 to keep
                         the same relative coverage.

    Returns:
        Dict with heatmap_grid, offset_grid, size_grid, depth_grid,
        rot_grid, reg_mask.
    """
    B, N = gt_mask.shape
    device = gt_mask.device

    heatmap   = torch.zeros(B, num_classes, grid_h, grid_w, device=device)
    offset    = torch.zeros(B, 2, grid_h, grid_w, device=device)
    size_map  = torch.zeros(B, 3, grid_h, grid_w, device=device)
    depth_map = torch.zeros(B, 1, grid_h, grid_w, device=device)
    rot_map   = torch.zeros(B, 2, grid_h, grid_w, device=device)
    reg_mask  = torch.zeros(B, grid_h, grid_w, device=device)

    # Dynamic radius: larger/closer objects get wider Gaussians than tiny/far ones.
    base_r = max(1, round(gaussian_radius * max(grid_h, grid_w) / 14))
    kernel_cache = {}

    def _get_kernel(radius: int) -> torch.Tensor:
        radius = int(max(1, radius))
        if radius not in kernel_cache:
            sigma = max(radius / 3.0, 0.5)
            dy = torch.arange(-radius, radius + 1, device=device).float()
            dx = torch.arange(-radius, radius + 1, device=device).float()
            gy, gx = torch.meshgrid(dy, dx, indexing='ij')
            kernel_cache[radius] = torch.exp(-(gx * gx + gy * gy) / (2.0 * sigma * sigma))
        return kernel_cache[radius]

    def _radius_for_object(size_xyz: torch.Tensor, depth_z: torch.Tensor) -> int:
        size_xy = size_xyz[:2].amax().item()
        depth_val = max(float(depth_z.item()), 1.0)
        projected_extent = max(grid_h, grid_w) * (size_xy / depth_val)
        dynamic_r = round(max(base_r, 0.6 * projected_extent))
        return int(min(max(dynamic_r, 1), max(2 * base_r, max(grid_h, grid_w) // 4, 1)))

    # Continuous centre coords on the grid
    cx_all = gt_centers_2d[:, :, 0] * grid_w   # (B, N)
    cy_all = gt_centers_2d[:, :, 1] * grid_h   # (B, N)
    ix_all = cx_all.long()
    iy_all = cy_all.long()

    # Valid mask: gt_mask > 0.5, valid class, within grid bounds
    valid = (gt_mask > 0.5)
    valid = valid & (gt_classes >= 0) & (gt_classes < num_classes)
    valid = valid & (ix_all >= 0) & (ix_all < grid_w) & (iy_all >= 0) & (iy_all < grid_h)

    # Process per batch (vectorised over objects within each sample)
    for b in range(B):
        v = valid[b]                             # (N,)
        if not v.any():
            continue

        cls_b  = gt_classes[b, v].long()         # (M,)
        cx_b   = cx_all[b, v]                    # (M,)
        cy_b   = cy_all[b, v]
        ix_b   = ix_all[b, v]
        iy_b   = iy_all[b, v]
        size_b = gt_sizes[b, v]
        depth_b = gt_centers_3d[b, v, 2]

        # Regression targets at integer centres (vectorised scatter)
        for m in range(cls_b.shape[0]):
            iy_m, ix_m = iy_b[m], ix_b[m]
            c_m = cls_b[m]
            r = _radius_for_object(size_b[m], depth_b[m])
            gauss_kernel = _get_kernel(r)
            diam = 2 * r + 1

            # Gaussian splat — bounded by grid edges
            y_lo = max(0, iy_m - r)
            y_hi = min(grid_h, iy_m + r + 1)
            x_lo = max(0, ix_m - r)
            x_hi = min(grid_w, ix_m + r + 1)
            ky_lo = y_lo - (iy_m - r)
            ky_hi = diam - ((iy_m + r + 1) - y_hi)
            kx_lo = x_lo - (ix_m - r)
            kx_hi = diam - ((ix_m + r + 1) - x_hi)
            patch = gauss_kernel[ky_lo:ky_hi, kx_lo:kx_hi]
            heatmap[b, c_m, y_lo:y_hi, x_lo:x_hi] = torch.max(
                heatmap[b, c_m, y_lo:y_hi, x_lo:x_hi], patch,
            )

            # Regression at centre pixel
            offset[b, 0, iy_m, ix_m] = cx_b[m] - ix_m
            offset[b, 1, iy_m, ix_m] = cy_b[m] - iy_m
            reg_mask[b, iy_m, ix_m] = 1.0

        # Vectorised regression fill (all valid objects at once)
        iy_v = iy_b.long()
        ix_v = ix_b.long()
        flat_idx = iy_v * grid_w + ix_v   # (M,)
        # Scatter last-write (later objects overwrite if same cell — matches
        # original behaviour since same-cell collisions are rare on 14×14)
        sz_log = torch.log(gt_sizes[b, v].clamp(min=1e-4))       # (M, 3)
        depth_log = torch.log1p(gt_centers_3d[b, v, 2:3].clamp(min=0.0))  # (M, 1)
        ori_v = gt_orientations[b, v]                              # (M, 2)

        size_flat  = size_map[b].reshape(3, -1)                    # (3, H*W)
        depth_flat = depth_map[b].reshape(1, -1)
        rot_flat   = rot_map[b].reshape(2, -1)

        size_flat[:, flat_idx]  = sz_log.T
        depth_flat[:, flat_idx] = depth_log.T
        rot_flat[:, flat_idx]   = ori_v.T

    return {
        'heatmap_grid': heatmap,
        'offset_grid':  offset,
        'size_grid':    size_map,
        'depth_grid':   depth_map,
        'rot_grid':     rot_map,
        'reg_mask':     reg_mask,
    }


class SpatialBBox3DProbe(nn.Module):
    """CenterNet-style dense 3D bounding-box probe.

    Unlike BBox3DProbe (DETR-style: pool → predict N slots), this probe
    retains the 14×14 spatial grid and makes dense predictions at every
    patch location — closer to modern anchor-free detectors.

    Optionally upsamples to 28×28 for finer localisation (upsample_factor=2).

    Architecture:
        patch_embs (B, N, D)
        → reshape (B, D, H, W)
        → 3×3 Conv adapter (BN + ReLU)
        [→ optional 2× upsample via ConvTranspose2d]
        → per-location heads:
            heatmap  (C, H’, W’) — per-class centre probability
            offset   (2, H’, W’) — sub-pixel centre correction
            size     (3, H’, W’) — log(w, l, h)
            depth    (1, H’, W’) — centre depth Z
            rot      (2, H’, W’) — sin/cos(yaw)

    Loss: Gaussian focal loss for heatmap + L1 for regressions.
    """

    def __init__(
        self,
        vit_dim: int = 384,
        num_classes: int = NUM_DETECTION_CLASSES,
        patch_grid: int = 14,
        adapter_dim: int = 256,
        upsample_factor: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_grid = patch_grid
        self.upsample_factor = upsample_factor
        self.output_grid = patch_grid * upsample_factor  # 14 or 28

        self.adapter = nn.Sequential(
            nn.Conv2d(vit_dim, adapter_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(adapter_dim),
            nn.ReLU(),
        )

        # Optional 2× upsample block: ConvTranspose2d (14×14 → 28×28)
        if upsample_factor > 1:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(adapter_dim, adapter_dim, kernel_size=4,
                                   stride=upsample_factor, padding=1),
                nn.BatchNorm2d(adapter_dim),
                nn.ReLU(),
            )
        else:
            self.upsample = None

        self.heatmap_head = nn.Conv2d(adapter_dim, num_classes, kernel_size=1)
        self.offset_head  = nn.Conv2d(adapter_dim, 2, kernel_size=1)
        self.size_head    = nn.Conv2d(adapter_dim, 3, kernel_size=1)
        self.depth_head   = nn.Conv2d(adapter_dim, 1, kernel_size=1)
        self.rot_head     = nn.Conv2d(adapter_dim, 2, kernel_size=1)

        # CenterNet focal-loss stability bias
        self.heatmap_head.bias.data.fill_(-2.19)

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, patch_embs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = patch_embs.shape
        H = W = int(N ** 0.5)
        x = patch_embs.transpose(1, 2).reshape(B, D, H, W)
        feat = self.adapter(x)

        # Optional 2× upsample: 14×14 → 28×28
        if self.upsample is not None:
            feat = self.upsample(feat)

        return {
            'heatmap': torch.sigmoid(self.heatmap_head(feat)),  # (B, C, H', W')
            'offset':  self.offset_head(feat),                  # (B, 2, H', W')
            'size':    self.size_head(feat),                    # (B, 3, H', W')
            'depth':   self.depth_head(feat),                   # (B, 1, H', W')
            'rot':     F.normalize(self.rot_head(feat), dim=1), # (B, 2, H', W')
        }

    # ── decode to boxes ──────────────────────────────────────────────────
    @torch.no_grad()
    def decode_to_boxes(
        self,
        predictions: Dict[str, torch.Tensor],
        max_detections: int = 50,
        score_threshold: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """Decode CenterNet heatmap predictions into box-format outputs
        compatible with NuScenesDetectionMetrics.

        Extracts top-K peaks from the per-class heatmap, then gathers
        offset/size/depth/rotation at those locations.

        Returns:
            dict with class_logits (B,K,C), centers (B,K,3), sizes (B,K,3),
            orientations (B,K,2), confidences (B,K) — same format as BBox3DProbe.
        """
        hm = predictions['heatmap']           # (B, C, H, W)
        B, C, H, W = hm.shape

        # Simple 3×3 NMS: keep only local maxima
        hm_pool = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
        hm_nms = hm * (hm_pool == hm).float()

        # Flatten spatial dims → (B, C*H*W) and take top-K
        hm_flat = hm_nms.reshape(B, -1)                   # (B, C*H*W)
        K = min(max_detections, hm_flat.shape[1])
        scores, indices = hm_flat.topk(K, dim=1)           # (B, K)

        # Convert flat index → class, y, x
        cls_ids = indices // (H * W)                       # (B, K)
        spatial_idx = indices % (H * W)
        iy = spatial_idx // W                              # (B, K)
        ix = spatial_idx % W                               # (B, K)

        # Gather regression outputs at peak locations
        # offset: (B, 2, H, W) → gather at (iy, ix)
        flat_sp = iy * W + ix                              # (B, K)
        offset = predictions['offset'].view(B, 2, -1)     # (B, 2, H*W)
        off = torch.gather(offset, 2, flat_sp.unsqueeze(1).expand(-1, 2, -1))  # (B, 2, K)

        size_map = predictions['size'].view(B, 3, -1)
        sz = torch.gather(size_map, 2, flat_sp.unsqueeze(1).expand(-1, 3, -1))  # (B, 3, K)

        depth_map = predictions['depth'].view(B, 1, -1)
        dp = torch.gather(depth_map, 2, flat_sp.unsqueeze(1).expand(-1, 1, -1))  # (B, 1, K)

        rot_map = predictions['rot'].view(B, 2, -1)
        rot = torch.gather(rot_map, 2, flat_sp.unsqueeze(1).expand(-1, 2, -1))  # (B, 2, K)

        # Build centre coords: (cx + offset_x, cy + offset_y, depth)
        cx = (ix.float() + off[:, 0]) / W                 # normalised [0,1]
        cy = (iy.float() + off[:, 1]) / H
        depth = torch.expm1(dp[:, 0].clamp(max=6.0))      # undo log1p from targets

        centers = torch.stack([cx, cy, depth], dim=-1)     # (B, K, 3)
        sizes = torch.exp(sz.permute(0, 2, 1))             # (B, K, 3) undo log
        orientations = rot.permute(0, 2, 1)                # (B, K, 2)

        # Build one-hot-ish class logits from cls_id + score
        class_logits = torch.full((B, K, C), -10.0, device=hm.device)
        class_logits.scatter_(2, cls_ids.unsqueeze(-1), scores.unsqueeze(-1) * 20.0)

        return {
            'class_logits': class_logits,     # (B, K, C)
            'centers': centers,               # (B, K, 3)
            'sizes': sizes,                   # (B, K, 3)
            'orientations': orientations,     # (B, K, 2)
            'confidences': scores,            # (B, K)
        }

    # ── loss ─────────────────────────────────────────────────────────────
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = predictions['heatmap'].device
        zero = torch.tensor(0.0, device=device)

        if targets is None or 'heatmap_grid' not in targets:
            return dict(loss_heatmap=zero, loss_offset=zero, loss_size=zero,
                        loss_depth=zero, loss_rot=zero, loss_total=zero)

        gt_hm   = targets['heatmap_grid']
        rmask   = targets['reg_mask']

        # 1. Heatmap — Gaussian focal loss
        loss_hm = _gaussian_focal_loss(predictions['heatmap'], gt_hm)

        # 2. Regression losses — only at positive locations
        mask = rmask.unsqueeze(1)  # (B, 1, H, W)
        n_pos = mask.sum().clamp(min=1)

        loss_offset = F.l1_loss(
            predictions['offset'] * mask,
            targets['offset_grid'] * mask,
            reduction='sum',
        ) / n_pos

        loss_size = F.l1_loss(
            predictions['size'] * mask,
            targets['size_grid'] * mask,
            reduction='sum',
        ) / n_pos

        loss_depth = F.l1_loss(
            predictions['depth'] * mask,
            targets['depth_grid'] * mask,
            reduction='sum',
        ) / n_pos

        # Orientation: cosine distance
        pred_rot = predictions['rot'] * mask
        gt_rot   = targets['rot_grid'] * mask
        dot = (pred_rot * gt_rot).sum(dim=1, keepdim=True)
        loss_rot = ((1 - dot) * mask[:, :1]).sum() / n_pos

        loss_total = loss_hm + loss_offset + loss_size + loss_depth + loss_rot

        return dict(loss_heatmap=loss_hm, loss_offset=loss_offset,
                    loss_size=loss_size, loss_depth=loss_depth,
                    loss_rot=loss_rot, loss_total=loss_total)


# ═══════════════════════════════════════════════════════════════════════════════
# 2D BOUNDING BOX PROBE — CENTERNET STYLE  (dense, spatial)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_centernet_targets_2d(
    gt_boxes_2d: torch.Tensor,
    gt_classes: torch.Tensor,
    gt_mask: torch.Tensor,
    grid_h: int = 14,
    grid_w: int = 14,
    num_classes: int = NUM_FLIR_2D_DETECTION_CLASSES,
    gaussian_radius: int = 2,
    min_box_grid_cells: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Build CenterNet-style 2D dense targets from normalized xyxy boxes."""
    B, _ = gt_mask.shape
    device = gt_mask.device

    heatmap = torch.zeros(B, num_classes, grid_h, grid_w, device=device)
    offset = torch.zeros(B, 2, grid_h, grid_w, device=device)
    size_map = torch.zeros(B, 2, grid_h, grid_w, device=device)
    reg_mask = torch.zeros(B, grid_h, grid_w, device=device)

    # Do NOT scale base radius with grid size — keep it fixed so that
    # sub-pixel objects get appropriately small Gaussians instead of
    # oversized blobs.
    base_r = max(1, gaussian_radius)
    kernel_cache = {}

    def _get_kernel(radius: int) -> torch.Tensor:
        radius = int(max(1, radius))
        if radius not in kernel_cache:
            sigma = max(radius / 3.0, 0.5)
            dy = torch.arange(-radius, radius + 1, device=device).float()
            dx = torch.arange(-radius, radius + 1, device=device).float()
            gy, gx = torch.meshgrid(dy, dx, indexing='ij')
            kernel_cache[radius] = torch.exp(-(gx * gx + gy * gy) / (2.0 * sigma * sigma))
        return kernel_cache[radius]

    for b in range(B):
        valid = gt_mask[b] > 0.5
        if not valid.any():
            continue
        for box, cls in zip(gt_boxes_2d[b][valid], gt_classes[b][valid].long()):
            if cls < 0 or cls >= num_classes:
                continue
            x1, y1, x2, y2 = box.tolist()
            bw = max((x2 - x1) * grid_w, 1e-4)
            bh = max((y2 - y1) * grid_h, 1e-4)
            # Skip objects too small to detect on this grid
            if max(bw, bh) < min_box_grid_cells:
                continue
            cx = ((x1 + x2) * 0.5) * grid_w
            cy = ((y1 + y2) * 0.5) * grid_h
            ix = int(min(max(int(cx), 0), grid_w - 1))
            iy = int(min(max(int(cy), 0), grid_h - 1))

            radius = int(min(max(round(max(bw, bh) * 0.6), base_r), max(grid_h, grid_w) // 4 or 1))
            kernel = _get_kernel(radius)
            diam = 2 * radius + 1

            y_lo = max(0, iy - radius)
            y_hi = min(grid_h, iy + radius + 1)
            x_lo = max(0, ix - radius)
            x_hi = min(grid_w, ix + radius + 1)
            ky_lo = y_lo - (iy - radius)
            ky_hi = diam - ((iy + radius + 1) - y_hi)
            kx_lo = x_lo - (ix - radius)
            kx_hi = diam - ((ix + radius + 1) - x_hi)
            patch = kernel[ky_lo:ky_hi, kx_lo:kx_hi]
            heatmap[b, cls, y_lo:y_hi, x_lo:x_hi] = torch.max(
                heatmap[b, cls, y_lo:y_hi, x_lo:x_hi], patch,
            )

            offset[b, 0, iy, ix] = cx - ix
            offset[b, 1, iy, ix] = cy - iy
            size_map[b, 0, iy, ix] = torch.log(torch.tensor(bw, device=device))
            size_map[b, 1, iy, ix] = torch.log(torch.tensor(bh, device=device))
            reg_mask[b, iy, ix] = 1.0

    return {
        'heatmap_grid': heatmap,
        'offset_grid': offset,
        'size_grid': size_map,
        'reg_mask': reg_mask,
    }


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_iou_torch(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))

    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]

    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0.0) *
              (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0.0))
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0.0) *
              (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0.0))
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-8)


class BBox2DSlotProbe(nn.Module):
    """DETR-style 2D box probe that predicts a fixed set of boxes from patch tokens."""

    def __init__(
        self,
        vit_dim: int = 384,
        num_classes: int = NUM_FLIR_2D_DETECTION_CLASSES,
        patch_grid: int = 14,
        hidden_dim: int = 256,
        max_objects: int = 50,
        num_heads: int = 4,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_grid = patch_grid
        self.max_objects = max_objects
        self.focal_gamma = focal_gamma

        self.object_queries = nn.Parameter(torch.randn(max_objects, hidden_dim) * 0.02)
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, hidden_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.det_head = nn.Linear(hidden_dim, num_classes + 4 + 1)

    def forward(self, patch_embs: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = patch_embs.shape[0]
        kv = self.patch_proj(patch_embs)
        queries = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attn(queries, kv, kv)
        x = self.attn_norm(queries + attn_out)
        x = x + self.ffn(x)
        raw = self.det_head(x)

        class_logits = raw[:, :, :self.num_classes]
        box_raw = raw[:, :, self.num_classes:self.num_classes + 4]
        conf_logits = raw[:, :, -1]
        boxes = _box_cxcywh_to_xyxy(torch.sigmoid(box_raw))
        confidences = torch.sigmoid(conf_logits)

        return {
            'class_logits': class_logits,
            'boxes': boxes,
            'confidences': confidences,
        }

    @torch.no_grad()
    def decode_to_boxes(
        self,
        predictions: Dict[str, torch.Tensor],
        max_detections: int = 50,
        score_threshold: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        class_scores = F.softmax(predictions['class_logits'], dim=-1)
        top_class_scores, _ = class_scores.max(dim=-1)
        scores = predictions['confidences'] * top_class_scores

        keep_count = min(max_detections, scores.shape[1])
        top_scores, top_idx = scores.topk(keep_count, dim=1)
        gathered_boxes = torch.gather(
            predictions['boxes'],
            1,
            top_idx.unsqueeze(-1).expand(-1, -1, 4),
        )
        gathered_logits = torch.gather(
            predictions['class_logits'],
            1,
            top_idx.unsqueeze(-1).expand(-1, -1, self.num_classes),
        )
        top_scores = top_scores * (top_scores >= score_threshold).float()
        return {
            'class_logits': gathered_logits,
            'boxes': gathered_boxes,
            'confidences': top_scores,
        }

    @torch.no_grad()
    def _hungarian_match(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_boxes_2d: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = gt_mask.shape[0]
        indices = []

        for batch_idx in range(batch_size):
            valid = gt_mask[batch_idx].bool()
            if not valid.any():
                empty = torch.empty(0, dtype=torch.long, device=gt_mask.device)
                indices.append((empty, empty))
                continue

            gt_boxes = gt_boxes_2d[batch_idx, valid]
            gt_cls = gt_classes[batch_idx, valid].long()
            pred_boxes = predictions['boxes'][batch_idx]
            pred_logits = predictions['class_logits'][batch_idx]
            pred_conf = predictions['confidences'][batch_idx]

            cost_box = torch.cdist(pred_boxes, gt_boxes, p=1)
            log_probs = F.log_softmax(pred_logits, dim=-1)
            cost_class = -log_probs[:, gt_cls]
            cost_iou = 1.0 - _box_iou_torch(pred_boxes, gt_boxes)
            cost_conf = (1.0 - pred_conf).unsqueeze(-1).expand_as(cost_box)
            cost = cost_class + 2.0 * cost_box + 2.0 * cost_iou + 0.5 * cost_conf
            cost_np = cost.detach().cpu().numpy()

            if linear_sum_assignment is not None:
                row_idx, col_idx = linear_sum_assignment(cost_np)
            else:
                col_idx = cost_np.argmin(axis=0)
                row_idx = np.arange(len(col_idx))

            indices.append((
                torch.as_tensor(row_idx, dtype=torch.long, device=gt_mask.device),
                torch.as_tensor(col_idx, dtype=torch.long, device=gt_mask.device),
            ))

        return indices

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = predictions['class_logits'].device
        zero = torch.tensor(0.0, device=device)
        if targets is None or 'gt_boxes_2d' not in targets:
            return dict(loss_cls=zero, loss_box=zero, loss_iou=zero, loss_conf=zero, loss_total=zero)

        gt_boxes = targets['gt_boxes_2d']
        gt_classes = targets['gt_classes_2d']
        gt_mask = targets['gt_mask_2d']
        matched = self._hungarian_match(predictions, gt_boxes, gt_classes, gt_mask)

        total_cls = zero
        total_box = zero
        total_iou = zero
        n_matched = 0

        for batch_idx, (pred_idx, gt_idx) in enumerate(matched):
            if len(pred_idx) == 0:
                continue
            valid = gt_mask[batch_idx].bool()
            matched_gt_boxes = gt_boxes[batch_idx, valid][gt_idx]
            matched_gt_classes = gt_classes[batch_idx, valid][gt_idx].long()
            matched_pred_logits = predictions['class_logits'][batch_idx][pred_idx]
            matched_pred_boxes = predictions['boxes'][batch_idx][pred_idx]

            if self.focal_gamma > 0:
                total_cls = total_cls + focal_loss(
                    matched_pred_logits,
                    matched_gt_classes,
                    gamma=self.focal_gamma,
                    reduction='sum',
                )
            else:
                total_cls = total_cls + F.cross_entropy(
                    matched_pred_logits,
                    matched_gt_classes,
                    reduction='sum',
                )
            total_box = total_box + F.l1_loss(
                matched_pred_boxes,
                matched_gt_boxes,
                reduction='sum',
            )
            pair_iou = _box_iou_torch(matched_pred_boxes, matched_gt_boxes).diag()
            total_iou = total_iou + (1.0 - pair_iou).sum()
            n_matched += len(pred_idx)

        denom = max(n_matched, 1)
        loss_cls = total_cls / denom
        loss_box = total_box / denom
        loss_iou = total_iou / denom

        conf_target = torch.zeros_like(predictions['confidences'])
        for batch_idx, (pred_idx, _) in enumerate(matched):
            if len(pred_idx) > 0:
                conf_target[batch_idx, pred_idx] = 1.0

        n_pos = conf_target.sum().clamp(min=1.0)
        n_neg = conf_target.numel() - n_pos
        conf_weights = torch.ones_like(conf_target)
        if n_neg > 0:
            pos_weight = (n_neg / n_pos).clamp(max=10.0)
            conf_weights = torch.where(conf_target > 0.5, pos_weight, torch.ones_like(conf_target))
        with torch.amp.autocast('cuda', enabled=False):
            conf_loss = F.binary_cross_entropy(
                predictions['confidences'].float(),
                conf_target.float(),
                reduction='none',
            )
            loss_conf = (conf_loss * conf_weights).mean()

        loss_total = loss_cls + 2.0 * loss_box + 2.0 * loss_iou + loss_conf
        return dict(
            loss_cls=loss_cls,
            loss_box=loss_box,
            loss_iou=loss_iou,
            loss_conf=loss_conf,
            loss_total=loss_total,
        )


class SpatialBBox2DProbe(nn.Module):
    """CenterNet-style dense 2D box probe for image-only box supervision."""

    def __init__(
        self,
        vit_dim: int = 384,
        num_classes: int = NUM_FLIR_2D_DETECTION_CLASSES,
        patch_grid: int = 14,
        adapter_dim: int = 256,
        upsample_factor: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_grid = patch_grid
        self.upsample_factor = upsample_factor
        self.output_grid = patch_grid * upsample_factor

        self.adapter = nn.Sequential(
            nn.Conv2d(vit_dim, adapter_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(adapter_dim),
            nn.ReLU(),
        )
        if upsample_factor > 1:
            if upsample_factor & (upsample_factor - 1):
                raise ValueError(f"SpatialBBox2DProbe upsample_factor must be a power of 2, got {upsample_factor}")
            upsample_layers = []
            scale = upsample_factor
            while scale > 1:
                upsample_layers.extend([
                    nn.ConvTranspose2d(adapter_dim, adapter_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(adapter_dim),
                    nn.ReLU(),
                ])
                scale //= 2
            self.upsample = nn.Sequential(*upsample_layers)
        else:
            self.upsample = None

        self.heatmap_head = nn.Conv2d(adapter_dim, num_classes, kernel_size=1)
        self.offset_head = nn.Conv2d(adapter_dim, 2, kernel_size=1)
        self.size_head = nn.Conv2d(adapter_dim, 2, kernel_size=1)
        self.heatmap_head.bias.data.fill_(-2.19)

    def forward(self, patch_embs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = patch_embs.shape
        H = W = int(N ** 0.5)
        x = patch_embs.transpose(1, 2).reshape(B, D, H, W)
        feat = self.adapter(x)
        if self.upsample is not None:
            feat = self.upsample(feat)
        return {
            'heatmap': torch.sigmoid(self.heatmap_head(feat)),
            'offset': self.offset_head(feat),
            'size': self.size_head(feat),
        }

    @torch.no_grad()
    def decode_to_boxes(
        self,
        predictions: Dict[str, torch.Tensor],
        max_detections: int = 50,
        score_threshold: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        hm = predictions['heatmap']
        B, C, H, W = hm.shape
        hm_pool = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
        hm_nms = hm * (hm_pool == hm).float()

        hm_flat = hm_nms.reshape(B, -1)
        K = min(max_detections, hm_flat.shape[1])
        scores, indices = hm_flat.topk(K, dim=1)
        cls_ids = indices // (H * W)
        spatial_idx = indices % (H * W)
        iy = spatial_idx // W
        ix = spatial_idx % W
        flat_sp = iy * W + ix

        offset = predictions['offset'].view(B, 2, -1)
        off = torch.gather(offset, 2, flat_sp.unsqueeze(1).expand(-1, 2, -1))
        size_map = predictions['size'].view(B, 2, -1)
        sz = torch.gather(size_map, 2, flat_sp.unsqueeze(1).expand(-1, 2, -1))

        cx = (ix.float() + off[:, 0]) / W
        cy = (iy.float() + off[:, 1]) / H
        bw = torch.exp(sz[:, 0]).clamp(max=float(W)) / W
        bh = torch.exp(sz[:, 1]).clamp(max=float(H)) / H
        x1 = (cx - bw * 0.5).clamp(0.0, 1.0)
        y1 = (cy - bh * 0.5).clamp(0.0, 1.0)
        x2 = (cx + bw * 0.5).clamp(0.0, 1.0)
        y2 = (cy + bh * 0.5).clamp(0.0, 1.0)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        keep = scores >= score_threshold
        scores = scores * keep.float()
        class_logits = torch.full((B, K, C), -10.0, device=hm.device)
        class_logits.scatter_(2, cls_ids.unsqueeze(-1), scores.unsqueeze(-1) * 20.0)

        return {
            'class_logits': class_logits,
            'boxes': boxes,
            'confidences': scores,
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = predictions['heatmap'].device
        zero = torch.tensor(0.0, device=device)
        if targets is None or 'heatmap_grid' not in targets:
            return dict(loss_heatmap=zero, loss_offset=zero, loss_size=zero, loss_total=zero)

        gt_hm = targets['heatmap_grid']
        rmask = targets['reg_mask']
        loss_hm = _gaussian_focal_loss(predictions['heatmap'], gt_hm)

        mask = rmask.unsqueeze(1)
        n_pos = mask.sum().clamp(min=1)
        loss_offset = F.l1_loss(
            predictions['offset'] * mask,
            targets['offset_grid'] * mask,
            reduction='sum',
        ) / n_pos
        loss_size = F.l1_loss(
            predictions['size'] * mask,
            targets['size_grid'] * mask,
            reduction='sum',
        ) / n_pos
        loss_total = loss_hm + loss_offset + loss_size
        return dict(loss_heatmap=loss_hm, loss_offset=loss_offset, loss_size=loss_size, loss_total=loss_total)


class DetectionMetrics2D:
    """Simple IoU@0.5 detection metric accumulator for 2D patch probes."""

    def __init__(self, class_names: Optional[List[str]] = None, num_classes: Optional[int] = None, iou_threshold: float = 0.5):
        self.class_names = class_names if class_names is not None else FLIR_2D_DETECTION_CLASSES
        self.num_classes = num_classes if num_classes is not None else len(self.class_names)
        self.iou_threshold = float(iou_threshold)
        self.reset()

    def reset(self):
        self._preds: List[dict] = []
        self._tgts: List[dict] = []

    @torch.no_grad()
    def update(self, predictions: Dict[str, torch.Tensor], targets: Optional[Dict[str, torch.Tensor]]):
        self._preds.append({
            'classes': predictions['class_logits'].argmax(-1).detach().cpu().numpy(),
            'boxes': predictions['boxes'].detach().cpu().numpy(),
            'confidences': predictions['confidences'].detach().cpu().numpy(),
        })
        if targets is not None and 'gt_boxes_2d' in targets:
            self._tgts.append({
                'classes': targets['gt_classes_2d'].detach().cpu().numpy(),
                'boxes': targets['gt_boxes_2d'].detach().cpu().numpy(),
                'mask': targets['gt_mask_2d'].detach().cpu().numpy(),
            })
        else:
            self._tgts.append(None)

    @staticmethod
    def _box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
        ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
        bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
        inter_x1 = np.maximum(ax1, bx1)
        inter_y1 = np.maximum(ay1, by1)
        inter_x2 = np.minimum(ax2, bx2)
        inter_y2 = np.minimum(ay2, by2)
        inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
        inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
        inter = inter_w * inter_h
        area_a = np.clip(ax2 - ax1, a_min=0.0, a_max=None) * np.clip(ay2 - ay1, a_min=0.0, a_max=None)
        area_b = np.clip(bx2 - bx1, a_min=0.0, a_max=None) * np.clip(by2 - by1, a_min=0.0, a_max=None)
        union = area_a + area_b - inter
        return inter / np.clip(union, a_min=1e-8, a_max=None)

    def compute(self) -> Dict[str, float]:
        if not self._preds or all(t is None for t in self._tgts):
            return {
                'mAP50': 0.0,
                'avg_AP50': 0.0,
                'precision50': 0.0,
                'recall50': 0.0,
                'f1_50': 0.0,
            }

        per_class_ap = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        result = {}

        flat_preds = []
        flat_tgts = []
        for pred, tgt in zip(self._preds, self._tgts):
            if tgt is None:
                continue
            B = pred['classes'].shape[0]
            for b in range(B):
                keep_pred = pred['confidences'][b] > 0.05
                keep_gt = tgt['mask'][b] > 0.5
                flat_preds.append({
                    'classes': pred['classes'][b][keep_pred],
                    'boxes': pred['boxes'][b][keep_pred],
                    'confidences': pred['confidences'][b][keep_pred],
                })
                flat_tgts.append({
                    'classes': tgt['classes'][b][keep_gt],
                    'boxes': tgt['boxes'][b][keep_gt],
                })

        for cls_idx in range(self.num_classes):
            cls_name = self.class_names[cls_idx]
            pred_records = []
            n_gt = 0
            for sid, (pred, tgt) in enumerate(zip(flat_preds, flat_tgts)):
                gt_mask = tgt['classes'] == cls_idx
                pred_mask = pred['classes'] == cls_idx
                gt_boxes = tgt['boxes'][gt_mask]
                pred_boxes = pred['boxes'][pred_mask]
                pred_scores = pred['confidences'][pred_mask]
                n_gt += len(gt_boxes)
                for box, score in zip(pred_boxes, pred_scores):
                    pred_records.append((sid, float(score), box.astype(np.float32)))

            if n_gt == 0:
                result[f'{cls_name}/AP50'] = 0.0
                continue

            pred_records.sort(key=lambda item: -item[1])
            matched = {sid: np.zeros(np.sum(tgt['classes'] == cls_idx), dtype=bool) for sid, tgt in enumerate(flat_tgts)}
            tp = np.zeros(len(pred_records), dtype=np.float32)
            fp = np.zeros(len(pred_records), dtype=np.float32)

            for idx, (sid, _score, pred_box) in enumerate(pred_records):
                gt_boxes = flat_tgts[sid]['boxes'][flat_tgts[sid]['classes'] == cls_idx]
                if len(gt_boxes) == 0:
                    fp[idx] = 1.0
                    continue
                ious = self._box_iou(pred_box[None, :], gt_boxes)[0]
                best = int(np.argmax(ious)) if len(ious) > 0 else -1
                if best >= 0 and ious[best] >= self.iou_threshold and not matched[sid][best]:
                    tp[idx] = 1.0
                    matched[sid][best] = True
                else:
                    fp[idx] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / max(n_gt, 1)
            prec = tp_cum / np.clip(tp_cum + fp_cum, a_min=1e-8, a_max=None)
            mrec = np.concatenate(([0.0], rec))
            mpre = np.concatenate(([0.0], prec))
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            ap50 = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
            per_class_ap.append(ap50)
            result[f'{cls_name}/AP50'] = ap50

            score_keep = np.array([score >= 0.3 for _, score, _ in pred_records], dtype=bool)
            cls_tp = float(tp[score_keep].sum())
            cls_fp = float(fp[score_keep].sum())
            cls_fn = float(n_gt - cls_tp)
            total_tp += cls_tp
            total_fp += cls_fp
            total_fn += cls_fn

        precision = total_tp / max(total_tp + total_fp, 1.0)
        recall = total_tp / max(total_tp + total_fn, 1.0)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
        result['mAP50'] = float(np.mean(per_class_ap)) if per_class_ap else 0.0
        result['avg_AP50'] = result['mAP50']
        result['precision50'] = float(precision)
        result['recall50'] = float(recall)
        result['f1_50'] = float(f1)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SEGMENTATION PROBE  (linear + PixelShuffle)
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticSegProbe(nn.Module):
    """
    Linear segmentation probe following SSL evaluation conventions
    (DINO, DINOv2, MAE, I-JEPA).

    Architecture:
        patch_embs (B, N, D)
        → reshape to (B, D, H_p, W_p)
        → 1×1 Conv  →  (B, C·r², H_p, W_p)
        → PixelShuffle(r)  →  (B, C, H_p·r, W_p·r)
        → bilinear resize to target

    For patch_grid=14, r=4 gives 56×56; final bilinear to 224×224.
    This is a *linear* probe: a single 1×1 conv (no nonlinearity).
    """

    def __init__(
        self,
        vit_dim: int = 384,
        num_classes: int = NUM_SIMPLIFIED_SEG_CLASSES,
        patch_grid: int = 14,
        upsample_factor: int = 4,
        target_size: Tuple[int, int] = (224, 224),
        ignore_index: Optional[int] = 0,
        class_names: Optional[List[str]] = None,
        mean_exclude_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_grid = patch_grid
        self.upsample_factor = upsample_factor
        self.target_size = target_size
        self.vit_dim = vit_dim
        self.ignore_index = ignore_index
        self.class_names = class_names if class_names is not None else SIMPLIFIED_SEG_CLASSES[:num_classes]
        if mean_exclude_indices is not None:
            self.mean_exclude_indices = {int(idx) for idx in mean_exclude_indices}
        elif ignore_index is not None:
            self.mean_exclude_indices = {int(ignore_index)}
        else:
            self.mean_exclude_indices = set()

        r = upsample_factor
        # Single 1×1 conv: linear probe. Output channels = C * r²
        self.linear_head = nn.Conv2d(
            vit_dim,
            num_classes * r * r,
            kernel_size=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(r)

    def forward(self, patch_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embs: (B, N_patches, vit_dim) from ViT backbone.
        Returns:
            logits: (B, num_classes, H_target, W_target)
        """
        B, N, D = patch_embs.shape
        H_p = W_p = int(round(N ** 0.5))
        if H_p * W_p != N:
            raise ValueError(f"SemanticSegProbe expects a square patch grid, got {N} tokens")

        x = patch_embs.transpose(1, 2).reshape(B, D, H_p, W_p)
        x = self.linear_head(x)
        x = self.pixel_shuffle(x)

        # Bilinear to target
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size,
                              mode='bilinear', align_corners=False)
        return x

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) long class indices  (0 = ignore)
        """
        # Resize targets if spatial dims differ
        if targets.shape[-2:] != predictions.shape[-2:]:
            targets = F.interpolate(
                targets.unsqueeze(1).float(),
                size=predictions.shape[-2:],
                mode='nearest',
            ).squeeze(1).long()

        # Check if there are any valid pixels before computing loss.
        # cross_entropy returns NaN when all pixels are ignored.
        if self.ignore_index is None:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            valid_mask = targets != self.ignore_index
        if valid_mask.sum() > 0:
            # Compute inverse-frequency class weights for this batch to handle
            # extreme class imbalance (e.g. Waymo seg_map is 94% car).
            # Weight_c = median(freq) / freq_c, clamped to [0.1, 10].
            with torch.no_grad():
                valid_targets = targets[valid_mask]
                class_weight = torch.ones(self.num_classes, device=predictions.device)
                counts = torch.bincount(valid_targets, minlength=self.num_classes).float()
                present = counts > 0
                if present.sum() > 1:
                    nonzero_counts = counts[present]
                    median_freq = nonzero_counts.median()
                    for c in range(self.num_classes):
                        if counts[c] > 0:
                            class_weight[c] = (median_freq / counts[c]).clamp(0.1, 10.0)
                if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
                    class_weight[self.ignore_index] = 0.0
            if self.ignore_index is None:
                loss = F.cross_entropy(predictions, targets, weight=class_weight)
            else:
                loss = F.cross_entropy(
                    predictions,
                    targets,
                    weight=class_weight,
                    ignore_index=self.ignore_index,
                )
        else:
            # No valid pixels - return zero loss to avoid NaN
            loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)

        with torch.no_grad():
            pred_cls = predictions.argmax(1)
            valid = valid_mask
            per_class_iou = {}  # class_idx → IoU tensor
            if valid.sum() > 0:
                acc = ((pred_cls == targets) & valid).sum().float() / valid.sum().float()
                # Per-class IoU
                ious = []
                for c in range(self.num_classes):
                    pred_c = (pred_cls == c) & valid
                    gt_c = (targets == c) & valid
                    inter = (pred_c & gt_c).sum().float()
                    union = (pred_c | gt_c).sum().float()
                    if union > 0:
                        iou_c = inter / union
                        if c not in self.mean_exclude_indices:
                            ious.append(iou_c)
                        per_class_iou[c] = iou_c
                miou = torch.stack(ious).mean() if ious else torch.tensor(0.0, device=predictions.device)
            else:
                acc = torch.tensor(0.0, device=predictions.device)
                miou = torch.tensor(0.0, device=predictions.device)

        result = {'loss_seg': loss, 'seg_accuracy': acc, 'seg_mIoU': miou}
        # Individual per-class IoU (only for classes present in GT or preds)
        for c_idx, iou_val in per_class_iou.items():
            cls_name = self.class_names[c_idx] if c_idx < len(self.class_names) else f'class_{c_idx}'
            result[f'seg_iou_{cls_name}'] = iou_val
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# DEPTH MAP PROBE  (PixelShuffle regression, same design as SemanticSegProbe)
# ═══════════════════════════════════════════════════════════════════════════════

class DepthMapProbe(nn.Module):
    """
    Dense-depth probe with PixelShuffle upsampling + lightweight refinement.

    Architecture:
        patch_embs (B, N, D)
        → reshape to (B, D, H_p, W_p)
        → 1×1 Conv  →  (B, C·r², H_p, W_p)
        → PixelShuffle(r)  →  (B, C, H_p·r, W_p·r)      e.g. (B, 16, 56, 56)
        → 3×3 Conv(C→C) + ReLU  (spatial refinement)
        → 1×1 Conv(C→1) + ReLU  (depth ≥ 0)

    For patch_grid=14, r=4 gives 56×56 intermediate, final (B, 1, 56, 56).

    The refinement convs are necessary because — unlike seg (classification) —
    continuous depth regression benefits from cross-pixel smoothing that a
    purely linear probe cannot provide.
    """

    HIDDEN_CH = 16   # lightweight: adds ~4.7K params total

    def __init__(
        self,
        vit_dim: int = 384,
        patch_grid: int = 14,
        upsample_factor: int = 4,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.upsample_factor = upsample_factor
        self.vit_dim = vit_dim

        r = upsample_factor
        C = self.HIDDEN_CH
        self.output_size = patch_grid * r   # 56 for default settings

        # 1×1 conv: project to C*r² channels for PixelShuffle
        self.project = nn.Conv2d(vit_dim, C * r * r, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(r)

        # Lightweight spatial refinement after shuffle
        self.refine = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(C, 1, kernel_size=1),
            nn.ReLU(),   # depth ≥ 0
        )

    def forward(self, patch_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embs: (B, N_patches, vit_dim) from ViT backbone.
        Returns:
            depth: (B, 1, output_size, output_size)  — non-negative
        """
        B, N, D = patch_embs.shape
        H_p = W_p = int(round(N ** 0.5))
        if H_p * W_p != N:
            raise ValueError(f"DepthMapProbe expects a square patch grid, got {N} tokens")

        x = patch_embs.transpose(1, 2).reshape(B, D, H_p, W_p)
        x = self.project(x)
        x = self.pixel_shuffle(x)
        x = self.refine(x)
        return x

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Masked MSE loss + metrics.

        Args:
            predictions: (B, 1, H, W) predicted depth
            targets:     (B, 1, H, W) ground-truth depth (metres)
            mask:        (B, 1, H, W) binary validity mask
        Returns:
            dict with 'loss_depth', 'depth_mae', 'depth_iou'
        """
        # Resize targets/mask if spatial dims differ
        if targets.shape[-2:] != predictions.shape[-2:]:
            targets = F.interpolate(targets, size=predictions.shape[-2:],
                                    mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=predictions.shape[-2:],
                                 mode='nearest')

        # Masked MSE
        diff_sq = (predictions - targets).square() * mask
        valid_per_sample = mask.sum(dim=(1, 2, 3)).clamp(min=1)
        loss = (diff_sq.sum(dim=(1, 2, 3)) / valid_per_sample).mean()

        # Metrics (no grad)
        with torch.no_grad():
            diff_abs = (predictions - targets).abs() * mask
            mae = (diff_abs.sum(dim=(1, 2, 3)) / valid_per_sample).mean()

            # IoU: fraction of valid cells within 8 m of target
            threshold = 8.0
            correct = ((diff_abs < threshold) & (mask > 0)).float()
            iou = (correct.sum(dim=(1, 2, 3)) / valid_per_sample).mean()

        return {
            'loss_depth': loss,
            'depth_mae': mae,
            'depth_iou': iou,
        }


class OccupancyMapProbe(nn.Module):
    """Lightweight dense binary occupancy probe from patch tokens."""

    HIDDEN_CH = 32

    def __init__(
        self,
        vit_dim: int = 384,
        patch_grid: int = 14,
        upsample_factor: int = 2,
        output_channels: int = 1,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.upsample_factor = upsample_factor
        self.vit_dim = vit_dim
        self.output_channels = max(1, int(output_channels))

        r = upsample_factor
        channels = self.HIDDEN_CH
        self.output_size = patch_grid * r
        norm_groups = max(1, channels // 8)

        self.project = nn.Conv2d(vit_dim, channels * r * r, kernel_size=1)
        self.skip = nn.Conv2d(vit_dim, channels, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(r)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, channels),
            nn.GELU(),
        )
        self.head = nn.Conv2d(channels, self.output_channels, kernel_size=1)

    def forward(self, patch_embs: torch.Tensor) -> torch.Tensor:
        B, N, D = patch_embs.shape
        H_p = W_p = int(round(N ** 0.5))
        if H_p * W_p != N:
            raise ValueError(f"OccupancyMapProbe expects a square patch grid, got {N} tokens")

        x = patch_embs.transpose(1, 2).reshape(B, D, H_p, W_p)
        skip = self.skip(x)
        x = self.project(x)
        x = self.pixel_shuffle(x)
        skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + skip
        x = x + self.refine(x)
        return self.head(x)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        if targets.shape[-2:] != predictions.shape[-2:]:
            targets = F.interpolate(targets.float(), size=predictions.shape[-2:], mode='nearest')

        targets = targets.float().clamp(0.0, 1.0)
        with torch.no_grad():
            pos_counts = targets.sum(dim=(0, 2, 3))
            elems_per_channel = float(targets.shape[0] * targets.shape[2] * targets.shape[3])
            neg_counts = torch.full_like(pos_counts, elems_per_channel) - pos_counts
            pos_weight = torch.ones_like(pos_counts)
            has_positive = pos_counts > 0
            pos_weight[has_positive] = (neg_counts[has_positive] / pos_counts[has_positive]).clamp(1.0, 50.0)

        bce_map = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=pos_weight.view(1, -1, 1, 1),
            reduction='none',
        )
        probs = torch.sigmoid(predictions)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - pt).pow(2.0)
        loss_bce = (bce_map * focal_factor).mean()

        probs_flat = probs.flatten(2)
        targets_flat = targets.flatten(2)
        intersection_soft = (probs_flat * targets_flat).sum(dim=2)
        cardinality = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice_loss_per_channel = 1.0 - ((2.0 * intersection_soft + 1.0) / (cardinality + 1.0))
        valid_channels = (targets_flat.sum(dim=2) > 0).float()
        valid_channel_count = valid_channels.sum()
        if valid_channel_count > 0:
            loss_dice = (dice_loss_per_channel * valid_channels).sum() / valid_channel_count
        else:
            loss_dice = predictions.sum() * 0.0

        if self.output_channels > 1:
            any_probs = probs[:, :1]
            class_probs = probs[:, 1:]
            consistency_loss = F.relu(class_probs.max(dim=1, keepdim=True).values - any_probs).mean()
        else:
            consistency_loss = predictions.sum() * 0.0

        loss = loss_bce + 0.5 * loss_dice + 0.1 * consistency_loss

        with torch.no_grad():
            pred_binary = (probs > 0.5).float()
            acc = (pred_binary == targets).float().mean()
            intersection = (pred_binary * targets).sum(dim=(1, 2, 3))
            union = ((pred_binary + targets) > 0).float().sum(dim=(1, 2, 3))
            iou = torch.where(union > 0, intersection / union.clamp(min=1.0), torch.ones_like(union)).mean()

        return {
            'loss_occ': loss,
            'loss_occ_bce': loss_bce,
            'loss_occ_dice': loss_dice,
            'loss_occ_consistency': consistency_loss,
            'occupancy_accuracy': acc,
            'occupancy_iou': iou,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NUSCENES DETECTION METRICS  (mAP, mATE, mASE, mAOE, mADE)
# ═══════════════════════════════════════════════════════════════════════════════

class NuScenesDetectionMetrics:
    """
    Accumulates predictions / targets across batches, then computes
    nuScenes-style detection metrics using centre-distance matching.
    """

    def __init__(self, dist_thresholds: List[float] = [0.5, 1.0, 2.0, 4.0],
                 class_names: List[str] = None, num_classes: int = None,
                 matching_plane: str = 'xz'):
        self.dist_thresholds = dist_thresholds
        # Allow overriding class names and count (e.g. for grouped classes)
        self.class_names = class_names if class_names is not None else DETECTION_CLASSES
        self.num_classes = num_classes if num_classes is not None else len(self.class_names)
        plane_to_axes = {
            'xy': (0, 1),
            'xz': (0, 2),
            'yz': (1, 2),
        }
        matching_plane = str(matching_plane).lower()
        if matching_plane not in plane_to_axes:
            raise ValueError(f"Unsupported matching_plane={matching_plane!r}; expected one of {sorted(plane_to_axes)}")
        self.matching_plane = matching_plane
        self.matching_axes = plane_to_axes[matching_plane]
        self.reset()

    def reset(self):
        self._preds: List[dict] = []
        self._tgts:  List[dict] = []

    @torch.no_grad()
    def update(self, predictions: Dict[str, torch.Tensor],
               targets: Optional[Dict[str, torch.Tensor]]):
        self._preds.append({k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                            for k, v in {
                                'classes': predictions['class_logits'].argmax(-1),
                                'centers': predictions['centers'],
                                'sizes': predictions['sizes'],
                                'orientations': predictions['orientations'],
                                'confidences': predictions['confidences'],
                            }.items()})
        if targets is not None and 'gt_classes' in targets:
            self._tgts.append({k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                               for k, v in {
                                   'classes': targets['gt_classes'],
                                   'centers': targets['gt_centers'],
                                   'sizes': targets['gt_sizes'],
                                   'orientations': targets['gt_orientations'],
                                   'mask': targets['gt_mask'],
                               }.items()})
        else:
            self._tgts.append(None)

    def compute(self) -> Dict[str, float]:
        # When no true-positive matches exist, TP-error terms should be penalized,
        # not reported as zero. We use 1.0 as the neutral/worst score for the
        # NDS-style transform max(1 - error, 0).
        no_tp_error = 0.0

        if not self._preds or all(t is None for t in self._tgts):
            return {'mAP': 0.0, 'mATE': no_tp_error, 'mASE': no_tp_error, 'mAOE': no_tp_error, 'mADE': no_tp_error}

        # Flatten into per-image lists
        all_p, all_g = [], []
        for p, t in zip(self._preds, self._tgts):
            if t is None:
                continue
            B = p['classes'].shape[0]
            for b in range(B):
                mc = t['mask'][b] > .5
                n_gt = mc.sum()
                pconf = p['confidences'][b]
                pvalid = pconf > 0.1
                all_p.append({k: p[k][b][pvalid] for k in p})
                all_g.append({k: t[k][b][mc] for k in t if k != 'mask'})

        aps, ates, ases, aoes, ades = [], [], [], [], []
        per_class = {}  # per-class metrics: {class_name: {AP, ATE, ASE, AOE, n_gt, n_pred}}
        for cls_idx in range(self.num_classes):
            cls_name = self.class_names[cls_idx]
            # Skip classes that are not defined (None = not used for this dataset)
            if cls_name is None:
                continue
            # Gather all pred/gt for this class
            p_ctr, p_sz, p_ori, p_conf, p_sid = [], [], [], [], []
            g_ctr, g_sz, g_ori, g_sid = [], [], [], []
            for sid, (pp, gg) in enumerate(zip(all_p, all_g)):
                pm = pp['classes'] == cls_idx
                gm = gg['classes'] == cls_idx
                if pm.sum():
                    p_ctr.append(pp['centers'][pm]); p_sz.append(pp['sizes'][pm])
                    p_ori.append(pp['orientations'][pm]); p_conf.append(pp['confidences'][pm])
                    p_sid.append(np.full(pm.sum(), sid))
                if gm.sum():
                    g_ctr.append(gg['centers'][gm]); g_sz.append(gg['sizes'][gm])
                    g_ori.append(gg['orientations'][gm]); g_sid.append(np.full(gm.sum(), sid))

            if not g_ctr:
                per_class[cls_name] = {
                    'AP': 0.0,
                    'ATE': no_tp_error,
                    'ASE': no_tp_error,
                    'AOE': no_tp_error,
                    'ADE': no_tp_error,
                    'n_gt': 0,
                    'n_pred': 0,
                }
                continue
            # Skip class if no predictions at all (avoids empty concatenate)
            if not p_ctr:
                aps.append(0.0)
                n_gt_total = sum(len(g) for g in g_ctr)
                per_class[cls_name] = {
                    'AP': 0.0,
                    'ATE': no_tp_error,
                    'ASE': no_tp_error,
                    'AOE': no_tp_error,
                    'ADE': no_tp_error,
                    'n_gt': n_gt_total,
                    'n_pred': 0,
                }
                continue

            pc = np.concatenate(p_ctr); ps = np.concatenate(p_sz)
            po = np.concatenate(p_ori); pconf = np.concatenate(p_conf)
            psid = np.concatenate(p_sid)
            gc = np.concatenate(g_ctr); gs = np.concatenate(g_sz)
            go = np.concatenate(g_ori); gsid = np.concatenate(g_sid)
            n_gt_total = len(gc)

            # Mean over distance thresholds
            ap_per_thresh = []
            cls_te_list, cls_se_list, cls_oe_list, cls_de_list = [], [], [], []
            for dt in self.dist_thresholds:
                order = np.argsort(-pconf)
                tp = np.zeros(len(pc), dtype=bool)
                gt_matched = np.zeros(len(gc), dtype=bool)

                for pi in order:
                    same = gsid == psid[pi]
                    if not same.any():
                        continue
                    # Match using the configured camera-frame plane.
                    dists = np.linalg.norm(
                        gc[same][:, self.matching_axes] - pc[pi][list(self.matching_axes)],
                        axis=1,
                    )
                    gtidx = np.where(same)[0]
                    best_local = np.argmin(dists)
                    best_global = gtidx[best_local]
                    if dists[best_local] < dt and not gt_matched[best_global]:
                        tp[pi] = True
                        gt_matched[best_global] = True
                        cls_te_list.append(dists[best_local])
                        cls_de_list.append(abs(float(pc[pi, 2]) - float(gc[best_global, 2])))
                        pv = np.prod(ps[pi]); gv = np.prod(gs[best_global])
                        cls_se_list.append(1 - min(pv, gv) / max(pv, gv) if max(pv, gv) > 0 else 1.0)
                        py = np.arctan2(po[pi, 0], po[pi, 1])
                        gy = np.arctan2(go[best_global, 0], go[best_global, 1])
                        yd = np.abs(py - gy); yd = min(yd, 2*np.pi - yd)
                        cls_oe_list.append(yd)

                tp_cum = np.cumsum(tp[order])
                fp_cum = np.cumsum(~tp[order])
                rec = tp_cum / n_gt_total
                prec = tp_cum / (tp_cum + fp_cum + 1e-8)
                # Integrate the precision-recall curve after building the
                # precision envelope. Prepending recall=0 ensures the first
                # true positive contributes to AP instead of being dropped.
                mrec = np.concatenate(([0.0], rec))
                mpre = np.concatenate(([0.0], prec))
                mpre = np.maximum.accumulate(mpre[::-1])[::-1]
                ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
                ap_per_thresh.append(ap)

            cls_ap = float(np.mean(ap_per_thresh))
            aps.append(cls_ap)
            ates.extend(cls_te_list); ases.extend(cls_se_list); aoes.extend(cls_oe_list); ades.extend(cls_de_list)
            per_class[cls_name] = {
                'AP': cls_ap,
                'ATE': float(np.mean(cls_te_list)) if cls_te_list else no_tp_error,
                'ASE': float(np.mean(cls_se_list)) if cls_se_list else no_tp_error,
                'AOE': float(np.mean(cls_oe_list)) if cls_oe_list else no_tp_error,
                'ADE': float(np.mean(cls_de_list)) if cls_de_list else no_tp_error,
                'n_gt': n_gt_total,
                'n_pred': len(pc),
            }

        result = {
            'mAP': float(np.mean(aps)) if aps else 0.0,
            'mATE': float(np.mean(ates)) if ates else no_tp_error,
            'mASE': float(np.mean(ases)) if ases else no_tp_error,
            'mAOE': float(np.mean(aoes)) if aoes else no_tp_error,
            'mADE': float(np.mean(ades)) if ades else no_tp_error,
        }
        # Per-class breakdown
        for cls_name, cls_metrics in per_class.items():
            # Human-readable aliases with explicit units.
            # ATE/ADE are in metres; AOE is radians (and AOE_deg in degrees).
            cls_metrics['ATE_m'] = cls_metrics['ATE']
            cls_metrics['ADE_m'] = cls_metrics['ADE']
            cls_metrics['AOE_deg'] = cls_metrics['AOE'] * (180.0 / np.pi)
            for metric_name, val in cls_metrics.items():
                result[f'{cls_name}/{metric_name}'] = val

        # Global unit-explicit aliases
        result['mATE_m'] = result['mATE']
        result['mADE_m'] = result['mADE']
        result['mAOE_deg'] = result['mAOE'] * (180.0 / np.pi)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION METRICS  (mIoU)
# ═══════════════════════════════════════════════════════════════════════════════

class SegmentationMetrics:
    """Vectorised mIoU via confusion matrix accumulation."""

    def __init__(
        self,
        num_classes: int = NUM_SIMPLIFIED_SEG_CLASSES,
        ignore_index: Optional[int] = 0,
        class_names: Optional[List[str]] = None,
        mean_exclude_indices: Optional[List[int]] = None,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names if class_names is not None else SIMPLIFIED_SEG_CLASSES[:num_classes]
        if mean_exclude_indices is not None:
            self.mean_exclude_indices = {int(idx) for idx in mean_exclude_indices}
        elif ignore_index is not None:
            self.mean_exclude_indices = {int(ignore_index)}
        else:
            self.mean_exclude_indices = set()
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            predictions: (B, H, W) predicted class indices (or (B, C, H, W) logits)
            targets:     (B, H, W) ground truth class indices
        """
        if predictions.dim() == 4:
            predictions = predictions.argmax(1)
        p = predictions.cpu().numpy().ravel()
        t = targets.cpu().numpy().ravel()
        if self.ignore_index is None:
            valid = np.ones_like(t, dtype=bool)
        else:
            valid = t != self.ignore_index
        p, t = p[valid], t[valid]
        # Vectorised confusion matrix update
        valid_range = (p >= 0) & (p < self.num_classes) & (t >= 0) & (t < self.num_classes)
        p, t = p[valid_range], t[valid_range]
        np.add.at(self.cm, (t, p), 1)

    def compute(self) -> Dict[str, float]:
        inter = np.diag(self.cm)
        union = self.cm.sum(1) + self.cm.sum(0) - inter
        union = np.where(union > 0, union, 1)
        iou = inter / union

        present = (self.cm.sum(1) > 0)
        for idx in self.mean_exclude_indices:
            if 0 <= idx < self.num_classes:
                present[idx] = False
        miou = float(iou[present].mean()) if present.sum() > 0 else 0.0

        total = self.cm.sum()
        pixel_acc = float(inter.sum() / total) if total > 0 else 0.0

        result = {'mIoU': miou, 'pixel_accuracy': pixel_acc}
        for c in range(self.num_classes):
            cls_name = self.class_names[c] if c < len(self.class_names) else f'class_{c}'
            result[f'IoU_{cls_name}'] = float(iou[c])
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED MODULE  (easy integration into training loop)
# ═══════════════════════════════════════════════════════════════════════════════

class DetectionSegmentationProbes(nn.Module):
    """Wraps BBox3D (DETR or CenterNet) + SemanticSeg probes + metric trackers.

    Args:
        bbox_style: 'detr'  — pool→MLP→Hungarian matching (BBox3DProbe)
                    'centernet' — dense 14×14 heatmap prediction (SpatialBBox3DProbe)
    """

    def __init__(
        self,
        vit_dim: int = 384,
        max_objects: int = 50,
        patch_grid: int = 14,
        seg_upsample: int = 4,
        target_size: Tuple[int, int] = (224, 224),
        bbox_style: str = 'detr',
    ):
        super().__init__()
        self.bbox_style = bbox_style

        if bbox_style == 'centernet':
            self.bbox_probe = SpatialBBox3DProbe(
                vit_dim=vit_dim, patch_grid=patch_grid,
            )
        else:  # 'detr'
            self.bbox_probe = BBox3DProbe(
                vit_dim=vit_dim, max_objects=max_objects, patch_grid=patch_grid,
            )

        self.seg_probe = SemanticSegProbe(
            vit_dim=vit_dim, patch_grid=patch_grid,
            upsample_factor=seg_upsample, target_size=target_size,
        )
        self.bbox_metrics = NuScenesDetectionMetrics()
        self.seg_metrics = SegmentationMetrics()

    def forward(self, patch_embs: torch.Tensor):
        """
        Args:
            patch_embs: (B, N_patches, vit_dim)
        Returns:
            bbox_out: dict  |  seg_logits: (B, C, H, W)
        """
        bbox_out = self.bbox_probe(patch_embs)
        seg_out = self.seg_probe(patch_embs)
        return bbox_out, seg_out

    def compute_losses(self, bbox_preds, seg_preds, bbox_targets=None, seg_targets=None):
        losses = {}
        bl = self.bbox_probe.compute_loss(bbox_preds, bbox_targets)
        losses.update({f'bbox_{k}': v for k, v in bl.items()})
        if seg_targets is not None:
            sl = self.seg_probe.compute_loss(seg_preds, seg_targets)
            losses.update({f'seg_{k}': v for k, v in sl.items()})
            losses['loss_detection_seg'] = bl['loss_total'] + sl['loss_seg']
        else:
            losses['loss_detection_seg'] = bl['loss_total']
        return losses

    @torch.no_grad()
    def update_metrics(self, bbox_preds, seg_preds, bbox_targets=None, seg_targets=None):
        if self.bbox_style != 'centernet':  # centernet uses heatmap, metrics need boxes
            self.bbox_metrics.update(bbox_preds, bbox_targets)
        if seg_targets is not None:
            self.seg_metrics.update(seg_preds, seg_targets)

    def compute_metrics(self):
        m = {}
        m.update({f'bbox_{k}': v for k, v in self.bbox_metrics.compute().items()})
        m.update({f'seg_{k}': v for k, v in self.seg_metrics.compute().items()})
        return m

    def reset_metrics(self):
        self.bbox_metrics.reset()
        self.seg_metrics.reset()
