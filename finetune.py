#!/usr/bin/env python3
"""
Fine-tune pre-trained Le MuMo JEPA backbones for downstream detection.

Loads a checkpoint from the sweep experiments, attaches a powerful
DETR-style multi-layer transformer decoder, and fine-tunes end-to-end
with differential learning rates (low for encoder, high for decoder).

Supports testing with different fractions of training data (1%, 10%, 50%, 100%)
to measure sample efficiency of learned representations.

Usage:
    # List available checkpoints
    python finetune.py --list_checkpoints

    # Fine-tune with 10% training data
    python finetune.py \
        --checkpoint saved_models/charmed-sweep-10/mmlejepa_C_lamb0.1_final.pt \
        --data_fraction 0.1 \
        --epochs 20

    # Debug mode (random data, no dataloader)
    python finetune.py --debug --epochs 1

    # Sweep agent mode
    SWEEP_ID='abc123' python finetune.py
"""

import sys
import os
import gc
import argparse
import time
import random
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# Auto-detect project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available, logging to console only")

from scipy.optimize import linear_sum_assignment

# --- Sweep agent configuration (env vars, matching sweep_agent_novel.py) ---
SWEEP_ID = os.environ.get('SWEEP_ID', None)
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'le-mumo-jepa-finetune-bbox3d-waymo')
SWEEP_COUNT = int(os.environ.get('SWEEP_COUNT', '100'))

from src.encoder import create_mm_encoder, get_vit_config, MMEncoderC_FusionTokens, MMEncoderC_LiDARRoPE
from src.dataset import MMNuScenesDataset, mm_collate_fn
from src.baseline_encoders import DINOv3FrozenEncoder
from src.detection_probes import (
    NUM_DETECTION_CLASSES,
    DETECTION_CLASSES,
    CATEGORY_TO_DETECTION,
    NuScenesDetectionMetrics,
)
from train import (
    ViTEncoder,
    extract_patch_tokens,
    get_input_stats,
    to_device,
)

# FLIR 2D detection classes (from flir_dataset.py)
FLIR_2D_DET_CLASS_NAMES = [
    "person", "bike", "car", "motor", "bus", "truck", "light", "sign",
]
NUM_FLIR_2D_CLASSES = len(FLIR_2D_DET_CLASS_NAMES)

# Grouped FLIR classes for comparison with Waymo/NuScenes grouping
FLIR_GROUPED_CLASSES = ['vehicle', 'pedestrian', 'cyclist']
NUM_FLIR_GROUPED_CLASSES = len(FLIR_GROUPED_CLASSES)
FLIR_CLASS_GROUP_MAP = {
    0: 1,  # person -> pedestrian
    1: 2,  # bike -> cyclist
    2: 0,  # car -> vehicle
    3: 2,  # motor -> cyclist
    4: 0,  # bus -> vehicle
    5: 0,  # truck -> vehicle
    6: -1, # light -> drop
    7: -1, # sign -> drop
}

_FLIR_CLASS_GROUP_TENSOR = None

def get_flir_class_group_tensor(device):
    """Return a (NUM_FLIR_2D_CLASSES,) tensor mapping old FLIR cls -> grouped cls."""
    global _FLIR_CLASS_GROUP_TENSOR
    if _FLIR_CLASS_GROUP_TENSOR is None or _FLIR_CLASS_GROUP_TENSOR.device != device:
        _FLIR_CLASS_GROUP_TENSOR = torch.tensor(
            [FLIR_CLASS_GROUP_MAP[i] for i in range(NUM_FLIR_2D_CLASSES)],
            dtype=torch.long, device=device,
        )
    return _FLIR_CLASS_GROUP_TENSOR

def remap_flir_gt_classes(gt_classes: torch.Tensor, gt_mask: torch.Tensor, device):
    """Remap FLIR 2D gt_classes using FLIR_CLASS_GROUP_MAP. Drop unmapped classes."""
    mapping = get_flir_class_group_tensor(device)
    remapped = mapping[gt_classes.clamp(0, NUM_FLIR_2D_CLASSES - 1)]
    drop_mask = (remapped == -1)
    remapped = remapped.clamp(min=0)
    new_mask = gt_mask.clone()
    new_mask[drop_mask] = 0.0
    return remapped, new_mask


# ═══════════════════════════════════════════════════════════════════════════════
# Class grouping for clustered experiments
# ═══════════════════════════════════════════════════════════════════════════════
GROUPED_CLASSES = ['vehicle', 'pedestrian', 'cyclist']
NUM_GROUPED_CLASSES = len(GROUPED_CLASSES)

# Map original DETECTION_CLASSES index -> grouped index (-1 = drop)
CLASS_GROUP_MAP = {
    0: 0,  # car -> vehicle
    1: 0,  # truck -> vehicle
    2: 0,  # bus -> vehicle
    3: 0,  # trailer -> vehicle
    4: 0,  # construction_vehicle -> vehicle
    5: 1,  # pedestrian -> pedestrian
    6: 2,  # motorcycle -> cyclist
    7: 2,  # bicycle -> cyclist
    8: -1, # traffic_cone -> drop
    9: -1, # barrier -> drop
}

_CLASS_GROUP_TENSOR = None  # lazily built on correct device

def get_class_group_tensor(device):
    """Return a (NUM_DETECTION_CLASSES,) tensor mapping old cls -> new cls.
    Values of -1 mean the class should be dropped."""
    global _CLASS_GROUP_TENSOR
    if _CLASS_GROUP_TENSOR is None or _CLASS_GROUP_TENSOR.device != device:
        _CLASS_GROUP_TENSOR = torch.tensor(
            [CLASS_GROUP_MAP[i] for i in range(NUM_DETECTION_CLASSES)],
            dtype=torch.long, device=device,
        )
    return _CLASS_GROUP_TENSOR

def remap_gt_classes(gt_classes: torch.Tensor, gt_mask: torch.Tensor, device):
    """Remap gt_classes using CLASS_GROUP_MAP. Drop classes mapped to -1."""
    mapping = get_class_group_tensor(device)
    remapped = mapping[gt_classes.clamp(0, NUM_DETECTION_CLASSES - 1)]
    # Mask out dropped classes (traffic_cone, barrier)
    drop_mask = (remapped == -1)
    remapped = remapped.clamp(min=0)  # avoid negative indices
    new_mask = gt_mask.clone()
    new_mask[drop_mask] = 0.0
    return remapped, new_mask


def _waymo_annotation_to_detection_index(annotation: Dict) -> int:
    """Map a Waymo annotation dict to the unified detection class index."""
    class_name = str(annotation.get('class_name', '')).lower()
    if class_name == 'vehicle':
        return 0
    if class_name == 'pedestrian':
        return 5
    if class_name == 'cyclist':
        return 7

    ann_type = int(annotation.get('type', 0) or 0)
    if ann_type == 1:
        return 0
    if ann_type == 2:
        return 5
    if ann_type == 4:
        return 7
    return -1


def _extract_sample_detection_classes(base_ds, pair: Dict, class_grouping: str) -> set:
    """Return the set of detection classes present in a sample across datasets."""
    cls_set = set()

    if isinstance(pair, dict) and 'annotations' in pair:
        # Waymo-style per-sample annotations are embedded directly in the pair.
        for annotation in pair.get('annotations', []):
            det_idx = _waymo_annotation_to_detection_index(annotation)
            if det_idx < 0:
                continue
            if class_grouping == 'clustered':
                mapped = CLASS_GROUP_MAP.get(det_idx, -1)
                if mapped >= 0:
                    cls_set.add(mapped)
            else:
                cls_set.add(det_idx)
        return cls_set

    if isinstance(pair, dict) and 'rgb_annotations' in pair:
        # FLIR-style 2D annotations.
        for annotation in pair.get('rgb_annotations', []):
            cat_id = int(annotation[0]) if len(annotation) > 0 else -1
            mapped = FLIR_CLASS_GROUP_MAP.get(cat_id, -1)
            if class_grouping == 'clustered':
                if mapped >= 0:
                    cls_set.add(mapped)
            elif 0 <= cat_id < NUM_FLIR_2D_CLASSES:
                cls_set.add(cat_id)
        return cls_set

    sample_token = pair.get('sample_token', '') if isinstance(pair, dict) else ''
    if hasattr(base_ds, 'annotation_store') and base_ds.annotation_store is not None:
        annots = base_ds.annotation_store.get_annotations(sample_token)
    elif hasattr(base_ds, 'sample_annotations'):
        annots = base_ds.sample_annotations.get(sample_token, [])
    else:
        annots = []

    for ann in annots:
        inst_tok = ann.get('instance_token', '')
        if hasattr(base_ds, 'annotation_store') and base_ds.annotation_store is not None:
            cat_name = base_ds.annotation_store.get_category(inst_tok)
        elif hasattr(base_ds, 'instance_to_category') and hasattr(base_ds, 'category_names'):
            cat_tok = base_ds.instance_to_category.get(inst_tok)
            cat_name = base_ds.category_names.get(cat_tok, '')
        else:
            cat_name = ''
        det_name = CATEGORY_TO_DETECTION.get(cat_name)
        if det_name and det_name in DETECTION_CLASSES:
            det_idx = DETECTION_CLASSES.index(det_name)
            if class_grouping == 'clustered':
                mapped = CLASS_GROUP_MAP.get(det_idx, -1)
                if mapped >= 0:
                    cls_set.add(mapped)
            else:
                cls_set.add(det_idx)
    return cls_set


def _count_sample_detection_objects(base_ds, pair: Dict) -> int:
    """Return the number of valid detection objects in a sample across datasets."""
    if isinstance(pair, dict) and 'annotations' in pair:
        return sum(1 for annotation in pair.get('annotations', []) if _waymo_annotation_to_detection_index(annotation) >= 0)

    if isinstance(pair, dict) and 'rgb_annotations' in pair:
        return sum(
            1 for annotation in pair.get('rgb_annotations', [])
            if len(annotation) > 0 and FLIR_CLASS_GROUP_MAP.get(int(annotation[0]), -1) >= 0
        )

    sample_token = pair.get('sample_token', '') if isinstance(pair, dict) else ''
    if hasattr(base_ds, 'annotation_store') and base_ds.annotation_store is not None:
        annots = base_ds.annotation_store.get_annotations(sample_token)
    elif hasattr(base_ds, 'sample_annotations'):
        annots = base_ds.sample_annotations.get(sample_token, [])
    else:
        annots = []

    n_obj = 0
    for ann in annots:
        inst_tok = ann.get('instance_token', '')
        if hasattr(base_ds, 'annotation_store') and base_ds.annotation_store is not None:
            cat_name = base_ds.annotation_store.get_category(inst_tok)
        elif hasattr(base_ds, 'instance_to_category') and hasattr(base_ds, 'category_names'):
            cat_tok = base_ds.instance_to_category.get(inst_tok)
            cat_name = base_ds.category_names.get(cat_tok, '')
        else:
            cat_name = ''
        if CATEGORY_TO_DETECTION.get(cat_name) in DETECTION_CLASSES:
            n_obj += 1
    return n_obj


def convert_centernet_predictions_to_metric_space(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert CenterNet-decoded UVZ predictions into camera-frame XYZ for metrics."""
    if 'centers' not in predictions or 'gt_centers_2d' not in targets:
        return predictions

    pred_centers = predictions['centers'].clone()
    gt_mask = targets['gt_mask'] > 0.5
    gt_uv = targets['gt_centers_2d']
    gt_cam = targets['gt_centers']

    for batch_idx in range(pred_centers.shape[0]):
        valid = gt_mask[batch_idx]
        n_valid = int(valid.sum().item())
        uv_pred = pred_centers[batch_idx, :, :2]
        z_pred = pred_centers[batch_idx, :, 2].clamp(min=1.0)

        if n_valid >= 2:
            uv_gt = gt_uv[batch_idx][valid].float()
            cam_gt = gt_cam[batch_idx][valid].float()
            z_gt = cam_gt[:, 2].clamp(min=1.0)
            x_over_z = cam_gt[:, 0] / z_gt
            y_over_z = cam_gt[:, 1] / z_gt
            ones = torch.ones(n_valid, 1, device=uv_gt.device, dtype=uv_gt.dtype)
            theta_x = torch.linalg.lstsq(torch.cat([uv_gt[:, 0:1], ones], dim=1), x_over_z.unsqueeze(1)).solution
            theta_y = torch.linalg.lstsq(torch.cat([uv_gt[:, 1:2], ones], dim=1), y_over_z.unsqueeze(1)).solution
            pred_centers[batch_idx, :, 0] = (uv_pred[:, 0] * theta_x[0, 0] + theta_x[1, 0]) * z_pred
            pred_centers[batch_idx, :, 1] = (uv_pred[:, 1] * theta_y[0, 0] + theta_y[1, 0]) * z_pred
        elif n_valid == 1:
            uv_gt = gt_uv[batch_idx][valid][0].float()
            cam_gt = gt_cam[batch_idx][valid][0].float()
            z_gt = cam_gt[2].clamp(min=1.0)
            a_default = 0.71
            b_x = cam_gt[0] / z_gt - a_default * uv_gt[0]
            b_y = cam_gt[1] / z_gt - a_default * uv_gt[1]
            pred_centers[batch_idx, :, 0] = (uv_pred[:, 0] * a_default + b_x) * z_pred
            pred_centers[batch_idx, :, 1] = (uv_pred[:, 1] * a_default + b_y) * z_pred
        else:
            pred_centers[batch_idx, :, 0] = (uv_pred[:, 0] * 0.71 - 0.37) * z_pred
            pred_centers[batch_idx, :, 1] = (uv_pred[:, 1] * 0.71 - 0.39) * z_pred

    converted = dict(predictions)
    converted['centers'] = pred_centers
    return converted


def _print_plane_class_breakdown(val_metrics: Dict[str, float], class_names: List[str], primary_plane: str = 'xz') -> None:
    """Print per-class AP/ADE for XY and XZ planes in a symmetric format."""
    if not class_names:
        return
    compare_plane = 'xy' if primary_plane == 'xz' else 'xz'
    for cls_name in class_names:
        xz_ap = val_metrics.get(f'xz_match/{cls_name}/AP', val_metrics.get(f'{cls_name}/AP' if primary_plane == 'xz' else None, float('nan')))
        xz_ade = val_metrics.get(f'xz_match/{cls_name}/ADE', val_metrics.get(f'{cls_name}/ADE' if primary_plane == 'xz' else None, float('nan')))
        xy_ap = val_metrics.get(f'xy_match/{cls_name}/AP', val_metrics.get(f'{cls_name}/AP' if primary_plane == 'xy' else None, float('nan')))
        xy_ade = val_metrics.get(f'xy_match/{cls_name}/ADE', val_metrics.get(f'{cls_name}/ADE' if primary_plane == 'xy' else None, float('nan')))
        if any(math.isfinite(v) for v in (xy_ap, xz_ap, xy_ade, xz_ade)):
            print(
                f"    {cls_name}: XY AP={xy_ap:.4f}, ADE={xy_ade:.4f} | "
                f"XZ AP={xz_ap:.4f}, ADE={xz_ade:.4f}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Focal loss for class-imbalanced detection
# ═══════════════════════════════════════════════════════════════════════════════
def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0, reduction: str = 'sum') -> torch.Tensor:
    """Focal loss for multi-class classification.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    Down-weights well-classified examples, focuses on hard ones.
    """
    ce = F.cross_entropy(logits, targets, reduction='none')
    p_t = torch.exp(-ce)
    loss = ((1 - p_t) ** gamma) * ce
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced 3D BBox Decoder (Multi-Layer DETR)
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedBBox3DDecoder(nn.Module):
    """
    Multi-layer transformer decoder for 3D bounding box prediction.

    Architecture:
        patch_embs (B, N, D) → project → 3× TransformerDecoderLayer
        → learnable object queries cross-attend to projected patches
        → predict fixed set of boxes with Hungarian matching

    Significantly more powerful than BBox3DProbe (single cross-attention layer).
    """

    PER_BOX_DIM = NUM_DETECTION_CLASSES + 3 + 3 + 2 + 1  # cls + center + size + orient + conf = 19

    def __init__(
        self,
        vit_dim: int = 384,
        hidden_dim: int = 256,
        max_objects: int = 100,
        num_classes: int = NUM_DETECTION_CLASSES,
        num_heads: int = 8,
        num_decoder_layers: int = 3,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        matching: str = 'hungarian',
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.max_objects = max_objects
        self.num_classes = num_classes
        self.vit_dim = vit_dim
        self.hidden_dim = hidden_dim
        self.matching = str(matching).lower()
        if self.matching not in {'hungarian', 'greedy'}:
            print(f"⚠️ Unknown matching='{matching}', falling back to 'hungarian'")
            self.matching = 'hungarian'
        self.focal_gamma = focal_gamma

        # Project patch embeddings to hidden_dim
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, hidden_dim),
        )

        # Learnable object queries
        self.object_queries = nn.Parameter(torch.randn(max_objects, hidden_dim) * 0.02)

        # Positional encoding for patches (learnable)
        self.patch_pos = nn.Parameter(torch.randn(1, 196, hidden_dim) * 0.02)  # 14x14=196

        # Multi-layer transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Detection heads
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.center_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.orient_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_embs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = patch_embs.shape

        # Project patches to decoder dim
        memory = self.patch_proj(patch_embs)  # (B, N, hidden_dim)

        # Add positional encoding (handle different N sizes)
        if N <= self.patch_pos.shape[1]:
            memory = memory + self.patch_pos[:, :N, :]
        else:
            # Interpolate positional encoding for different grid sizes
            memory = memory + F.interpolate(
                self.patch_pos.transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

        # Expand object queries for batch
        queries = self.object_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N_obj, hidden)

        # Multi-layer decoder: queries attend to patch tokens
        decoded = self.decoder(queries, memory)  # (B, N_obj, hidden)
        decoded = self.decoder_norm(decoded)

        # Per-query predictions via separate heads
        class_logits = self.class_head(decoded)        # (B, N_obj, num_classes)
        centers = self.center_head(decoded)             # (B, N_obj, 3)
        sizes = F.softplus(self.size_head(decoded))     # (B, N_obj, 3) - positive
        orientations = F.normalize(self.orient_head(decoded), dim=-1)  # (B, N_obj, 2)
        confidences = torch.sigmoid(self.confidence_head(decoded).squeeze(-1))  # (B, N_obj)

        return {
            'class_logits': class_logits,
            'centers': centers,
            'sizes': sizes,
            'orientations': orientations,
            'confidences': confidences,
        }

    @torch.no_grad()
    def _hungarian_match(
        self,
        pred: Dict[str, torch.Tensor],
        gt_centers: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Bipartite matching between predictions and GT per sample."""
        B = gt_centers.shape[0]
        indices = []

        for b in range(B):
            mask_b = gt_mask[b].bool()
            n_gt = mask_b.sum().item()
            if n_gt == 0:
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue

            gt_c = gt_centers[b, mask_b]
            gt_cl = gt_classes[b, mask_b]
            pred_c = pred['centers'][b]
            pred_cl = pred['class_logits'][b]
            pred_conf = pred['confidences'][b]

            # Cost matrix
            cost_center = torch.cdist(pred_c, gt_c, p=1)
            log_probs = F.log_softmax(pred_cl, dim=-1)
            cost_class = -log_probs[:, gt_cl.long()]
            cost_conf = (1 - pred_conf).unsqueeze(-1).expand_as(cost_center)

            cost = cost_center + cost_class + 0.5 * cost_conf

            if self.matching == 'hungarian':
                cost_np = cost.detach().cpu().numpy()
                row_idx, col_idx = linear_sum_assignment(cost_np)
                indices.append((
                    torch.as_tensor(row_idx, dtype=torch.long, device=gt_centers.device),
                    torch.as_tensor(col_idx, dtype=torch.long, device=gt_centers.device),
                ))
            else:
                # Greedy GPU matching: repeatedly pick global minimum and mask row/col.
                n_pred, n_gt_local = cost.shape
                k = min(n_pred, n_gt_local)
                work = cost.clone()
                row_sel = []
                col_sel = []
                inf = torch.tensor(float('inf'), device=work.device, dtype=work.dtype)
                for _ in range(k):
                    flat_idx = torch.argmin(work)
                    r = int(flat_idx // n_gt_local)
                    c = int(flat_idx % n_gt_local)
                    if torch.isinf(work[r, c]):
                        break
                    row_sel.append(r)
                    col_sel.append(c)
                    work[r, :] = inf
                    work[:, c] = inf

                if row_sel:
                    indices.append((
                        torch.as_tensor(row_sel, dtype=torch.long, device=gt_centers.device),
                        torch.as_tensor(col_sel, dtype=torch.long, device=gt_centers.device),
                    ))
                else:
                    indices.append((
                        torch.empty(0, dtype=torch.long, device=gt_centers.device),
                        torch.empty(0, dtype=torch.long, device=gt_centers.device),
                    ))
        return indices

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

        gt_classes = targets['gt_classes']
        gt_centers = targets['gt_centers']
        gt_sizes = targets['gt_sizes']
        gt_orient = targets['gt_orientations']
        gt_mask = targets['gt_mask']

        matched = self._hungarian_match(predictions, gt_centers, gt_classes, gt_mask)

        total_cls = zero
        total_ctr = zero
        total_sz = zero
        total_ori = zero
        n_matched = 0

        for b, (pi, gi) in enumerate(matched):
            if len(pi) == 0:
                continue
            n = len(pi)
            n_matched += n

            # Classification
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
            # Center L1 (depth in log-space)
            pred_ctr = predictions['centers'][b][pi]
            gt_ctr = gt_centers[b][gt_mask[b].bool()][gi]
            total_ctr = total_ctr + F.l1_loss(pred_ctr[:, :2], gt_ctr[:, :2], reduction='sum')
            total_ctr = total_ctr + F.l1_loss(
                torch.log1p(pred_ctr[:, 2:].clamp(min=0.0)),
                torch.log1p(gt_ctr[:, 2:].clamp(min=0.0)),
                reduction='sum',
            )
            # Size L1 (log-space)
            gt_sz_log = torch.log(gt_sizes[b][gt_mask[b].bool()][gi].clamp(min=1e-4))
            pred_sz_log = torch.log(predictions['sizes'][b][pi].clamp(min=1e-4))
            total_sz = total_sz + F.l1_loss(pred_sz_log, gt_sz_log, reduction='sum')
            # Orientation cosine
            dot = (predictions['orientations'][b][pi] *
                   gt_orient[b][gt_mask[b].bool()][gi]).sum(-1)
            total_ori = total_ori + (1.0 - dot).sum()

        denom = max(n_matched, 1)
        loss_cls = total_cls / denom
        loss_center = total_ctr / denom
        loss_size = total_sz / denom
        loss_orient = total_ori / denom

        # Confidence target with balanced weighting for objectness
        # Positive targets are rare (~4 per sample), negatives are many (~96)
        # Weight positive samples more heavily to counter class imbalance
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
        
        with torch.amp.autocast('cuda', enabled=False):
            # Weighted BCE: multiply per-element loss by weight
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
# Enhanced CenterNet Decoder (dense, spatial heatmap-based)
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalize_per_class: bool = False,
) -> torch.Tensor:
    """CenterNet-style focal loss for Gaussian heatmaps.
    
    Modified FL where positive targets follow Gaussian distribution.
    Penalty-reduced loss for positions near (but not at) GT centers.
    
    When normalize_per_class=True, apply a mild class-frequency correction to
    the positive term only. This helps rare classes without allowing the dense
    negative term to dominate the total loss scale.
    """
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()
    
    # Numerical stability
    pred = pred.clamp(min=1e-6, max=1-1e-6)
    
    pos_loss = -((1 - pred) ** 2) * torch.log(pred) * pos_mask
    neg_loss = -((1 - target) ** 4) * (pred ** 2) * torch.log(1 - pred) * neg_mask

    n_pos_total = pos_mask.sum().clamp(min=1e-4)
    pos_loss_sum = pos_loss.sum()
    neg_loss_sum = neg_loss.sum()

    if normalize_per_class:
        class_pos = pos_mask.sum(dim=(0, 2, 3))
        active = class_pos > 0
        if active.any():
            active_pos = class_pos[active].clamp(min=1e-4)
            class_pos_loss = pos_loss.sum(dim=(0, 2, 3))[active]
            inv_sqrt_freq = active_pos.rsqrt()
            weights = inv_sqrt_freq / inv_sqrt_freq.mean()
            weights = weights.clamp(min=0.5, max=2.0)
            pos_loss_sum = (class_pos_loss * weights).sum()

    loss = (pos_loss_sum + neg_loss_sum) / n_pos_total
    return loss


class EnhancedCenterNetDecoder(nn.Module):
    """CenterNet-style decoder with multi-scale feature extraction.
    
    Unlike DETR (slot-based), this retains spatial structure and makes
    dense predictions at every location. Better for detecting many objects.
    
    Architecture:
        patch_embs (B, N, D) → reshape (B, D, H, W)
        → FPN-like multi-scale feature pyramid
        → per-location heads:
            heatmap  (C, H', W') — per-class centre probability
            offset   (2, H', W') — sub-pixel centre correction
            size     (3, H', W') — w, l, h (in meters)
            depth    (1, H', W') — centre depth Z
            rot      (2, H', W') — sin/cos(yaw)
        
    Training: Gaussian focal loss for heatmap + L1 for regressions.
    Inference: NMS peaks → gather attributes → convert to boxes.
    """
    
    def __init__(
        self,
        vit_dim: int = 384,
        hidden_dim: int = 256,
        num_classes: int = NUM_DETECTION_CLASSES,
        patch_grid: int = 14,
        upsample_factor: int = 2,  # 14→28 for finer localization
        num_conv_layers: int = 3,  # More layers for richer features
        focal_gamma: float = 2.0,
        max_objects: int = 100,  # For inference compatibility
        log_space_3d: bool = True,
        per_class_heatmap_loss: bool = False,
        min_gaussian_radius: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_grid = patch_grid
        self.upsample_factor = upsample_factor
        self.output_grid = patch_grid * upsample_factor
        self.focal_gamma = focal_gamma
        self.max_objects = max_objects
        self.hidden_dim = hidden_dim
        self.log_space_3d = log_space_3d
        self.per_class_heatmap_loss = per_class_heatmap_loss
        self.min_gaussian_radius = max(1, int(min_gaussian_radius))
        
        # Multi-layer feature extractor (FPN-like)
        layers = []
        in_ch = vit_dim
        for i in range(num_conv_layers):
            out_ch = hidden_dim
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.feature_extractor = nn.Sequential(*layers)
        
        # Upsample: 14×14 → 28×28 via transposed convolution
        if upsample_factor > 1:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4,
                                   stride=upsample_factor, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.upsample = nn.Identity()
        
        # Per-task head networks (deeper than probe version)
        def make_head(out_dim):
            return nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_dim, 1),
            )
        
        self.heatmap_head = make_head(num_classes)
        self.offset_head = make_head(2)
        self.size_head = make_head(3)
        self.depth_head = make_head(1)
        self.rot_head = make_head(2)
        
        # CenterNet bias initialization for heatmap stability
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)
        # Depth head bias: initialize to mean log1p(depth) ≈ 3.76 so the
        # initial prediction is already in the right range.  Without this,
        # certain GPU random initializations can produce very large depth
        # outputs that cause loss_depth to dominate all other losses.
        if log_space_3d:
            nn.init.constant_(self.depth_head[-1].bias, 3.76)
    
    def forward(self, patch_embs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = patch_embs.shape
        H = W = int(N ** 0.5)
        x = patch_embs.transpose(1, 2).reshape(B, D, H, W)
        
        feat = self.feature_extractor(x)
        feat = self.upsample(feat)

        size_pred = self.size_head(feat)
        if not self.log_space_3d:
            size_pred = F.softplus(size_pred)
        
        return {
            'heatmap': torch.sigmoid(self.heatmap_head(feat)),
            'offset': self.offset_head(feat),
            'size': size_pred,
            'depth': self.depth_head(feat),
            'rot': F.normalize(self.rot_head(feat), dim=1),
        }
    
    def generate_targets(
        self,
        gt_centers_2d: torch.Tensor,  # (B, max_obj, 2) normalized [0,1]
        gt_depths: torch.Tensor,       # (B, max_obj)
        gt_sizes: torch.Tensor,        # (B, max_obj, 3)
        gt_orientations: torch.Tensor, # (B, max_obj, 2)
        gt_classes: torch.Tensor,      # (B, max_obj)
        gt_mask: torch.Tensor,         # (B, max_obj)
        output_grid: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate dense CenterNet training targets from sparse GT."""
        B = gt_centers_2d.shape[0]
        if output_grid is None:
            H = W = self.output_grid
        else:
            H, W = int(output_grid[0]), int(output_grid[1])
        device = gt_centers_2d.device
        
        heatmap = torch.zeros(B, self.num_classes, H, W, device=device)
        offset = torch.zeros(B, 2, H, W, device=device)
        size = torch.zeros(B, 3, H, W, device=device)
        depth = torch.zeros(B, 1, H, W, device=device)
        rot = torch.zeros(B, 2, H, W, device=device)
        reg_mask = torch.zeros(B, 1, H, W, device=device)

        kernel_cache = {}

        def _get_kernel(radius: int) -> torch.Tensor:
            radius = int(max(1, radius))
            if radius not in kernel_cache:
                sigma = max(radius / 3.0, 0.5)
                dy = torch.arange(-radius, radius + 1, device=device).float()
                dx = torch.arange(-radius, radius + 1, device=device).float()
                gy, gx = torch.meshgrid(dy, dx, indexing='ij')
                kernel_cache[radius] = torch.exp(-(gx * gx + gy * gy) / (2 * sigma * sigma))
            return kernel_cache[radius]

        def _radius_for_object(size_xyz: torch.Tensor, depth_z: torch.Tensor) -> int:
            size_xy = size_xyz[:2].amax().item()
            depth_val = max(float(depth_z.item()), 1.0)
            projected_extent = max(H, W) * (size_xy / depth_val)
            radius_floor = self.min_gaussian_radius
            dynamic_r = round(max(radius_floor, 0.6 * projected_extent))
            return int(min(max(dynamic_r, radius_floor), max(4, max(H, W) // 4)))
        
        for b in range(B):
            for i in range(gt_mask.shape[1]):
                if gt_mask[b, i] < 0.5:
                    continue
                
                cx, cy = gt_centers_2d[b, i]
                if cx < 0 or cx > 1 or cy < 0 or cy > 1:
                    continue
                    
                ix = int(cx * (W - 1))
                iy = int(cy * (H - 1))
                ix = min(max(ix, 0), W - 1)
                iy = min(max(iy, 0), H - 1)
                
                cls_idx = int(gt_classes[b, i].item())
                if cls_idx < 0 or cls_idx >= self.num_classes:
                    continue
                
                # Size-aware Gaussian splat for heatmap
                radius = _radius_for_object(gt_sizes[b, i], gt_depths[b, i])
                kernel = _get_kernel(radius)
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        py, px = iy + dy, ix + dx
                        if 0 <= py < H and 0 <= px < W:
                            g = float(kernel[dy + radius, dx + radius].item())
                            heatmap[b, cls_idx, py, px] = max(
                                heatmap[b, cls_idx, py, px], g
                            )
                
                # Store regression targets at center
                offset[b, 0, iy, ix] = cx * (W - 1) - ix
                offset[b, 1, iy, ix] = cy * (H - 1) - iy
                if self.log_space_3d:
                    size[b, :, iy, ix] = torch.log(gt_sizes[b, i].clamp(min=1e-4))
                    depth[b, 0, iy, ix] = torch.log1p(gt_depths[b, i].clamp(min=0.0))
                else:
                    size[b, :, iy, ix] = gt_sizes[b, i]
                    depth[b, 0, iy, ix] = gt_depths[b, i]
                rot[b, :, iy, ix] = gt_orientations[b, i]
                reg_mask[b, 0, iy, ix] = 1.0
        
        return {
            'heatmap': heatmap,
            'offset': offset,
            'size': size,
            'depth': depth,
            'rot': rot,
            'reg_mask': reg_mask,
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute CenterNet-style loss."""
        loss_hm = gaussian_focal_loss(
            predictions['heatmap'],
            targets['heatmap'],
            normalize_per_class=self.per_class_heatmap_loss,
        )
        
        reg_mask = targets['reg_mask']
        n_pos = reg_mask.sum().clamp(min=1)
        
        loss_offset = (F.l1_loss(predictions['offset'], targets['offset'], reduction='none') 
                       * reg_mask).sum() / n_pos
        loss_size = (F.l1_loss(predictions['size'], targets['size'], reduction='none')
                     * reg_mask).sum() / n_pos
        # Clamp depth predictions to prevent GPU-specific init outliers from
        # causing catastrophic loss_depth (observed 42 vs expected ~3.8).
        depth_pred = predictions['depth']
        if self.log_space_3d:
            depth_pred = depth_pred.clamp(-1.0, 7.0)  # log1p(0)=0 .. log1p(1096)=7
        loss_depth = (F.l1_loss(depth_pred, targets['depth'], reduction='none')
                      * reg_mask).sum() / n_pos
        loss_rot = (F.l1_loss(predictions['rot'], targets['rot'], reduction='none')
                    * reg_mask).sum() / n_pos
        
        w = loss_weights or {}
        loss_total = (
            w.get('hm', 1.0) * loss_hm
            + w.get('offset', 1.0) * loss_offset
            + w.get('size', 1.0) * loss_size
            + w.get('depth', 1.0) * loss_depth
            + w.get('rot', 1.0) * loss_rot
        )
        
        return {
            'loss_hm': loss_hm,
            'loss_offset': loss_offset,
            'loss_size': loss_size,
            'loss_depth': loss_depth,
            'loss_rot': loss_rot,
            'loss_total': loss_total,
        }
    
    @torch.no_grad()
    def decode_to_boxes(
        self,
        predictions: Dict[str, torch.Tensor],
        max_detections: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """Decode heatmap predictions into box-format outputs."""
        hm = predictions['heatmap']
        B, C, H, W = hm.shape
        
        # 3×3 NMS
        hm_pool = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
        hm_nms = hm * (hm_pool == hm).float()
        
        # Top-K across all classes
        hm_flat = hm_nms.reshape(B, -1)
        K = min(max_detections, hm_flat.shape[1])
        scores, indices = hm_flat.topk(K, dim=1)
        
        cls_ids = indices // (H * W)
        spatial_idx = indices % (H * W)
        iy = spatial_idx // W
        ix = spatial_idx % W
        
        # One-hot class logits for compatibility
        class_logits = torch.zeros(B, K, C, device=hm.device)
        class_logits.scatter_(2, cls_ids.unsqueeze(-1), 1.0)
        
        # Gather attributes
        offset_pred = predictions['offset']  # (B, 2, H, W)
        size_pred = predictions['size']      # (B, 3, H, W)
        depth_pred = predictions['depth']    # (B, 1, H, W)
        rot_pred = predictions['rot']        # (B, 2, H, W)
        
        # Convert to float indices for grid_sample
        # Normalize to [-1, 1]
        grid_x = (ix.float() / (W - 1)) * 2 - 1
        grid_y = (iy.float() / (H - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, K, 1, 2)
        
        def sample(feat):
            # feat: (B, C, H, W)
            sampled = F.grid_sample(feat, grid, mode='bilinear', align_corners=True)
            return sampled.squeeze(-1).transpose(1, 2)  # (B, K, C)
        
        offsets = sample(offset_pred)  # (B, K, 2)
        sizes = sample(size_pred)      # (B, K, 3)
        depths = sample(depth_pred)    # (B, K, 1)
        rots = sample(rot_pred)        # (B, K, 2)
        
        # Refine centers with offsets
        cx = (ix.float() + offsets[..., 0]) / (W - 1)
        cy = (iy.float() + offsets[..., 1]) / (H - 1)

        if self.log_space_3d:
            sizes = torch.exp(sizes)
            depths = torch.expm1(depths.clamp(max=6.0))
        
        # Build centers (cx_norm, cy_norm, depth)
        centers = torch.stack([cx, cy, depths.squeeze(-1)], dim=-1)
        
        return {
            'class_logits': class_logits,
            'centers': centers,
            'sizes': sizes,
            'orientations': F.normalize(rots, dim=-1),
            'confidences': scores,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint utilities
# ═══════════════════════════════════════════════════════════════════════════════

SAVED_MODELS_DIR = Path(PROJECT_ROOT) / "saved_models"
SPECIAL_CHECKPOINTS = {
    '__dinov3_frozen__': 'DINOv3 frozen RGB baseline',
    'dinov3_frozen': 'DINOv3 frozen RGB baseline',
}


def _resolve_checkpoint_path(checkpoint_path):
    """Robustly resolve checkpoint paths (matching sweep_agent_novel logic).

    Supports:
    - Absolute paths to .pt files
    - Relative paths resolved against PROJECT_ROOT
    - Run directory names under saved_models/ (picks latest.pt or first .pt)
    """
    if not checkpoint_path:
        return None

    raw_path = Path(str(checkpoint_path)).expanduser()
    candidates = []
    seen = set()

    def add_candidate(path_obj):
        key = str(path_obj)
        if key not in seen:
            candidates.append(path_obj)
            seen.add(key)

    add_candidate(raw_path)
    if not raw_path.is_absolute():
        add_candidate(Path(PROJECT_ROOT) / raw_path)
        add_candidate(Path(PROJECT_ROOT) / 'saved_models' / raw_path)

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.is_dir():
            latest_path = candidate / 'latest.pt'
            if latest_path.exists():
                return latest_path.resolve()
            pt_files = sorted(candidate.glob('*.pt'))
            if pt_files:
                return pt_files[-1].resolve()
            continue
        return candidate.resolve()
    return None


def _infer_scenario_from_checkpoint_file(checkpoint_path):
    """Try to infer scenario and dataset from a saved checkpoint's config dict."""
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved_cfg = ckpt.get('config', {})
        if not isinstance(saved_cfg, dict):
            return None, None, None
        scenario = saved_cfg.get('scenario')
        dataset = saved_cfg.get('dataset', saved_cfg.get('dataset_name'))
        run_name = ckpt.get('run_name')
        return scenario, dataset, run_name
    except Exception:
        return None, None, None


def list_checkpoints():
    """List all available checkpoints with their metadata."""
    if not SAVED_MODELS_DIR.exists():
        print("No saved_models directory found!")
        checkpoints = []
    else:
        checkpoints = []

        for run_dir in sorted(SAVED_MODELS_DIR.iterdir()):
            if not run_dir.is_dir():
                continue
            for pt_file in run_dir.glob("*.pt"):
                try:
                    meta = torch.load(pt_file, map_location='cpu', weights_only=False)
                    arch = meta.get('arch', '?')
                    run_name = meta.get('run_name', run_dir.name)
                    config = meta.get('config', {})
                    scenario = config.get('scenario', 'unknown') if isinstance(config, dict) else 'unknown'
                    vit_size = config.get('vit_size', 'small') if isinstance(config, dict) else 'small'
                    checkpoints.append({
                        'path': str(pt_file),
                        'run_name': run_name,
                        'arch': arch,
                        'scenario': scenario,
                        'vit_size': vit_size,
                    })
                except Exception as e:
                    print(f"  Skipping {pt_file}: {e}")

    checkpoints.append({
        'path': '__dinov3_frozen__',
        'run_name': 'DINOv3 frozen baseline',
        'arch': 'C',
        'scenario': 'dinov3_frozen',
        'vit_size': 'small/base/large',
    })
    return checkpoints


def load_special_encoder(checkpoint_name: str, device: str = 'cuda'):
    """Construct baseline encoders that do not come from saved training checkpoints."""
    key = str(checkpoint_name).strip().lower()
    if key not in SPECIAL_CHECKPOINTS:
        raise FileNotFoundError(f"Unknown special checkpoint: {checkpoint_name}")

    vit_size = 'small'
    encoder = DINOv3FrozenEncoder(
        proj_dim=16,
        img_size=224,
        vit_size=vit_size,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)
    vit_cfg = get_vit_config(vit_size)
    vit_dim = vit_cfg['vit_dim']
    config = {
        'scenario': 'dinov3_frozen',
        'arch': 'C',
        'vit_size': vit_size,
        'aligned_mode': True,
        'lidar_mode': 'depth',
        'modality2_channels': 1,
        'baseline_encoder': 'dinov3_frozen',
        'simple_baseline': False,
    }
    print("📦 Using special baseline encoder: DINOv3 frozen RGB")
    return encoder, 'C', vit_size, vit_dim, config


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """
    Load a checkpoint and reconstruct the encoder.

    Returns:
        encoder: nn.Module (on device)
        arch: str
        vit_size: str
        vit_dim: int
        config: dict
    """
    checkpoint_key = str(checkpoint_path).strip().lower()
    if checkpoint_key in SPECIAL_CHECKPOINTS:
        return load_special_encoder(checkpoint_key, device)

    checkpoint_path = Path(checkpoint_path)

    # Support run name or full path
    if not checkpoint_path.exists():
        # Try as run name
        run_dir = SAVED_MODELS_DIR / checkpoint_path
        if run_dir.is_dir():
            pt_files = list(run_dir.glob("*.pt"))
            if pt_files:
                checkpoint_path = pt_files[0]
            else:
                raise FileNotFoundError(f"No .pt files in {run_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"📦 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    arch = ckpt.get('arch', 'B')
    config = ckpt.get('config', {})
    if not isinstance(config, dict):
        config = dict(config) if hasattr(config, '__iter__') else {}

    vit_size = str(config.get('vit_size', 'small')).lower()
    proj_dim = int(config.get('proj_dim', 16))
    aligned_mode = config.get('aligned_mode', False)  # Older checkpoints default to non-aligned

    # Detect special encoder types from config
    fusion_tokens_sigreg = config.get('fusion_tokens_sigreg', False)
    lidar_rope_rgb = config.get('lidar_rope_rgb', False)
    fusion_tokens_variant = str(config.get('fusion_tokens_variant', 'prune_after_first')).lower()
    simple_baseline = config.get('simple_baseline', False)
    simple_modality = config.get('simple_modality', 'rgb')

    lidar_mode = config.get('lidar_mode', 'depth')
    
    # Auto-detect modality2 channels from saved weights if possible
    encoder_state = ckpt['encoder']
    range_patch_key = 'range_patch_embed.weight'
    if range_patch_key in encoder_state:
        modality2_channels = encoder_state[range_patch_key].shape[1]
        print(f"  ✓ Auto-detected modality2_channels={modality2_channels} from weights")
    elif lidar_mode == 'depth' or aligned_mode:
        modality2_channels = 1
    else:
        modality2_channels = 5

    # Reconstruct encoder
    if simple_baseline:
        in_channels = 1 if simple_modality == 'lidar' else 3
        encoder = ViTEncoder(
            proj_dim=proj_dim,
            img_size=224,
            in_channels=in_channels,
            vit_size=vit_size,
        )
    elif fusion_tokens_sigreg:
        encoder = MMEncoderC_FusionTokens(
            proj_dim=proj_dim,
            img_size=224,
            range_channels=modality2_channels,
            aligned_mode=aligned_mode,
            vit_size=vit_size,
            attention_mode=fusion_tokens_variant,
        )
    elif lidar_rope_rgb:
        encoder = MMEncoderC_LiDARRoPE(
            proj_dim=proj_dim,
            img_size=224,
            aligned_mode=aligned_mode,
            vit_size=vit_size,
        )
    else:
        encoder = create_mm_encoder(
            arch=arch,
            proj_dim=proj_dim,
            img_size=224,
            second_modality_channels=modality2_channels,
            aligned_mode=aligned_mode,
            vit_size=vit_size,
        )

    # Load weights (encoder_state already extracted above for auto-detection)
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"  ⚠️  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    vit_cfg = get_vit_config(vit_size)
    vit_dim = vit_cfg['vit_dim']

    encoder = encoder.to(device)
    config['modality2_channels'] = modality2_channels
    print(f"  ✓ Encoder loaded: arch={arch}, vit_size={vit_size}, vit_dim={vit_dim}")
    print(f"  ✓ Config: scenario={config.get('scenario', 'unknown')}")

    return encoder, arch, vit_size, vit_dim, config


# ═══════════════════════════════════════════════════════════════════════════════
# Dummy batch for debug mode
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dummy_batch(batch_size: int, arch: str, device: str = 'cuda',
                         simple_baseline: bool = False, simple_modality: str = 'rgb',
                         modality2_channels: int = 1):
    """Generate random dummy data for debug mode."""
    img_size = 224

    if simple_baseline:
        in_c = 1 if simple_modality == 'lidar' else 3
        cam_views = torch.randn(batch_size, 1, in_c, img_size, img_size, device=device)
        modality2 = torch.randn(batch_size, 1, modality2_channels, img_size, img_size, device=device)
    elif arch == 'A':
        cam_views = torch.randn(batch_size, 1, 3, img_size, img_size, device=device)
        modality2 = torch.randn(batch_size, 16384, 3, device=device)
    elif arch == 'D':
        cam_views = torch.randn(batch_size, 1, 4, img_size, img_size, device=device)
        modality2 = torch.randn(batch_size, 1, modality2_channels, img_size, img_size, device=device)
    else:
        cam_views = torch.randn(batch_size, 1, 3, img_size, img_size, device=device)
        modality2 = torch.randn(batch_size, 1, modality2_channels, img_size, img_size, device=device)

    max_objects = 50
    labels = {
        'gt_classes': torch.randint(0, NUM_DETECTION_CLASSES, (batch_size, max_objects)),
        'gt_centers': torch.randn(batch_size, max_objects, 3) * 20,
        'gt_sizes': torch.rand(batch_size, max_objects, 3) * 5 + 0.5,
        'gt_orientations': F.normalize(torch.randn(batch_size, max_objects, 2), dim=-1),
        'gt_mask': (torch.rand(batch_size, max_objects) > 0.7).float(),
        'gt_centers_2d': torch.rand(batch_size, max_objects, 2),
        'seg_map': torch.randint(0, 16, (batch_size, 14, 14)),
        'scene': torch.randint(0, 10, (batch_size,)),
        'camera': torch.randint(0, 6, (batch_size,)),
        'location': torch.randint(0, 4, (batch_size,)),
    }

    return cam_views, modality2, labels


def compute_nds(metrics: Dict[str, float]) -> float:
    """Compute nuScenes Detection Score (NDS).
    
    NDS = 1/10 * [5 * mAP + Σ max(1 - metric, 0)]
    where metric ∈ {mATE, mASE, mAOE, mAVE, mAAE}
    We skip mAVE (velocity) and mAAE (attribute) as we don't predict those.
    """
    mAP = metrics.get('mAP', 0.0)
    tp_errors = [
        metrics.get('mATE', 1.0),
        metrics.get('mASE', 1.0),
        metrics.get('mAOE', 1.0),
    ]
    # NDS formula: (5 * mAP + sum(max(1 - e, 0) for e in tp_errors)) / (5 + len(tp_errors))
    tp_score = sum(max(1.0 - e, 0.0) for e in tp_errors)
    nds = (5.0 * mAP + tp_score) / (5.0 + len(tp_errors))
    return nds


def _align_modal_view_counts(cam_views, modality2):
    """Align camera/modality2 view-count (dim=1) to avoid encoder shape mismatches.

    This guards against dataset branches that may emit different numbers of global/local
    views between modalities (e.g. V=1 + range mode with extra clean probe view).
    """
    def _slice_to_v(t: torch.Tensor, target_v: int) -> torch.Tensor:
        if not isinstance(t, torch.Tensor) or t.dim() != 5:
            return t
        current_v = t.shape[1]
        if current_v == target_v:
            return t
        if target_v == 1 and current_v > 1:
            # Prefer last view when collapsing to one (typically the clean/full probe view).
            return t[:, -1:, ...]
        return t[:, :target_v, ...]

    changed = False

    if isinstance(cam_views, dict) and isinstance(modality2, dict):
        cam_out = dict(cam_views)
        mod_out = dict(modality2)
        for key in ('global', 'local'):
            c = cam_out.get(key)
            m = mod_out.get(key)
            if isinstance(c, torch.Tensor) and isinstance(m, torch.Tensor) and c.dim() == 5 and m.dim() == 5:
                target_v = min(c.shape[1], m.shape[1])
                if target_v > 0 and (c.shape[1] != target_v or m.shape[1] != target_v):
                    cam_out[key] = _slice_to_v(c, target_v)
                    mod_out[key] = _slice_to_v(m, target_v)
                    changed = True
        return cam_out, mod_out, changed

    if isinstance(cam_views, torch.Tensor) and isinstance(modality2, torch.Tensor):
        if cam_views.dim() == 5 and modality2.dim() == 5:
            target_v = min(cam_views.shape[1], modality2.shape[1])
            if target_v > 0 and (cam_views.shape[1] != target_v or modality2.shape[1] != target_v):
                cam_views = _slice_to_v(cam_views, target_v)
                modality2 = _slice_to_v(modality2, target_v)
                changed = True
        return cam_views, modality2, changed

    return cam_views, modality2, changed


def _prefer_probe_views(views):
    """Use explicit probe views when present, otherwise keep the original views."""
    if isinstance(views, dict):
        probe = views.get('probe')
        if isinstance(probe, torch.Tensor) and probe.numel() > 0:
            return probe
    return views


# ═══════════════════════════════════════════════════════════════════════════════
# Main fine-tuning loop
# ═══════════════════════════════════════════════════════════════════════════════

def main(sweep_config: Optional[Dict] = None):
    """Main fine-tuning function. Called standalone or by wandb.agent().
    
    Args:
        sweep_config: If provided (from wandb.agent), overrides CLI args with sweep params.
    """
    parser = argparse.ArgumentParser(description='Fine-tune Le MuMo JEPA for downstream detection')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pt checkpoint or run name in saved_models/')
    parser.add_argument('--list_checkpoints', action='store_true',
                        help='List all available checkpoints and exit')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                        help='Fraction of training data to use (e.g. 0.01, 0.1, 0.5, 1.0)')
    parser.add_argument('--encoder_lr', type=float, default=1e-5,
                        help='Learning rate for encoder (backbone)')
    parser.add_argument('--decoder_lr', type=float, default=1e-4,
                        help='Learning rate for decoder head')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--bs', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--dataroot', type=str,
                        default=os.path.join(PROJECT_ROOT, 'nuscenes_data'),
                        help='Path to nuScenes data')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='le-mumo-jepa-finetune-bbox3d',
                        help='WandB project name')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: random data, 1 batch')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--decoder_layers', type=int, default=3,
                        help='Number of transformer decoder layers')
    parser.add_argument('--decoder_dim', type=int, default=256,
                        help='Decoder hidden dimension')
    parser.add_argument('--matching', type=str, default='hungarian', choices=['hungarian', 'greedy'],
                        help='Assignment strategy for training loss matching')
    parser.add_argument('--max_objects', type=int, default=100,
                        help='Max detection slots (object queries)')
    parser.add_argument('--finetune_crop_scale', type=float, nargs=2, default=(0.8, 1.0),
                        metavar=('MIN', 'MAX'),
                        help='RandomResizedCrop scale range used when finetune_mode is enabled')
    parser.add_argument('--decode_max_detections', type=int, default=100,
                        help='Max decoded CenterNet detections per image during validation')
    parser.add_argument('--val_freq', type=int, default=0,
                        help='Validate every N batches (0=end of epoch only)')
    parser.add_argument('--val_batches_limit', type=int, default=0,
                        help='Max validation batches at epoch end (0=full validation set)')
    parser.add_argument('--mid_epoch_val_batches_limit', type=int, default=20,
                        help='Max validation batches for in-between validation passes (0=full validation set)')
    parser.add_argument('--skip_val_loss', action='store_true',
                        help='Skip expensive validation loss computation (metrics only)')
    parser.add_argument('--mid_epoch_official_like_breakdown', action='store_true',
                        help='Enable per-threshold AP breakdown during in-between validation passes')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder entirely (linear probe mode)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (0=disabled)')
    parser.add_argument('--official_like_breakdown', action='store_true',
                        help='Log additional nuScenes-style AP breakdown at distance thresholds (0.5/1/2/4m)')
    # Dataset + augmentation args
    parser.add_argument('--dataset', type=str, default='nuscenes',
                        choices=['nuscenes', 'waymo', 'flir'],
                        help='Dataset to use')
    parser.add_argument('--waymo_dataroot', type=str, default=None,
                        help='Waymo data root (default: PROJECT_ROOT/waymo_data)')
    parser.add_argument('--flir_dataroot', type=str, default=None,
                        help='FLIR ADAS data root (default: PROJECT_ROOT/flir_adas_v2)')
    parser.add_argument('--flir_train_split', type=str, default='train',
                        help='FLIR training split (e.g. train, train+video_test)')
    parser.add_argument('--flir_val_split', type=str, default='val',
                        help='FLIR validation split')
    parser.add_argument('--flir_img_size', type=int, default=224,
                        help='FLIR auxiliary global-view image size')
    parser.add_argument('--probe_img_size', type=int, default=None,
                        help='Clean probe view size. Used as FLIR encoder input during fine-tuning; also passed through to dataset probe-view pipelines on other datasets')
    parser.add_argument('--probe_train_img_size', type=int, default=None,
                        help='Training probe view size. Defaults to probe_img_size when unset, enabling train/eval resolution splits such as 224 train / 640 eval')
    parser.add_argument('--probe_forward_chunk_size', type=int, default=0,
                        help='Chunk size for probe/patch-token encoder forward passes. 0 disables chunking.')
    parser.add_argument('--encoder_warmup_epochs', type=int, default=1,
                        help='Number of initial epochs to train only the decoder before enabling encoder gradients')
    parser.add_argument('--min_gaussian_radius', type=int, default=2,
                        help='Minimum CenterNet Gaussian radius for object centers. Larger values help small objects produce broader heatmap targets.')
    parser.add_argument('--depth_loss_weight', type=float, default=0.2,
                        help='Weight for depth regression loss relative to other losses. '
                             'Default 0.2 since raw loss_depth is typically ~3-4x larger than other regression losses')
    parser.add_argument('--flir_resize_mode', type=str, default='letterbox',
                        choices=['center_crop', 'letterbox'],
                        help='FLIR image resize mode for probe-style views; ignored by FLIR fine-tuning, which uses a full-image stretch for label alignment')
    parser.add_argument('--precomputed_labels_path', type=str, default=None,
                        help='Path to precomputed det/seg labels (Waymo only). If unset, auto-uses cache/det_seg_labels_v2 when available')
    parser.add_argument('--split_strategy', type=str, default='official',
                        choices=['official', 'scene', 'random'],
                        help='Dataset split strategy')
    parser.add_argument('--official_val_mode', type=str, default='auto',
                        help='Official validation mode (auto, strict)')
    parser.add_argument('--finetune_task', type=str, default='auto',
                        choices=['auto', 'bbox3d', 'bbox2d'],
                        help="Finetune task: 'auto' picks bbox3d for nuscenes/waymo, bbox2d for flir")
    parser.add_argument('--lidar_aug_preset', type=str, default='none',
                        choices=['none', 'light', 'moderate', 'strong'],
                        help='LiDAR augmentation preset')
    parser.add_argument('--copy_paste_preset', type=str, default='none',
                        choices=['none', 'light', 'moderate', 'strong'],
                        help='Copy-paste augmentation preset')
    parser.add_argument('--gt_database_path', type=str, default=None,
                        help='Path to GT database for copy-paste augmentation')
    # Experiment flags (sweep experiments)
    parser.add_argument('--focal_loss', action='store_true',
                        help='Use focal loss for classification (helps with class imbalance)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss (higher = more focus on hard examples)')
    parser.add_argument('--per_class_heatmap_loss', dest='per_class_heatmap_loss', action='store_true',
                        help='Normalize CenterNet heatmap loss per active class so common classes do not dominate the batch loss.')
    parser.add_argument('--no_per_class_heatmap_loss', dest='per_class_heatmap_loss', action='store_false',
                        help='Disable per-class normalization for CenterNet heatmap loss.')
    parser.add_argument('--front_camera_only', action='store_true',
                        help='Train and validate only on CAM_FRONT (no side/rear cameras)')
    parser.add_argument('--class_grouping', type=str, default='none',
                        choices=['none', 'clustered', 'simplified'],
                        help="'clustered'/'simplified': merge into 3 super-classes (vehicle/pedestrian/cyclist) like Waymo")
    parser.add_argument('--class_balanced_sampling', action='store_true',
                        help='Use inverse-frequency weighted sampling to oversample rare classes')
    parser.add_argument('--object_weighted_sampling', action='store_true',
                        help='Sample images proportionally to their object count (more objects = more likely)')
    parser.add_argument('--decoder_type', type=str, default='detr',
                        choices=['detr', 'centernet'],
                        help="'detr': slot-based with Hungarian matching; 'centernet': dense heatmap-based")
    parser.set_defaults(per_class_heatmap_loss=False)

    args = parser.parse_args()

    def _coerce_sweep_override(action, value):
        if value is None:
            return None
        if isinstance(action, argparse._StoreTrueAction):
            if isinstance(value, str):
                return value.strip().lower() in {'1', 'true', 'yes', 'on'}
            return bool(value)
        if isinstance(action, argparse._StoreFalseAction):
            if isinstance(value, str):
                return value.strip().lower() not in {'1', 'true', 'yes', 'on'}
            return not bool(value)
        if action.type is None:
            return value
        if action.nargs in (None, '?'):
            return action.type(value)
        if isinstance(value, (list, tuple)):
            return type(value)(action.type(item) for item in value)
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(',') if part.strip()]
            return [action.type(part) for part in parts]
        return value

    action_by_dest = {
        action.dest: action
        for action in parser._actions
        if getattr(action, 'dest', None)
    }

    # ── Override CLI args with sweep config (wandb.agent mode) ────────────
    if sweep_config is not None:
        for k, v in sweep_config.items():
            if hasattr(args, k):
                action = action_by_dest.get(k)
                coerced = _coerce_sweep_override(action, v) if action is not None else v
                setattr(args, k, coerced)
        args.wandb = True  # Always log in sweep mode

        # If focal_loss is set globally in sweep config, apply it
        if sweep_config.get('focal_loss', False):
            args.focal_loss = True
            args.focal_gamma = sweep_config.get('focal_gamma', 2.0)

        # Interpret the "experiment" pseudo-parameter to activate the right flags
        experiment = sweep_config.get('experiment', None)
        if experiment == 'focal_loss':
            args.focal_loss = True
            args.focal_gamma = sweep_config.get('focal_gamma', 2.0)
        elif experiment == 'front_camera_only':
            args.front_camera_only = True
        elif experiment in ('clustered_classes', 'simplified_classes'):
            args.class_grouping = 'clustered'
        elif experiment == 'class_balanced_sampling':
            args.class_balanced_sampling = True
        elif experiment == 'object_weighted_sampling':
            args.object_weighted_sampling = True
        elif experiment == 'centernet_decoder':
            args.decoder_type = 'centernet'
        
        # Also support simplified as alias for clustered
        if args.class_grouping == 'simplified':
            args.class_grouping = 'clustered'

    # ── Auto-resolve finetune_task based on dataset ──────────────────────
    finetune_task = getattr(args, 'finetune_task', 'auto')
    if finetune_task == 'auto':
        if getattr(args, 'dataset', 'nuscenes') == 'flir':
            finetune_task = 'bbox2d'
        else:
            finetune_task = 'bbox3d'
    args.finetune_task = finetune_task

    # ── Resolve dataset roots ─────────────────────────────────────────────
    dataset_name = getattr(args, 'dataset', 'nuscenes')
    if dataset_name == 'waymo':
        if not getattr(args, 'waymo_dataroot', None):
            args.waymo_dataroot = os.path.join(PROJECT_ROOT, 'waymo_data')
    elif dataset_name == 'flir':
        if not getattr(args, 'flir_dataroot', None):
            args.flir_dataroot = os.path.join(PROJECT_ROOT, 'flir_adas_v2')

    # ── List checkpoints ─────────────────────────────────────────────────
    if args.list_checkpoints:
        ckpts = list_checkpoints()
        if not ckpts:
            print("No checkpoints found.")
            return
        print(f"\n{'='*80}")
        print(f"{'Run Name':<35} {'Arch':<6} {'ViT':<8} {'Scenario':<40}")
        print(f"{'='*80}")
        for c in ckpts:
            print(f"{c['run_name']:<35} {c['arch']:<6} {c['vit_size']:<8} {c['scenario']:<40}")
        print(f"\nTotal: {len(ckpts)} checkpoints")
        return

    # ── Setup ─────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Detection Fine-tuning ({args.finetune_task})")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Dataset: {getattr(args, 'dataset', 'nuscenes')}")
    print(f"Task: {args.finetune_task}")
    print(f"Data fraction: {args.data_fraction}")
    print(f"Encoder LR: {args.encoder_lr}")
    print(f"Decoder LR: {args.decoder_lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.bs}")
    print(f"Encoder warmup epochs: {args.encoder_warmup_epochs}")

    # ── Load encoder ──────────────────────────────────────────────────────
    if args.checkpoint:
        # Use robust checkpoint resolution (like sweep_agent_novel)
        resolved = _resolve_checkpoint_path(args.checkpoint)
        if resolved is not None:
            args.checkpoint = str(resolved)
        encoder, arch, vit_size, vit_dim, ckpt_config = load_checkpoint(args.checkpoint, device)
        simple_baseline = ckpt_config.get('simple_baseline', False)
        simple_modality = ckpt_config.get('simple_modality', 'rgb')
        if ckpt_config.get('baseline_encoder') == 'dinov3_frozen' and not args.freeze_encoder:
            args.freeze_encoder = True
            print("🔒 Auto-forcing freeze_encoder=True for DINOv3 frozen baseline")
        # Auto-infer dataset from checkpoint if not explicitly set
        inferred_dataset = ckpt_config.get('dataset', ckpt_config.get('dataset_name'))
        if inferred_dataset and sweep_config is not None and 'dataset' not in sweep_config:
            args.dataset = inferred_dataset
            print(f"  📦 Auto-inferred dataset from checkpoint: {inferred_dataset}")
        # Log scenario info
        scenario = ckpt_config.get('scenario', 'unknown')
        print(f"  📦 Checkpoint scenario: {scenario}")
    elif args.debug:
        # Debug without checkpoint: create fresh encoder
        arch = 'C'
        vit_size = 'small'
        vit_cfg = get_vit_config(vit_size)
        vit_dim = vit_cfg['vit_dim']
        ckpt_config = {'scenario': 'debug', 'arch': arch, 'vit_size': vit_size,
                       'aligned_mode': True, 'lidar_mode': 'depth', 'modality2_channels': 1}
        encoder = create_mm_encoder(arch=arch, proj_dim=16, vit_size=vit_size, aligned_mode=True).to(device)
        simple_baseline = False
        simple_modality = 'rgb'
        print("🐞 DEBUG: Using fresh encoder (no checkpoint)")
    else:
        print("❌ Must specify --checkpoint or --debug")
        parser.print_help()
        return

    if arch == 'A' and not simple_baseline:
        print("❌ arch='A' does not expose patch tokens for this decoder pipeline.")
        print("   Use checkpoints from arch B/C/D/E/F or a simple baseline checkpoint.")
        return

    # Freeze encoder if requested
    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        print("🔒 Encoder FROZEN (linear probe mode)")

    # ── Create decoder ────────────────────────────────────────────────────
    # Determine num_classes based on dataset + grouping
    dataset_name = getattr(args, 'dataset', 'nuscenes')
    is_flir = (dataset_name == 'flir')
    encoder_probe_img_size = None
    if is_flir:
        encoder_probe_img_size = int(getattr(args, 'probe_img_size', 0) or getattr(args, 'flir_img_size', 224))
        if encoder_probe_img_size % 16 != 0:
            raise ValueError(f"FLIR fine-tune probe_img_size must be divisible by 16, got {encoder_probe_img_size}")
    elif dataset_name == 'waymo' and getattr(args, 'probe_img_size', None):
        encoder_probe_img_size = int(args.probe_img_size)
        if encoder_probe_img_size % 16 != 0:
            raise ValueError(f"Waymo fine-tune probe_img_size must be divisible by 16, got {encoder_probe_img_size}")

    train_probe_img_size = None
    if getattr(args, 'probe_train_img_size', None):
        train_probe_img_size = int(args.probe_train_img_size)
        if train_probe_img_size % 16 != 0:
            raise ValueError(f"probe_train_img_size must be divisible by 16, got {train_probe_img_size}")

    if is_flir:
        if args.class_grouping == 'clustered':
            effective_num_classes = NUM_FLIR_GROUPED_CLASSES
            effective_class_names = FLIR_GROUPED_CLASSES
            print(f"🏷️  FLIR class grouping: clustered → {effective_num_classes} classes: {effective_class_names}")
        else:
            effective_num_classes = NUM_FLIR_2D_CLASSES
            effective_class_names = FLIR_2D_DET_CLASS_NAMES
            print(f"🏷️  FLIR 2D detection: {effective_num_classes} classes: {effective_class_names}")
        # Force CenterNet for FLIR 2D (heatmap-based is natural for 2D detection)
        if args.decoder_type == 'detr' and args.finetune_task == 'bbox2d':
            args.decoder_type = 'centernet'
            print("  ↳ Auto-switching to CenterNet decoder for FLIR 2D detection")
    elif args.class_grouping == 'clustered':
        effective_num_classes = NUM_GROUPED_CLASSES
        effective_class_names = GROUPED_CLASSES
        print(f"🏷️  Class grouping: clustered → {effective_num_classes} classes: {effective_class_names}")
    else:
        effective_num_classes = NUM_DETECTION_CLASSES
        effective_class_names = DETECTION_CLASSES

    # Create decoder based on type
    if args.decoder_type == 'centernet':
        decoder_patch_grid = (encoder_probe_img_size // 16) if encoder_probe_img_size else 14
        decoder = EnhancedCenterNetDecoder(
            vit_dim=vit_dim,
            hidden_dim=args.decoder_dim,
            num_classes=effective_num_classes,
            patch_grid=decoder_patch_grid,
            upsample_factor=2,  # 14→28 for finer localization
            num_conv_layers=args.decoder_layers if args.decoder_layers <= 6 else 3,
            focal_gamma=args.focal_gamma if args.focal_loss else 2.0,
            max_objects=args.max_objects,
            log_space_3d=(not is_flir),
            per_class_heatmap_loss=args.per_class_heatmap_loss,
            min_gaussian_radius=args.min_gaussian_radius,
        ).to(device)
        print(
            f"🎯 Decoder: CenterNet (heatmap-based, {args.decoder_dim}D, "
            f"{decoder.output_grid}×{decoder.output_grid} grid, min_radius={args.min_gaussian_radius}, "
            f"per_class_heatmap_loss={args.per_class_heatmap_loss})"
        )
    else:  # detr
        decoder = EnhancedBBox3DDecoder(
            vit_dim=vit_dim,
            hidden_dim=args.decoder_dim,
            max_objects=args.max_objects,
            num_classes=effective_num_classes,
            num_decoder_layers=args.decoder_layers,
            matching=args.matching,
            focal_gamma=args.focal_gamma if args.focal_loss else 0.0,
        ).to(device)
        print(f"🎯 Decoder: DETR ({args.decoder_layers} layers, {args.max_objects} queries, {args.matching} matching)")

    n_enc_params = sum(p.numel() for p in encoder.parameters())
    n_dec_params = sum(p.numel() for p in decoder.parameters())
    n_trainable_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n📊 Model parameters:")
    print(f"  Encoder: {n_enc_params/1e6:.1f}M ({n_trainable_enc/1e6:.1f}M trainable)")
    print(f"  Decoder: {n_dec_params/1e6:.1f}M")
    print(f"  Total:   {(n_enc_params + n_dec_params)/1e6:.1f}M")

    # ── Create datasets ───────────────────────────────────────────────────
    lidar_mode = ckpt_config.get('lidar_mode', 'depth')

    if args.debug:
        print("🐞 DEBUG: Using dummy data")
        train_loader = None
        test_loader = None
    else:
        # ── Dataset + augmentation configuration ──────────────────
        dataset_name = getattr(args, 'dataset', 'nuscenes')
        lidar_aug_preset = getattr(args, 'lidar_aug_preset', 'none')
        copy_paste_preset = getattr(args, 'copy_paste_preset', 'none')
        gt_database_path = getattr(args, 'gt_database_path', None)
        precomputed_labels_path = getattr(args, 'precomputed_labels_path', None)

        if dataset_name == 'waymo' and not precomputed_labels_path:
            default_precomputed = os.path.join(PROJECT_ROOT, 'cache', 'det_seg_labels_v2')
            if os.path.exists(default_precomputed):
                precomputed_labels_path = default_precomputed
                print(f"⚡ Using default precomputed labels: {precomputed_labels_path}")

        val_precomputed_labels_path = precomputed_labels_path
        if dataset_name == 'waymo' and precomputed_labels_path:
            validation_cache_dir = Path(precomputed_labels_path) / 'validation'
            if validation_cache_dir.exists() and any(validation_cache_dir.glob('shard_*.zip')):
                val_precomputed_labels_path = str(validation_cache_dir)

        aug_kwargs = dict(
            lidar_aug_preset=lidar_aug_preset,
            copy_paste_preset=copy_paste_preset,
            gt_database_path=gt_database_path,
        )

        if dataset_name == 'waymo':
            from src.waymo_dataset import WaymoDataset, waymo_collate_fn
            waymo_root = getattr(args, 'waymo_dataroot', None) or args.dataroot
            probe_img_size = int(getattr(args, 'probe_img_size', 0) or 224)
            probe_train_img_size = int(train_probe_img_size or probe_img_size)
            train_ds = WaymoDataset(
                waymo_root, split='train', arch=arch, lidar_mode=lidar_mode,
                V=1, local_crops_number=0, modality_dropout=0.0,
                probe_img_size=probe_train_img_size,
                finetune_mode=True,
                include_probe_view=True,
                finetune_crop_scale=getattr(args, 'finetune_crop_scale', (0.8, 1.0)),
                det_seg_label_mode='bbox_only',
                precomputed_labels_path=precomputed_labels_path,
                **aug_kwargs,
            )
            test_ds = WaymoDataset(
                waymo_root, split='val', arch=arch, lidar_mode=lidar_mode,
                V=1, local_crops_number=0, modality_dropout=0.0,
                probe_img_size=probe_img_size,
                finetune_mode=True, include_probe_view=True,
                det_seg_label_mode='bbox_only',
                precomputed_labels_path=val_precomputed_labels_path,
            )
            _collate_fn = waymo_collate_fn
            if probe_train_img_size != probe_img_size:
                print(f"📐 Waymo probe resolution split: train={probe_train_img_size}, val={probe_img_size}")
            print("🎯 Waymo fine-tune uses deterministic probe views for encoder inputs to keep image geometry aligned with center-crop labels")
        elif dataset_name == 'flir':
            from src.flir_dataset import FlirAdasDataset, flir_collate_fn
            flir_root = getattr(args, 'flir_dataroot', None) or os.path.join(PROJECT_ROOT, 'flir_adas_v2')
            flir_train_split = getattr(args, 'flir_train_split', 'train')
            flir_val_split = getattr(args, 'flir_val_split', 'val')
            flir_img_size = int(getattr(args, 'flir_img_size', 224))
            flir_probe_img_size = int(getattr(args, 'probe_img_size', 0) or flir_img_size)
            flir_probe_train_img_size = int(train_probe_img_size or flir_probe_img_size)
            flir_resize_mode = str(getattr(args, 'flir_resize_mode', 'letterbox'))
            # FLIR uses thermal as second modality (depth-like 1-channel)
            flir_label_source = 'rgb'
            if simple_baseline and simple_modality in ('lidar', 'thermal'):
                flir_label_source = 'thermal'
            # FLIR fine-tuning uses the explicit clean probe view as encoder
            # input, matching the probe-training path in sweep_agent_novel.
            # finetune_mode=True disables random flips on auxiliary global
            # views, while include_probe_view=True keeps the synchronized
            # full-image probe tensor available at probe_img_size.
            train_ds = FlirAdasDataset(
                flir_root,
                split=flir_train_split,
                arch=arch,
                lidar_mode='depth',
                V=1,
                global_crops_scale=(1.0, 1.0),
                local_crops_number=0,
                img_size=flir_img_size,
                probe_img_size=flir_probe_train_img_size,
                modality_dropout=0.0,
                include_probe_view=True,
                det_seg_label_mode='bbox_only',
                detection_label_source=flir_label_source,
                resize_mode=flir_resize_mode,
                finetune_mode=True,
            )
            test_ds = FlirAdasDataset(
                flir_root,
                split=flir_val_split,
                arch=arch,
                lidar_mode='depth',
                V=1,
                global_crops_scale=(1.0, 1.0),
                local_crops_number=0,
                img_size=flir_img_size,
                probe_img_size=flir_probe_img_size,
                modality_dropout=0.0,
                include_probe_view=True,
                det_seg_label_mode='bbox_only',
                detection_label_source=flir_label_source,
                resize_mode=flir_resize_mode,
                finetune_mode=True,
            )
            _collate_fn = flir_collate_fn
            print(f"📷 FLIR dataset: train={flir_train_split}, val={flir_val_split}, "
                f"global_img_size={flir_img_size}, train_probe_img_size={flir_probe_train_img_size}, "
                f"val_probe_img_size={flir_probe_img_size}, labels={flir_label_source}")
            print(f"  ↳ encoder input uses explicit clean probe view at {flir_probe_train_img_size}x{flir_probe_train_img_size} during training")
            if flir_probe_train_img_size != flir_probe_img_size:
                print(f"  ↳ validation keeps explicit clean probe view at {flir_probe_img_size}x{flir_probe_img_size}")
            print(f"  ↳ flir_resize_mode='{flir_resize_mode}' is ignored during FLIR fine-tuning")
        else:
            nuscenes_split_strategy = getattr(args, 'split_strategy', 'official')
            nuscenes_val_mode = getattr(args, 'official_val_mode', 'auto')
            train_ds = MMNuScenesDataset(
                args.dataroot, split='train', arch=arch, lidar_mode=lidar_mode,
                V=1, local_crops_number=0, modality_dropout=0.0,
                split_strategy=nuscenes_split_strategy,
                official_val_mode=nuscenes_val_mode,
                finetune_mode=True,
                finetune_crop_scale=getattr(args, 'finetune_crop_scale', (0.8, 1.0)),
                det_seg_label_mode='bbox_only',
                **aug_kwargs,
            )
            test_ds = MMNuScenesDataset(
                args.dataroot, split='val', arch=arch, lidar_mode=lidar_mode,
                V=1, local_crops_number=0, modality_dropout=0.0,
                split_strategy=nuscenes_split_strategy,
                official_val_mode=nuscenes_val_mode,
                finetune_mode=True, det_seg_label_mode='bbox_only',
            )
            _collate_fn = mm_collate_fn

        # ── Front-camera-only filtering ─────────────────────────────
        if args.front_camera_only:
            preferred_front_name = 'FRONT' if dataset_name == 'waymo' else 'CAM_FRONT'

            def _is_front_camera(camera_name: str) -> bool:
                if not isinstance(camera_name, str):
                    return False
                cam = camera_name.strip().upper()
                # Support both naming schemes: nuScenes (CAM_FRONT) and Waymo (FRONT)
                return cam in {'CAM_FRONT', 'FRONT'}

            def _get_front_indices(ds):
                """Return indices of front-camera samples only."""
                if hasattr(ds, 'pairs_camera_names') and ds.pairs_camera_names is not None:
                    cam_names = np.asarray(ds.pairs_camera_names)
                    if cam_names.size == 0:
                        return []
                    cam_names_upper = np.char.upper(cam_names.astype(str))
                    mask = (cam_names_upper == preferred_front_name) | (cam_names_upper == 'CAM_FRONT') | (cam_names_upper == 'FRONT')
                    return np.where(mask)[0].tolist()
                elif hasattr(ds, 'pairs') and ds.pairs is not None:
                    return [i for i, p in enumerate(ds.pairs) if _is_front_camera(p.get('camera_name', ''))]
                else:
                    # Reconstruct from compact pairs
                    return [i for i in range(len(ds)) if _is_front_camera(ds._get_pair(i).get('camera_name', ''))]

            train_front_idx = _get_front_indices(train_ds)
            test_front_idx = _get_front_indices(test_ds)
            train_ds = Subset(train_ds, train_front_idx)
            test_ds = Subset(test_ds, test_front_idx)
            print(f"📸 Front-camera-only: train={len(train_ds)}, val={len(test_ds)}")

        # Guard against empty training split after filtering/config selection
        n_total = len(train_ds)
        if n_total == 0:
            raise ValueError(
                "Training dataset is empty (0 samples) after split/filtering. "
                f"dataset={dataset_name}, dataroot={args.dataroot}, "
                f"waymo_dataroot={getattr(args, 'waymo_dataroot', None)}, "
                f"front_camera_only={args.front_camera_only}. "
                "Check dataset paths, split contents, and filtering flags."
            )

        # Apply data fraction
        n_subset = max(1, int(n_total * args.data_fraction))
        if args.data_fraction < 1.0:
            rng = random.Random(args.seed)
            indices = list(range(n_total))
            rng.shuffle(indices)
            indices = sorted(indices[:n_subset])  # Sort for locality
            train_ds_subset = Subset(train_ds, indices)
            print(f"\n📊 Training data: {n_subset}/{n_total} samples ({args.data_fraction*100:.1f}%)")
        else:
            train_ds_subset = train_ds
            print(f"\n📊 Training data: {n_total} samples (100%)")

        if len(train_ds_subset) == 0:
            raise ValueError(
                "Training subset is empty after applying --data_fraction and filtering. "
                f"n_total={n_total}, data_fraction={args.data_fraction}."
            )

        print(f"📊 Validation data: {len(test_ds)} samples (100%)")

        subset_object_counts = None
        if dataset_name in ('waymo', 'nuscenes', 'flir'):
            _base_ds = train_ds_subset.dataset if isinstance(train_ds_subset, Subset) else train_ds_subset
            _indices = train_ds_subset.indices if isinstance(train_ds_subset, Subset) else list(range(len(train_ds_subset)))
            subset_object_counts = np.array([
                _count_sample_detection_objects(
                    _base_ds,
                    _base_ds._get_pair(idx) if hasattr(_base_ds, '_get_pair') else _base_ds.pairs[idx],
                )
                for idx in _indices
            ], dtype=np.int32)
            if subset_object_counts.size > 0:
                print(
                    "📦 Detection subset composition: "
                    f"min={subset_object_counts.min()}, max={subset_object_counts.max()}, "
                    f"mean={subset_object_counts.mean():.1f}, median={np.median(subset_object_counts):.0f}"
                )
                print(
                    f"  Samples with 0 objects: {(subset_object_counts == 0).sum()} "
                    f"({100 * (subset_object_counts == 0).mean():.1f}%)"
                )
                print(
                    f"  Samples with 1+ objects: {(subset_object_counts > 0).sum()} "
                    f"({100 * (subset_object_counts > 0).mean():.1f}%)"
                )
                print(
                    f"  Samples with 5+ objects: {(subset_object_counts >= 5).sum()} "
                    f"({100 * (subset_object_counts >= 5).mean():.1f}%)"
                )
                if (subset_object_counts > 0).sum() == 0:
                    print("⚠️  Detection subset contains no positive samples; results will be degenerate.")

        # ── Class-balanced sampling ──────────────────────────────────
        train_sampler = None
        if args.class_balanced_sampling:
            print("⚖️  Computing class-balanced sample weights...")
            # Get the underlying dataset (unwrap Subset if needed)
            _base_ds = train_ds_subset.dataset if isinstance(train_ds_subset, Subset) else train_ds_subset
            _indices = train_ds_subset.indices if isinstance(train_ds_subset, Subset) else list(range(len(train_ds_subset)))

            # Count per-class object frequency across the dataset
            class_counts = np.zeros(effective_num_classes, dtype=np.float64)
            sample_class_flags = []  # per-sample: set of classes present
            for idx in _indices:
                pair = _base_ds._get_pair(idx) if hasattr(_base_ds, '_get_pair') else _base_ds.pairs[idx]
                cls_set = _extract_sample_detection_classes(_base_ds, pair, args.class_grouping)
                sample_class_flags.append(cls_set)
                for c in cls_set:
                    class_counts[c] += 1

            # Inverse frequency weights (with smoothing)
            class_counts = np.maximum(class_counts, 1.0)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_weights)

            # Per-sample weight = max class weight among its objects (focus on rare)
            sample_weights = []
            for cls_set in sample_class_flags:
                if cls_set:
                    w = max(class_weights[c] for c in cls_set)
                else:
                    w = 1.0  # background/empty samples get neutral weight
                sample_weights.append(w)

            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            print(f"  Class counts: {dict(enumerate(class_counts.astype(int).tolist()))}")
            print(f"  Class weights: {dict(enumerate(np.round(class_weights, 2).tolist()))}")

        # ── Object-weighted sampling (sample more images with more objects) ──
        elif args.object_weighted_sampling:
            print("📦 Computing object-count-weighted sample weights...")
            _base_ds = train_ds_subset.dataset if isinstance(train_ds_subset, Subset) else train_ds_subset
            _indices = train_ds_subset.indices if isinstance(train_ds_subset, Subset) else list(range(len(train_ds_subset)))

            sample_weights = []
            object_counts = []
            for idx in _indices:
                pair = _base_ds._get_pair(idx) if hasattr(_base_ds, '_get_pair') else _base_ds.pairs[idx]
                n_obj = _count_sample_detection_objects(_base_ds, pair)
                
                object_counts.append(n_obj)
                # Weight = sqrt(object_count + 1) to favor samples with objects
                # sqrt dampens extreme differences (e.g., 50 objects vs 5)
                w = np.sqrt(n_obj + 1)
                sample_weights.append(w)

            # Normalize weights
            total_w = sum(sample_weights)
            sample_weights = [w / total_w * len(sample_weights) for w in sample_weights]

            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            
            if subset_object_counts is None:
                obj_counts_arr = np.array(object_counts)
                print(f"  Object count stats: min={obj_counts_arr.min()}, max={obj_counts_arr.max()}, "
                    f"mean={obj_counts_arr.mean():.1f}, median={np.median(obj_counts_arr):.0f}")
                print(f"  Samples with 0 objects: {(obj_counts_arr == 0).sum()} ({100*(obj_counts_arr == 0).mean():.1f}%)")
                print(f"  Samples with 5+ objects: {(obj_counts_arr >= 5).sum()} ({100*(obj_counts_arr >= 5).mean():.1f}%)")

        train_drop_last = len(train_ds_subset) >= args.bs
        if not train_drop_last:
            print(
                f"⚠️  Training subset ({len(train_ds_subset)}) is smaller than batch size ({args.bs}); "
                "using drop_last=False to keep at least one batch."
            )

        train_loader = DataLoader(
            train_ds_subset,
            batch_size=args.bs,
            shuffle=(train_sampler is None),  # sampler and shuffle are mutually exclusive
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=train_drop_last,
            persistent_workers=(args.num_workers > 0),
            collate_fn=_collate_fn,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
            collate_fn=_collate_fn,
        )

    # ── Optimizer with differential LR ─────────────────────────────────
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    decoder_params = list(decoder.parameters())

    param_groups = []
    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': args.encoder_lr})
    param_groups.append({'params': decoder_params, 'lr': args.decoder_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)

    # Learning rate schedule: warmup (5%) + cosine annealing
    total_steps = args.epochs * (len(train_loader) if train_loader else 10)
    warmup_steps = max(1, total_steps // 20)  # 5% warmup

    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])

    scaler = GradScaler()

    # ── WandB ─────────────────────────────────────────────────────────────
    use_wandb = WANDB_AVAILABLE and args.wandb and not args.debug
    if use_wandb:
        if wandb.run is None:
            wandb_key_path = os.path.join(PROJECT_ROOT, '.wandb_key')
            if os.path.exists(wandb_key_path):
                with open(wandb_key_path) as f:
                    wandb.login(key=f.read().strip())

            run = wandb.init(
                project=args.wandb_project,
                config={
                    'checkpoint': str(args.checkpoint),
                    'arch': arch,
                    'vit_size': vit_size,
                    'scenario': ckpt_config.get('scenario', 'unknown'),
                    'dataset': dataset_name,
                    'finetune_task': args.finetune_task,
                    'data_fraction': args.data_fraction,
                    'encoder_lr': args.encoder_lr,
                    'decoder_lr': args.decoder_lr,
                    'epochs': args.epochs,
                    'batch_size': args.bs,
                    'probe_img_size': args.probe_img_size,
                    'probe_train_img_size': args.probe_train_img_size,
                    'probe_forward_chunk_size': args.probe_forward_chunk_size,
                    'encoder_warmup_epochs': args.encoder_warmup_epochs,
                    'decoder_layers': args.decoder_layers,
                    'decoder_dim': args.decoder_dim,
                    'matching': args.matching,
                    'finetune_crop_scale': tuple(args.finetune_crop_scale),
                    'max_objects': args.max_objects,
                    'freeze_encoder': args.freeze_encoder,
                    'seed': args.seed,
                    'focal_loss': args.focal_loss,
                    'focal_gamma': args.focal_gamma if args.focal_loss else 0.0,
                    'front_camera_only': args.front_camera_only,
                    'class_grouping': args.class_grouping,
                    'class_balanced_sampling': args.class_balanced_sampling,
                    'effective_num_classes': effective_num_classes,
                    'decode_max_detections': args.decode_max_detections,
                },
            )
        else:
            run = wandb.run
        run_name = wandb.run.name
    else:
        run_name = f"finetune_{arch}_{args.data_fraction}"

    # ── Detection metrics ─────────────────────────────────────────────────
    if is_flir:
        if args.class_grouping == 'clustered':
            det_metrics = NuScenesDetectionMetrics(
                class_names=FLIR_GROUPED_CLASSES, num_classes=NUM_FLIR_GROUPED_CLASSES,
                matching_plane='xy')
        else:
            det_metrics = NuScenesDetectionMetrics(
                class_names=FLIR_2D_DET_CLASS_NAMES, num_classes=NUM_FLIR_2D_CLASSES,
                matching_plane='xy')
    elif args.class_grouping == 'clustered':
        det_metrics = NuScenesDetectionMetrics(
            class_names=GROUPED_CLASSES, num_classes=NUM_GROUPED_CLASSES)
    else:
        det_metrics = NuScenesDetectionMetrics()

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting fine-tuning: {run_name}")
    print(f"{'='*60}")

    best_mAP = 0.0
    best_NDS = 0.0 if not is_flir else None
    epochs_without_improvement = 0
    encoder_warmup_epochs = 0 if args.freeze_encoder else max(0, int(getattr(args, 'encoder_warmup_epochs', 0)))
    encoder_trainable = not args.freeze_encoder
    centernet_loss_weights = {'depth': getattr(args, 'depth_loss_weight', 0.2)} if args.decoder_type == 'centernet' else {}

    for epoch in range(args.epochs):
        epoch_start = time.time()
        should_train_encoder = (not args.freeze_encoder) and (epoch >= encoder_warmup_epochs)
        if should_train_encoder != encoder_trainable:
            for p in encoder.parameters():
                p.requires_grad = should_train_encoder
            encoder_trainable = should_train_encoder
            state = 'UNFROZEN' if encoder_trainable else 'decoder warmup only'
            print(f"🔁 Encoder state changed: {state}")

        encoder.train()
        decoder.train()

        epoch_loss = 0.0
        epoch_steps = 0

        if args.debug:
            # Debug: single batch of random data
            n_batches = 1
        else:
            n_batches = len(train_loader)

        if args.debug:
            data_iter = range(n_batches)
        else:
            data_iter = enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"))

        for step_data in data_iter:
            if args.debug:
                batch_idx = 0
                cam_views, modality2, labels = generate_dummy_batch(
                    args.bs, arch, device, simple_baseline, simple_modality,
                    modality2_channels=ckpt_config.get('modality2_channels', 1))
            else:
                batch_idx, (cam_views, modality2, labels) = step_data
                cam_views = to_device(cam_views, device)
                modality2 = to_device(modality2, device)

            use_probe_encoder_views = is_flir or dataset_name == 'waymo'
            encoder_cam_views = _prefer_probe_views(cam_views) if use_probe_encoder_views else cam_views
            encoder_modality2 = _prefer_probe_views(modality2) if use_probe_encoder_views else modality2
            encoder_cam_views, encoder_modality2, _ = _align_modal_view_counts(encoder_cam_views, encoder_modality2)

            optimizer.zero_grad()

            with autocast(enabled=(device == 'cuda')):
                # Extract patch tokens from encoder
                patch_tokens = extract_patch_tokens(
                    encoder,
                    encoder_cam_views,
                    encoder_modality2,
                    arch,
                    simple_baseline=simple_baseline,
                    simple_modality=simple_modality if simple_baseline else 'rgb',
                    batch_chunk_size=args.probe_forward_chunk_size,
                )

                if isinstance(encoder_cam_views, (dict, torch.Tensor)) and isinstance(encoder_modality2, (dict, torch.Tensor)):
                    _, _, aligned = _align_modal_view_counts(encoder_cam_views, encoder_modality2)
                    if aligned and batch_idx == 0:
                        print("  ⚠️  Auto-aligned camera/LiDAR view counts for patch extraction")

                if patch_tokens is None:
                    print(f"  ⚠️  No patch tokens at step {batch_idx}, skipping")
                    continue

                # Handle multi-view (take last view)
                cam_stats_src = encoder_cam_views
                B_train, V_train = get_input_stats(cam_stats_src)
                if V_train > 1 and patch_tokens.shape[0] == B_train * V_train:
                    patch_tokens = patch_tokens.view(B_train, V_train, patch_tokens.shape[1], patch_tokens.shape[2])[:, -1]

                # Decoder forward
                predictions = decoder(patch_tokens)

                # Build targets (adapt for FLIR 2D vs NuScenes/Waymo 3D)
                if is_flir:
                    # FLIR 2D detection: use gt_classes_2d, gt_centers_2d, gt_mask_2d
                    det_targets = {
                        'gt_classes': labels['gt_classes_2d'].to(device).long(),
                        'gt_centers': torch.cat([
                            labels['gt_centers_2d'].to(device).float(),
                            torch.zeros(labels['gt_centers_2d'].shape[0], labels['gt_centers_2d'].shape[1], 1, device=device),
                        ], dim=-1),  # (B, N, 3) with z=0 for 2D
                        'gt_sizes': labels.get('gt_sizes', torch.zeros(labels['gt_classes_2d'].shape[0], labels['gt_classes_2d'].shape[1], 3)).to(device).float(),
                        'gt_orientations': labels.get('gt_orientations', torch.zeros(labels['gt_classes_2d'].shape[0], labels['gt_classes_2d'].shape[1], 2)).to(device).float(),
                        'gt_mask': labels['gt_mask_2d'].to(device).float(),
                        'gt_centers_2d': labels['gt_centers_2d'].to(device).float(),
                    }
                    if args.class_grouping == 'clustered':
                        det_targets['gt_classes'], det_targets['gt_mask'] = remap_flir_gt_classes(
                            det_targets['gt_classes'], det_targets['gt_mask'], device)
                else:
                    det_targets = {
                        'gt_classes': labels['gt_classes'].to(device).long(),
                        'gt_centers': labels['gt_centers'].to(device).float(),
                        'gt_sizes': labels['gt_sizes'].to(device).float(),
                        'gt_orientations': labels['gt_orientations'].to(device).float(),
                        'gt_mask': labels['gt_mask'].to(device).float(),
                    }
                    # Remap classes for clustered grouping
                    if args.class_grouping == 'clustered':
                        det_targets['gt_classes'], det_targets['gt_mask'] = remap_gt_classes(
                            det_targets['gt_classes'], det_targets['gt_mask'], device)

                # Compute loss (CenterNet needs dense targets; DETR uses direct)
                if args.decoder_type == 'centernet':
                    # CenterNet needs 2D normalized centers for heatmap generation
                    gt_centers_2d = det_targets.get('gt_centers_2d',
                        labels.get('gt_centers_2d', det_targets['gt_centers'][:, :, :2])).to(device).float()
                    gt_depths = det_targets['gt_centers'][:, :, 2].float()
                    centernet_targets = decoder.generate_targets(
                        gt_centers_2d=gt_centers_2d,
                        gt_depths=gt_depths,
                        gt_sizes=det_targets['gt_sizes'],
                        gt_orientations=det_targets['gt_orientations'],
                        gt_classes=det_targets['gt_classes'],
                        gt_mask=det_targets['gt_mask'],
                        output_grid=predictions['heatmap'].shape[-2:],
                    )
                    losses = decoder.compute_loss(predictions, centernet_targets,
                                                   loss_weights=centernet_loss_weights)
                else:
                    losses = decoder.compute_loss(predictions, det_targets)
                loss = losses['loss_total']

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1

            # First-batch debug diagnostics
            if epoch == 0 and batch_idx == 0 and args.decoder_type == 'centernet':
                with torch.no_grad():
                    _rm = centernet_targets['reg_mask']
                    _np = _rm.sum().item()
                    _dp = predictions['depth']
                    _dt = centernet_targets['depth']
                    _dp_pos = _dp[_rm.bool()]
                    _dt_pos = _dt[_rm.bool()]
                    print(f"  🔍 Step-0 debug: n_pos={_np:.0f}, "
                          f"depth_pred=[{_dp_pos.mean():.2f}±{_dp_pos.std():.2f}], "
                          f"depth_tgt=[{_dt_pos.mean():.2f}±{_dt_pos.std():.2f}], "
                          f"loss_depth={losses['loss_depth'].item():.3f}")

            # Periodic logging
            if use_wandb and batch_idx % 10 == 0:
                lr_enc = optimizer.param_groups[0]['lr'] if encoder_params else 0
                lr_dec = optimizer.param_groups[-1]['lr']
                log_dict = {
                    'train/loss': loss.item(),
                    'train/lr_encoder': lr_enc,
                    'train/lr_decoder': lr_dec,
                    'train/epoch': epoch,
                }
                # Log decoder-specific losses
                if args.decoder_type == 'centernet':
                    log_dict.update({
                        'train/loss_hm': losses.get('loss_hm', torch.tensor(0)).item(),
                        'train/loss_offset': losses.get('loss_offset', torch.tensor(0)).item(),
                        'train/loss_size': losses.get('loss_size', torch.tensor(0)).item(),
                        'train/loss_depth': losses.get('loss_depth', torch.tensor(0)).item(),
                        'train/loss_rot': losses.get('loss_rot', torch.tensor(0)).item(),
                    })
                else:
                    log_dict.update({
                        'train/loss_cls': losses.get('loss_cls', torch.tensor(0)).item(),
                        'train/loss_center': losses.get('loss_center', torch.tensor(0)).item(),
                        'train/loss_size': losses.get('loss_size', torch.tensor(0)).item(),
                        'train/loss_orient': losses.get('loss_orient', torch.tensor(0)).item(),
                        'train/loss_conf': losses.get('loss_conf', torch.tensor(0)).item(),
                    })
                wandb.log(log_dict)

            # Periodic validation
            if args.val_freq > 0 and (batch_idx + 1) % args.val_freq == 0 and not args.debug:
                mid_epoch_limit = None if args.mid_epoch_val_batches_limit <= 0 else args.mid_epoch_val_batches_limit
                val_metrics = validate(encoder, decoder, test_loader, device, arch,
                                       simple_baseline, simple_modality, det_metrics,
                                       compute_loss=(not args.skip_val_loss),
                                       official_like_breakdown=args.mid_epoch_official_like_breakdown,
                                       limit_batches=mid_epoch_limit,
                                       class_grouping=args.class_grouping,
                                       decoder_type=args.decoder_type,
                                       decode_max_detections=args.decode_max_detections,
                                       probe_forward_chunk_size=args.probe_forward_chunk_size,
                                       is_flir=is_flir,
                                       loss_weights=centernet_loss_weights)
                if use_wandb:
                    mid_val_log = {f'val_mid/{k}': v for k, v in val_metrics.items()}
                    mid_val_log.update({'val_mid/epoch': epoch, 'val_mid/step': batch_idx + 1})
                    wandb.log(mid_val_log)
                mid_map = val_metrics.get('mAP', 0.0)
                print(f"  Mid-epoch val @ step {batch_idx + 1}: mAP={mid_map:.4f}")
                encoder.train()
                decoder.train()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n  Epoch {epoch+1}/{args.epochs} — avg loss: {avg_loss:.4f} ({epoch_time:.1f}s)")

        # Full validation at end of epoch
        if not args.debug and test_loader is not None:
            is_last_epoch = (epoch == args.epochs - 1)
            epoch_val_limit = None if is_last_epoch else (args.val_batches_limit if args.val_batches_limit > 0 else None)
            val_metrics = validate(encoder, decoder, test_loader, device, arch,
                                   simple_baseline, simple_modality, det_metrics,
                                   limit_batches=epoch_val_limit,
                                   compute_loss=(not args.skip_val_loss),
                                   official_like_breakdown=args.official_like_breakdown,
                                   class_grouping=args.class_grouping,
                                   decoder_type=args.decoder_type,
                                   decode_max_detections=args.decode_max_detections,
                                   probe_forward_chunk_size=args.probe_forward_chunk_size,
                                   is_flir=is_flir,
                                   loss_weights=centernet_loss_weights)
            mAP = val_metrics.get('mAP', 0.0)

            # NDS is only meaningful for 3D detection (NuScenes/Waymo) where
            # size/orientation predictions are real.  For FLIR 2D detection
            # the geometry fields are zero-filled placeholders, so NDS would
            # be misleading — we report mAP only.
            if is_flir:
                nds = None
                print(f"  Val mAP: {mAP:.4f} (FLIR 2D — NDS not applicable)")
            else:
                nds = compute_nds(val_metrics)
                val_metrics['NDS'] = nds
                print(f"  Val mAP: {mAP:.4f} | NDS: {nds:.4f} | mATE: {val_metrics.get('mATE', 0):.4f} | "
                      f"mASE: {val_metrics.get('mASE', 0):.4f} | mAOE: {val_metrics.get('mAOE', 0):.4f}")

            if 'mAP_xy_match' in val_metrics or 'mAP_xz_match' in val_metrics:
                xy_map = val_metrics.get('mAP_xy_match', float('nan'))
                xz_map = val_metrics.get('mAP_xz_match', float('nan'))
                xy_ade = val_metrics.get('mADE_xy_match', float('nan'))
                xz_ade = val_metrics.get('mADE_xz_match', float('nan'))
                print(f"  Match planes: XY mAP={xy_map:.4f}, mADE={xy_ade:.4f} | XZ mAP={xz_map:.4f}, mADE={xz_ade:.4f}")
                if not is_flir:
                    primary_plane = 'xz'
                    _print_plane_class_breakdown(val_metrics, effective_class_names, primary_plane=primary_plane)

            if use_wandb:
                # Core metrics under val/
                log_dict = {
                    'val/mAP': mAP,
                    'val/loss': val_metrics.get('loss', 0),
                    'val/epoch': epoch,
                    'val/epoch_time_s': epoch_time,
                    # Match naming from sweep_agent_novel for easy comparison
                    'val/detr_mAP': mAP,
                }
                if not is_flir:
                    # 3D-specific metrics (meaningless for FLIR 2D)
                    log_dict.update({
                        'val/NDS': nds,
                        'val/mATE': val_metrics.get('mATE', 0),
                        'val/mASE': val_metrics.get('mASE', 0),
                        'val/mAOE': val_metrics.get('mAOE', 0),
                        'val/mADE': val_metrics.get('mADE', 0),
                    })
                for k, v in val_metrics.items():
                    if '/' not in k and k not in {'loss', 'mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mADE'}:
                        log_dict[f'val/{k}'] = v
                # Per-class metrics under det/finetune/<class>/
                for k, v in val_metrics.items():
                    if '/' in k:  # e.g. "car/AP", "truck/ATE"
                        log_dict[f'det/finetune/{k}'] = v
                wandb.log(log_dict)

            if mAP > best_mAP:
                best_mAP = mAP
                if not is_flir:
                    best_NDS = nds
                epochs_without_improvement = 0
                # Save best model
                save_dir = SAVED_MODELS_DIR / f"finetune_{run_name}"
                save_dir.mkdir(parents=True, exist_ok=True)
                best_config = {
                    **ckpt_config,
                    'dataset': dataset_name,
                    'finetune_task': args.finetune_task,
                    'data_fraction': args.data_fraction,
                    'encoder_lr': args.encoder_lr,
                    'decoder_lr': args.decoder_lr,
                    'best_mAP': best_mAP,
                    'epoch': epoch,
                }
                if not is_flir:
                    best_config['best_NDS'] = best_NDS
                torch.save({
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'config': best_config,
                    'arch': arch,
                    'run_name': run_name,
                }, save_dir / 'best.pt')
                if is_flir:
                    print(f"  ✓ New best! mAP={best_mAP:.4f}, saved to {save_dir / 'best.pt'}")
                else:
                    print(f"  ✓ New best! mAP={best_mAP:.4f}, NDS={best_NDS:.4f}, saved to {save_dir / 'best.pt'}")
            else:
                epochs_without_improvement += 1
                if args.patience > 0 and epochs_without_improvement >= args.patience:
                    print(f"  ⏹ Early stopping: no improvement for {args.patience} epochs")
                    break

        # Save last checkpoint every epoch
        if not args.debug:
            save_dir = SAVED_MODELS_DIR / f"finetune_{run_name}"
            save_dir.mkdir(parents=True, exist_ok=True)
            last_config = {
                **ckpt_config,
                'dataset': dataset_name,
                'finetune_task': args.finetune_task,
                'data_fraction': args.data_fraction,
                'encoder_lr': args.encoder_lr,
                'decoder_lr': args.decoder_lr,
                'best_mAP': best_mAP,
                'epoch': epoch,
            }
            if not is_flir:
                last_config['best_NDS'] = best_NDS
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'config': last_config,
                'arch': arch,
                'run_name': run_name,
            }, save_dir / 'last.pt')

    # ── Cleanup ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if is_flir:
        print(f"Fine-tuning complete! Best mAP: {best_mAP:.4f}")
    else:
        print(f"Fine-tuning complete! Best mAP: {best_mAP:.4f} | Best NDS: {best_NDS:.4f}")
    print(f"{'='*60}")

    if use_wandb:
        summary_log = {'best_mAP': best_mAP}
        if not is_flir:
            summary_log['best_NDS'] = best_NDS
        wandb.log(summary_log)
        wandb.finish()

    # Cleanup
    del encoder, decoder, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate(encoder, decoder, test_loader, device, arch,
             simple_baseline=False, simple_modality='rgb',
             det_metrics=None, limit_batches=None,
             compute_loss: bool = True,
             official_like_breakdown: bool = False,
             class_grouping: str = 'none',
             decoder_type: str = 'detr',
             decode_max_detections: int = 100,
             probe_forward_chunk_size: int = 0,
             is_flir: bool = False,
             loss_weights: Optional[Dict[str, float]] = None):
    """
    Run validation and compute detection metrics.
    Supports 3D detection (NuScenes/Waymo) and 2D detection (FLIR).
    """
    encoder.eval()
    decoder.eval()

    if det_metrics is not None:
        det_metrics.reset()

    metric_kwargs = {}
    if is_flir:
        if class_grouping == 'clustered':
            metric_kwargs = dict(class_names=FLIR_GROUPED_CLASSES, num_classes=NUM_FLIR_GROUPED_CLASSES)
        else:
            metric_kwargs = dict(class_names=FLIR_2D_DET_CLASS_NAMES, num_classes=NUM_FLIR_2D_CLASSES)
    elif class_grouping == 'clustered':
        metric_kwargs = dict(class_names=GROUPED_CLASSES, num_classes=NUM_GROUPED_CLASSES)

    det_metrics_compare = None
    if det_metrics is not None and decoder_type == 'centernet' and not is_flir:
        compare_plane = 'xy' if getattr(det_metrics, 'matching_plane', 'xz') != 'xy' else 'xz'
        det_metrics_compare = NuScenesDetectionMetrics(matching_plane=compare_plane, **metric_kwargs)

    det_metrics_by_thresh = None
    if official_like_breakdown:
        if is_flir:
            # FLIR 2D: centers are normalized to [0,1], so thresholds are
            # fractions of the image — e.g. 0.05 = 5% of image width/height.
            thresholds = {0.05: [0.05], 0.1: [0.1], 0.2: [0.2], 0.5: [0.5]}
        else:
            # NuScenes/Waymo 3D: thresholds in meters
            thresholds = {0.5: [0.5], 1.0: [1.0], 2.0: [2.0], 4.0: [4.0]}
        det_metrics_by_thresh = {
            thr: NuScenesDetectionMetrics(dist_thresholds=dists, matching_plane='xy' if is_flir else 'xz', **metric_kwargs)
            for thr, dists in thresholds.items()
        }

    total_loss = 0.0
    n_batches = 0

    with torch.inference_mode():
        for i, (cam_views, modality2, labels) in enumerate(test_loader):
            if limit_batches is not None and i >= limit_batches:
                break

            cam_views = to_device(cam_views, device)
            modality2 = to_device(modality2, device)
            is_waymo = getattr(getattr(test_loader, 'dataset', None), '__class__', type('', (), {})).__name__ == 'WaymoDataset'
            use_probe_encoder_views = is_flir or is_waymo
            encoder_cam_views = _prefer_probe_views(cam_views) if use_probe_encoder_views else cam_views
            encoder_modality2 = _prefer_probe_views(modality2) if use_probe_encoder_views else modality2
            encoder_cam_views, encoder_modality2, _ = _align_modal_view_counts(encoder_cam_views, encoder_modality2)

            with autocast(enabled=(device == 'cuda')):
                # Extract patch tokens
                patch_tokens = extract_patch_tokens(
                    encoder,
                    encoder_cam_views,
                    encoder_modality2,
                    arch,
                    simple_baseline=simple_baseline,
                    simple_modality=simple_modality if simple_baseline else 'rgb',
                    batch_chunk_size=probe_forward_chunk_size,
                )

                if patch_tokens is None:
                    continue

                # Handle multi-view
                cam_stats_src = encoder_cam_views
                B_val, V_val = get_input_stats(cam_stats_src)
                if V_val > 1 and patch_tokens.shape[0] == B_val * V_val:
                    patch_tokens = patch_tokens.view(B_val, V_val, patch_tokens.shape[1], patch_tokens.shape[2])[:, -1]

                predictions = decoder(patch_tokens)

                if is_flir:
                    # FLIR 2D: use gt_classes_2d, gt_centers_2d, gt_mask_2d
                    max_obj_2d = labels.get('gt_classes_2d', torch.zeros(B_val, 50)).shape[1]
                    det_targets = {
                        'gt_classes': labels.get('gt_classes_2d', torch.zeros(B_val, max_obj_2d, dtype=torch.long)).to(device),
                        'gt_centers': torch.cat([
                            labels.get('gt_centers_2d', torch.zeros(B_val, max_obj_2d, 2)).to(device).float(),
                            torch.zeros(B_val, max_obj_2d, 1, device=device),
                        ], dim=-1),
                        'gt_centers_2d': labels.get('gt_centers_2d', torch.zeros(B_val, max_obj_2d, 2)).to(device).float(),
                        'gt_sizes': labels.get('gt_sizes', torch.zeros(B_val, max_obj_2d, 3)).to(device),
                        'gt_orientations': labels.get('gt_orientations', torch.zeros(B_val, max_obj_2d, 2)).to(device),
                        'gt_mask': labels.get('gt_mask_2d', torch.zeros(B_val, max_obj_2d, dtype=torch.float32)).to(device),
                    }
                    if class_grouping == 'clustered':
                        det_targets['gt_classes'], det_targets['gt_mask'] = remap_flir_gt_classes(
                            det_targets['gt_classes'], det_targets['gt_mask'], device)
                else:
                    det_targets = {
                        'gt_classes': labels.get('gt_classes', torch.zeros(B_val, 50, dtype=torch.long)).to(device),
                        'gt_centers': labels.get('gt_centers', torch.zeros(B_val, 50, 3)).to(device),
                        'gt_centers_2d': labels.get('gt_centers_2d', torch.zeros(B_val, 50, 2)).to(device).float(),
                        'gt_sizes': labels.get('gt_sizes', torch.zeros(B_val, 50, 3)).to(device),
                        'gt_orientations': labels.get('gt_orientations', torch.zeros(B_val, 50, 2)).to(device),
                        'gt_mask': labels.get('gt_mask', torch.zeros(B_val, 50, dtype=torch.float32)).to(device),
                    }

                # Remap classes for clustered grouping (non-FLIR only; FLIR handled above)
                if class_grouping == 'clustered' and not is_flir:
                    det_targets['gt_classes'], det_targets['gt_mask'] = remap_gt_classes(
                        det_targets['gt_classes'], det_targets['gt_mask'], device)

                if compute_loss:
                    if decoder_type == 'centernet':
                        gt_centers_2d = det_targets.get('gt_centers_2d', det_targets['gt_centers'][:, :, :2])
                        if isinstance(gt_centers_2d, np.ndarray):
                            gt_centers_2d = torch.from_numpy(gt_centers_2d)
                        gt_centers_2d = gt_centers_2d.to(device).float()
                        gt_depths = det_targets['gt_centers'][:, :, 2].float()
                        centernet_targets = decoder.generate_targets(
                            gt_centers_2d=gt_centers_2d,
                            gt_depths=gt_depths,
                            gt_sizes=det_targets['gt_sizes'],
                            gt_orientations=det_targets['gt_orientations'],
                            gt_classes=det_targets['gt_classes'],
                            gt_mask=det_targets['gt_mask'],
                            output_grid=predictions['heatmap'].shape[-2:],
                        )
                        losses = decoder.compute_loss(predictions, centernet_targets,
                                                       loss_weights=loss_weights)
                    else:
                        losses = decoder.compute_loss(predictions, det_targets)
                    total_loss += losses['loss_total'].item()
                n_batches += 1

                # For metrics, CenterNet needs decode_to_boxes
                if decoder_type == 'centernet':
                    predictions = decoder.decode_to_boxes(
                        predictions,
                        max_detections=decode_max_detections,
                    )
                    predictions = convert_centernet_predictions_to_metric_space(predictions, det_targets)

                if det_metrics is not None:
                    det_metrics.update(predictions, det_targets)
                if det_metrics_compare is not None:
                    det_metrics_compare.update(predictions, det_targets)
                if det_metrics_by_thresh is not None:
                    for metrics_obj in det_metrics_by_thresh.values():
                        metrics_obj.update(predictions, det_targets)

    results = {}
    if compute_loss:
        avg_loss = total_loss / max(n_batches, 1)
        results['loss'] = avg_loss

    if det_metrics is not None:
        try:
            metrics = det_metrics.compute()
            results.update(metrics)
            primary_plane = getattr(det_metrics, 'matching_plane', 'xz')
            for key, value in metrics.items():
                if '/' in key:
                    results[f'{primary_plane}_match/{key}'] = value
            for key in ('mAP', 'mATE', 'mASE', 'mAOE', 'mADE'):
                if key in metrics:
                    results[f'{key}_{primary_plane}_match'] = metrics[key]
        except Exception as e:
            print(f"  ⚠️  Metrics computation failed: {e}")

    if det_metrics_compare is not None:
        try:
            compare_metrics = det_metrics_compare.compute()
            compare_plane = det_metrics_compare.matching_plane
            for key, value in compare_metrics.items():
                results[f'{compare_plane}_match/{key}'] = value
            for key in ('mAP', 'mATE', 'mASE', 'mAOE', 'mADE'):
                if key in compare_metrics:
                    results[f'{key}_{compare_plane}_match'] = compare_metrics[key]
        except Exception as e:
            print(f"  ⚠️  Comparison metrics computation failed: {e}")

    if det_metrics_by_thresh is not None:
        for thr, metrics_obj in det_metrics_by_thresh.items():
            try:
                m = metrics_obj.compute()
                if is_flir:
                    # FLIR: thresholds are normalized image fractions, not meters
                    results[f'mAP@{thr:.2f}norm'] = float(m.get('mAP', 0.0))
                else:
                    results[f'mAP@{thr:.1f}m'] = float(m.get('mAP', 0.0))
            except Exception as e:
                unit = 'norm' if is_flir else 'm'
                print(f"  ⚠️  Threshold metrics ({thr}{unit}) failed: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep agent entry point
# ═══════════════════════════════════════════════════════════════════════════════

def sweep_train():
    """Training function called by wandb.agent().
    
    Reads hyperparameters from wandb.config (set by the sweep controller),
    then calls main() with those params.
    """
    try:
        wandb_key_path = os.path.join(PROJECT_ROOT, '.wandb_key')
        if os.path.exists(wandb_key_path):
            with open(wandb_key_path) as f:
                wandb.login(key=f.read().strip())

        run = wandb.init()
        config = dict(wandb.config)
        print(f"\n{'='*60}")
        print(f"Sweep run: {wandb.run.name}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        main(sweep_config=config)

    except Exception as e:
        print(f"Sweep train error: {e}")
        import traceback
        traceback.print_exc()
        wandb.log({'training_failed': 1, 'error': str(e)})
        raise
    finally:
        # Aggressive cleanup between sweep runs (matching sweep_agent_novel)
        print("🧹 Running cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            run.finish()
        except Exception:
            pass
        print("Waiting 5s for cleanup...")
        time.sleep(5)
        gc.collect()


if __name__ == '__main__':
    if SWEEP_ID:
        # Sweep agent mode: requires wandb
        if not WANDB_AVAILABLE:
            print("ERROR: wandb is required for sweep mode. Install with: pip install wandb")
            sys.exit(1)
        print(f"Starting sweep agent: {PROJECT_NAME}/{SWEEP_ID}")
        wandb_key_path = os.path.join(PROJECT_ROOT, '.wandb_key')
        if os.path.exists(wandb_key_path):
            with open(wandb_key_path) as f:
                wandb.login(key=f.read().strip())
        try:
            wandb.agent(
                f'{PROJECT_NAME}/{SWEEP_ID}',
                function=sweep_train,
                count=SWEEP_COUNT,
            )
        except Exception as e:
            print(f"Sweep error: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Standalone mode: use CLI args
        main()
