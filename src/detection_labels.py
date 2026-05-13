"""
detection_labels.py  (v2)

Core label generation + loading for 3D detection and segmentation probes.

Two modes:
1. **Precompute** (called from precompute_det_seg_labels.py):
   - compute_bbox_labels()  → GT arrays for detection probe
   - compute_seg_map()      → projected LiDAR seg map for seg probe
   - save_det_seg_labels()  → pack into .npz

2. **Load** (called from dataset / training loop):
   - load_det_seg_labels()  → read from shard-zip or individual .npz

Label convention:
    BBox:  gt_classes (N,) int, gt_centers (N,3), gt_sizes (N,3),
           gt_orientations (N,2), gt_mask (N,) float
    Seg:   seg_map (H,W) int64  (0 = ignore)
"""

import numpy as np
import zipfile
import io
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.detection_probes import (
    DETECTION_CLASSES,
    CATEGORY_TO_DETECTION,
    LIDARSEG_TO_SIMPLIFIED,
    NUM_DETECTION_CLASSES,
    NUM_SIMPLIFIED_SEG_CLASSES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# QUATERNION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def quat_to_rot_matrix(q) -> np.ndarray:
    """Convert quaternion [w, x, y, z] (nuScenes format) to 3×3 rotation matrix.
    Pure numpy — no scipy dependency."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[0], q[1], q[2], q[3]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (y2 + z2), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (x2 + z2), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (x2 + y2)],
    ], dtype=np.float64)


def compute_lidar_to_cam_transform(
    cam_calib: Dict,
    lidar_calib: Dict,
    cam_ego: Dict,
    lidar_ego: Dict,
) -> np.ndarray:
    """
    Full 4×4 transform: LiDAR → Ego(lidar) → World → Ego(cam) → Camera.
    """
    # LiDAR → Ego
    L2E = np.eye(4)
    L2E[:3, :3] = quat_to_rot_matrix(lidar_calib["rotation"])
    L2E[:3, 3] = np.asarray(lidar_calib["translation"])

    # Ego(lidar) → World
    E2W = np.eye(4)
    E2W[:3, :3] = quat_to_rot_matrix(lidar_ego["rotation"])
    E2W[:3, 3] = np.asarray(lidar_ego["translation"])

    # World → Ego(cam)
    W2CE = np.eye(4)
    R_ce = quat_to_rot_matrix(cam_ego["rotation"])
    W2CE[:3, :3] = R_ce.T
    W2CE[:3, 3] = -R_ce.T @ np.asarray(cam_ego["translation"])

    # Ego(cam) → Camera
    CE2C = np.eye(4)
    R_c = quat_to_rot_matrix(cam_calib["rotation"])
    CE2C[:3, :3] = R_c.T
    CE2C[:3, 3] = -R_c.T @ np.asarray(cam_calib["translation"])

    return (CE2C @ W2CE @ E2W @ L2E).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# CENTER-CROP GEOMETRY  (matches dataset Resize+CenterCrop for probe view)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_center_crop_region(
    img_hw: Tuple[int, int] = (900, 1600),
    target_size: int = 224,
) -> Tuple[int, int, int, int]:
    """Compute the center-square crop region in original pixel coords.

    Mirrors `_apply_test_aug` in mm_dataset.py:
        1. Resize smallest edge to target_size → scale factor
        2. CenterCrop(target_size) on the resized image
        3. Back-project crop to original pixel coords

    Returns:
        (i, j, h, w) — top, left, height, width in original image pixels.
        This is the square region that the model's probe view actually sees.
    """
    orig_h, orig_w = img_hw
    scale = target_size / min(orig_h, orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    # CenterCrop in resized space
    top_new = (new_h - target_size) // 2
    left_new = (new_w - target_size) // 2

    # Back-project to original coords
    i_orig = int(top_new / scale)
    j_orig = int(left_new / scale)
    h_orig = int(target_size / scale)
    w_orig = int(target_size / scale)
    # Clamp
    h_orig = min(h_orig, orig_h - i_orig)
    w_orig = min(w_orig, orig_w - j_orig)

    return (i_orig, j_orig, h_orig, w_orig)


# ═══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE: 3D BOUNDING BOX LABELS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bbox_labels(
    annotations: List[Dict],
    cam_calib: Dict,
    cam_ego: Dict,
    get_category_fn,          # callable(instance_token) → category string
    max_objects: int = 50,
    img_hw: Tuple[int, int] = (900, 1600),
    crop_region: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, np.ndarray]:
    """Compute 3D bounding-box labels in camera frame.

    When *crop_region* is provided (i, j, h, w in original pixels — from
    `compute_center_crop_region()`), gt_centers_2d are stored as normalised
    coordinates within that crop, and objects outside the crop are filtered.
    Without crop_region, coordinates are normalised to the full image.

    Returns dict with gt_classes, gt_centers, gt_sizes, gt_orientations,
    gt_mask, gt_centers_2d.
    """
    H, W = img_hw
    gt_classes = np.full(max_objects, -1, dtype=np.int64)
    gt_centers = np.zeros((max_objects, 3), dtype=np.float32)
    gt_sizes = np.zeros((max_objects, 3), dtype=np.float32)
    gt_orientations = np.zeros((max_objects, 2), dtype=np.float32)
    gt_mask = np.zeros(max_objects, dtype=np.float32)
    gt_centers_2d = np.zeros((max_objects, 2), dtype=np.float32)

    intrinsic = cam_calib.get("intrinsic")
    if intrinsic is None:
        return _pack_bbox(gt_classes, gt_centers, gt_sizes, gt_orientations, gt_mask, gt_centers_2d)

    intrinsic = np.asarray(intrinsic, dtype=np.float64)

    # Crop region (defaults to full image)
    if crop_region is not None:
        ci, cj, ch, cw = crop_region
    else:
        ci, cj, ch, cw = 0, 0, H, W

    # Extrinsics
    ego_R = quat_to_rot_matrix(cam_ego["rotation"])
    ego_t = np.asarray(cam_ego["translation"])
    cam_R = quat_to_rot_matrix(cam_calib["rotation"])
    cam_t = np.asarray(cam_calib["translation"])

    idx = 0
    n_no_fields = 0
    n_no_category = 0
    n_behind = 0
    n_outside_crop = 0
    
    for ann in annotations:
        if idx >= max_objects:
            break
        if "translation" not in ann or "size" not in ann or "rotation" not in ann:
            n_no_fields += 1
            continue

        inst_token = ann.get("instance_token", "")
        cat_name = get_category_fn(inst_token) if callable(get_category_fn) else ""

        # Map to detection class
        det_cls = None
        for cat_key, det_name in CATEGORY_TO_DETECTION.items():
            if cat_key in cat_name.lower():
                det_cls = DETECTION_CLASSES.index(det_name)
                break
        if det_cls is None:
            n_no_category += 1
            continue

        # World → Ego → Camera
        center_w = np.asarray(ann["translation"], dtype=np.float64)
        center_ego = ego_R.T @ (center_w - ego_t)
        center_cam = cam_R.T @ (center_ego - cam_t)

        if center_cam[2] <= 0:   # behind camera
            n_behind += 1
            continue

        # Project to full image 2D
        proj = intrinsic @ center_cam
        u, v = proj[0] / proj[2], proj[1] / proj[2]

        # Filter: must be within the crop region (small margin for partially visible)
        margin_frac = 0.05  # 5% of crop dimension
        margin_u = cw * margin_frac
        margin_v = ch * margin_frac
        if not (cj - margin_u < u < cj + cw + margin_u and
                ci - margin_v < v < ci + ch + margin_v):
            n_outside_crop += 1
            continue

        # Orientation → yaw angle in camera frame
        rot_w = quat_to_rot_matrix(ann["rotation"])
        rot_cam = cam_R.T @ ego_R.T @ rot_w
        # Fix: Yaw is rotation around Y-axis in Camera frame (XZ plane).
        # Previous: arctan2(rot_cam[1, 0], rot_cam[0, 0]) was screen-plane angle (XY).
        # Correct: arctan2(-rot_cam[2, 0], rot_cam[0, 0]) extracts Y-rotation.
        yaw = np.arctan2(-rot_cam[2, 0], rot_cam[0, 0])

        gt_classes[idx] = det_cls
        gt_centers[idx] = center_cam.astype(np.float32)
        gt_sizes[idx] = np.asarray(ann["size"], dtype=np.float32)
        gt_orientations[idx] = [np.sin(yaw), np.cos(yaw)]
        # Normalise 2D coords relative to the crop region [0, 1]
        gt_centers_2d[idx] = [(u - cj) / cw, (v - ci) / ch]
        gt_mask[idx] = 1.0
        idx += 1

    # Debug output
    # if len(annotations) > 0:
    #     print(f"      compute_bbox_labels: {len(annotations)} ann → "
    #           f"{idx} valid, {n_no_fields} no_fields, {n_no_category} no_cat, "
    #           f"{n_behind} behind, {n_outside_crop} outside_crop")

    return _pack_bbox(gt_classes, gt_centers, gt_sizes, gt_orientations, gt_mask, gt_centers_2d)


def _pack_bbox(cls, ctr, sz, ori, msk, ctr2d=None):
    d = {
        "gt_classes": cls,
        "gt_centers": ctr,
        "gt_sizes": sz,
        "gt_orientations": ori,
        "gt_mask": msk,
    }
    if ctr2d is not None:
        d["gt_centers_2d"] = ctr2d
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE: SEMANTIC SEGMENTATION MAP
# ═══════════════════════════════════════════════════════════════════════════════

def load_lidarseg_bin(lidarseg_dir: Path, lidar_token: str) -> Optional[np.ndarray]:
    """Load (N,) uint8 per-point class labels from the lidarseg binary file."""
    path = Path(lidarseg_dir) / f"{lidar_token}_lidarseg.bin"
    if path.exists():
        return np.fromfile(str(path), dtype=np.uint8)
    return None


def nearest_neighbor_fill(seg_map: np.ndarray, max_dist: int = 10) -> np.ndarray:
    """Fill unlabeled pixels (0) with the nearest non-zero label.
    
    Uses scipy.ndimage if available, else falls back to iterative dilation.
    
    Args:
        seg_map: (H, W) int64 with 0 = unlabeled
        max_dist: Maximum distance (in pixels) to propagate labels
        
    Returns:
        Filled seg_map with same shape
    """
    if seg_map.max() == 0:
        return seg_map
    
    filled = seg_map.copy()
    
    try:
        from scipy import ndimage
        # Use distance transform for true nearest-neighbor
        labeled = seg_map > 0
        if not labeled.any():
            return seg_map
        
        # Get indices of nearest labeled pixel for each unlabeled pixel
        dist, indices = ndimage.distance_transform_edt(
            ~labeled, return_indices=True
        )
        # Only fill within max_dist
        fill_mask = (seg_map == 0) & (dist <= max_dist)
        filled[fill_mask] = seg_map[indices[0][fill_mask], indices[1][fill_mask]]
        
    except ImportError:
        # Fallback: iterative morphological dilation
        from collections import deque
        
        H, W = seg_map.shape
        # BFS from labeled pixels
        queue = deque()
        dist_map = np.full((H, W), np.inf)
        
        # Initialize with labeled pixels
        for r in range(H):
            for c in range(W):
                if seg_map[r, c] > 0:
                    dist_map[r, c] = 0
                    queue.append((r, c, 0))
        
        # 4-connected neighbors
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]
        
        while queue:
            r, c, d = queue.popleft()
            if d >= max_dist:
                continue
            for i in range(4):
                nr, nc = r + dr[i], c + dc[i]
                if 0 <= nr < H and 0 <= nc < W:
                    if dist_map[nr, nc] > d + 1:
                        dist_map[nr, nc] = d + 1
                        filled[nr, nc] = filled[r, c]
                        queue.append((nr, nc, d + 1))
    
    return filled


def compute_seg_map(
    lidar_points: np.ndarray,
    lidarseg_labels: np.ndarray,
    intrinsic: np.ndarray,
    lidar_to_cam: np.ndarray,
    img_hw: Tuple[int, int] = (900, 1600),
    target_hw: Tuple[int, int] = (224, 224),
    crop_region: Optional[Tuple[int, int, int, int]] = None,
    fill_max_dist: int = 0,
    dataset_type: str = 'nuscenes',
) -> np.ndarray:
    """Z-buffer project LiDAR segmentation labels onto the camera image plane.

    When *crop_region* = (i, j, h, w) is provided, only LiDAR points that
    fall inside that crop of the original image are kept, and their pixel
    coordinates are remapped to the crop before scaling to *target_hw*.
    This produces a seg map that matches the model's center-crop probe view.

    Without crop_region the full original image is used (legacy behaviour).
    
    Args:
        fill_max_dist: If > 0, fill unlabeled pixels with nearest-neighbor
                       labels up to this distance (in pixels). This reduces
                       sparsity for training. 0 = no fill (sparse).
        dataset_type: 'nuscenes' or 'waymo' - determines label mapping.

    Returns:
        seg_map: (H_tgt, W_tgt) int64   (0 = ignore/empty)
    """
    from src.detection_probes import waymo_label_to_simplified
    
    H_orig, W_orig = img_hw
    H_out, W_out = target_hw

    pts_3d = lidar_points[:, :3].astype(np.float64)
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    pts_cam = (lidar_to_cam @ pts_h.T).T[:, :3]

    # Keep points in front of camera
    in_front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[in_front]
    labels = lidarseg_labels[in_front]

    # Project to full-image 2D
    pts_2d_h = (intrinsic @ pts_cam.T).T
    u = pts_2d_h[:, 0] / (pts_2d_h[:, 2] + 1e-8)
    v = pts_2d_h[:, 1] / (pts_2d_h[:, 2] + 1e-8)

    # Determine the source region (full image or crop)
    if crop_region is not None:
        ci, cj, ch, cw = crop_region
    else:
        ci, cj, ch, cw = 0, 0, H_orig, W_orig

    # Filter to crop bounds
    ok = (u >= cj) & (u < cj + cw) & (v >= ci) & (v < ci + ch)
    u, v, labels, depths = u[ok], v[ok], labels[ok], pts_cam[ok, 2]

    # Remap to crop-relative coords, then scale to target resolution
    u_rel = u - cj   # [0, cw)
    v_rel = v - ci   # [0, ch)
    u_s = np.clip((u_rel * W_out / cw).astype(np.int32), 0, W_out - 1)
    v_s = np.clip((v_rel * H_out / ch).astype(np.int32), 0, H_out - 1)

    # Simplify classes based on dataset type
    if dataset_type == 'waymo':
        simplified = np.array([waymo_label_to_simplified(int(l)) for l in labels],
                              dtype=np.int64)
    else:
        simplified = np.array([LIDARSEG_TO_SIMPLIFIED.get(int(l), 0) for l in labels],
                              dtype=np.int64)

    # Z-buffer render (closest point wins)
    seg_map = np.zeros((H_out, W_out), dtype=np.int64)
    zbuf = np.full((H_out, W_out), np.inf, dtype=np.float64)
    for i in range(len(u_s)):
        r, c = v_s[i], u_s[i]
        if depths[i] < zbuf[r, c]:
            zbuf[r, c] = depths[i]
            seg_map[r, c] = simplified[i]

    # Optional: nearest-neighbor fill to reduce sparsity
    if fill_max_dist > 0:
        seg_map = nearest_neighbor_fill(seg_map, max_dist=fill_max_dist)

    return seg_map


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD  (shard-compatible, follows prepare_labels.py convention)
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_VERSION = "det_seg_v2"
CACHE_DIR_NAME = "det_seg_labels_v2"


def label_filename(sample_token: str, cam_name: str) -> str:
    """Canonical filename inside a shard zip or on disk."""
    return f"{sample_token}_{cam_name}_{CACHE_VERSION}.npz"


def save_det_seg_labels(
    path: Path,
    bbox_labels: Dict[str, np.ndarray],
    seg_map: np.ndarray,
    panoptic_seg_map: Optional[np.ndarray] = None,
):
    """Save bbox + seg labels to a single .npz file."""
    save_kwargs = dict(
        gt_classes=bbox_labels["gt_classes"],
        gt_centers=bbox_labels["gt_centers"],
        gt_sizes=bbox_labels["gt_sizes"],
        gt_orientations=bbox_labels["gt_orientations"],
        gt_mask=bbox_labels["gt_mask"],
        seg_map=seg_map,
    )
    if panoptic_seg_map is not None:
        save_kwargs["panoptic_seg_map"] = panoptic_seg_map
        
    if "gt_centers_2d" in bbox_labels:
        save_kwargs["gt_centers_2d"] = bbox_labels["gt_centers_2d"]
    np.savez_compressed(str(path), **save_kwargs)


def load_det_seg_labels(
    sample_token: str,
    cam_name: str,
    cache_dir: Path,
    has_shards: bool = True,
    required_keys: Optional[List[str]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load precomputed det/seg labels from shard zip or individual file.

    Returns dict with gt_classes, gt_centers, gt_sizes, gt_orientations,
    gt_mask, seg_map — or None if not found.
    """
    fname = label_filename(sample_token, cam_name)

    # Strategy 1: sharded zip
    if has_shards:
        from src.dataset import get_shard_id
        shard_id = get_shard_id(sample_token)
        shard_path = cache_dir / f"shard_{shard_id}.zip"
        if shard_path.exists():
            try:
                with zipfile.ZipFile(shard_path, "r") as zf:
                    with zf.open(fname) as f:
                        data = np.load(io.BytesIO(f.read()))
                        if required_keys is not None:
                            out = {}
                            for k in required_keys:
                                if k in data.files:
                                    out[k] = data[k]
                            return out if out else None
                        return {k: data[k] for k in data.files}
            except (KeyError, Exception):
                pass

    # Strategy 2: individual file
    ind_path = cache_dir / fname
    if ind_path.exists():
        try:
            data = np.load(str(ind_path))
            if required_keys is not None:
                out = {}
                for k in required_keys:
                    if k in data.files:
                        out[k] = data[k]
                return out if out else None
            return {k: data[k] for k in data.files}
        except Exception:
            pass

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# COLLATE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def collate_det_seg_labels(
    batch_labels: List[Dict[str, np.ndarray]],
) -> Dict:
    """Stack a list of label dicts into batched tensors."""
    import torch
    keys = ["gt_classes", "gt_centers", "gt_sizes", "gt_orientations", "gt_mask",
            "gt_centers_2d", "seg_map"]
    out = {}
    for k in keys:
        if k in batch_labels[0]:
            out[k] = torch.from_numpy(np.stack([b[k] for b in batch_labels]))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing detection_labels v2 ...")
    # Identity quaternion
    R = quat_to_rot_matrix([1, 0, 0, 0])
    assert np.allclose(R, np.eye(3)), "identity quat should give I"

    # Center-crop geometry
    crop = compute_center_crop_region(img_hw=(900, 1600), target_size=224)
    print(f"  center crop (900x1600→224): i={crop[0]}, j={crop[1]}, h={crop[2]}, w={crop[3]}")
    assert crop[2] == crop[3], "crop should be square"
    assert crop[0] == 0, "for landscape, top should be ~0"
    assert crop[1] > 0, "for landscape, left should offset horizontally"

    # Seg map — full image
    pts = np.random.randn(500, 5).astype(np.float32)
    pts[:, 2] = np.abs(pts[:, 2]) + 1.0  # positive Z
    seg = np.random.randint(0, 32, 500, dtype=np.uint8)
    K = np.array([[1000, 0, 800], [0, 1000, 450], [0, 0, 1]], dtype=np.float64)
    out = compute_seg_map(pts, seg, K, np.eye(4))
    print(f"  seg_map (full)  shape={out.shape}, unique={np.unique(out)[:6]}...")

    # Seg map — with crop
    out_c = compute_seg_map(pts, seg, K, np.eye(4), crop_region=crop)
    print(f"  seg_map (crop)  shape={out_c.shape}, unique={np.unique(out_c)[:6]}...")

    print("  ✅ All OK")
