"""
lidar_augmentations.py

LiDAR & Camera-Synchronized Augmentation Pipeline for 3D Object Detection.

Two levels of augmentation:

1. **Scene-level augmentations** (applied to every sample):
   - Global scaling (uniform random scale factor)
   - Global rotation (small yaw perturbation)
   - Global translation noise (small shift of entire scene)
   - Point-level jitter (per-point Gaussian noise on xyz)
   - Intensity noise / dropout
   - Ground-plane random flip along X or Y axis
   - Random point dropout (thin out a fraction of points)
   - Frustum dropout (drop a random angular wedge of points)

2. **Copy-paste augmentation** (for class imbalance):
   - Extracts objects (points inside 3D bounding boxes) from "donor" scenes
   - Pastes them into the current scene (both LiDAR points AND camera image)
   - Handles occlusion: removes existing points inside pasted boxes
   - Projects pasted object into camera to paste corresponding RGB crop

All augmentations operate on raw LiDAR points in LIDAR SENSOR frame
(before any projection) and modify annotations consistently.

Usage:
    from src.lidar_augmentations import LiDARSceneAugmentor, CopyPasteAugmentor

    scene_aug = LiDARSceneAugmentor(cfg)
    points, annotations = scene_aug(points, annotations)

    cp_aug = CopyPasteAugmentor(database_path, cfg)
    points, annotations, image = cp_aug(points, annotations, image,
                                         intrinsic, lidar_to_cam)
"""

import numpy as np
import copy
import pickle
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix for rotation around Z axis (yaw)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float64)


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix for rotation around Y axis (pitch)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float64)


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix for rotation around X axis (roll)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ], dtype=np.float64)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(y2+z2), 2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),     1 - 2*(x2+z2), 2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),     1 - 2*(x2+y2)],
    ], dtype=np.float64)


def get_box_corners_3d(center: np.ndarray, size: np.ndarray,
                       rotation: np.ndarray) -> np.ndarray:
    """
    Compute 8 corners of a 3D bounding box.

    Args:
        center: (3,) xyz center of box in world/lidar frame
        size: (3,) width, length, height (nuScenes convention: w, l, h)
        rotation: (4,) quaternion [w, x, y, z]

    Returns:
        corners: (8, 3) array of corner coordinates
    """
    w, l, h = size[0], size[1], size[2]
    # Half extents
    dx, dy, dz = w / 2, l / 2, h / 2

    # 8 corners in object-local frame
    corners_local = np.array([
        [-dx, -dy, -dz],
        [ dx, -dy, -dz],
        [ dx,  dy, -dz],
        [-dx,  dy, -dz],
        [-dx, -dy,  dz],
        [ dx, -dy,  dz],
        [ dx,  dy,  dz],
        [-dx,  dy,  dz],
    ])

    R = quat_to_rot(rotation)
    corners_world = (R @ corners_local.T).T + center
    return corners_world


def get_box_corners_3d_nuscenes(center: np.ndarray, size_wlh: np.ndarray,
                               rotation: np.ndarray) -> np.ndarray:
    """nuScenes box geometry helper.

    nuScenes stores size as (w, l, h), but in practice the heading/rotation in this
    codebase aligns the *length* axis with local-x and *width* axis with local-y.
    Using (dx=l/2, dy=w/2) makes projected boxes and in-box point masks consistent.
    """
    w, l, h = size_wlh[0], size_wlh[1], size_wlh[2]
    dx, dy, dz = l / 2, w / 2, h / 2

    corners_local = np.array([
        [-dx, -dy, -dz],
        [ dx, -dy, -dz],
        [ dx,  dy, -dz],
        [-dx,  dy, -dz],
        [-dx, -dy,  dz],
        [ dx, -dy,  dz],
        [ dx,  dy,  dz],
        [-dx,  dy,  dz],
    ])

    R = quat_to_rot(rotation)
    return (R @ corners_local.T).T + center


def points_in_box(points_xyz: np.ndarray, center: np.ndarray,
                  size: np.ndarray, rotation: np.ndarray,
                  margin: float = 0.0) -> np.ndarray:
    """
    Boolean mask of which points lie inside a 3D bounding box.

    Args:
        points_xyz: (N, 3) point coordinates
        center: (3,) box center
        size: (3,) box size (w, l, h)
        rotation: (4,) quaternion [w, x, y, z]
        margin: extra margin around box (meters)

    Returns:
        mask: (N,) boolean array
    """
    R = quat_to_rot(rotation)
    # Transform points to box-local frame
    local = (R.T @ (points_xyz - center).T).T  # (N, 3)

    w, l, h = size[0], size[1], size[2]
    hx, hy, hz = w / 2 + margin, l / 2 + margin, h / 2 + margin

    mask = (
        (local[:, 0] >= -hx) & (local[:, 0] <= hx) &
        (local[:, 1] >= -hy) & (local[:, 1] <= hy) &
        (local[:, 2] >= -hz) & (local[:, 2] <= hz)
    )
    return mask


def points_in_box_nuscenes(points_xyz: np.ndarray, center: np.ndarray,
                           size_wlh: np.ndarray, rotation: np.ndarray,
                           margin: float = 0.0) -> np.ndarray:
    """nuScenes-consistent points-in-box.

    Uses (hx=l/2, hy=w/2, hz=h/2) to match get_box_corners_3d_nuscenes.
    """
    R = quat_to_rot(rotation)
    local = (R.T @ (points_xyz - center).T).T
    w, l, h = size_wlh[0], size_wlh[1], size_wlh[2]
    hx, hy, hz = l / 2 + margin, w / 2 + margin, h / 2 + margin
    return (
        (local[:, 0] >= -hx) & (local[:, 0] <= hx) &
        (local[:, 1] >= -hy) & (local[:, 1] <= hy) &
        (local[:, 2] >= -hz) & (local[:, 2] <= hz)
    )


def rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE-LEVEL AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class LiDARSceneAugmentor:
    """
    Applies scene-level augmentations to raw LiDAR point clouds and
    their corresponding 3D bounding box annotations (in LiDAR sensor frame).

    All augmentations are stochastic and applied with independent probabilities.

    Config dict keys:
        global_scaling_range: (min, max) uniform scale factor, e.g. (0.95, 1.05)
        global_rotation_range: (min, max) yaw in radians, e.g. (-pi/12, pi/12)
        global_translation_std: std-dev for xyz translation noise (meters)
        object_translation_std: std-dev for per-object xyz noise (meters)
        point_jitter_std: per-point xyz Gaussian noise std (meters)
        intensity_noise_std: std-dev for intensity noise
        random_flip_x: probability of flipping along X axis
        random_flip_y: probability of flipping along Y axis
        point_dropout_rate: fraction of points to randomly drop
        frustum_dropout_prob: probability of applying frustum dropout
        frustum_dropout_angle: angular range (degrees) of the dropped wedge
    """

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        # Global scene transforms
        self.global_scaling_range = cfg.get('global_scaling_range', (0.95, 1.05))
        self.global_rotation_range = cfg.get('global_rotation_range',
                                              (-np.pi / 12, np.pi / 12))  # ±15°
        self.global_translation_std = cfg.get('global_translation_std', 0.2)  # meters
        self.object_translation_std = cfg.get('object_translation_std', 0.1)  # meters

        # Point-level noise
        self.point_jitter_std = cfg.get('point_jitter_std', 0.02)  # meters
        # Extra per-object point jitter (applied only to points inside each box).
        # This is useful when you want small object-specific perturbations without
        # rigidly moving objects (which would break camera alignment).
        self.object_point_jitter_std = cfg.get('object_point_jitter_std', 0.0)
        self.intensity_noise_std = cfg.get('intensity_noise_std', 0.05)

        # Flipping
        self.random_flip_x = cfg.get('random_flip_x', 0.5)
        self.random_flip_y = cfg.get('random_flip_y', 0.0)  # usually off for driving

        # Dropout
        self.point_dropout_rate = cfg.get('point_dropout_rate', 0.0)
        self.frustum_dropout_prob = cfg.get('frustum_dropout_prob', 0.0)
        self.frustum_dropout_angle = cfg.get('frustum_dropout_angle', 30.0)  # degrees

        # Enable/disable flags (allow partial augmentation)
        self.enable_global_scaling = cfg.get('enable_global_scaling', True)
        self.enable_global_rotation = cfg.get('enable_global_rotation', True)
        self.enable_global_translation = cfg.get('enable_global_translation', True)
        self.enable_object_translation = cfg.get('enable_object_translation', True)
        self.enable_point_jitter = cfg.get('enable_point_jitter', True)
        self.enable_intensity_noise = cfg.get('enable_intensity_noise', True)

    def __call__(
        self,
        points: np.ndarray,
        annotations: List[Dict],
        alignment_safe: bool = False,
        lidarseg_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Dict], ...]:
        """
        Apply scene-level augmentations.

        Args:
            points: (N, 5) array [x, y, z, intensity, ring_index]
            annotations: list of annotation dicts, each containing
                'translation' (3,), 'size' (3,), 'rotation' (4,) in LiDAR frame

        Returns:
            augmented_points: (M, 5) augmented point cloud
            augmented_annotations: list of augmented annotation dicts
        """
        points = points.copy()
        annotations = copy.deepcopy(annotations)

        xyz = points[:, :3]

        # ── 1. Global Random Flip ───────────────────────────────────
        # NOTE: Flips/rigid transforms break camera-projected alignment unless
        # you apply the same transform to RGB or update the extrinsics.
        if (not alignment_safe) and self.random_flip_x > 0 and np.random.random() < self.random_flip_x:
            xyz[:, 0] = -xyz[:, 0]
            for ann in annotations:
                t = np.asarray(ann['translation'], dtype=np.float64)
                t[0] = -t[0]
                ann['translation'] = t
                # Flip yaw: q = [w, x, y, z] → flip x component of yaw
                R = quat_to_rot(np.asarray(ann['rotation']))
                flip_x = np.diag([-1, 1, 1]).astype(np.float64)
                R_new = flip_x @ R @ flip_x
                ann['rotation'] = rot_matrix_to_quat(R_new)

        if (not alignment_safe) and self.random_flip_y > 0 and np.random.random() < self.random_flip_y:
            xyz[:, 1] = -xyz[:, 1]
            for ann in annotations:
                t = np.asarray(ann['translation'], dtype=np.float64)
                t[1] = -t[1]
                ann['translation'] = t
                R = quat_to_rot(np.asarray(ann['rotation']))
                flip_y = np.diag([1, -1, 1]).astype(np.float64)
                R_new = flip_y @ R @ flip_y
                ann['rotation'] = rot_matrix_to_quat(R_new)

        # ── 2. Global Scaling ───────────────────────────────────────
        if (not alignment_safe) and self.enable_global_scaling:
            scale = np.random.uniform(*self.global_scaling_range)
            xyz *= scale
            for ann in annotations:
                ann['translation'] = np.asarray(ann['translation'], dtype=np.float64) * scale
                ann['size'] = np.asarray(ann['size'], dtype=np.float64) * scale

        # ── 3. Global Rotation (yaw only) ──────────────────────────
        if (not alignment_safe) and self.enable_global_rotation:
            angle = np.random.uniform(*self.global_rotation_range)
            R_aug = rotation_matrix_z(angle)
            xyz = (R_aug @ xyz.T).T
            for ann in annotations:
                t = np.asarray(ann['translation'], dtype=np.float64)
                ann['translation'] = R_aug @ t
                R_box = quat_to_rot(np.asarray(ann['rotation']))
                ann['rotation'] = rot_matrix_to_quat(R_aug @ R_box)

        # ── 4. Global Translation Noise ─────────────────────────────
        if (not alignment_safe) and self.enable_global_translation and self.global_translation_std > 0:
            trans = np.random.normal(0, self.global_translation_std, size=3)
            xyz += trans
            for ann in annotations:
                ann['translation'] = np.asarray(ann['translation'], dtype=np.float64) + trans

        # ── 5. Per-Object Translation Noise ─────────────────────────
        if (not alignment_safe) and self.enable_object_translation and self.object_translation_std > 0:
            for ann in annotations:
                center = np.asarray(ann['translation'], dtype=np.float64)
                size = np.asarray(ann['size'], dtype=np.float64)
                rot = np.asarray(ann['rotation'], dtype=np.float64)

                # Find points inside this box
                mask = points_in_box(xyz, center, size, rot, margin=0.1)
                if mask.sum() < 3:
                    continue

                obj_noise = np.random.normal(0, self.object_translation_std, size=3)
                # Only translate in x,y (keep z stable for driving)
                obj_noise[2] *= 0.1

                xyz[mask] += obj_noise
                ann['translation'] = center + obj_noise

        # ── 5b. Alignment-safe per-object point jitter (no box motion) ─────────
        if alignment_safe and self.object_point_jitter_std and self.object_point_jitter_std > 0:
            for ann in annotations:
                center = np.asarray(ann['translation'], dtype=np.float64)
                size = np.asarray(ann['size'], dtype=np.float64)
                rot = np.asarray(ann['rotation'], dtype=np.float64)
                mask = points_in_box(xyz, center, size, rot, margin=0.05)
                n_in = int(mask.sum())
                if n_in < 3:
                    continue
                jitter = np.random.normal(0, self.object_point_jitter_std, size=(n_in, 3))
                # Keep z more stable
                jitter[:, 2] *= 0.25
                xyz[mask] += jitter

        points[:, :3] = xyz

        # ── 6. Point-Level XYZ Jitter ───────────────────────────────
        if self.enable_point_jitter and self.point_jitter_std > 0:
            jitter = np.random.normal(0, self.point_jitter_std,
                                       size=(len(points), 3))
            points[:, :3] += jitter.astype(points.dtype)

        # ── 7. Intensity Noise ──────────────────────────────────────
        if self.enable_intensity_noise and self.intensity_noise_std > 0:
            noise = np.random.normal(0, self.intensity_noise_std,
                                      size=len(points))
            points[:, 3] += noise.astype(points.dtype)
            points[:, 3] = np.clip(points[:, 3], 0, 1)

        # ── 8. Random Point Dropout ─────────────────────────────────
        if self.point_dropout_rate > 0:
            keep = np.random.random(len(points)) > self.point_dropout_rate
            if keep.sum() > 100:  # Don't drop too many
                points = points[keep]
                if lidarseg_labels is not None:
                    lidarseg_labels = lidarseg_labels[keep]

        # ── 9. Frustum Dropout ──────────────────────────────────────
        if self.frustum_dropout_prob > 0 and np.random.random() < self.frustum_dropout_prob:
            angles = np.arctan2(points[:, 1], points[:, 0])  # angle in XY plane
            center_angle = np.random.uniform(-np.pi, np.pi)
            half_span = np.deg2rad(self.frustum_dropout_angle / 2)
            # Circular range check
            diff = np.abs(np.arctan2(
                np.sin(angles - center_angle),
                np.cos(angles - center_angle)
            ))
            keep = diff > half_span
            if keep.sum() > 100:
                points = points[keep]
                if lidarseg_labels is not None:
                    lidarseg_labels = lidarseg_labels[keep]

        if lidarseg_labels is not None:
            return points, annotations, lidarseg_labels
        return points, annotations


# ═══════════════════════════════════════════════════════════════════════════════
# GROUND-TRUTH DATABASE FOR COPY-PASTE
# ═══════════════════════════════════════════════════════════════════════════════

class GTDatabaseBuilder:
    """
    Builds a database of isolated objects (LiDAR points + metadata) from
    training set for use in copy-paste augmentation.

    For each annotated object in the training set, stores:
    - Points belonging to that object (in object-local frame)
    - Category name, box size, number of points
    - Optionally: 2D image crop of the object from each visible camera

    The database is saved to disk as a pickle file and can be loaded
    efficiently during training.
    """

    def __init__(self, dataroot: str, output_dir: str,
                 min_points: int = 5, dataset_name: str = 'nuscenes'):
        self.dataroot = Path(dataroot)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_points = min_points
        self.dataset_name = dataset_name

    def build_from_nuscenes(self, sample_pairs: List[Dict],
                            sample_annotations: Dict,
                            calibrations,
                            ego_poses,
                            lidar_loader_fn,
                            image_loader_fn=None,
                            category_fn=None,
                            include_lidarseg_labels: bool = False,
                            progress_every: int = 1000):
        """
        Build GT database by iterating over all training samples.

        Args:
            sample_pairs: list of sample dicts (from dataset._build_sample_pairs)
            sample_annotations: sample_token → list of annotation dicts
            calibrations: calibration store/dict
            ego_poses: ego pose store/dict
            lidar_loader_fn: callable(pair) → (N, 5) numpy array
            image_loader_fn: callable(pair) → PIL.Image (optional, for image crops)
            category_fn: callable(instance_token) → category_name
        """
        from src.lidar_utils import load_lidar_bin

        import time

        database = {}  # category → list of object dicts
        # Avoid redundant work: if we are NOT extracting image crops, process each sample once.
        # If we ARE extracting image crops, allow multiple cameras per sample, but cache LiDAR.
        seen_keys = set()
        lidar_points_cache = {}
        lidarseg_cache = {}

        lidarseg_dir = None
        if include_lidarseg_labels:
            for candidate in [
                self.dataroot / "lidarseg" / "v1.0-trainval",
                self.dataroot / "lidarseg" / "v1.0-mini",
            ]:
                if candidate.exists():
                    lidarseg_dir = candidate
                    break
            if lidarseg_dir is None:
                print("[GTDB] lidarseg not found; semantic labels will not be stored", flush=True)

        t0 = time.time()
        n_pairs_total = len(sample_pairs)
        n_pairs_seen = 0
        n_objects_kept = 0

        print(
            f"[GTDB] start: pairs={n_pairs_total} image_crops={'on' if image_loader_fn is not None else 'off'}",
            flush=True,
        )

        for pair in sample_pairs:
            n_pairs_seen += 1
            if progress_every and (n_pairs_seen % int(progress_every)) == 0:
                dt = max(1e-6, time.time() - t0)
                rate = n_pairs_seen / dt
                print(
                    f"[GTDB] pairs={n_pairs_seen}/{n_pairs_total} "
                    f"({100.0*n_pairs_seen/max(1,n_pairs_total):.1f}%) "
                    f"objs={n_objects_kept} rate={rate:.1f} pairs/s"
                    , flush=True
                )
            sample_token = pair['sample_token']
            camera_name = pair.get('camera_name', None)
            if image_loader_fn is None:
                key = sample_token
            else:
                key = (sample_token, camera_name)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Load raw LiDAR (cache per sample)
            if sample_token in lidar_points_cache:
                lidar_points = lidar_points_cache[sample_token]
            else:
                try:
                    lidar_points = lidar_loader_fn(pair)
                except Exception:
                    continue
                lidar_points_cache[sample_token] = lidar_points

            lidarseg_labels_full = None
            if lidarseg_dir is not None and include_lidarseg_labels:
                # lidarseg labels are keyed by lidar sample_data token
                lidar_sd_token = pair.get('lidar_sd_token', '')
                if lidar_sd_token:
                    if lidar_sd_token in lidarseg_cache:
                        lidarseg_labels_full = lidarseg_cache[lidar_sd_token]
                    else:
                        try:
                            from src.detection_labels import load_lidarseg_bin
                            lidarseg_labels_full = load_lidarseg_bin(lidarseg_dir, lidar_sd_token)
                        except Exception:
                            lidarseg_labels_full = None
                        lidarseg_cache[lidar_sd_token] = lidarseg_labels_full

            if lidar_points is None or len(lidar_points) < 10:
                continue

            annotations = sample_annotations.get(sample_token, [])
            if not annotations:
                continue

            # Get calibration for lidar-to-world transform
            lidar_calib_token = pair.get('lidar_calib_token')
            lidar_ego_token = pair.get('lidar_ego_token')

            # Build world<->lidar transforms for this sample so we can convert
            # global nuScenes annotations into the LiDAR sensor frame.
            try:
                if hasattr(calibrations, 'get_calib'):
                    lidar_calib = calibrations.get_calib(lidar_calib_token)
                    # In this codebase, ego poses are stored in the same compact store.
                    lidar_ego = calibrations.get_ego(lidar_ego_token)
                else:
                    lidar_calib = calibrations.get(lidar_calib_token, {})
                    lidar_ego = ego_poses.get(lidar_ego_token, {})

                lidar_to_ego = np.eye(4, dtype=np.float64)
                lidar_to_ego[:3, :3] = quat_to_rot(np.asarray(lidar_calib['rotation'], dtype=np.float64))
                lidar_to_ego[:3, 3] = np.asarray(lidar_calib['translation'], dtype=np.float64)

                ego_to_world = np.eye(4, dtype=np.float64)
                ego_to_world[:3, :3] = quat_to_rot(np.asarray(lidar_ego['rotation'], dtype=np.float64))
                ego_to_world[:3, 3] = np.asarray(lidar_ego['translation'], dtype=np.float64)

                lidar_to_world = ego_to_world @ lidar_to_ego
                world_to_lidar = np.linalg.inv(lidar_to_world)
            except Exception:
                lidar_to_world = None
                world_to_lidar = None

            for ann in annotations:
                cat_name = category_fn(ann.get('instance_token', '')) if category_fn else ''
                if not cat_name:
                    continue

                # nuScenes annotations are in world coordinates; convert to LiDAR sensor frame.
                center_w = np.asarray(ann['translation'], dtype=np.float64)
                size = np.asarray(ann['size'], dtype=np.float64)
                rotation_w = np.asarray(ann['rotation'], dtype=np.float64)

                if world_to_lidar is not None:
                    center_l = (world_to_lidar @ np.append(center_w, 1.0))[:3]
                    Rw = quat_to_rot(rotation_w)
                    Rl = world_to_lidar[:3, :3] @ Rw
                    rotation_l = rot_matrix_to_quat(Rl)
                    center = center_l
                    rotation = rotation_l
                else:
                    # Fallback (may yield poor DB, but won't crash)
                    center = center_w
                    rotation = rotation_w

                # Find points inside this box
                mask = points_in_box_nuscenes(lidar_points[:, :3], center, size, rotation,
                                             margin=0.0)
                n_pts = mask.sum()
                if n_pts < self.min_points:
                    continue

                obj_points = lidar_points[mask].copy()
                obj_lidarseg = None
                if lidarseg_labels_full is not None and len(lidarseg_labels_full) == len(lidar_points):
                    try:
                        obj_lidarseg = np.asarray(lidarseg_labels_full[mask], dtype=np.uint8).copy()
                    except Exception:
                        obj_lidarseg = None
                # Store in object-local frame (rotation-neutralized, centered at box center)
                # local_xyz = R^T (xyz - center)
                R = quat_to_rot(rotation)
                obj_points[:, :3] = (R.T @ (obj_points[:, :3] - center).T).T

                obj_entry = {
                    'points': obj_points,       # (M, 5) in local frame
                    'points_frame': 'object',
                    'category': cat_name,
                    'center': center.copy(),
                    'size': size.copy(),
                    'rotation': rotation.copy(),
                    'num_points': int(n_pts),
                    'sample_token': sample_token,
                }

                if obj_lidarseg is not None:
                    # Per-point semantic labels (nuScenes lidarseg ids) aligned with obj_entry['points'].
                    obj_entry['lidarseg_labels'] = obj_lidarseg

                # Optionally store image crop
                if image_loader_fn is not None:
                    try:
                        cam_calib_token = pair.get('cam_calib_token')
                        cam_ego_token = pair.get('cam_ego_token')
                        img = image_loader_fn(pair)
                        if img is not None:
                            crop = self._extract_image_crop(
                                img, center, size, rotation,
                                calibrations, ego_poses,
                                cam_calib_token, cam_ego_token,
                                lidar_calib_token, lidar_ego_token,
                                source_camera_name=camera_name,
                            )
                            if crop is not None:
                                obj_entry['image_crop'] = crop
                    except Exception:
                        pass

                if cat_name not in database:
                    database[cat_name] = []
                database[cat_name].append(obj_entry)
                n_objects_kept += 1

        # Summary
        total = sum(len(v) for v in database.values())
        print(f"\n{'='*60}")
        print(f"GT Database Built: {total} objects across {len(database)} classes")
        for cat, objects in sorted(database.items(), key=lambda x: -len(x[1])):
            pts_stats = [o['num_points'] for o in objects]
            print(f"  {cat:30s}: {len(objects):5d} objects, "
                  f"pts: {np.mean(pts_stats):.0f} avg, {np.min(pts_stats)}-{np.max(pts_stats)} range")
        print(f"{'='*60}\n")

        # Save
        db_path = self.output_dir / f"gt_database_{self.dataset_name}.pkl"
        with open(db_path, 'wb') as f:
            pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved GT database to {db_path}")

        return database

    def _extract_image_crop(self, img, center, size, rotation,
                            calibrations, ego_poses,
                            cam_calib_token, cam_ego_token,
                            lidar_calib_token, lidar_ego_token,
                            source_camera_name: Optional[str] = None):
        """Extract tight 2D crop of object from camera image."""
        try:
            # Get calibrations
            if hasattr(calibrations, 'get_calib'):
                cam_calib = calibrations.get_calib(cam_calib_token)
                lidar_calib = calibrations.get_calib(lidar_calib_token)
            else:
                cam_calib = calibrations.get(cam_calib_token, {})
                lidar_calib = calibrations.get(lidar_calib_token, {})

            if hasattr(ego_poses, 'get_ego'):
                cam_ego = ego_poses.get_ego(cam_ego_token)
                lidar_ego = ego_poses.get_ego(lidar_ego_token)
            else:
                cam_ego = ego_poses.get(cam_ego_token, {})
                lidar_ego = ego_poses.get(lidar_ego_token, {})

            intrinsic = np.asarray(cam_calib.get('intrinsic'), dtype=np.float64)
            if intrinsic is None:
                return None

            # Build lidar-to-camera transform
            from src.detection_labels import compute_lidar_to_cam_transform
            lidar_to_cam = compute_lidar_to_cam_transform(
                cam_calib, lidar_calib, cam_ego, lidar_ego)

            # Project 3D box corners to 2D
            corners = get_box_corners_3d_nuscenes(center, size, rotation)
            corners_cam = (lidar_to_cam[:3, :3] @ corners.T + lidar_to_cam[:3, 3:4]).T
            # Filter behind camera
            valid = corners_cam[:, 2] > 0.1
            if valid.sum() < 4:
                return None

            cam_v = corners_cam[valid]
            # Project with 3x3 intrinsics
            u = intrinsic[0, 0] * cam_v[:, 0] / cam_v[:, 2] + intrinsic[0, 2]
            v = intrinsic[1, 1] * cam_v[:, 1] / cam_v[:, 2] + intrinsic[1, 2]
            uv = np.column_stack([u, v])

            img_w, img_h = img.size
            u_min = max(0, int(np.floor(uv[:, 0].min())))
            u_max = min(img_w, int(np.ceil(uv[:, 0].max())))
            v_min = max(0, int(np.floor(uv[:, 1].min())))
            v_max = min(img_h, int(np.ceil(uv[:, 1].max())))

            if u_max - u_min < 4 or v_max - v_min < 4:
                return None

            crop = img.crop((u_min, v_min, u_max, v_max))
            return {
                'image': np.array(crop),
                'bbox_2d': (u_min, v_min, u_max, v_max),
                'source_camera_name': source_camera_name,
            }
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# COPY-PASTE AUGMENTOR
# ═══════════════════════════════════════════════════════════════════════════════

class CopyPasteAugmentor:
    """
    Copy-paste augmentor that:
    1. Samples objects from a GT database (oversampling rare classes)
    2. Places them at random valid locations in the current scene
    3. Removes existing points occluded by pasted objects
    4. Pastes corresponding image pixels into the camera image

    This addresses class imbalance by enriching training scenes with
    underrepresented categories (e.g., bicycle, motorcycle, trailer).

    Config dict keys:
        max_paste_objects: maximum number of objects to paste per sample
        class_weights: dict mapping category → sampling weight (higher = more likely)
        min_distance: minimum distance from ego vehicle for placement
        max_distance: maximum distance from ego vehicle for placement
        collision_threshold: IoU threshold to reject overlapping placements
        paste_image: whether to also paste image crops (requires crops in DB)
        height_offset_std: std-dev for random height offset when pasting
    """

    def __init__(self, database_path: str, cfg: Optional[Dict] = None):
        """
        Args:
            database_path: path to gt_database_*.pkl
            cfg: augmentor configuration dict
        """
        cfg = cfg or {}
        self.max_paste = cfg.get('max_paste_objects', 15)
        # RGB pasting is optional and only works when the GT DB contains compatible crops.
        self.paste_image = cfg.get('paste_image', False)
        self.min_distance = cfg.get('min_distance', 3.0)
        self.max_distance = cfg.get('max_distance', 60.0)
        self.collision_threshold = cfg.get('collision_threshold', 0.0)  # No overlap
        self.height_offset_std = cfg.get('height_offset_std', 0.0)

        # Camera-aware placement: prefer/require pasted objects to be visible in the
        # current camera view when intrinsics + lidar_to_cam are provided.
        self.require_in_view = cfg.get('require_in_view', True)
        self.min_box_px = cfg.get('min_box_px', 8)  # min(w,h) in pixels
        self.max_trials_per_object = cfg.get('max_trials_per_object', 30)
        self.view_angle_std = cfg.get('view_angle_std', np.deg2rad(35.0))
        self.min_paste_in_view = cfg.get('min_paste_in_view', 3)

        # Scene-aware placement (simple realism heuristics)
        # - If there's a close obstacle in front of the camera, don't paste new objects.
        # - Otherwise, only paste into locally free BEV regions (low point density).
        self.enable_scene_aware_placement = cfg.get('enable_scene_aware_placement', True)
        # Camera-view proximity gate (simple): if a close object is in the image center, skip pasting.
        self.front_clearance_m = float(cfg.get('front_clearance_m', 10.0))
        self.image_gate_mode = str(cfg.get('image_gate_mode', 'full'))  # 'center' or 'full'
        self.image_gate_center_frac = float(cfg.get('image_gate_center_frac', 0.6))
        self.image_gate_min_box_px = int(cfg.get('image_gate_min_box_px', 24))

        # Image-space realism constraints to reduce clutter.
        self.max_pasted_area_frac = float(cfg.get('max_pasted_area_frac', 0.18))
        self.max_single_box_area_frac = float(cfg.get('max_single_box_area_frac', 0.20))
        self.soft_big_box_area_frac = float(cfg.get('soft_big_box_area_frac', 0.12))
        self.soft_big_box_keep_prob = float(cfg.get('soft_big_box_keep_prob', 0.15))
        self.max_pasted_iou = float(cfg.get('max_pasted_iou', 0.10))
        self.max_existing_iou = float(cfg.get('max_existing_iou', 0.25))
        self.max_existing_boxes_in_view = int(cfg.get('max_existing_boxes_in_view', 10))

        # RGB crop realism: avoid extreme warps when reusing a crop at a very different
        # projected aspect/scale (especially noticeable for vehicles).
        self.max_aspect_log_diff = float(cfg.get('max_aspect_log_diff', np.log(2.0)))  # ~2x aspect change
        self.max_aspect_log_diff_vehicle = float(cfg.get('max_aspect_log_diff_vehicle', np.log(1.6)))
        self.max_scale_factor = float(cfg.get('max_scale_factor', 3.0))
        self.max_scale_factor_vehicle = float(cfg.get('max_scale_factor_vehicle', 2.5))

        # Optional BEV free-space check (can be too strict; off by default)
        self.enable_free_space_check = bool(cfg.get('enable_free_space_check', False))

        # Z band for occupancy grid / clearance heuristics
        self.front_z_min = float(cfg.get('front_z_min', -2.5))
        self.front_z_max = float(cfg.get('front_z_max', 2.5))
        self.free_grid_cell_m = float(cfg.get('free_grid_cell_m', 0.5))
        self.free_region_radius_m = float(cfg.get('free_region_radius_m', 1.5))
        self.occ_points_thresh = int(cfg.get('occ_points_thresh', 6))
        self.grid_extent_m = float(cfg.get('grid_extent_m', 70.0))

        # LiDAR occlusion simulation (approximate): after paste, keep only the
        # closest point per (pitch,yaw) bin, similar to a range-image z-buffer.
        # This makes pasted objects behind other geometry disappear.
        self.simulate_lidar_occlusion = cfg.get('simulate_lidar_occlusion', True)
        self.occlusion_H = int(cfg.get('occlusion_H', 64))
        self.occlusion_W = int(cfg.get('occlusion_W', 1024))
        self.occlusion_fov_up = float(cfg.get('occlusion_fov_up', 10.0))
        self.occlusion_fov_down = float(cfg.get('occlusion_fov_down', -30.0))
        # Expand occlusion beyond exact bin overlap to better mimic beam divergence
        # and quantization (removes points behind within a small angular neighborhood).
        self.occlusion_bin_radius = int(cfg.get('occlusion_bin_radius', 1))
        self.occlusion_depth_margin_m = float(cfg.get('occlusion_depth_margin_m', 0.25))

        # Debug info from the last call (useful for visualization/verification)
        self.last_debug = {}

        # Class sampling weights (higher = paste more often)
        # Default: oversample rare classes
        self.class_weights = cfg.get('class_weights', {
            'human.pedestrian.adult': 1.0,
            'human.pedestrian.child': 3.0,
            'human.pedestrian.construction_worker': 2.0,
            'vehicle.car': 0.5,
            'vehicle.truck': 1.5,
            'vehicle.bus.bendy': 2.0,
            'vehicle.bus.rigid': 2.0,
            'vehicle.trailer': 3.0,
            'vehicle.construction': 3.0,
            'vehicle.motorcycle': 3.0,
            'vehicle.bicycle': 3.0,
            'movable_object.traffic_cone': 1.0,
            'movable_object.barrier': 0.5,
        })

        # Load GT database
        self.database = {}
        if database_path and Path(database_path).exists():
            with open(database_path, 'rb') as f:
                self.database = pickle.load(f)
            total = sum(len(v) for v in self.database.values())
            print(f"Loaded GT database: {total} objects, {len(self.database)} classes")
        else:
            print(f"Warning: GT database not found at {database_path}. "
                  "Copy-paste augmentation disabled.")

        # Build weighted sampling pool
        self._build_sampling_pool()

    def _build_sampling_pool(self):
        """Build a weighted list for efficient sampling."""
        self.sampling_pool = []
        self.sampling_weights = []

        for cat, objects in self.database.items():
            weight = self._get_class_weight(cat)
            for obj in objects:
                self.sampling_pool.append(obj)
                self.sampling_weights.append(weight)

        if self.sampling_weights:
            total = sum(self.sampling_weights)
            self.sampling_weights = [w / total for w in self.sampling_weights]
        else:
            self.sampling_weights = []

    def _get_class_weight(self, category: str) -> float:
        """Get sampling weight for a category, with prefix matching."""
        # Exact match
        if category in self.class_weights:
            return self.class_weights[category]
        # Prefix match (e.g., 'vehicle.car' matches 'vehicle.car.sedan')
        for key, weight in self.class_weights.items():
            if category.startswith(key) or key.startswith(category):
                return weight
        return 1.0  # Default weight

    def __call__(self, points: np.ndarray, annotations: List[Dict],
                 image: Optional[Image.Image] = None,
                 intrinsic: Optional[np.ndarray] = None,
                 lidar_to_cam: Optional[np.ndarray] = None,
                 current_camera_name: Optional[str] = None,
                 lidarseg_labels: Optional[np.ndarray] = None,
                 ) -> Tuple[np.ndarray, List[Dict], Optional[Image.Image]]:
        """
        Apply copy-paste augmentation.

        Args:
            points: (N, 5) raw LiDAR points in sensor frame
            annotations: list of annotation dicts in LiDAR frame
            image: optional PIL camera image to also paste into
            intrinsic: (3, 3) camera intrinsic matrix
            lidar_to_cam: (4, 4) LiDAR → camera transform

        Returns:
            points: augmented point cloud
            annotations: augmented annotation list
            image: augmented camera image (or None)
        """
        if not self.sampling_pool:
            if lidarseg_labels is None:
                return points, annotations, image
            return points, annotations, image, lidarseg_labels

        n_paste = np.random.randint(1, self.max_paste + 1)

        img_w = None
        img_h = None
        if image is not None:
            img_w, img_h = image.size

        cam_ok = (
            intrinsic is not None and lidar_to_cam is not None and img_w is not None and img_h is not None
        )

        use_nuscenes_geom = bool(current_camera_name is not None and str(current_camera_name).startswith('CAM_'))
        corners_fn = get_box_corners_3d_nuscenes if use_nuscenes_geom else get_box_corners_3d
        in_box_fn = points_in_box_nuscenes if use_nuscenes_geom else points_in_box

        # Compute camera forward direction in LiDAR frame for biased sampling.
        cam_forward_yaw = None
        if cam_ok:
            try:
                cam_to_lidar = np.linalg.inv(np.asarray(lidar_to_cam, dtype=np.float64))
                forward_lidar = cam_to_lidar[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
                cam_forward_yaw = float(np.arctan2(forward_lidar[1], forward_lidar[0]))
            except Exception:
                cam_forward_yaw = None

        # ── Scene-aware gating (camera-view close object) ──────────
        close_obj_found = False
        close_obj_depth_m = None
        if self.enable_scene_aware_placement and cam_ok and image is not None and intrinsic is not None and lidar_to_cam is not None:
            try:
                img_w, img_h = image.size
                cx0 = (1.0 - self.image_gate_center_frac) * 0.5 * img_w
                cx1 = (1.0 + self.image_gate_center_frac) * 0.5 * img_w
                cy0 = (1.0 - self.image_gate_center_frac) * 0.5 * img_h
                cy1 = (1.0 + self.image_gate_center_frac) * 0.5 * img_h

                K = np.asarray(intrinsic, dtype=np.float64)
                T = np.asarray(lidar_to_cam, dtype=np.float64)

                for ann in annotations:
                    # Only consider existing objects (ignore pasted ones)
                    if str(ann.get('instance_token', '')).startswith('paste_'):
                        continue

                    c = np.asarray(ann.get('translation', [0, 0, 0]), dtype=np.float64)
                    s = np.asarray(ann.get('size', [1, 1, 1]), dtype=np.float64)
                    r = np.asarray(ann.get('rotation', [1, 0, 0, 0]), dtype=np.float64)
                    corners = corners_fn(c, s, r)
                    corners_h = np.hstack([corners, np.ones((8, 1), dtype=np.float64)])
                    cam = (T @ corners_h.T).T
                    valid = cam[:, 2] > 0.5
                    if valid.sum() < 4:
                        continue
                    zmin = float(np.min(cam[valid, 2]))
                    if close_obj_depth_m is None or zmin < close_obj_depth_m:
                        close_obj_depth_m = zmin

                    u = K[0, 0] * cam[valid, 0] / cam[valid, 2] + K[0, 2]
                    v = K[1, 1] * cam[valid, 1] / cam[valid, 2] + K[1, 2]
                    u0, u1 = float(np.min(u)), float(np.max(u))
                    v0, v1 = float(np.min(v)), float(np.max(v))

                    if (u1 - u0) < self.image_gate_min_box_px or (v1 - v0) < self.image_gate_min_box_px:
                        continue

                    # Gate mode:
                    # - 'full': any sufficiently large close object anywhere in the image blocks pasting
                    # - 'center': only objects overlapping a central ROI block pasting
                    if self.image_gate_mode == 'center':
                        overlaps = not (u1 < cx0 or u0 > cx1 or v1 < cy0 or v0 > cy1)
                    else:
                        # any overlap with image bounds counts
                        overlaps = not (u1 < 0 or u0 > (img_w - 1) or v1 < 0 or v0 > (img_h - 1))

                    if overlaps and zmin < self.front_clearance_m:
                        close_obj_found = True
                        break
            except Exception:
                close_obj_found = False

        # Collect existing 2D boxes in view for overlap/busyness checks
        existing_boxes_2d = []
        if self.enable_scene_aware_placement and cam_ok:
            try:
                for ann in annotations:
                    if str(ann.get('instance_token', '')).startswith('paste_'):
                        continue
                    bb = _box_bbox2d(
                        np.asarray(ann.get('translation', [0, 0, 0]), dtype=np.float64),
                        np.asarray(ann.get('size', [1, 1, 1]), dtype=np.float64),
                        np.asarray(ann.get('rotation', [1, 0, 0, 0]), dtype=np.float64),
                    )
                    if bb is None:
                        continue
                    u0, v0, u1, v1, _zmin = bb
                    if (u1 - u0) < self.image_gate_min_box_px or (v1 - v0) < self.image_gate_min_box_px:
                        continue
                    existing_boxes_2d.append((u0, v0, u1, v1, _zmin))
            except Exception:
                existing_boxes_2d = []

        # Build a simple BEV occupancy grid from existing points (used to avoid pasting into clutter)
        occ_grid = None
        occ_params = None
        if self.enable_scene_aware_placement and self.enable_free_space_check and points is not None and len(points) > 0:
            extent = float(self.grid_extent_m)
            cell = float(self.free_grid_cell_m)
            size = int(np.ceil((2 * extent) / cell))
            # only consider near-ground-ish points for occupancy
            z = points[:, 2].astype(np.float64)
            mask = (z > self.front_z_min) & (z < self.front_z_max)
            xy = points[mask, :2].astype(np.float64)
            if len(xy) > 0:
                ix = np.floor((xy[:, 0] + extent) / cell).astype(np.int32)
                iy = np.floor((xy[:, 1] + extent) / cell).astype(np.int32)
                valid = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
                ix = ix[valid]; iy = iy[valid]
                counts = np.zeros((size, size), dtype=np.uint16)
                np.add.at(counts, (ix, iy), 1)
                occ_grid = counts
                occ_params = (extent, cell, size)

        # Precompute K/T once (used heavily in placement trials)
        K = None
        T = None
        if cam_ok:
            try:
                K = np.asarray(intrinsic, dtype=np.float64)
                T = np.asarray(lidar_to_cam, dtype=np.float64)
            except Exception:
                K = None
                T = None

        def _precompute_corners_rel(size: np.ndarray, rot_q: np.ndarray) -> np.ndarray:
            """Return (8,3) corners in LiDAR frame relative to the box center."""
            corners = corners_fn(
                np.zeros(3, dtype=np.float64),
                np.asarray(size, dtype=np.float64),
                np.asarray(rot_q, dtype=np.float64),
            )
            return corners.astype(np.float64)

        def _project_corners(center_xyz: np.ndarray, corners_rel: np.ndarray):
            """Project 3D corners to image; returns (u,v,z) arrays for valid corners."""
            if not cam_ok or K is None or T is None:
                return None
            corners = corners_rel + np.asarray(center_xyz, dtype=np.float64)
            cam = (T[:3, :3] @ corners.T).T + T[:3, 3]
            valid = cam[:, 2] > 0.5
            if valid.sum() < 4:
                return None
            cam_v = cam[valid]
            u = K[0, 0] * cam_v[:, 0] / cam_v[:, 2] + K[0, 2]
            v = K[1, 1] * cam_v[:, 1] / cam_v[:, 2] + K[1, 2]
            z = cam_v[:, 2]
            return u, v, z

        def _box_in_view(center_xyz: np.ndarray, corners_rel: np.ndarray) -> bool:
            if not cam_ok:
                return True
            try:
                proj = _project_corners(center_xyz, corners_rel)
                if proj is None:
                    return False
                u, v, _z = proj
                u_min = np.min(u); u_max = np.max(u)
                v_min = np.min(v); v_max = np.max(v)
                if (u_max - u_min) < self.min_box_px or (v_max - v_min) < self.min_box_px:
                    return False
                # Require overlap with image bounds
                if u_max < 0 or u_min > (img_w - 1) or v_max < 0 or v_min > (img_h - 1):
                    return False
                # Prefer fully inside, but allow partial overlap
                return True
            except Exception:
                return False

        def _box_bbox2d(center_xyz: np.ndarray, corners_rel: np.ndarray):
            """Return (u0,v0,u1,v1,zmin) in pixels (unclipped) or None."""
            if not cam_ok:
                return None
            try:
                proj = _project_corners(center_xyz, corners_rel)
                if proj is None:
                    return None
                u, v, z = proj
                zmin = float(np.min(z))
                return float(np.min(u)), float(np.min(v)), float(np.max(u)), float(np.max(v)), zmin
            except Exception:
                return None

        def _iou(a, b) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            ix0 = max(ax0, bx0)
            iy0 = max(ay0, by0)
            ix1 = min(ax1, bx1)
            iy1 = min(ay1, by1)
            iw = max(0.0, ix1 - ix0)
            ih = max(0.0, iy1 - iy0)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
            area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
            denom = area_a + area_b - inter
            return float(inter / denom) if denom > 0 else 0.0

        # Collect existing box centers for collision checking
        existing_centers = []
        existing_sizes = []
        for ann in annotations:
            existing_centers.append(np.asarray(ann['translation'][:3], dtype=np.float64))
            existing_sizes.append(np.asarray(ann['size'], dtype=np.float64))

        new_points_list = []
        new_lidarseg_list = []
        new_annotations = []
        removal_masks = np.ones(len(points), dtype=bool)

        lidarseg_out = None
        if lidarseg_labels is not None:
            try:
                lidarseg_out = np.asarray(lidarseg_labels, dtype=np.uint8)
                if len(lidarseg_out) != len(points):
                    lidarseg_out = None
            except Exception:
                lidarseg_out = None

        if self.paste_image and image is not None:
            img_array = np.array(image)
            paste_to_image = True
        else:
            img_array = None
            paste_to_image = False

        # Sample objects to paste. If RGB pasting is enabled, prefer objects that
        # actually have a crop for the current camera so the RGB image changes.
        pool_n = len(self.sampling_pool)
        if pool_n == 0:
            return points, annotations, image

        eligible_rgb_indices = None
        if self.paste_image and paste_to_image and cam_ok and current_camera_name is not None:
            eligible_rgb_indices = []
            for j, obj in enumerate(self.sampling_pool):
                crop = obj.get('image_crop')
                if crop is None:
                    continue
                src_cam = crop.get('source_camera_name')
                if src_cam == current_camera_name:
                    eligible_rgb_indices.append(j)

        if eligible_rgb_indices is not None and len(eligible_rgb_indices) > 0:
            rep = n_paste > len(eligible_rgb_indices)
            indices = np.random.choice(eligible_rgb_indices, size=n_paste, replace=rep)
        else:
            indices = np.random.choice(
                pool_n,
                size=n_paste,
                replace=True,
                p=self.sampling_weights if self.sampling_weights else None,
            )

        accepted = 0
        accepted_in_view = 0
        rgb_paste_attempts = 0
        rgb_paste_applied = 0
        rgb_paste_skipped_warp = 0
        pasted_boxes_2d = []
        pasted_area_frac = 0.0

        # Gate pasting if the scene is "blocked" in front.
        skip_due_to_clearance = False
        if self.enable_scene_aware_placement and close_obj_found:
            skip_due_to_clearance = True
            indices = []

        skip_due_to_busyness = False
        if self.enable_scene_aware_placement and cam_ok and len(existing_boxes_2d) > self.max_existing_boxes_in_view:
            skip_due_to_busyness = True
            indices = []

        for idx in indices:
            obj = self.sampling_pool[idx]
            obj_points = obj['points'].copy()       # (M, 5) in local frame
            obj_size = obj['size'].copy()
            obj_rotation = obj['rotation'].copy()
            corners_rel = None
            if cam_ok:
                try:
                    corners_rel = _precompute_corners_rel(obj_size, obj_rotation)
                except Exception:
                    corners_rel = None
            obj_lidarseg = obj.get('lidarseg_labels', None)
            if obj_lidarseg is not None:
                try:
                    obj_lidarseg = np.asarray(obj_lidarseg, dtype=np.uint8)
                    if len(obj_lidarseg) != len(obj_points):
                        obj_lidarseg = None
                except Exception:
                    obj_lidarseg = None

            # Choose new placement location.
            # If camera intrinsics/extrinsics are available, bias to camera-forward
            # and optionally require the projected box to be in view.
            placed = False
            new_center = None
            for _trial in range(int(self.max_trials_per_object)):
                orig_dist = np.linalg.norm(np.asarray(obj['center'][:2], dtype=np.float64))
                new_dist = np.clip(orig_dist + np.random.normal(0, 5), self.min_distance, self.max_distance)

                if cam_forward_yaw is not None:
                    new_angle = np.random.normal(cam_forward_yaw, self.view_angle_std)
                else:
                    new_angle = np.random.uniform(-np.pi, np.pi)

                cand = np.array(
                    [
                        new_dist * np.cos(new_angle),
                        new_dist * np.sin(new_angle),
                        float(obj['center'][2]),
                    ],
                    dtype=np.float64,
                )

                if self.height_offset_std > 0:
                    cand[2] += np.random.normal(0, self.height_offset_std)

                if self.require_in_view and cam_ok:
                    if corners_rel is None or not _box_in_view(cand, corners_rel):
                        continue

                # Image-space size/overlap/busyness constraints
                if cam_ok:
                    if corners_rel is None:
                        continue
                    bb = _box_bbox2d(cand, corners_rel)
                    if bb is None:
                        continue
                    u0, v0, u1, v1, _zmin = bb

                    # Clip to image bounds for area/IoU
                    u0c = max(0.0, min(float(img_w - 1), u0))
                    v0c = max(0.0, min(float(img_h - 1), v0))
                    u1c = max(0.0, min(float(img_w - 1), u1))
                    v1c = max(0.0, min(float(img_h - 1), v1))
                    w = max(0.0, u1c - u0c)
                    h = max(0.0, v1c - v0c)
                    if w < self.min_box_px or h < self.min_box_px:
                        continue

                    img_area = float(img_w * img_h)
                    area_frac = (w * h) / max(1.0, img_area)

                    # Hard reject huge pasted boxes (screen takeover)
                    if area_frac > self.max_single_box_area_frac:
                        continue

                    # Make big boxes rare
                    if area_frac > self.soft_big_box_area_frac:
                        if np.random.random() > self.soft_big_box_keep_prob:
                            continue

                    # Prevent excessive total coverage
                    if (pasted_area_frac + area_frac) > self.max_pasted_area_frac:
                        continue

                    cand_box = (u0c, v0c, u1c, v1c)

                    # Avoid overlaps with existing objects
                    too_much_existing = False
                    for ex in existing_boxes_2d:
                        ex_box = (ex[0], ex[1], ex[2], ex[3])
                        if _iou(cand_box, ex_box) > self.max_existing_iou:
                            too_much_existing = True
                            break
                    if too_much_existing:
                        continue

                    # Avoid overlaps among pasted objects
                    too_much_pasted = False
                    for pb in pasted_boxes_2d:
                        if _iou(cand_box, pb) > self.max_pasted_iou:
                            too_much_pasted = True
                            break
                    if too_much_pasted:
                        continue

                # Free-space check in BEV: reject if local neighborhood is too occupied
                if self.enable_scene_aware_placement and self.enable_free_space_check and occ_grid is not None and occ_params is not None:
                    extent, cell, size = occ_params
                    gx = int(np.floor((cand[0] + extent) / cell))
                    gy = int(np.floor((cand[1] + extent) / cell))
                    rad = int(np.ceil(self.free_region_radius_m / cell))
                    if gx < 0 or gx >= size or gy < 0 or gy >= size:
                        continue
                    x0 = max(0, gx - rad); x1 = min(size, gx + rad + 1)
                    y0 = max(0, gy - rad); y1 = min(size, gy + rad + 1)
                    if int(np.max(occ_grid[x0:x1, y0:y1])) >= int(self.occ_points_thresh):
                        continue

                new_center = cand
                placed = True
                break

            if not placed or new_center is None:
                continue

            # Check collision with existing boxes
            collision = False
            for ec, es in zip(existing_centers, existing_sizes):
                dist = np.linalg.norm(new_center[:2] - ec[:2])
                min_clearance = (np.linalg.norm(obj_size[:2]) + np.linalg.norm(es[:2])) / 2
                if dist < min_clearance * (1 + self.collision_threshold):
                    collision = True
                    break

            if collision:
                continue

            # If camera-aware mode is enabled, track that we are actually pasting in view.
            in_view = True
            if self.require_in_view and cam_ok:
                in_view = (corners_rel is not None) and _box_in_view(new_center, corners_rel)
                if not in_view:
                    continue

            # Transform object points from local frame to new world position
            R_obj = quat_to_rot(obj_rotation)
            world_points = obj_points.copy()
            world_points[:, :3] = (R_obj @ obj_points[:, :3].T).T + new_center

            # Remove existing points inside the new box location
            removal = in_box_fn(points[:, :3], new_center, obj_size,
                                obj_rotation, margin=0.1)
            removal_masks &= ~removal

            new_points_list.append(world_points)
            if obj_lidarseg is not None:
                new_lidarseg_list.append(obj_lidarseg.copy())
            elif lidarseg_out is not None:
                # If we're tracking semantics but the object has none, fill unknown.
                new_lidarseg_list.append(np.zeros(len(world_points), dtype=np.uint8))

            # Create annotation for pasted object
            new_ann = {
                'translation': new_center.tolist(),
                'size': obj_size.tolist(),
                'rotation': obj_rotation.tolist(),
                'instance_token': f'paste_{idx}_{np.random.randint(1e6)}',
                'category_name': obj['category'],
            }
            new_annotations.append(new_ann)

            # Track pasted box coverage
            if cam_ok:
                bb = _box_bbox2d(new_center, corners_rel) if corners_rel is not None else None
                if bb is not None:
                    u0, v0, u1, v1, _ = bb
                    u0c = max(0.0, min(float(img_w - 1), u0))
                    v0c = max(0.0, min(float(img_h - 1), v0))
                    u1c = max(0.0, min(float(img_w - 1), u1))
                    v1c = max(0.0, min(float(img_h - 1), v1))
                    pasted_boxes_2d.append((u0c, v0c, u1c, v1c))
                    pasted_area_frac += (max(0.0, u1c - u0c) * max(0.0, v1c - v0c)) / max(1.0, float(img_w * img_h))

            existing_centers.append(new_center)
            existing_sizes.append(obj_size)

            # ── Image paste ─────────────────────────────────────────
            if paste_to_image and obj.get('image_crop') is not None and \
               intrinsic is not None and lidar_to_cam is not None:
                src_cam = obj['image_crop'].get('source_camera_name', None)
                if src_cam is not None and current_camera_name is not None and src_cam != current_camera_name:
                    continue
                rgb_paste_attempts += 1
                ok = self._paste_object_to_image(
                    img_array, obj, new_center, obj_size, obj_rotation,
                    intrinsic, lidar_to_cam, use_nuscenes_geom=use_nuscenes_geom
                )
                if ok:
                    rgb_paste_applied += 1
                else:
                    rgb_paste_skipped_warp += 1

            accepted += 1
            if in_view:
                accepted_in_view += 1

            # Encourage multiple in-view objects early
            if self.require_in_view and cam_ok and accepted_in_view >= self.min_paste_in_view and accepted >= n_paste:
                break

        # Apply point removal and concat new points
        points = points[removal_masks]
        if lidarseg_out is not None:
            lidarseg_out = lidarseg_out[removal_masks]
        pasted_flags = np.zeros(len(points), dtype=bool)
        if new_points_list:
            all_new = np.concatenate(new_points_list, axis=0)
            points = np.concatenate([points, all_new], axis=0)
            pasted_flags = np.concatenate([pasted_flags, np.ones(len(all_new), dtype=bool)])
            if lidarseg_out is not None:
                if new_lidarseg_list:
                    all_new_lbl = np.concatenate(new_lidarseg_list, axis=0)
                else:
                    all_new_lbl = np.zeros(len(all_new), dtype=np.uint8)
                lidarseg_out = np.concatenate([lidarseg_out, all_new_lbl], axis=0)

        self.last_debug = {
            'total_before_occlusion': int(len(points)),
            'pasted_before_occlusion': int(pasted_flags.sum()),
            'accepted': int(accepted),
            'accepted_in_view': int(accepted_in_view),
            'rgb_paste_attempts': int(rgb_paste_attempts),
            'rgb_paste_applied': int(rgb_paste_applied),
            'rgb_paste_skipped_warp': int(rgb_paste_skipped_warp),
            'rgb_eligible_pool': int(len(eligible_rgb_indices) if eligible_rgb_indices is not None else 0),
            'close_obj_found': bool(close_obj_found),
            'closest_obj_depth_m': (None if close_obj_depth_m is None else float(close_obj_depth_m)),
            'skip_due_to_clearance': bool(skip_due_to_clearance),
            'skip_due_to_busyness': bool(skip_due_to_busyness) if 'skip_due_to_busyness' in locals() else False,
            'existing_boxes_in_view': int(len(existing_boxes_2d)) if 'existing_boxes_2d' in locals() else 0,
            'pasted_area_frac': float(pasted_area_frac),
        }

        # Approximate LiDAR occlusion: keep closest point per angular bin.
        if self.simulate_lidar_occlusion and points is not None and len(points) > 0:
            keep_idx = self._paste_selective_occlusion_keep_indices(points, pasted_flags)
            points = points[keep_idx]
            pasted_flags = pasted_flags[keep_idx]
            if lidarseg_out is not None:
                lidarseg_out = lidarseg_out[keep_idx]
            self.last_debug.update({
                'total_after_occlusion': int(len(points)),
                'pasted_after_occlusion': int(pasted_flags.sum()),
                'pasted_removed_by_occlusion': int(self.last_debug['pasted_before_occlusion'] - int(pasted_flags.sum())),
            })
        else:
            self.last_debug.update({
                'total_after_occlusion': int(len(points)),
                'pasted_after_occlusion': int(pasted_flags.sum()),
                'pasted_removed_by_occlusion': 0,
            })

        annotations = annotations + new_annotations

        # Convert image back
        if paste_to_image and img_array is not None:
            image = Image.fromarray(img_array)

        if lidarseg_out is None:
            return points, annotations, image
        return points, annotations, image, lidarseg_out

    def _paste_selective_occlusion_keep_indices(self, points: np.ndarray, pasted_flags: np.ndarray) -> np.ndarray:
        """Selective occlusion caused by pasted objects.

        Keeps all points except those that are behind a pasted point in the same
        (pitch,yaw) neighborhood. This avoids globally thinning the point cloud.

        Also makes pasted objects occlude each other: pasted points farther than
        the nearest pasted return in a bin-neighborhood are removed.
        """
        try:
            xyz = points[:, :3].astype(np.float64)
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            depth = np.sqrt(x * x + y * y + z * z)
            valid = depth > 0.1
            if valid.sum() < 10:
                return np.arange(len(points), dtype=np.int64)

            x = x[valid]; y = y[valid]; z = z[valid]
            depth = depth[valid]

            yaw = np.arctan2(y, x)
            pitch = np.arcsin(np.clip(z / depth, -1.0, 1.0))

            fov_up = np.deg2rad(self.occlusion_fov_up)
            fov_down = np.deg2rad(self.occlusion_fov_down)
            fov = fov_up - fov_down
            H = self.occlusion_H
            W = self.occlusion_W

            col = ((yaw + np.pi) / (2 * np.pi)) * W
            col = np.clip(col, 0, W - 1).astype(np.int32)
            row = (1.0 - (pitch - fov_down) / max(1e-6, fov)) * H
            row = np.clip(row, 0, H - 1).astype(np.int32)

            # Compute per-bin minimum depth (closest return)
            lin = row * W + col
            pasted_v = pasted_flags[valid].astype(bool)

            # Min depth per bin for pasted points only
            pasted_min_flat = np.full(H * W, np.inf, dtype=np.float64)
            if pasted_v.any():
                np.minimum.at(pasted_min_flat, lin[pasted_v], depth[pasted_v])
            pasted_min_grid = pasted_min_flat.reshape(H, W)

            # Neighborhood min of pasted depths
            r = max(0, int(self.occlusion_bin_radius))
            if r == 0:
                pasted_neigh_min = pasted_min_grid
            else:
                pad = np.full((H + 2 * r, W + 2 * r), np.inf, dtype=np.float64)
                pad[r:r + H, r:r + W] = pasted_min_grid
                pasted_neigh_min = np.full((H, W), np.inf, dtype=np.float64)
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        pasted_neigh_min = np.minimum(
                            pasted_neigh_min,
                            pad[r + dy:r + dy + H, r + dx:r + dx + W],
                        )

            # If there's no pasted point in the neighborhood, keep the point.
            # If there is, keep only points within (pasted_neigh_min + margin).
            margin = float(self.occlusion_depth_margin_m)
            pasted_depth_here = pasted_neigh_min[row, col]
            has_paste = np.isfinite(pasted_depth_here)
            keep_valid = (~has_paste) | (depth <= (pasted_depth_here + margin))

            orig_idx = np.nonzero(valid)[0]
            keep_orig = orig_idx[keep_valid]
            return keep_orig.astype(np.int64)
        except Exception:
            return np.arange(len(points), dtype=np.int64)

    def _paste_object_to_image(self, img_array: np.ndarray,
                                obj: Dict,
                                new_center: np.ndarray,
                                obj_size: np.ndarray,
                                obj_rotation: np.ndarray,
                                intrinsic: np.ndarray,
                                lidar_to_cam: np.ndarray,
                                use_nuscenes_geom: bool = False):
        """
        Paste an object's image crop into the camera image at the
        projected position of the new 3D placement.

        Uses the stored 2D crop and reprojects based on new location
        to determine target region in the image.
        """
        try:
            crop_info = obj['image_crop']
            crop_img = crop_info['image']  # (H, W, 3) numpy

            # Project new 3D box corners to camera image
            corners = (get_box_corners_3d_nuscenes if use_nuscenes_geom else get_box_corners_3d)(
                new_center, obj_size, obj_rotation
            )

            # Transform to camera frame
            ones = np.ones((corners.shape[0], 1))
            corners_h = np.hstack([corners, ones])  # (8, 4)
            corners_cam = (lidar_to_cam @ corners_h.T).T  # (8, 4)

            # Filter points behind camera
            valid = corners_cam[:, 2] > 0.1
            if valid.sum() < 4:
                return

            uv = np.column_stack([
                intrinsic[0, 0] * corners_cam[valid, 0] / corners_cam[valid, 2] + intrinsic[0, 2],
                intrinsic[1, 1] * corners_cam[valid, 1] / corners_cam[valid, 2] + intrinsic[1, 2],
            ])

            img_h, img_w = img_array.shape[:2]
            u_min = max(0, int(np.floor(uv[:, 0].min())))
            u_max = min(img_w, int(np.ceil(uv[:, 0].max())))
            v_min = max(0, int(np.floor(uv[:, 1].min())))
            v_max = min(img_h, int(np.ceil(uv[:, 1].max())))

            target_w = u_max - u_min
            target_h = v_max - v_min

            if target_w < 4 or target_h < 4:
                return False

            # Reject extreme warps (aspect + scale) to reduce weird-looking vehicles.
            src_h, src_w = crop_img.shape[0], crop_img.shape[1]
            if src_w < 4 or src_h < 4:
                return False

            import math
            src_aspect = float(src_w) / float(src_h)
            tgt_aspect = float(target_w) / float(target_h)
            aspect_log_diff = abs(math.log(max(1e-6, tgt_aspect)) - math.log(max(1e-6, src_aspect)))

            scale_w = float(target_w) / float(src_w)
            scale_h = float(target_h) / float(src_h)
            scale = max(scale_w, scale_h)

            cat = str(obj.get('category', ''))
            is_vehicle = cat.startswith('vehicle')
            max_aspect = self.max_aspect_log_diff_vehicle if is_vehicle else self.max_aspect_log_diff
            max_scale = self.max_scale_factor_vehicle if is_vehicle else self.max_scale_factor

            if aspect_log_diff > max_aspect:
                return False
            if scale > max_scale:
                return False

            # Resize crop to target size
            crop_pil = Image.fromarray(crop_img)
            crop_resized = crop_pil.resize((target_w, target_h), Image.BILINEAR)
            crop_array = np.array(crop_resized)

            # Paste with simple overwrite (depth ordering approximation:
            # objects closer to camera should be pasted last)
            img_array[v_min:v_min+target_h, u_min:u_min+target_w] = crop_array[:target_h, :target_w]

            return True

        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: Build GT Database Script Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def build_gt_database_nuscenes(dataroot: str, output_dir: str = None):
    """
    Convenience function to build the GT database from a NuScenes dataset.

    Usage:
        python lidar_augmentations.py --dataroot /path/to/nuscenes_data
    """
    from src.dataset import MMNuScenesDataset
    from src.lidar_utils import load_lidar_bin

    if output_dir is None:
        output_dir = str(Path(dataroot).parent / "cache" / "gt_database")

    # Create a minimal dataset just to access metadata
    ds = MMNuScenesDataset(
        dataroot=dataroot,
        split='train',
        arch='B',
        lidar_mode='depth',
        V=1,
        local_crops_number=0,
        legacy_mode=True,
        split_strategy='official',
    )

    # Create loader functions
    def lidar_loader(pair):
        pts = None
        if ds.has_data_shards:
            pts = ds._load_lidar_from_shard(pair)
        if pts is None:
            pts = load_lidar_bin(str(pair['lidar']))
        return pts

    def image_loader(pair):
        img = None
        if ds.has_data_shards:
            img = ds._load_image_from_shard(pair)
        if img is None:
            try:
                img = Image.open(pair['camera']).convert('RGB')
            except Exception:
                return None
        return img

    def category_fn(instance_token):
        if hasattr(ds, 'annotation_store') and ds.annotation_store is not None:
            return ds.annotation_store.get_category(instance_token)
        cat_token = ds.instance_to_category.get(instance_token, "")
        return ds.category_names.get(cat_token, "")

    # Get annotations
    if hasattr(ds, 'annotation_store') and ds.annotation_store is not None:
        sample_annotations = {}
        for i in range(len(ds)):
            pair = ds._get_pair(i)
            st = pair['sample_token']
            if st not in sample_annotations:
                sample_annotations[st] = ds.annotation_store.get_annotations(st)
    else:
        sample_annotations = ds.sample_annotations

    builder = GTDatabaseBuilder(dataroot, output_dir,
                                min_points=5, dataset_name='nuscenes')
    builder.build_from_nuscenes(
        sample_pairs=[ds._get_pair(i) for i in range(len(ds))],
        sample_annotations=sample_annotations,
        calibrations=ds.calib_store if hasattr(ds, 'calib_store') else ds.calibrations,
        ego_poses=ds.calib_store if hasattr(ds, 'calib_store') else ds.ego_poses,
        lidar_loader_fn=lidar_loader,
        image_loader_fn=image_loader,
        category_fn=category_fn,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION CONFIG PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

AUGMENTATION_PRESETS = {
    'none': {
        'enable_global_scaling': False,
        'enable_global_rotation': False,
        'enable_global_translation': False,
        'enable_object_translation': False,
        'enable_point_jitter': False,
        'enable_intensity_noise': False,
        'random_flip_x': 0.0,
        'random_flip_y': 0.0,
        'point_dropout_rate': 0.0,
        'frustum_dropout_prob': 0.0,
    },
    'light': {
        'global_scaling_range': (0.97, 1.03),
        'global_rotation_range': (-np.pi / 20, np.pi / 20),  # ±9°
        'global_translation_std': 0.1,
        'object_translation_std': 0.05,
        'point_jitter_std': 0.01,
        'object_point_jitter_std': 0.005,
        'intensity_noise_std': 0.03,
        'random_flip_x': 0.0,
        'random_flip_y': 0.0,
        'point_dropout_rate': 0.0,
        'frustum_dropout_prob': 0.0,
    },
    'moderate': {
        'global_scaling_range': (0.95, 1.05),
        'global_rotation_range': (-np.pi / 12, np.pi / 12),  # ±15°
        'global_translation_std': 0.2,
        'object_translation_std': 0.1,
        'point_jitter_std': 0.02,
        'object_point_jitter_std': 0.02,
        'intensity_noise_std': 0.05,
        'random_flip_x': 0.0,
        'random_flip_y': 0.0,
        'point_dropout_rate': 0.05,
        'frustum_dropout_prob': 0.1,
        'frustum_dropout_angle': 30.0,
    },
    'strong': {
        'global_scaling_range': (0.90, 1.10),
        'global_rotation_range': (-np.pi / 6, np.pi / 6),  # ±30°
        'global_translation_std': 0.5,
        'object_translation_std': 0.25,
        'point_jitter_std': 0.05,
        'object_point_jitter_std': 0.06,
        'intensity_noise_std': 0.10,
        'random_flip_x': 0.0,
        'random_flip_y': 0.0,
        'point_dropout_rate': 0.10,
        'frustum_dropout_prob': 0.2,
        'frustum_dropout_angle': 45.0,
    },
}

COPY_PASTE_PRESETS = {
    'none': {
        'max_paste_objects': 0,
    },
    'light': {
        'max_paste_objects': 3,
        'paste_image': True,
        'min_distance': 5.0,
        'max_distance': 50.0,
        'require_in_view': True,
        'min_paste_in_view': 3,
        'simulate_lidar_occlusion': True,
        'occlusion_bin_radius': 1,
        'occlusion_depth_margin_m': 0.25,
        'enable_scene_aware_placement': True,
        'front_clearance_m': 10.0,
        'image_gate_mode': 'full',
        'enable_free_space_check': False,
        'max_pasted_area_frac': 0.10,
        'soft_big_box_area_frac': 0.10,
        'soft_big_box_keep_prob': 0.10,
        'max_single_box_area_frac': 0.18,
        'max_pasted_iou': 0.08,
        'max_aspect_log_diff_vehicle': np.log(1.4),
        'max_scale_factor_vehicle': 2.2,
    },
    'moderate': {
        'max_paste_objects': 8,
        'paste_image': True,
        'min_distance': 3.0,
        'max_distance': 60.0,
        'require_in_view': True,
        'min_paste_in_view': 5,
        'simulate_lidar_occlusion': True,
        'occlusion_bin_radius': 1,
        'occlusion_depth_margin_m': 0.35,
        'enable_scene_aware_placement': True,
        'front_clearance_m': 9.0,
        'image_gate_mode': 'full',
        'enable_free_space_check': False,
        'max_pasted_area_frac': 0.14,
        'soft_big_box_area_frac': 0.11,
        'soft_big_box_keep_prob': 0.12,
        'max_single_box_area_frac': 0.18,
        'max_pasted_iou': 0.08,
        'max_aspect_log_diff_vehicle': np.log(1.45),
        'max_scale_factor_vehicle': 2.3,
    },
    'strong': {
        'max_paste_objects': 12,
        'paste_image': True,
        'min_distance': 2.0,
        'max_distance': 70.0,
        'require_in_view': True,
        'min_paste_in_view': 8,
        'simulate_lidar_occlusion': True,
        'occlusion_bin_radius': 2,
        'occlusion_depth_margin_m': 0.5,
        'enable_scene_aware_placement': True,
        'front_clearance_m': 8.0,
        'image_gate_mode': 'full',
        'enable_free_space_check': False,
        'max_pasted_area_frac': 0.18,
        'soft_big_box_area_frac': 0.12,
        'soft_big_box_keep_prob': 0.12,
        'max_single_box_area_frac': 0.20,
        'max_pasted_iou': 0.10,
        'max_aspect_log_diff_vehicle': np.log(1.5),
        'max_scale_factor_vehicle': 2.5,
    },
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build GT database for copy-paste augmentation')
    parser.add_argument('--dataroot', type=str, default='/path/to/nuscenes_data',
                        help='Path to NuScenes data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for GT database')
    parser.add_argument('--with_image_crops', action='store_true',
                        help='Also extract RGB crops for later RGB copy-paste (much slower / more IO)')
    parser.add_argument('--with_lidarseg_labels', action='store_true',
                        help='Also store per-point nuScenes lidarseg labels for each object (enables seg_map consistency under copy-paste)')
    parser.add_argument('--progress_every', type=int, default=1000,
                        help='Print progress every N pairs (default: 1000)')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Optional cap on number of pairs (for debugging / quick DB)')
    args = parser.parse_args()

    # Build GT DB with optional image-crops and progress.
    from src.dataset import MMNuScenesDataset
    from src.lidar_utils import load_lidar_bin

    if args.output_dir is None:
        args.output_dir = str(Path(args.dataroot).parent / "cache" / "gt_database")

    ds = MMNuScenesDataset(
        dataroot=args.dataroot,
        split='train',
        arch='B',
        lidar_mode='depth',
        V=1,
        local_crops_number=0,
        legacy_mode=True,
        split_strategy='official',
    )

    def lidar_loader(pair):
        pts = None
        if ds.has_data_shards:
            pts = ds._load_lidar_from_shard(pair)
        if pts is None:
            pts = load_lidar_bin(str(pair['lidar']))
        return pts

    if args.with_image_crops:
        def _image_loader(pair):
            img = None
            if ds.has_data_shards:
                img = ds._load_image_from_shard(pair)
            if img is None:
                try:
                    img = Image.open(pair['camera']).convert('RGB')
                except Exception:
                    return None
            return img
        image_loader_fn = _image_loader
    else:
        image_loader_fn = None

    def category_fn(instance_token):
        if hasattr(ds, 'annotation_store') and ds.annotation_store is not None:
            return ds.annotation_store.get_category(instance_token)
        cat_token = ds.instance_to_category.get(instance_token, "")
        return ds.category_names.get(cat_token, "")

    # Build annotation map
    sample_annotations = {}
    for i in range(len(ds)):
        pair = ds._get_pair(i)
        st = pair['sample_token']
        if st not in sample_annotations:
            sample_annotations[st] = ds.annotation_store.get_annotations(st)

    pairs = [ds._get_pair(i) for i in range(len(ds))]
    if args.max_pairs is not None:
        pairs = pairs[:int(args.max_pairs)]

    builder = GTDatabaseBuilder(args.dataroot, args.output_dir, min_points=5, dataset_name='nuscenes')
    builder.build_from_nuscenes(
        sample_pairs=pairs,
        sample_annotations=sample_annotations,
        calibrations=ds.calib_store,
        ego_poses=ds.calib_store,
        lidar_loader_fn=lidar_loader,
        image_loader_fn=image_loader_fn,
        category_fn=category_fn,
        include_lidarseg_labels=bool(args.with_lidarseg_labels),
        progress_every=int(args.progress_every),
    )
