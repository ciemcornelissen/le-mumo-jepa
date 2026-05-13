"""
waymo_dataset.py

Waymo Open Dataset loader for MM-LeJEPA.

Provides the same interface as MMNuScenesDataset (returns (cam_views, modality2, labels))
so the training loop, encoders, and probes work without modification.

Key differences from NuScenes:
- 5 cameras (FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT) vs 6
- Different calibration format (direct 4x4 extrinsic + [fx,fy,cx,cy,...] intrinsic)
- Different annotation format (center + size + heading instead of quaternion)
- Different class taxonomy (vehicle, pedestrian, cyclist, sign)
- Data stored as extracted images/lidar + JSON metadata (from download_waymo.py)

Usage:
    from src.waymo_dataset import WaymoDataset, waymo_collate_fn

    ds = WaymoDataset('/path/to/waymo_data', split='train',
                      arch='B', lidar_mode='depth')
    loader = DataLoader(ds, batch_size=32, collate_fn=waymo_collate_fn)
"""

import torch
import numpy as np
import json
import io
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple, Optional, List

from src.lidar_utils import (
    load_lidar_bin,
    lidar_to_range_image,
    lidar_to_depth_map_full,
    lidar_to_aligned_points,
    subsample_points,
    normalize_points,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

WAYMO_CAMERAS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']

# Waymo → unified detection classes (same 10 classes as NuScenes probes)
WAYMO_TO_DETECTION_CLASS = {
    'vehicle': 0,       # → car (idx 0): covers cars, trucks, buses
    'pedestrian': 5,    # → pedestrian (idx 5)
    'cyclist': 7,       # → bicycle (idx 7)
    'sign': -1,         # → not in detection set (skip)
}

# Expanded class mapping for finer-grained Waymo types
WAYMO_TYPE_TO_DETECTION = {
    1: 0,   # TYPE_VEHICLE → car
    2: 5,   # TYPE_PEDESTRIAN → pedestrian
    3: -1,  # TYPE_SIGN → skip
    4: 7,   # TYPE_CYCLIST → bicycle
}

# Waymo v2 Camera Panoptic Segmentation → simplified 16 classes (same as NuScenes)
# Used for panoptic_seg_map from camera segmentation masks
WAYMO_PANOPTIC_TO_SIMPLIFIED = {
    0: 0,    # undefined → ignore
    1: 0,    # ego_vehicle → ignore
    2: 6,    # car → car
    3: 10,   # truck → truck
    4: 5,    # bus → bus
    5: 0,    # other_large_vehicle → ignore
    6: 4,    # bicycle → bicycle
    7: 8,    # motorcycle → motorcycle
    8: 9,    # trailer → trailer
    9: 1,    # pedestrian → pedestrian
    10: 4,   # cyclist → bicycle (rider on bike)
    11: 8,   # motorcyclist → motorcycle (rider on motorcycle)
    12: 0,   # bird → ignore
    13: 0,   # ground_animal → ignore
    14: 3,   # construction_cone_pole → traffic_cone
    15: 0,   # pole → ignore
    16: 0,   # pedestrian_object → ignore
    17: 0,   # sign → ignore
    18: 0,   # traffic_light → ignore
    19: 14,  # building → manmade
    20: 11,  # road → driveable
    21: 11,  # lane_marker → driveable
    22: 11,  # road_marker → driveable
    23: 12,  # sidewalk → sidewalk
    24: 15,  # vegetation → vegetation
    25: 0,   # sky → ignore
    26: 13,  # ground → terrain
    27: 0,   # dynamic → ignore
    28: 14,  # static → manmade
}

# Waymo v1 LiDAR Segmentation → simplified 16 classes (same as NuScenes)
# Used for seg_map from LiDAR point cloud semantic labels (6th column)
WAYMO_SEG_TO_SIMPLIFIED = {
    0: 0,    # undefined → ignore
    1: 6,    # car → car
    2: 10,   # truck → truck
    3: 5,    # bus → bus
    4: 0,    # other vehicle → ignore
    5: 1,    # pedestrian → pedestrian
    6: 4,    # cyclist → bicycle
    7: 8,    # motorcycle → motorcycle
    8: 0,    # other ground → ignore
    9: 11,   # road → driveable
    10: 12,  # sidewalk → sidewalk
    11: 14,  # building → manmade
    12: 15,  # vegetation → vegetation
    13: 13,  # terrain → terrain  
    14: 0,   # pole → ignore (no direct mapping)
    15: 3,   # construction cone → traffic_cone
    16: 0,   # other object → ignore
}


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def heading_to_quat(heading: float) -> np.ndarray:
    """Convert yaw heading angle to quaternion [w, x, y, z]."""
    half = heading / 2
    return np.array([np.cos(half), 0, 0, np.sin(half)], dtype=np.float64)


def waymo_intrinsic_to_matrix(intrinsic_params: List[float],
                               width: int, height: int) -> np.ndarray:
    """
    Convert Waymo intrinsic parameters [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    to 3x3 intrinsic matrix.
    """
    fx, fy, cx, cy = intrinsic_params[:4]
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)
    return K


def load_waymo_lidar(path: str, num_cols: int = None, keep_semantic: bool = False) -> np.ndarray:
    """
    Load Waymo LiDAR binary file.
    Format: (N, 5 or 6) float32 [x, y, z, intensity, laser_index, (semantic)]
    Same layout as NuScenes [x, y, z, intensity, ring_index].
    
    Args:
        path: Path to the .bin file
        num_cols: Number of columns (5 or 6). If None, auto-detect.
        keep_semantic: If True and file has 6 columns, keep the semantic label column.
    
    Returns:
        Points array of shape (N, 5) or (N, 6) depending on keep_semantic.
    """
    raw = np.fromfile(path, dtype=np.float32)

    # Metadata may say 5 columns even when the file actually contains
    # panoptic labels in a 6th column. When semantic labels are requested,
    # re-run detection in that case instead of trusting stale metadata.
    if keep_semantic and num_cols == 5 and (len(raw) % 6 == 0):
        num_cols = None
    
    # Auto-detect column count if not specified
    # Prefer 6 columns (has semantic) when both 5 and 6 divide evenly
    if num_cols is None:
        div_by_5 = (len(raw) % 5 == 0)
        div_by_6 = (len(raw) % 6 == 0)
        if div_by_6 and div_by_5:
            # Both divide - check if 6 columns makes sense by looking at column values
            # 6th column stores panoptic labels: semantic_class + instance_id * 23
            # Values can reach hundreds (e.g. class 22 + instance 20 * 23 = 482)
            pts_6 = raw.reshape(-1, 6)
            col6_vals = pts_6[:, 5]
            # Labels are non-negative integers; coordinates have fractional parts
            if np.all((col6_vals >= -1) & (col6_vals == col6_vals.astype(int))):
                num_cols = 6
            else:
                num_cols = 5
        elif div_by_6:
            num_cols = 6
        elif div_by_5:
            num_cols = 5
        else:
            # Fallback: try 6 first, then 5
            num_cols = 6 if div_by_6 else 5
    
    pts = raw.reshape(-1, num_cols)
    
    # Drop semantic column unless explicitly requested
    if num_cols == 6 and not keep_semantic:
        pts = pts[:, :5]
    
    return pts


# ═══════════════════════════════════════════════════════════════════════════════
# WAYMO DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class WaymoDataset(Dataset):
    """
    Waymo Open Dataset loader with the same __getitem__ interface as
    MMNuScenesDataset.

    Returns: (cam_views, modality2, labels)
        cam_views: dict {'global': (V, 3, H, W), 'local': (V_l, 3, h, w)}
        modality2: dict for depth/range or tensor for points
        labels: dict with detection/segmentation/spatial labels

    Supports the same arch/lidar_mode combinations as the NuScenes dataset.
    """

    def __init__(
        self,
        dataroot: str,
        split: str = "train",
        arch: str = "B",
        lidar_mode: str = "auto",
        V: int = 2,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 4,
        img_size: int = 224,
        local_img_size: int = 96,
        probe_img_size: Optional[int] = None,
        occupancy_grid_size: int = 28,
        range_size: Tuple[int, int] = (64, 256),
        n_points: int = 16384,
        modality_dropout: float = 0.0,
        legacy_mode: bool = False,
        finetune_mode: bool = False,
        include_probe_view: Optional[bool] = None,
        encoder_only_labels: bool = False,
        finetune_crop_scale: Tuple[float, float] = (0.8, 1.0),
        det_seg_label_mode: str = "both",
        lidar_aug_preset: str = "none",
        lidar_aug_cfg: Optional[Dict] = None,
        copy_paste_preset: str = "none",
        copy_paste_cfg: Optional[Dict] = None,
        gt_database_path: str = None,
        target_cameras: Optional[List[str]] = None,
        precomputed_labels_path: Optional[str] = None,
        dino_aug_mode: str = "default",
        seg_filter_mode: str = "none",  # 'none', 'camera_seg', 'lidar_seg', 'any_seg'
        return_multimae_view_labels: bool = False,
    ):
        """
        Args:
            dataroot: Path to Waymo extracted data (output of download_waymo.py)
            split: 'train' or 'val'
            target_cameras: Which cameras to use. Default: all 5.
                           Use ['FRONT'] for single-camera experiments.
            seg_filter_mode: Filter samples by segmentation label availability.
                'none': No filtering (default)
                'camera_seg': Only samples with camera_seg PNG files
                'lidar_seg': Only samples with LiDAR semantic labels (6-col)
                'any_seg': Either camera_seg or lidar_seg available
        """
        self.dataroot = Path(dataroot)
        self.seg_filter_mode = seg_filter_mode
        self.split = split
        self.arch = arch
        self.V = V
        self.local_crops_number = local_crops_number
        self.total_views = V + local_crops_number
        self.img_size = img_size
        self.local_img_size = local_img_size
        self.probe_img_size = int(probe_img_size or img_size)
        self.occupancy_grid_size = max(1, int(occupancy_grid_size))
        self.range_size = range_size
        self.n_points = n_points
        self.modality_dropout = modality_dropout
        self.legacy_mode = legacy_mode
        self.finetune_mode = finetune_mode
        self.include_probe_view = (not self.finetune_mode) if include_probe_view is None else bool(include_probe_view)
        self.encoder_only_labels = bool(encoder_only_labels)
        self.dino_aug_mode = str(dino_aug_mode).lower()
        self.use_official_dino_augs = self.split == "train" and self.dino_aug_mode == "official"
        self.finetune_crop_scale = finetune_crop_scale
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.det_seg_label_mode = det_seg_label_mode
        self.return_multimae_view_labels = bool(return_multimae_view_labels)
        self.target_cameras = target_cameras or WAYMO_CAMERAS
        self._camera_names = list(self.target_cameras)

        # Determine lidar mode
        if lidar_mode == "auto":
            if arch == "D":
                self.lidar_mode = "depth"
            elif arch == "A":
                self.lidar_mode = "points"
            else:
                self.lidar_mode = "range"
        else:
            self.lidar_mode = lidar_mode

        # Precomputed labels
        self.precomputed_labels_path = None
        self.precomputed_labels_has_shards = False
        if precomputed_labels_path and os.path.exists(precomputed_labels_path):
            self.precomputed_labels_path = Path(precomputed_labels_path)
            self.precomputed_labels_has_shards = any(self.precomputed_labels_path.glob("shard_*.zip"))
            print(f"⚡ Using precomputed labels from: {self.precomputed_labels_path}")
            print(f"   └─ format: {'sharded zip' if self.precomputed_labels_has_shards else 'individual .npz files'}")
        elif precomputed_labels_path:
            print(f"⚠️ precomputed_labels_path not found, disabling precomputed labels: {precomputed_labels_path}")
        
        # Depth map caching (similar to NuScenes - prevents recomputing expensive projections)
        self.cache_res = 448  # Cached depth resolution (same as NuScenes)
        self.cache_dir = Path(__file__).parent / "cache" / f"waymo_depth_maps_v1_{self.cache_res}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Waymo depth cache: {self.cache_dir}")
        
        # Load metadata
        self._load_metadata()

        # Expose NuScenes-like dataset metadata fields expected by the training loop
        # (probe heads use these to size classification outputs)
        self.num_scenes = len(getattr(self, '_segments_sorted', [])) or 1
        self.num_cameras = len(self._camera_names) or len(WAYMO_CAMERAS)
        self.num_locations = 1

        # Build sample pairs (frame × camera)
        self._build_sample_pairs()

        # Split
        self._apply_split()

        # Optional: Filter to only samples with segmentation labels
        if self.seg_filter_mode != 'none':
            self._filter_seg_samples()

        # Setup transforms (same as NuScenes)
        self._setup_transforms()

        # LiDAR augmentation setup
        self._setup_augmentations(
            lidar_aug_preset, lidar_aug_cfg,
            copy_paste_preset, copy_paste_cfg, gt_database_path
        )

        print(f"WaymoDataset [arch={arch}, lidar_mode={self.lidar_mode}]: "
              f"{len(self.pairs)} samples, "
              f"Global={V}, Local={local_crops_number}, "
              f"Cameras={self.target_cameras}")

    def _load_metadata(self):
        """Load metadata JSON produced by download_waymo.py."""
        # Determine which metadata file to use
        split_name = 'training' if self.split == 'train' else 'validation'
        meta_path = self.dataroot / f'metadata_{split_name}.json'

        if not meta_path.exists():
            # Try alternate names
            for name in [f'metadata_{self.split}.json', 'metadata.json']:
                alt = self.dataroot / name
                if alt.exists():
                    meta_path = alt
                    break

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Waymo metadata not found at {meta_path}. "
                f"Run download_waymo.py first to extract the dataset.")

        print(f"Loading Waymo metadata from {meta_path}...")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        self.frames = metadata['frames']
        self.target_hz = metadata.get('target_hz', 2.0)
        self.num_segments = metadata.get('num_segments', 0)
        print(f"  {len(self.frames)} frames from {self.num_segments} segments "
              f"at {self.target_hz}Hz")

        # Build segment → frame indices for scene-based splitting
        self.segment_to_frames = {}
        for i, frame in enumerate(self.frames):
            seg = frame['segment_name']
            if seg not in self.segment_to_frames:
                self.segment_to_frames[seg] = []
            self.segment_to_frames[seg].append(i)

        # Cache stable segment ordering for consistent scene ids
        self._segments_sorted = sorted(self.segment_to_frames.keys())

    def _build_sample_pairs(self):
        """Build one sample per (frame, camera) combination."""
        self.pairs = []
        for frame_idx, frame in enumerate(self.frames):
            for cam_name in self.target_cameras:
                if cam_name not in frame.get('cameras', {}):
                    continue
                cam_info = frame['cameras'][cam_name]
                self.pairs.append({
                    'frame_idx': frame_idx,
                    'camera_name': cam_name,
                    'camera_path': str(self.dataroot / cam_info['path']),
                    'lidar_path': str(self.dataroot / frame['lidar_path']) if frame.get('lidar_path') else None,
                    'lidar_cols': frame.get('lidar_cols'),  # Number of columns in LiDAR binary (5 or 6) when known
                    'segment_name': frame['segment_name'],
                    'timestamp': frame['timestamp'],
                    'intrinsic': waymo_intrinsic_to_matrix(
                        cam_info['intrinsic'],
                        cam_info.get('width', 1920),
                        cam_info.get('height', 1280)
                    ),
                    'cam_extrinsic': np.array(cam_info['extrinsic']).reshape(4, 4) if len(cam_info.get('extrinsic', [])) == 16 else np.eye(4),
                    'ego_pose': np.array(frame['ego_pose']).reshape(4, 4) if np.array(frame.get('ego_pose', [])).size == 16 else np.eye(4),
                    'lidar_extrinsic': np.array(frame.get('lidar_extrinsic', np.eye(4).tolist())).reshape(4, 4) if np.array(frame.get('lidar_extrinsic', [])).size == 16 else np.eye(4),
                    'annotations': frame.get('annotations', []),
                    'img_width': cam_info.get('width', 1920),
                    'img_height': cam_info.get('height', 1280),
                    # Compatibility keys expected by shared code
                    'sample_token': f"{frame['segment_name']}_{frame['timestamp']}",
                })

    def _apply_split(self):
        """Scene-based train/val split (80/20 by segment)."""
        segments = getattr(self, '_segments_sorted', None) or sorted(self.segment_to_frames.keys())
        n_train = int(len(segments) * 0.8)

        if self.split == 'train':
            valid_segments = set(segments[:n_train])
        else:
            valid_segments = set(segments[n_train:])

        self.pairs = [p for p in self.pairs
                      if p['segment_name'] in valid_segments]

    def _filter_seg_samples(self):
        """Filter samples to only include those with segmentation labels available.
        
        Based on seg_filter_mode:
        - 'camera_seg': Only samples with camera_seg PNG files
        - 'lidar_seg': Only samples with LiDAR semantic labels (6-col)
        - 'any_seg': Either camera_seg or lidar_seg available
        """
        before_count = len(self.pairs)
        
        # Build camera_seg availability index
        camera_seg_available = set()
        camera_seg_dir = self.dataroot / "camera_seg"
        if camera_seg_dir.exists():
            for seg_dir in camera_seg_dir.iterdir():
                if seg_dir.is_dir():
                    seg_name = seg_dir.name
                    for f in seg_dir.iterdir():
                        if f.suffix == '.png':
                            # Extract timestamp from filename like "1552694250424216_FRONT.png"
                            parts = f.stem.split('_')
                            if len(parts) >= 2:
                                ts = parts[0]
                                cam = '_'.join(parts[1:])  # Handle camera names with underscores
                                camera_seg_available.add((seg_name, ts, cam))
        
        # Build frame_idx to has_lidar_seg mapping
        lidar_seg_frames = set()
        for frame_idx, frame in enumerate(self.frames):
            if frame.get('has_lidar_seg', False) or frame.get('lidar_cols', 0) >= 6:
                lidar_seg_frames.add(frame_idx)
        
        # Filter pairs
        filtered_pairs = []
        for p in self.pairs:
            has_camera_seg = (p['segment_name'], str(p['timestamp']), p['camera_name']) in camera_seg_available
            has_lidar_seg = p['frame_idx'] in lidar_seg_frames
            
            if self.seg_filter_mode == 'camera_seg' and has_camera_seg:
                filtered_pairs.append(p)
            elif self.seg_filter_mode == 'lidar_seg' and has_lidar_seg:
                filtered_pairs.append(p)
            elif self.seg_filter_mode == 'any_seg' and (has_camera_seg or has_lidar_seg):
                filtered_pairs.append(p)
        
        self.pairs = filtered_pairs
        after_count = len(self.pairs)
        
        print(f"🔬 Seg filter ({self.seg_filter_mode}): {before_count} → {after_count} samples "
              f"({100*after_count/before_count:.1f}% retained)")

    def _setup_transforms(self):
        """Setup image transforms matching NuScenes dataset."""
        from torchvision.transforms import v2
        from torchvision.transforms import InterpolationMode

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        to_tensor_norm = (
            v2.ToTensor()
            if not hasattr(v2, 'ToImage')
            else v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        )

        normalize = v2.Compose([
            to_tensor_norm,
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        color_jitter = v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
        ])
        self._dino_global_post_aug_1 = v2.Compose([
            color_jitter,
            v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            normalize,
        ])
        self._dino_global_post_aug_2 = v2.Compose([
            color_jitter,
            v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.1),
            v2.RandomSolarize(threshold=128, p=0.2),
            normalize,
        ])
        self._dino_local_post_aug = v2.Compose([
            color_jitter,
            v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            normalize,
        ])

        # Post-crop pipelines used for depth-synchronized crops (crop happens outside).
        self._cam_post_aug = v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
            to_tensor_norm,
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self._cam_post_test = v2.Compose([
            to_tensor_norm,
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # Global crop augmentation
        self.cam_global_aug = _AugWithParams(
            v2.Compose([
                v2.RandomResizedCrop(self.img_size,
                                     scale=self.global_crops_scale,
                                     interpolation=InterpolationMode.BICUBIC),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
                v2.RandomHorizontalFlip(),
                to_tensor_norm,
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
            self.img_size,
        )

        # Local crop augmentation
        self.cam_local_aug = _AugWithParams(
            v2.Compose([
                v2.RandomResizedCrop(self.local_img_size,
                                     scale=self.local_crops_scale,
                                     interpolation=InterpolationMode.BICUBIC),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
                v2.RandomHorizontalFlip(),
                to_tensor_norm,
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
            self.local_img_size,
        )

        # Test/probe transform (deterministic)
        self.cam_test = _AugWithParams(
            v2.Compose([
                v2.Resize(self.img_size, interpolation=InterpolationMode.BICUBIC),
                v2.CenterCrop(self.img_size),
                to_tensor_norm,
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
            self.img_size,
        )
        self.cam_probe_test = _AugWithParams(
            v2.Compose([
                v2.Resize(self.probe_img_size, interpolation=InterpolationMode.BICUBIC),
                v2.CenterCrop(self.probe_img_size),
                to_tensor_norm,
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
            self.probe_img_size,
        )

        # Range image transforms
        self.range_aug = v2.Compose([
            v2.RandomResizedCrop((self.img_size, self.img_size),
                                 scale=(0.5, 1.0)),
            v2.RandomHorizontalFlip(),
        ])
        self.range_test = v2.Compose([
            v2.Resize((self.img_size, self.img_size)),
        ])
        self.range_probe_test = v2.Compose([
            v2.Resize((self.probe_img_size, self.probe_img_size)),
        ])
        self.range_local_aug = v2.Compose([
            v2.RandomResizedCrop((self.local_img_size, self.local_img_size),
                                 scale=(0.5, 1.0)),
            v2.RandomHorizontalFlip(),
        ])

        # RGB augmentation without flip (for synchronized depth)
        self.rgb_aug_no_flip = v2.Compose([
            v2.ColorJitter(0.8, 0.8, 0.8, 0.2),
            v2.RandomGrayscale(p=0.2),
        ])

    def _select_rgb_post_aug(self, *, is_global: bool, view_index: int = 0):
        if not self.use_official_dino_augs:
            return self._cam_post_aug
        if is_global:
            return self._dino_global_post_aug_1 if (view_index % 2 == 0) else self._dino_global_post_aug_2
        return self._dino_local_post_aug

    def _apply_rgb_post_aug(self, img, *, is_global: bool, view_index: int = 0):
        return self._select_rgb_post_aug(is_global=is_global, view_index=view_index)(img)

    def _setup_augmentations(self, lidar_aug_preset, lidar_aug_cfg,
                              copy_paste_preset, copy_paste_cfg,
                              gt_database_path):
        """Setup LiDAR augmentations (same as NuScenes dataset)."""
        self.lidar_aug_enabled = False
        self.copy_paste_enabled = False
        self._scene_augmentor = None
        self._copy_paste_augmentor = None

        if self.split == "train":
            from src.lidar_augmentations import (
                LiDARSceneAugmentor, CopyPasteAugmentor,
                AUGMENTATION_PRESETS, COPY_PASTE_PRESETS,
            )
            if lidar_aug_preset != "none" or lidar_aug_cfg:
                aug_cfg = AUGMENTATION_PRESETS.get(lidar_aug_preset, {})
                if lidar_aug_cfg:
                    aug_cfg.update(lidar_aug_cfg)
                self._scene_augmentor = LiDARSceneAugmentor(aug_cfg)
                self.lidar_aug_enabled = True
                print(f"🎲 LiDAR scene augmentation: preset={lidar_aug_preset}")

            if copy_paste_preset != "none" or copy_paste_cfg:
                cp_cfg = COPY_PASTE_PRESETS.get(copy_paste_preset, {})
                if copy_paste_cfg:
                    cp_cfg.update(copy_paste_cfg)
                db_path = gt_database_path or str(
                    self.dataroot.parent / "cache" / "gt_database" / "gt_database_waymo.pkl")
                self._copy_paste_augmentor = CopyPasteAugmentor(db_path, cp_cfg)
                self.copy_paste_enabled = bool(self._copy_paste_augmentor.sampling_pool)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        labels = self._get_global_labels(pair)

        # Load camera image
        try:
            img = Image.open(pair['camera_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (pair['img_width'], pair['img_height']), 'black')

        orig_w, orig_h = img.size

        # Load LiDAR points (keep semantic column if present for seg_map computation)
        lidar_points = None
        if pair['lidar_path'] and os.path.exists(pair['lidar_path']):
            try:
                lidar_points = load_waymo_lidar(
                    pair['lidar_path'], 
                    num_cols=pair.get('lidar_cols'),
                    keep_semantic=True  # Keep 6th column for seg_map if present
                )
            except Exception:
                lidar_points = None

        # ── LiDAR Augmentation ──────────────────────────────────────
        annotations_for_labels = pair['annotations']
        _aug_active = False

        if lidar_points is not None and self.split == "train":
            # Convert Waymo annotations to augmentation format
            aug_anns = self._annotations_to_lidar_frame(pair)

            if self.lidar_aug_enabled and self._scene_augmentor is not None:
                import copy
                aug_anns_copy = copy.deepcopy(aug_anns)
                _alignment_safe = self.lidar_mode in ("depth", "aligned_points")
                lidar_points, aug_anns_copy = self._scene_augmentor(
                    lidar_points, aug_anns_copy, alignment_safe=_alignment_safe)
                aug_anns = aug_anns_copy
                _aug_active = True

            if self.copy_paste_enabled and self._copy_paste_augmentor is not None:
                import copy
                aug_anns_copy = copy.deepcopy(aug_anns) if not _aug_active else aug_anns
                lidar_points, aug_anns_copy, img = self._copy_paste_augmentor(
                    lidar_points, aug_anns_copy, img,
                    pair['intrinsic'],
                    self._compute_lidar_to_cam(pair),
                    current_camera_name=pair.get('camera_name'),
                )
                aug_anns = aug_anns_copy
                _aug_active = True

            if _aug_active:
                annotations_for_labels = self._aug_anns_to_waymo_format(aug_anns)

        # ── Camera Views ────────────────────────────────────────────
        if self.lidar_mode != "depth":
            cam_views_global = []
            cam_views_local = []
            cam_probe = torch.empty(0)

            for global_idx in range(self.V):
                if self.use_official_dino_augs:
                    i, j, h, w = v2.RandomResizedCrop.get_params(img, scale=self.global_crops_scale, ratio=(3.0 / 4.0, 4.0 / 3.0))
                    view = TF.resized_crop(img, i, j, h, w, (self.img_size, self.img_size), interpolation=InterpolationMode.BICUBIC)
                    if np.random.random() < 0.5:
                        view = TF.hflip(view)
                    view = self._apply_rgb_post_aug(view, is_global=True, view_index=global_idx)
                else:
                    view, _ = self.cam_global_aug(img, return_params=True)
                cam_views_global.append(view)

            if self.include_probe_view:
                view_clean, _ = self.cam_probe_test(img, return_params=True)
                cam_probe = view_clean.unsqueeze(0)

            for local_idx in range(self.local_crops_number):
                if self.use_official_dino_augs:
                    i, j, h, w = v2.RandomResizedCrop.get_params(img, scale=self.local_crops_scale, ratio=(3.0 / 4.0, 4.0 / 3.0))
                    view = TF.resized_crop(img, i, j, h, w, (self.local_img_size, self.local_img_size), interpolation=InterpolationMode.BICUBIC)
                    if np.random.random() < 0.5:
                        view = TF.hflip(view)
                    view = self._apply_rgb_post_aug(view, is_global=False, view_index=local_idx)
                else:
                    view, _ = self.cam_local_aug(img, return_params=True)
                cam_views_local.append(view)

            cam_views_global = torch.stack(cam_views_global) if cam_views_global else torch.empty(0)
            cam_views_local = torch.stack(cam_views_local) if cam_views_local else torch.empty(0)

            cam_views = {
                'global': cam_views_global,
                'local': cam_views_local,
                'probe': cam_probe,
            }
        else:
            # Depth mode builds synchronized RGB+depth views below.
            cam_views = {
                'global': torch.empty(0),
                'local': torch.empty(0),
                'probe': torch.empty(0),
            }

        # ── LiDAR Modality ─────────────────────────────────────────
        if self.lidar_mode == "range" and lidar_points is not None:
            range_img = lidar_to_range_image(
                lidar_points, H=self.range_size[0], W=self.range_size[1])
            range_tensor = torch.from_numpy(range_img).permute(2, 0, 1)

            range_views_global = []
            for _ in range(self.V):
                range_views_global.append(self.range_aug(range_tensor))
            range_probe = torch.empty(0)
            if self.include_probe_view:
                range_probe = self.range_probe_test(range_tensor).unsqueeze(0)

            range_views_local = []
            for _ in range(self.local_crops_number):
                range_views_local.append(self.range_local_aug(range_tensor))

            modality2 = {
                'global': torch.stack(range_views_global),
                'local': torch.stack(range_views_local) if range_views_local else torch.empty(0),
                'probe': range_probe,
            }

        elif self.lidar_mode == "depth" and lidar_points is not None:
            # ── Depth map caching (huge performance boost) ──────────────
            # Cache key: segment_timestamp_camera
            cache_key = f"{pair['segment_name']}_{pair['timestamp']}_{pair['camera_name']}"
            cache_path = self.cache_dir / f"{cache_key}.png"
            
            # When LiDAR augmentation is active, bypass cache (aug modifies point cloud)
            _aug_active = (
                self.split == "train" and (
                    getattr(self, "lidar_aug_enabled", False) or 
                    getattr(self, "copy_paste_enabled", False)
                )
            )
            
            depth_map = None
            need_computation = True
            
            # Try loading from cache (only when no augmentation)
            if not _aug_active and cache_path.exists():
                try:
                    depth_int = np.array(Image.open(cache_path))
                    depth_map = depth_int.astype(np.float32) / 256.0  # 16-bit to float
                    need_computation = False
                except Exception:
                    need_computation = True
            
            # Compute depth if not cached
            if need_computation:
                lidar_to_cam = self._compute_lidar_to_cam(pair)
                depth_map = lidar_to_depth_map_full(
                    lidar_points,
                    intrinsic=pair['intrinsic'],
                    lidar_to_cam_transform=lidar_to_cam.astype(np.float32),
                    img_size=(orig_h, orig_w),
                    max_depth=80.0,
                )
                
                # Save to cache (only when not augmenting and split is train for warm-up)
                if not _aug_active:
                    try:
                        depth_int = (depth_map * 256.0).astype(np.uint16)
                        Image.fromarray(depth_int).save(cache_path)
                    except Exception:
                        pass  # Cache write failed, continue anyway

            # Generate synchronized RGB+depth views (shared crop + flip params)
            depth_tensor_full = torch.from_numpy(depth_map).unsqueeze(0)  # (1, H, W)
            cam_views_global = []
            cam_views_local = []
            depth_views_global = []
            depth_views_local = []
            cam_probe = torch.empty(0)
            depth_probe = torch.empty(0)
            multimae_global_crop_rects = []
            multimae_probe_crop_rect = None

            ratio = (3.0 / 4.0, 4.0 / 3.0)
            for global_idx in range(self.V):
                i, j, h, w = v2.RandomResizedCrop.get_params(img, scale=self.global_crops_scale, ratio=ratio)
                img_crop = TF.resized_crop(
                    img, i, j, h, w,
                    (self.img_size, self.img_size),
                    interpolation=InterpolationMode.BICUBIC,
                )
                depth_crop = TF.resized_crop(
                    depth_tensor_full, i, j, h, w,
                    (self.img_size, self.img_size),
                    interpolation=InterpolationMode.BILINEAR,
                )
                if np.random.random() < 0.5:
                    img_crop = TF.hflip(img_crop)
                    depth_crop = TF.hflip(depth_crop)
                cam_views_global.append(self._apply_rgb_post_aug(img_crop, is_global=True, view_index=global_idx))
                depth_views_global.append(depth_crop)
                multimae_global_crop_rects.append((i, j, h, w))

            if self.include_probe_view:
                # Deterministic clean probe view at probe_img_size.
                scale = self.probe_img_size / min(orig_w, orig_h)
                new_h = int(round(orig_h * scale))
                new_w = int(round(orig_w * scale))
                img_clean = TF.center_crop(
                    TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BICUBIC),
                    (self.probe_img_size, self.probe_img_size),
                )
                depth_clean = TF.center_crop(
                    TF.resize(depth_tensor_full, (new_h, new_w), interpolation=InterpolationMode.BILINEAR),
                    (self.probe_img_size, self.probe_img_size),
                )
                cam_probe = self._cam_post_test(img_clean).unsqueeze(0)
                depth_probe = depth_clean.unsqueeze(0)
                top_new = (new_h - self.probe_img_size) // 2
                left_new = (new_w - self.probe_img_size) // 2
                crop_i = int(top_new / scale)
                crop_j = int(left_new / scale)
                crop_h = min(int(self.probe_img_size / scale), orig_h - crop_i)
                crop_w = min(int(self.probe_img_size / scale), orig_w - crop_j)
                multimae_probe_crop_rect = (crop_i, crop_j, crop_h, crop_w)

            for local_idx in range(self.local_crops_number):
                i, j, h, w = v2.RandomResizedCrop.get_params(img, scale=self.local_crops_scale, ratio=ratio)
                img_crop = TF.resized_crop(
                    img, i, j, h, w,
                    (self.local_img_size, self.local_img_size),
                    interpolation=InterpolationMode.BICUBIC,
                )
                depth_crop = TF.resized_crop(
                    depth_tensor_full, i, j, h, w,
                    (self.local_img_size, self.local_img_size),
                    interpolation=InterpolationMode.BILINEAR,
                )
                if np.random.random() < 0.5:
                    img_crop = TF.hflip(img_crop)
                    depth_crop = TF.hflip(depth_crop)
                cam_views_local.append(self._apply_rgb_post_aug(img_crop, is_global=False, view_index=local_idx))
                depth_views_local.append(depth_crop)

            cam_views = {
                'global': torch.stack(cam_views_global) if cam_views_global else torch.empty(0),
                'local': torch.stack(cam_views_local) if cam_views_local else torch.empty(0),
                'probe': cam_probe,
            }
            modality2 = {
                'global': torch.stack(depth_views_global) if depth_views_global else torch.empty(0),
                'local': torch.stack(depth_views_local) if depth_views_local else torch.empty(0),
                'probe': depth_probe,
            }

        elif self.lidar_mode == "points" and lidar_points is not None:
            pts = subsample_points(lidar_points, self.n_points)
            pts = normalize_points(pts)
            modality2 = torch.from_numpy(pts).float()

        elif self.lidar_mode == "aligned_points" and lidar_points is not None:
            lidar_to_cam = self._compute_lidar_to_cam(pair)
            aligned, uv = lidar_to_aligned_points(
                lidar_points,
                intrinsic=pair['intrinsic'],
                lidar_to_cam_transform=lidar_to_cam.astype(np.float32),
            )
            pts = subsample_points(aligned, self.n_points)
            pts = normalize_points(pts)
            modality2 = torch.from_numpy(pts).float()

        else:
            # Fallback: zeros
            if self.lidar_mode in ("range", "depth"):
                modality2 = {
                    'global': torch.zeros(self.V,
                                         1, self.img_size, self.img_size),
                    'local': torch.zeros(self.local_crops_number,
                                        1, self.local_img_size, self.local_img_size),
                    'probe': torch.zeros(0) if (self.finetune_mode or not self.include_probe_view) else torch.zeros(1, 1, self.probe_img_size, self.probe_img_size),
                }
                # Also generate fallback camera views for depth mode with missing lidar
                if self.lidar_mode == "depth" and cam_views['global'].numel() == 0:
                    cam_views_global = []
                    cam_views_local = []
                    cam_probe = torch.empty(0)
                    for global_idx in range(self.V):
                        if self.use_official_dino_augs:
                            i, j, h, w = v2.RandomResizedCrop.get_params(img, scale=self.global_crops_scale, ratio=(3.0 / 4.0, 4.0 / 3.0))
                            view = TF.resized_crop(img, i, j, h, w, (self.img_size, self.img_size), interpolation=InterpolationMode.BICUBIC)
                            if np.random.random() < 0.5:
                                view = TF.hflip(view)
                            view = self._apply_rgb_post_aug(view, is_global=True, view_index=global_idx)
                        else:
                            view, _ = self.cam_global_aug(img, return_params=True)
                        cam_views_global.append(view)
                    if self.include_probe_view:
                        view_clean, _ = self.cam_probe_test(img, return_params=True)
                        cam_probe = view_clean.unsqueeze(0)
                    for local_idx in range(self.local_crops_number):
                        if self.use_official_dino_augs:
                            i, j, h, w = v2.RandomResizedCrop.get_params(img, scale=self.local_crops_scale, ratio=(3.0 / 4.0, 4.0 / 3.0))
                            view = TF.resized_crop(img, i, j, h, w, (self.local_img_size, self.local_img_size), interpolation=InterpolationMode.BICUBIC)
                            if np.random.random() < 0.5:
                                view = TF.hflip(view)
                            view = self._apply_rgb_post_aug(view, is_global=False, view_index=local_idx)
                        else:
                            view, _ = self.cam_local_aug(img, return_params=True)
                        cam_views_local.append(view)
                    cam_views = {
                        'global': torch.stack(cam_views_global) if cam_views_global else torch.empty(0),
                        'local': torch.stack(cam_views_local) if cam_views_local else torch.empty(0),
                        'probe': cam_probe,
                    }
            else:
                modality2 = torch.zeros(self.n_points, 5)

        # ── Labels ──────────────────────────────────────────────────
        if not self.encoder_only_labels:
            labels.update(self._get_spatial_labels(pair, annotations_for_labels, lidar_points))

        if not self.legacy_mode and not self.encoder_only_labels:
            labels.update(self._get_det_labels(
                pair, annotations_for_labels, lidar_points))
            if self.return_multimae_view_labels and self.lidar_mode == "depth":
                if multimae_global_crop_rects:
                    global_seg = []
                    global_has_seg = []
                    global_panoptic = []
                    global_has_panoptic = []
                    for crop_rect in multimae_global_crop_rects:
                        seg_info = self._get_multimae_view_seg_labels(
                            pair,
                            lidar_points=lidar_points,
                            crop_rect=crop_rect,
                            target_size=self.img_size,
                        )
                        global_seg.append(seg_info['seg_map'])
                        global_has_seg.append(seg_info['has_seg_map'])
                        global_panoptic.append(seg_info['panoptic_seg_map'])
                        global_has_panoptic.append(seg_info['has_panoptic_seg_map'])
                    labels['multimae_global_seg_map'] = np.stack(global_seg).astype(np.int64)
                    labels['multimae_global_has_seg_map'] = np.asarray(global_has_seg, dtype=np.bool_)
                    labels['multimae_global_panoptic_seg_map'] = np.stack(global_panoptic).astype(np.int64)
                    labels['multimae_global_has_panoptic_seg_map'] = np.asarray(global_has_panoptic, dtype=np.bool_)
                if multimae_probe_crop_rect is not None:
                    seg_info = self._get_multimae_view_seg_labels(
                        pair,
                        lidar_points=lidar_points,
                        crop_rect=multimae_probe_crop_rect,
                        target_size=self.probe_img_size,
                    )
                    labels['multimae_probe_seg_map'] = seg_info['seg_map'].astype(np.int64)
                    labels['multimae_probe_has_seg_map'] = np.bool_(seg_info['has_seg_map'])
                    labels['multimae_probe_panoptic_seg_map'] = seg_info['panoptic_seg_map'].astype(np.int64)
                    labels['multimae_probe_has_panoptic_seg_map'] = np.bool_(seg_info['has_panoptic_seg_map'])

        return cam_views, modality2, labels

    # ═══════════════════════════════════════════════════════════════
    # LABEL COMPUTATION
    # ═══════════════════════════════════════════════════════════════

    def _get_global_labels(self, pair: Dict) -> Dict:
        """Global scene-level labels."""
        # Map segment to scene_id
        segments = getattr(self, '_segments_sorted', None) or sorted(self.segment_to_frames.keys())
        scene_id = segments.index(pair['segment_name']) if pair['segment_name'] in segments else 0

        cam_names = self._camera_names or WAYMO_CAMERAS
        cam_id = cam_names.index(pair['camera_name']) if pair['camera_name'] in cam_names else 0

        return {
            'scene': scene_id,
            'camera': cam_id,
            'location': 0,  # Waymo doesn't have location labels like NuScenes
        }

    def _get_spatial_labels(self, pair: Dict, annotations: List[Dict], 
                            lidar_points: Optional[np.ndarray] = None) -> Dict:
        """Spatial labels (object counts, grid occupancy, depth grids)."""
        
        # ── Label caching (huge performance boost) ──────────────────
        cache_key = f"{pair['segment_name']}_{pair['timestamp']}_{pair['camera_name']}"
        labels_cache_path = self.cache_dir / f"{cache_key}_labels.npz"
        
        # Only check cache when no LiDAR augmentation is active
        _aug_active = (
            self.split == "train" and (
                getattr(self, "lidar_aug_enabled", False) or 
                getattr(self, "copy_paste_enabled", False)
            )
        )
        
        # Try loading cached labels (depth grids only - object counts recomputed)
        cached_depth_grids = None
        if not _aug_active and labels_cache_path.exists():
            try:
                cached = np.load(labels_cache_path)
                cached_depth_grids = {
                    'mean_depth': float(cached['mean_depth']),
                    'depth_grid': cached['depth_grid'],
                    'depth_grid_mask': cached['depth_grid_mask'],
                    'depth_grid_hr': cached['depth_grid_hr'],
                    'depth_grid_mask_hr': cached['depth_grid_mask_hr'],
                }
            except Exception:
                cached_depth_grids = None
        
        num_cars = 0
        num_peds = 0
        num_objects = 0
        
        # Build box list for grid occupancy computation
        box_2d_list = []
        
        # Get camera parameters for 3D→2D projection
        intrinsic = np.asarray(pair['intrinsic'], dtype=np.float64)
        vehicle_to_cam_raw = np.linalg.inv(pair['cam_extrinsic']).astype(np.float64)
        
        # Apply axis swap to convert Waymo vehicle frame (X=forward, Y=left, Z=up)
        # to standard camera frame (X=right, Y=down, Z=forward)
        axis_swap = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        vehicle_to_cam = axis_swap @ vehicle_to_cam_raw
        
        img_w = pair.get('img_width', 1920)
        img_h = pair.get('img_height', 1280)
        
        # NuScenes-style center crop parameters (scale shortest edge to 224, then center crop)
        scale = 224 / min(img_w, img_h)
        scaled_w = int(round(img_w * scale))
        scaled_h = int(round(img_h * scale))
        crop_left = (scaled_w - 224) // 2
        crop_top = (scaled_h - 224) // 2

        for ann in annotations:
            cls = ann.get('class_name', '')
            atype = ann.get('type', 0)
            obj_class = None
            
            if cls == 'vehicle' or atype == 1:
                num_cars += 1
                obj_class = 'car'
            elif cls == 'pedestrian' or atype == 2:
                num_peds += 1
                obj_class = 'ped'
            else:
                num_objects += 1
                continue
            num_objects += 1
            
            # Project 3D box center to 2D for grid occupancy
            try:
                center_3d = np.array(ann.get('center', [0, 0, 0]), dtype=np.float64)
                size_3d = np.array(ann.get('size', [1, 1, 1]), dtype=np.float64)  # [l, w, h]
                
                # Transform to camera frame
                center_cam = vehicle_to_cam[:3, :3] @ center_3d + vehicle_to_cam[:3, 3]
                
                # Skip if behind camera
                if center_cam[2] <= 0:
                    continue
                
                # Project center to image
                cx_2d = intrinsic[0, 0] * center_cam[0] / center_cam[2] + intrinsic[0, 2]
                cy_2d = intrinsic[1, 1] * center_cam[1] / center_cam[2] + intrinsic[1, 2]
                
                # Approximate 2D box size from 3D dimensions
                depth = center_cam[2]
                box_w_2d = intrinsic[0, 0] * max(size_3d[0], size_3d[1]) / depth
                box_h_2d = intrinsic[1, 1] * size_3d[2] / depth
                
                # Convert to crop coordinates
                u_min = (cx_2d - box_w_2d / 2) * scale - crop_left
                u_max = (cx_2d + box_w_2d / 2) * scale - crop_left
                v_min = (cy_2d - box_h_2d / 2) * scale - crop_top
                v_max = (cy_2d + box_h_2d / 2) * scale - crop_top
                
                # Only include boxes that are at least partially visible in crop
                if u_max > 0 and u_min < 224 and v_max > 0 and v_min < 224:
                    box_2d_list.append({
                        "box": [u_min, v_min, u_max, v_max],
                        "class": obj_class
                    })
            except Exception:
                pass

        # Initialize default depth grids
        depth_grid = np.zeros(64, dtype=np.float32)
        depth_grid_mask = np.zeros(64, dtype=np.float32)
        depth_grid_hr = np.zeros(3136, dtype=np.float32)
        depth_grid_mask_hr = np.zeros(3136, dtype=np.float32)
        mean_depth = 0.0
        
        # Use cached depth grids if available (huge speedup)
        if cached_depth_grids is not None:
            mean_depth = cached_depth_grids['mean_depth']
            depth_grid = cached_depth_grids['depth_grid']
            depth_grid_mask = cached_depth_grids['depth_grid_mask']
            depth_grid_hr = cached_depth_grids['depth_grid_hr']
            depth_grid_mask_hr = cached_depth_grids['depth_grid_mask_hr']
        
        # Compute depth grids from LiDAR if not cached and lidar available
        elif lidar_points is not None and len(lidar_points) > 0:
            try:
                from src.lidar_utils import lidar_to_depth_map_full
                
                # Get image dimensions
                orig_h = pair.get('img_height', 1280)
                orig_w = pair.get('img_width', 1920)
                
                # Compute LiDAR-to-camera transform
                lidar_to_cam = self._compute_lidar_to_cam(pair)
                
                # Use only xyz+intensity columns for depth map computation
                pts_5d = lidar_points[:, :5] if lidar_points.shape[1] >= 5 else lidar_points
                
                # Compute full-resolution depth map
                depth_map_fullres = lidar_to_depth_map_full(
                    pts_5d,
                    intrinsic=pair['intrinsic'],
                    lidar_to_cam_transform=lidar_to_cam.astype(np.float32),
                    img_size=(orig_h, orig_w),
                    max_depth=80.0
                )
                
                # Compute depth grids from full resolution depth map
                d_tensor_full = torch.from_numpy(depth_map_fullres).unsqueeze(0).unsqueeze(0)
                valid_mask_full = (d_tensor_full > 0).float()
                valid_count_full = valid_mask_full.sum()
                
                if valid_count_full > 0:
                    mean_depth = (d_tensor_full.sum() / valid_count_full).item() * 80.0  # In Meters
                
                # 8x8 grid (64 cells)
                h, w = d_tensor_full.shape[-2:]
                grid_sum = torch.nn.functional.adaptive_avg_pool2d(d_tensor_full, (8, 8)) * (h * w / 64)
                grid_count = torch.nn.functional.adaptive_avg_pool2d(valid_mask_full, (8, 8)) * (h * w / 64)
                
                depth_grid_label = torch.zeros_like(grid_sum)
                mask_nz = grid_count > 0
                depth_grid_label[mask_nz] = grid_sum[mask_nz] / grid_count[mask_nz]
                
                depth_grid = (depth_grid_label * 80.0).flatten().numpy().astype(np.float32)
                depth_grid_mask = mask_nz.float().flatten().numpy().astype(np.float32)
                
                # 56x56 HR grid (3136 cells)
                grid_sum_hr = torch.nn.functional.adaptive_avg_pool2d(d_tensor_full, (56, 56)) * (h * w / 3136)
                grid_count_hr = torch.nn.functional.adaptive_avg_pool2d(valid_mask_full, (56, 56)) * (h * w / 3136)
                
                depth_grid_label_hr = torch.zeros_like(grid_sum_hr)
                mask_nz_hr = grid_count_hr > 0
                depth_grid_label_hr[mask_nz_hr] = grid_sum_hr[mask_nz_hr] / grid_count_hr[mask_nz_hr]
                
                depth_grid_hr = (depth_grid_label_hr * 80.0).flatten().numpy().astype(np.float32)
                depth_grid_mask_hr = mask_nz_hr.float().flatten().numpy().astype(np.float32)
                
                # Save to cache (when not augmenting)
                if not _aug_active:
                    try:
                        np.savez_compressed(
                            labels_cache_path,
                            mean_depth=mean_depth,
                            depth_grid=depth_grid,
                            depth_grid_mask=depth_grid_mask,
                            depth_grid_hr=depth_grid_hr,
                            depth_grid_mask_hr=depth_grid_mask_hr,
                        )
                    except Exception:
                        pass  # Cache write failed, continue anyway
                
            except Exception as e:
                # Leave defaults if computation fails
                pass

        # Compute 8x8 grid occupancy from 2D bounding boxes
        grid_occupancy = np.zeros(64, dtype=np.float32)
        grid_occupancy_car = np.zeros(64, dtype=np.float32)
        grid_occupancy_ped = np.zeros(64, dtype=np.float32)
        grid_occupancy_hr = np.zeros(self.occupancy_grid_size * self.occupancy_grid_size, dtype=np.float32)
        cell_size = 224 / 8  # 28 pixels per cell
        cell_size_hr = 224 / float(self.occupancy_grid_size)
        
        for box_info in box_2d_list:
            u_min, v_min, u_max, v_max = box_info["box"]
            obj_class = box_info["class"]
            
            # Find all grid cells that this box overlaps
            col_start = max(0, int(u_min / cell_size))
            col_end = min(7, int(u_max / cell_size))
            row_start = max(0, int(v_min / cell_size))
            row_end = min(7, int(v_max / cell_size))
            
            for row in range(row_start, row_end + 1):
                for col in range(col_start, col_end + 1):
                    cell_idx = row * 8 + col
                    grid_occupancy[cell_idx] = 1.0
                    if obj_class == "car":
                        grid_occupancy_car[cell_idx] = 1.0
                    elif obj_class == "ped":
                        grid_occupancy_ped[cell_idx] = 1.0

            col_start_hr = max(0, int(u_min / cell_size_hr))
            col_end_hr = min(self.occupancy_grid_size - 1, int(u_max / cell_size_hr))
            row_start_hr = max(0, int(v_min / cell_size_hr))
            row_end_hr = min(self.occupancy_grid_size - 1, int(v_max / cell_size_hr))
            for row in range(row_start_hr, row_end_hr + 1):
                for col in range(col_start_hr, col_end_hr + 1):
                    cell_idx = row * self.occupancy_grid_size + col
                    grid_occupancy_hr[cell_idx] = 1.0

        return {
            'num_cars': num_cars,
            'num_pedestrians': num_peds,
            'num_objects': num_objects,
            'depth_grid': depth_grid,
            'depth_grid_mask': depth_grid_mask,
            'depth_grid_hr': depth_grid_hr,
            'depth_grid_mask_hr': depth_grid_mask_hr,
            'mean_depth': mean_depth,
            'has_boxes': num_objects > 0,
            'grid_occupancy': grid_occupancy,
            'grid_occupancy_car': grid_occupancy_car,
            'grid_occupancy_ped': grid_occupancy_ped,
            'grid_occupancy_hr': grid_occupancy_hr,
        }

    def _get_det_labels(self, pair: Dict, annotations: List[Dict],
                        lidar_points: Optional[np.ndarray] = None) -> Dict:
        """
        Compute detection labels in the same format as NuScenes.

        Returns gt_classes, gt_centers, gt_sizes, gt_orientations, gt_mask,
        gt_centers_2d, seg_map.
        """
        max_objects = 50

        gt_classes = np.full(max_objects, -1, dtype=np.int64)
        gt_centers = np.zeros((max_objects, 3), dtype=np.float32)
        gt_sizes = np.zeros((max_objects, 3), dtype=np.float32)
        gt_orientations = np.zeros((max_objects, 2), dtype=np.float32)
        gt_mask = np.zeros(max_objects, dtype=np.float32)
        gt_centers_2d = np.zeros((max_objects, 2), dtype=np.float32)
        bbox_only = str(getattr(self, "det_seg_label_mode", "both")).lower() == "bbox_only"
        
        # ── Precomputed Labels ─────────────────────────────────────────
        if getattr(self, "precomputed_labels_path", None) is not None:
            from src.detection_labels import load_det_seg_labels
            required_keys = [
                'gt_classes', 'gt_centers', 'gt_sizes',
                'gt_orientations', 'gt_mask', 'gt_centers_2d'
            ] if bbox_only else None
            cached_data = load_det_seg_labels(
                sample_token=pair['sample_token'],
                cam_name=pair['camera_name'],
                cache_dir=self.precomputed_labels_path,
                has_shards=getattr(self, 'precomputed_labels_has_shards', False),
                required_keys=required_keys,
            )
            if cached_data is not None:
                if bbox_only:
                    cached_data['seg_map'] = np.zeros((224, 224), dtype=np.int64)
                    cached_data['panoptic_seg_map'] = np.zeros((224, 224), dtype=np.int64)
                    cached_data['has_seg_map'] = False
                    cached_data['has_panoptic_seg_map'] = False
                else:
                    # Always recompute panoptic_seg_map on-the-fly from camera_seg PNGs
                    # because precomputed panoptic labels may have been generated with
                    # an older/incorrect WAYMO_PANOPTIC_TO_SIMPLIFIED mapping.
                    panoptic_seg_map = np.zeros((224, 224), dtype=np.int64)
                    seg_path = self.dataroot / "camera_seg" / pair['segment_name'] / f"{pair['timestamp']}_{pair['camera_name']}.png"
                    if seg_path.exists():
                        try:
                            seg_img = Image.open(seg_path)
                            orig_w, orig_h = seg_img.size
                            scale = 224 / min(orig_w, orig_h)
                            new_w = int(round(orig_w * scale))
                            new_h = int(round(orig_h * scale))
                            seg_img = seg_img.resize((new_w, new_h), Image.NEAREST)
                            left = (new_w - 224) // 2
                            top = (new_h - 224) // 2
                            seg_img = seg_img.crop((left, top, left + 224, top + 224))
                            seg_arr = np.array(seg_img, dtype=np.int64) // 1000
                            mapper = np.zeros(256, dtype=np.int64)
                            for k, v in WAYMO_PANOPTIC_TO_SIMPLIFIED.items():
                                if k < 256: mapper[k] = v
                            panoptic_seg_map = mapper[seg_arr]
                        except Exception:
                            pass
                    cached_data['panoptic_seg_map'] = panoptic_seg_map
                    cached_data['has_panoptic_seg_map'] = panoptic_seg_map.sum() > 0

                    # Add seg_map availability flag if not present (backwards compat)
                    if 'has_seg_map' not in cached_data:
                        seg_map = cached_data.get('seg_map', None)
                        cached_data['has_seg_map'] = seg_map is not None and seg_map.sum() > 0
                return cached_data
        
        seg_map = np.zeros((224, 224), dtype=np.int64)
        panoptic_seg_map = np.zeros((224, 224), dtype=np.int64)
        
        # Load 2D Camera Panoptic Segmentation if it exists
        if not bbox_only:
            seg_path = self.dataroot / "camera_seg" / pair['segment_name'] / f"{pair['timestamp']}_{pair['camera_name']}.png"
            if seg_path.exists():
                try:
                    # NuScenes-style evaluation pipeline: Resize shortest edge, then CenterCrop
                    seg_img = Image.open(seg_path)
                    orig_w, orig_h = seg_img.size
                    scale = 224 / min(orig_w, orig_h)
                    new_w = int(round(orig_w * scale))
                    new_h = int(round(orig_h * scale))
                    
                    # 1. Resize shortest edge
                    seg_img = seg_img.resize((new_w, new_h), Image.NEAREST)
                    
                    # 2. Center crop
                    left = (new_w - 224) // 2
                    top = (new_h - 224) // 2
                    seg_img = seg_img.crop((left, top, left + 224, top + 224))
                    
                    # Waymo panoptic labels = semantic_id * 1000 + instance_id
                    seg_arr = np.array(seg_img, dtype=np.int64) // 1000
                    
                    # Fast mapping via lookup array - use PANOPTIC mapping for camera masks
                    mapper = np.zeros(256, dtype=np.int64)
                    for k, v in WAYMO_PANOPTIC_TO_SIMPLIFIED.items():
                        if k < 256: mapper[k] = v
                    panoptic_seg_map = mapper[seg_arr]
                except Exception as e:
                    print(f"Warning: Failed to load mask {seg_path}: {e}")
        
        # ── LiDAR Semantic Segmentation (from 6th column if present) ─
        if (not bbox_only) and lidar_points is not None and lidar_points.shape[1] >= 6:
            try:
                from src.detection_labels import compute_seg_map, compute_center_crop_region
                
                lidarseg_labels = lidar_points[:, 5].astype(np.int64)
                points_5d = lidar_points[:, :5].copy()
                
                img_w = pair.get('img_width', 1920)
                img_h = pair.get('img_height', 1280)
                crop_region = compute_center_crop_region(
                    img_hw=(img_h, img_w),
                    target_size=224,
                )
                
                lidar_to_cam = self._compute_lidar_to_cam(pair)
                
                # compute_seg_map with dataset_type='waymo' handles Waymo label encoding
                # internally (extracts semantic class from panoptic label via mod 23)
                seg_map = compute_seg_map(
                    lidar_points=points_5d,
                    lidarseg_labels=lidarseg_labels,
                    intrinsic=np.asarray(pair['intrinsic'], dtype=np.float64),
                    lidar_to_cam=lidar_to_cam.astype(np.float64),
                    img_hw=(img_h, img_w),
                    target_hw=(224, 224),
                    crop_region=crop_region,
                    fill_max_dist=5,
                    dataset_type='waymo',
                )
                
            except Exception as e:
                print(f"Warning computing lidar seg_map: {e}")
                pass # Leave seg_map empty

        intrinsic = pair['intrinsic']
        vehicle_to_cam = np.linalg.inv(pair['cam_extrinsic']).astype(np.float64)

        obj_idx = 0
        for ann in annotations:
            if obj_idx >= max_objects:
                break

            # Get detection class
            atype = ann.get('type', 0)
            det_cls = WAYMO_TYPE_TO_DETECTION.get(atype, -1)
            if det_cls < 0:
                continue

            # Center in vehicle frame
            center = np.array(ann['center'], dtype=np.float64)
            size = np.array(ann['size'], dtype=np.float64)  # w, l, h
            heading = float(ann.get('heading', 0))

            # Transform center (vehicle frame) to camera frame
            center_h = np.append(center, 1.0)
            center_cam_waymo = (vehicle_to_cam @ center_h)[:3]

            # Waymo sensor frame is X-forward, Y-left, Z-up
            # Standard camera frame is X-right, Y-down, Z-forward
            center_cam = np.array([
                -center_cam_waymo[1],  # X_std = -Y_waymo (right)
                -center_cam_waymo[2],  # Y_std = -Z_waymo (down)
                center_cam_waymo[0]    # Z_std = X_waymo (forward)
            ], dtype=np.float32)

            # Skip if behind camera
            if center_cam[2] <= 0:
                continue

            # Project to 2D
            u = intrinsic[0, 0] * center_cam[0] / center_cam[2] + intrinsic[0, 2]
            v = intrinsic[1, 1] * center_cam[1] / center_cam[2] + intrinsic[1, 2]

            img_w = pair.get('img_width', 1920)
            img_h = pair.get('img_height', 1280)
            
            # Use same center crop as evaluation
            from src.detection_labels import compute_center_crop_region
            crop_i, crop_j, crop_h, crop_w = compute_center_crop_region(
                img_hw=(img_h, img_w),
                target_size=224, # Note: Using default 224 since spatial_labels doesn't know evaluation target
            )
            
            # Check if in centered crop bounds (matching evaluation behavior!)
            if u < crop_j or u >= crop_j + crop_w or v < crop_i or v >= crop_i + crop_h:
                continue

            gt_classes[obj_idx] = det_cls
            gt_centers[obj_idx] = center_cam[:3].astype(np.float32)
            gt_sizes[obj_idx] = size.astype(np.float32)
            gt_orientations[obj_idx] = np.array(
                [np.sin(heading), np.cos(heading)], dtype=np.float32)
            gt_mask[obj_idx] = 1.0
            
            # Normalize 2D centers strictly inside the crop box
            gt_centers_2d[obj_idx] = np.array(
                [(u - crop_j) / crop_w, (v - crop_i) / crop_h], dtype=np.float32)

            obj_idx += 1

        # Track which seg labels are actually available (have non-zero pixels)
        # This allows training to skip loss computation for missing labels
        has_seg_map = (not bbox_only) and (seg_map.sum() > 0)  # LiDAR-projected semantic labels
        has_panoptic_seg_map = (not bbox_only) and (panoptic_seg_map.sum() > 0)  # Camera panoptic labels

        return {
            'gt_classes': gt_classes,
            'gt_centers': gt_centers,
            'gt_sizes': gt_sizes,
            'gt_orientations': gt_orientations,
            'gt_mask': gt_mask,
            'gt_centers_2d': gt_centers_2d,
            'seg_map': seg_map,
            'panoptic_seg_map': panoptic_seg_map,
            'has_seg_map': has_seg_map,
            'has_panoptic_seg_map': has_panoptic_seg_map,
        }

    def _get_multimae_view_seg_labels(
        self,
        pair: Dict,
        lidar_points: Optional[np.ndarray] = None,
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
        target_size: Optional[int] = None,
    ) -> Dict:
        from src.detection_labels import compute_center_crop_region, compute_seg_map

        target_size = int(target_size or self.img_size)
        seg_map = np.zeros((target_size, target_size), dtype=np.int64)
        panoptic_seg_map = np.zeros((target_size, target_size), dtype=np.int64)
        bbox_only = str(getattr(self, "det_seg_label_mode", "both")).lower() == "bbox_only"
        img_w = pair.get('img_width', 1920)
        img_h = pair.get('img_height', 1280)
        crop_region = crop_rect
        if crop_region is None:
            crop_region = compute_center_crop_region(img_hw=(img_h, img_w), target_size=target_size)

        if not bbox_only:
            seg_path = self.dataroot / "camera_seg" / pair['segment_name'] / f"{pair['timestamp']}_{pair['camera_name']}.png"
            if seg_path.exists():
                try:
                    seg_img = Image.open(seg_path)
                    crop_i, crop_j, crop_h, crop_w = crop_region
                    seg_img = seg_img.crop((crop_j, crop_i, crop_j + crop_w, crop_i + crop_h))
                    seg_img = seg_img.resize((target_size, target_size), Image.NEAREST)
                    seg_arr = np.array(seg_img, dtype=np.int64) // 1000
                    mapper = np.zeros(256, dtype=np.int64)
                    for key, value in WAYMO_PANOPTIC_TO_SIMPLIFIED.items():
                        if key < 256:
                            mapper[key] = value
                    panoptic_seg_map = mapper[seg_arr]
                except Exception:
                    pass

        if (not bbox_only) and lidar_points is not None and lidar_points.shape[1] >= 6:
            try:
                lidarseg_labels = lidar_points[:, 5].astype(np.int64)
                points_5d = lidar_points[:, :5].copy()
                lidar_to_cam = self._compute_lidar_to_cam(pair)
                seg_map = compute_seg_map(
                    lidar_points=points_5d,
                    lidarseg_labels=lidarseg_labels,
                    intrinsic=np.asarray(pair['intrinsic'], dtype=np.float64),
                    lidar_to_cam=lidar_to_cam.astype(np.float64),
                    img_hw=(img_h, img_w),
                    target_hw=(target_size, target_size),
                    crop_region=crop_region,
                    fill_max_dist=5,
                    dataset_type='waymo',
                )
            except Exception:
                pass

        return {
            'seg_map': seg_map,
            'panoptic_seg_map': panoptic_seg_map,
            'has_seg_map': bool(seg_map.sum() > 0),
            'has_panoptic_seg_map': bool(panoptic_seg_map.sum() > 0),
        }

    # ═══════════════════════════════════════════════════════════════
    # GEOMETRY
    # ═══════════════════════════════════════════════════════════════

    def _compute_lidar_to_cam(self, pair: Dict) -> np.ndarray:
        """
        Compute LiDAR → Camera transform that outputs STANDARD CAMERA coordinates.

        Waymo coordinate conventions:
         - Vehicle frame: X=forward, Y=left, Z=up
         - Sensor frame (camera/lidar): SAME as vehicle (X=forward, Y=left, Z=up)
         - Standard camera frame for projection: X=right, Y=down, Z=forward

        Waymo extrinsics are sensor→vehicle transforms.
        
        Steps:
         1. lidar_to_vehicle = lidar_extrinsic
         2. vehicle_to_waymo_cam = inv(cam_extrinsic)
         3. waymo_cam_to_std_cam = axis swap matrix
         
        Result: lidar points → standard camera frame for pinhole projection.
        """
        lidar_ext = pair['lidar_extrinsic']  # lidar → vehicle
        cam_ext = pair['cam_extrinsic']      # cam → vehicle

        # cam_ext is sensor-to-vehicle, we need vehicle-to-sensor
        cam_ext_inv = np.linalg.inv(cam_ext)
        
        # This gives us points in Waymo sensor frame (X=forward, Y=left, Z=up)
        lidar_to_waymo_cam = cam_ext_inv @ lidar_ext
        
        # Axis swap: Waymo sensor (X=fwd, Y=left, Z=up) → standard camera (X=right, Y=down, Z=fwd)
        # std_cam_x = -waymo_y  (left → right)
        # std_cam_y = -waymo_z  (up → down)
        # std_cam_z = waymo_x   (forward stays forward)
        axis_swap = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        return (axis_swap @ lidar_to_waymo_cam).astype(np.float64)

    def _annotations_to_lidar_frame(self, pair: Dict) -> List[Dict]:
        """
        Convert Waymo annotations (in vehicle frame) to augmentation format.

        Returns list of dicts with 'translation', 'size', 'rotation' keys
        (same format as NuScenes annotations for LiDARSceneAugmentor).
        """
        # Waymo annotations are already in vehicle frame (same as lidar frame
        # since lidar_extrinsic is identity for TOP lidar in most cases)
        lidar_ext_inv = np.linalg.inv(pair['lidar_extrinsic'])

        result = []
        for ann in pair['annotations']:
            center = np.array(ann['center'], dtype=np.float64)
            # Transform from vehicle to lidar frame
            center_h = np.append(center, 1.0)
            center_lidar = (lidar_ext_inv @ center_h)[:3]

            heading = float(ann.get('heading', 0))
            quat = heading_to_quat(heading)

            result.append({
                'translation': center_lidar.tolist(),
                'size': ann['size'],
                'rotation': quat.tolist(),
                'instance_token': ann.get('id', ''),
                'category_name': ann.get('class_name', ''),
                'type': ann.get('type', 0),
            })
        return result

    def _aug_anns_to_waymo_format(self, aug_anns: List[Dict]) -> List[Dict]:
        """Convert augmented annotations back to Waymo format."""
        result = []
        for ann in aug_anns:
            q = ann.get('rotation', [1.0, 0.0, 0.0, 0.0])
            if isinstance(q, np.ndarray):
                q = q.tolist()
            if isinstance(q, (list, tuple)) and len(q) == 4:
                w, x, y, z = [float(v) for v in q]
                heading = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
            else:
                heading = 0.0
            result.append({
                'center': list(ann['translation']),
                'size': list(ann['size']),
                'heading': heading,
                'class_name': ann.get('category_name', ''),
                'type': ann.get('type', 0),
                'id': ann.get('instance_token', ''),
            })
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION HELPER (wraps torchvision transform to return crop params)
# ═══════════════════════════════════════════════════════════════════════════════

class _AugWithParams:
    """Wrapper that returns (augmented_image, crop_params) tuple."""

    def __init__(self, transform, target_size):
        self.transform = transform
        self.target_size = target_size

    def __call__(self, img, return_params=False):
        result = self.transform(img)
        if return_params:
            # Return dummy crop params (full image)
            w, h = img.size if hasattr(img, 'size') else (224, 224)
            return result, (0, 0, h, w)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# COLLATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def waymo_collate_fn(batch):
    """
    Custom collate function with the same interface as mm_collate_fn.
    Handles mixed dict/tensor modality2 properly.
    """
    cam_views_list, modality2_list, labels_list = zip(*batch)

    # Stack camera views
    global_views = torch.stack([cv['global'] for cv in cam_views_list])
    local_views_list = [cv['local'] for cv in cam_views_list]
    if all(lv.numel() > 0 for lv in local_views_list):
        local_views = torch.stack(local_views_list)
    else:
        local_views = torch.empty(0)

    probe_views_list = [cv.get('probe', torch.empty(0)) for cv in cam_views_list]
    if all(pv.numel() > 0 for pv in probe_views_list):
        probe_views = torch.stack(probe_views_list)
    else:
        probe_views = torch.empty(0)

    cam_views = {'global': global_views, 'local': local_views, 'probe': probe_views}

    # Stack modality2
    if isinstance(modality2_list[0], dict):
        mod2_global = torch.stack([m['global'] for m in modality2_list])
        mod2_local_list = [m['local'] for m in modality2_list]
        if all(ml.numel() > 0 for ml in mod2_local_list):
            mod2_local = torch.stack(mod2_local_list)
        else:
            mod2_local = torch.empty(0)
        mod2_probe_list = [m.get('probe', torch.empty(0)) for m in modality2_list]
        if all(mp.numel() > 0 for mp in mod2_probe_list):
            mod2_probe = torch.stack(mod2_probe_list)
        else:
            mod2_probe = torch.empty(0)
        modality2 = {'global': mod2_global, 'local': mod2_local, 'probe': mod2_probe}
    else:
        modality2 = torch.stack(modality2_list)

    # Stack labels
    labels = {}
    for key in labels_list[0].keys():
        vals = [l[key] for l in labels_list]
        if isinstance(vals[0], (int, float)):
            labels[key] = torch.tensor(vals)
        elif isinstance(vals[0], np.ndarray):
            labels[key] = torch.from_numpy(np.stack(vals))
        elif isinstance(vals[0], torch.Tensor):
            labels[key] = torch.stack(vals)
        elif isinstance(vals[0], (bool, np.bool_)):
            labels[key] = torch.tensor(vals, dtype=torch.bool)
        else:
            labels[key] = vals

    return cam_views, modality2, labels


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_dataset(dataset_name: str, dataroot: str, **kwargs):
    """
    Factory function to create NuScenes, Waymo, or FLIR datasets.

    Args:
        dataset_name: 'nuscenes', 'waymo', or 'flir'
        dataroot: path to dataset root
        **kwargs: passed to dataset constructor

    Returns:
        (dataset, collate_fn)
    """
    if dataset_name.lower() == 'nuscenes':
        from src.dataset import MMNuScenesDataset, mm_collate_fn
        return MMNuScenesDataset(dataroot, **kwargs), mm_collate_fn
    elif dataset_name.lower() == 'waymo':
        return WaymoDataset(dataroot, **kwargs), waymo_collate_fn
    elif dataset_name.lower() == 'flir':
        from src.flir_dataset import FlirAdasDataset, flir_collate_fn
        return FlirAdasDataset(dataroot, **kwargs), flir_collate_fn
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'nuscenes', 'waymo', or 'flir'.")
