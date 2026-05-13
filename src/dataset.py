"""
Multi-Modal NuScenes Dataset for MM-LeJEPA.

Pairs camera images with corresponding LiDAR scans via sample_token.
Supports both Option A (raw points) and Option B (range images).

Memory Optimization (v2):
- Uses numpy arrays instead of Python dicts for metadata
- Stores paths as string arrays to avoid Path object overhead
- Memory-mapped calibration data for cross-worker sharing
"""

import torch
import numpy as np
import json
import zipfile
import hashlib
import io
import gc
import os
import pickle
import tempfile
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple, Optional, List
import tqdm


from src.lidar_utils import (
    load_lidar_bin, 
    lidar_to_range_image, 
    lidar_to_depth_map,
    lidar_to_aligned_points,
    lidar_to_depth_map_full, # Import at module level
    subsample_points,
    normalize_points
)


def quat_to_rot_numpy(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix using numpy.
    Faster than scipy.spatial.transform.Rotation for simple ops.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Pre-calculate squares
    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    return np.array([
        [1 - 2*(y2 + z2), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(x2 + z2), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(x2 + y2)]
    ], dtype=np.float32)

NUM_SHARDS = 64

def get_shard_id(token: str) -> int:
    """Deterministic shard assignment."""
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % NUM_SHARDS


# =============================================================================
# MEMORY-EFFICIENT METADATA STORAGE
# =============================================================================

class CompactCalibrationStore:
    """
    Memory-efficient storage for calibrations and ego poses.
    Uses numpy arrays instead of Python dicts to reduce memory by ~5x.
    Can optionally be memory-mapped for shared access across workers.
    """
    
    def __init__(self, mmap_path: Optional[str] = None):
        self.token_to_idx = {}
        self.translations = None  # (N, 3) float32
        self.rotations = None     # (N, 4) float32 (quaternions)
        self.intrinsics = None    # (N, 3, 3) float32 (None for non-cameras)
        self.has_intrinsic = None # (N,) bool
        self.mmap_path = mmap_path
        
    def build_from_dicts(self, calibrations: Dict, ego_poses: Dict):
        """
        Convert dict-based storage to compact numpy arrays.
        """
        # Combine both calibrations and ego_poses into single storage
        all_tokens = list(calibrations.keys()) + list(ego_poses.keys())
        n_total = len(all_tokens)
        
        self.token_to_idx = {t: i for i, t in enumerate(all_tokens)}
        self.translations = np.zeros((n_total, 3), dtype=np.float32)
        self.rotations = np.zeros((n_total, 4), dtype=np.float32)
        self.intrinsics = np.zeros((n_total, 3, 3), dtype=np.float32)
        self.has_intrinsic = np.zeros(n_total, dtype=bool)
        
        # Fill calibrations
        for token, data in calibrations.items():
            idx = self.token_to_idx[token]
            self.translations[idx] = data["translation"]
            self.rotations[idx] = data["rotation"]
            if data.get("intrinsic") is not None:
                self.intrinsics[idx] = data["intrinsic"]
                self.has_intrinsic[idx] = True
        
        # Fill ego poses
        for token, data in ego_poses.items():
            idx = self.token_to_idx[token]
            self.translations[idx] = data["translation"]
            self.rotations[idx] = data["rotation"]
            # ego_poses don't have intrinsics
    
    def get_calib(self, token: str) -> Dict:
        """Get calibration data for a token (dict-like interface for compatibility)."""
        idx = self.token_to_idx.get(token)
        if idx is None:
            return {}
        return {
            "translation": self.translations[idx],
            "rotation": self.rotations[idx],
            "intrinsic": self.intrinsics[idx] if self.has_intrinsic[idx] else None
        }
    
    def get_ego(self, token: str) -> Dict:
        """Get ego pose for a token."""
        idx = self.token_to_idx.get(token)
        if idx is None:
            return {}
        return {
            "translation": self.translations[idx],
            "rotation": self.rotations[idx],
        }
    
    def save_mmap(self, path: str):
        """Save to memory-mapped files for cross-worker sharing."""
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "translations.npy"), self.translations)
        np.save(os.path.join(path, "rotations.npy"), self.rotations)
        np.save(os.path.join(path, "intrinsics.npy"), self.intrinsics)
        np.save(os.path.join(path, "has_intrinsic.npy"), self.has_intrinsic)
        with open(os.path.join(path, "token_to_idx.pkl"), "wb") as f:
            pickle.dump(self.token_to_idx, f)
    
    @classmethod
    def load_mmap(cls, path: str) -> 'CompactCalibrationStore':
        """Load from memory-mapped files (shared across workers)."""
        store = cls()
        store.mmap_path = path
        store.translations = np.load(os.path.join(path, "translations.npy"), mmap_mode='r')
        store.rotations = np.load(os.path.join(path, "rotations.npy"), mmap_mode='r')
        store.intrinsics = np.load(os.path.join(path, "intrinsics.npy"), mmap_mode='r')
        store.has_intrinsic = np.load(os.path.join(path, "has_intrinsic.npy"), mmap_mode='r')
        with open(os.path.join(path, "token_to_idx.pkl"), "rb") as f:
            store.token_to_idx = pickle.load(f)
        return store


class CompactAnnotationStore:
    """
    Memory-efficient storage for sample annotations.
    Instead of storing full annotation dicts, store only the fields we need.
    """
    
    def __init__(self):
        self.sample_to_range = {}  # sample_token -> (start_idx, count)
        self.instance_tokens = None  # (N,) string array (compact)
        self.translations = None     # (N, 3) float32
        self.sizes = None            # (N, 3) float32
        self.rotations = None        # (N, 4) float32
        
    def build_from_dicts(self, sample_annotations: Dict[str, List], 
                         instance_to_category: Dict, category_names: Dict):
        """Build compact arrays from dict-based annotations."""
        # First pass: count total annotations
        total = sum(len(v) for v in sample_annotations.values())
        
        # Pre-allocate arrays
        self.instance_tokens = np.empty(total, dtype=object)
        self.translations = np.zeros((total, 3), dtype=np.float32)
        self.sizes = np.zeros((total, 3), dtype=np.float32)
        self.rotations = np.zeros((total, 4), dtype=np.float32)
        
        # Build category lookup
        self.instance_to_category = instance_to_category
        self.category_names = category_names
        
        # Fill arrays
        idx = 0
        for sample_token, annots in sample_annotations.items():
            start_idx = idx
            for annot in annots:
                self.instance_tokens[idx] = annot.get("instance_token", "")
                self.translations[idx] = annot.get("translation", [0, 0, 0])
                self.sizes[idx] = annot.get("size", [0, 0, 0])
                self.rotations[idx] = annot.get("rotation", [1, 0, 0, 0])
                idx += 1
            self.sample_to_range[sample_token] = (start_idx, idx - start_idx)
    
    def get_annotations(self, sample_token: str) -> List[Dict]:
        """Get annotations for a sample (dict-like interface for compatibility)."""
        if sample_token not in self.sample_to_range:
            return []
        
        start, count = self.sample_to_range[sample_token]
        annotations = []
        for i in range(start, start + count):
            annotations.append({
                "instance_token": self.instance_tokens[i],
                "translation": self.translations[i],
                "size": self.sizes[i],
                "rotation": self.rotations[i],
            })
        return annotations
    
    def get_category(self, instance_token: str) -> str:
        """Get category name for an instance."""
        cat_token = self.instance_to_category.get(instance_token)
        if cat_token:
            return self.category_names.get(cat_token, "")
        return ""


class MMNuScenesDataset(Dataset):
    """
    Multi-modal NuScenes dataset for MM-LeJEPA training.
    
    Provides paired camera-LiDAR samples with labels for:
    - Scene ID, Camera view, Location
    - Object counts (car, pedestrian, total)
    - Grid occupancy
    - Sparse depth (for depth probe)
    
    Supports:
    - Option A: Returns (camera_views, lidar_points, labels)
    - Option B: Returns (camera_views, range_views, labels)
    - Modality dropout for robustness testing
    """
    
    CAMERA_VIEWS = [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    ]
    
    LOCATION_MAP = {
        "n015": 0,  # Singapore
        "n008": 1,  # Boston
    }
    
    def __init__(
        self, 
        dataroot: str, 
        split: str = "train",
        arch: str = "B",  # A=separate encoders, B=shared ViT, C=true fusion, D=RGBD
        lidar_mode: str = "auto",  # "range"=360° panoramic, "depth"=aligned projection, "auto"=default per arch
        V: int = 2,       # views per modality
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 4,
        img_size: int = 224,
        local_img_size: int = 96,  # Smaller resolution for local crops
        range_size: Tuple[int, int] = (64, 256),  # (H, W) for range images
        n_points: int = 16384,  # points for Option A
        modality_dropout: float = 0.0,  # probability of dropping one modality
        split_strategy: str = "random",  # "random" or "scene_based"
        cache_in_memory: bool = False,
        legacy_mode: bool = False,  # If True, skip expensive detection/segmentation labels
        official_val_mode: str = "auto",  # "auto", "full", or "mini" (only used when split_strategy="official")
        finetune_mode: bool = False,  # If True, skip clean probe view, support augmented single views
        include_probe_view: Optional[bool] = None,
        encoder_only_labels: bool = False,
        finetune_crop_scale: Tuple[float, float] = (0.8, 1.0),  # Large crops for finetune augmentation
        det_seg_label_mode: str = "both",  # "both" (bbox+seg) or "bbox_only"
        lidar_aug_preset: str = "none",  # "none", "light", "moderate", "strong" or custom dict
        lidar_aug_cfg: Optional[Dict] = None,  # Override preset with custom config
        copy_paste_preset: str = "none",  # "none", "light", "moderate", "strong"
        copy_paste_cfg: Optional[Dict] = None,  # Override preset with custom config
        gt_database_path: str = None,  # Path to GT database for copy-paste
        precomputed_labels_path: Optional[str] = None,
        dino_aug_mode: str = "default",  # "default" or "official"
        cameras: Optional[List[str]] = None,  # Filter to specific cameras (e.g., ["CAM_FRONT"])
        return_multimae_view_labels: bool = False,
    ):
        """
        Args:
            dataroot: Path to nuScenes data directory
            split: 'train' or 'val' (80/20 split)
            arch: Architecture option - A/B/C/D for different fusion methods
            lidar_mode: LiDAR representation mode:
                - "range": 360° panoramic range image (works with B/C)
                - "depth": Aligned projected depth (spatially aligned with camera)
                - "points": Raw point cloud (works with A only)
                - "aligned_points": 3D points visible in camera FOV (aligned A)
                - "auto": Default based on arch (D→depth, A→points, B/C→range)
            V: Number of GLOBAL views per modality (default 2)
            global_crops_scale: Scale range for global crops (e.g. 0.4-1.0)
            local_crops_scale: Scale range for local crops (e.g. 0.05-0.4)
            local_crops_number: Number of local crops (default 4)
            img_size: Output image size for camera
            range_size: (H, W) for range image output
            n_points: Number of points for Option A
            modality_dropout: Probability of dropping one modality during training
            cache_in_memory: Whether to cache data in RAM
        """

        self.arch = arch
        self.split = split
        
        # Determine lidar_mode
        if lidar_mode == "auto":
            if arch == "D":
                self.lidar_mode = "depth"
            elif arch == "A":
                self.lidar_mode = "points"
            else:
                self.lidar_mode = "range"
        else:
            self.lidar_mode = lidar_mode
        
        self.legacy_mode = legacy_mode
        self.finetune_mode = finetune_mode
        self.include_probe_view = (not self.finetune_mode) if include_probe_view is None else bool(include_probe_view)
        self.encoder_only_labels = bool(encoder_only_labels)
        self.finetune_crop_scale = finetune_crop_scale
        self.V = V  # Global views
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.total_views = self.V + self.local_crops_number # Total views per sample
        
        self.img_size = img_size
        self.local_img_size = local_img_size
        self.range_size = range_size
        self.n_points = n_points
        self.modality_dropout = modality_dropout
        self.split_strategy = split_strategy
        self.dino_aug_mode = str(dino_aug_mode).lower()
        self.use_official_dino_augs = self.split == "train" and self.dino_aug_mode == "official"
        self.official_val_mode = str(official_val_mode).lower()
        if self.official_val_mode not in {"auto", "full", "mini"}:
            print(f"⚠️ Unknown official_val_mode='{official_val_mode}', falling back to 'auto'")
            self.official_val_mode = "auto"
        self.det_seg_label_mode = str(det_seg_label_mode).lower()
        if self.det_seg_label_mode not in {"both", "bbox_only"}:
            print(f"⚠️ Unknown det_seg_label_mode='{det_seg_label_mode}', falling back to 'both'")
            self.det_seg_label_mode = "both"
        self.return_multimae_view_labels = bool(return_multimae_view_labels)
        self.precomputed_labels_path = None
        self.precomputed_labels_has_shards = False
        if precomputed_labels_path and os.path.exists(precomputed_labels_path):
            self.precomputed_labels_path = Path(precomputed_labels_path)
            self.precomputed_labels_has_shards = any(self.precomputed_labels_path.glob("shard_*.zip"))
            print(f"⚡ Using precomputed labels from: {self.precomputed_labels_path}")
            print(f"   └─ format: {'sharded zip' if self.precomputed_labels_has_shards else 'individual .npz files'}")
        elif precomputed_labels_path:
            print(f"⚠️ precomputed_labels_path not found, disabling precomputed labels: {precomputed_labels_path}")
        self.cache_in_memory = cache_in_memory
        self.cache = {}
        
        # Camera filter: If specified, only include samples from these cameras
        # Useful to focus on high-object-density cameras like CAM_FRONT
        self.cameras_filter = cameras
        if cameras is not None:
            # Validate camera names
            valid_cams = set(self.CAMERA_VIEWS)
            for c in cameras:
                if c not in valid_cams:
                    raise ValueError(f"Invalid camera name: {c}. Valid: {self.CAMERA_VIEWS}")
            print(f"📷 Camera filter enabled: only using {cameras}")

        # ── LiDAR Augmentation Setup ──────────────────────────────
        self.lidar_aug_enabled = False
        self.copy_paste_enabled = False
        self._scene_augmentor = None
        self._copy_paste_augmentor = None

        if split == "train":  # Only augment training data
            from src.lidar_augmentations import (
                LiDARSceneAugmentor, CopyPasteAugmentor,
                AUGMENTATION_PRESETS, COPY_PASTE_PRESETS,
            )
            # Scene-level augmentation
            if lidar_aug_preset != "none" or lidar_aug_cfg:
                aug_cfg = AUGMENTATION_PRESETS.get(lidar_aug_preset, {})
                if lidar_aug_cfg:
                    aug_cfg.update(lidar_aug_cfg)
                self._scene_augmentor = LiDARSceneAugmentor(aug_cfg)
                self.lidar_aug_enabled = True
                print(f"🎲 LiDAR scene augmentation: preset={lidar_aug_preset}")

            # Copy-paste augmentation
            if copy_paste_preset != "none" or copy_paste_cfg:
                cp_cfg = COPY_PASTE_PRESETS.get(copy_paste_preset, {})
                if copy_paste_cfg:
                    cp_cfg.update(copy_paste_cfg)
                db_path = gt_database_path or str(
                    Path(dataroot).parent / "cache" / "gt_database" / "gt_database_nuscenes.pkl")
                self._copy_paste_augmentor = CopyPasteAugmentor(db_path, cp_cfg)
                self.copy_paste_enabled = bool(self._copy_paste_augmentor.sampling_pool)
                if self.copy_paste_enabled:
                    print(f"🧬 Copy-paste augmentation: preset={copy_paste_preset}")
                else:
                    print(f"⚠️ Copy-paste: GT database empty/missing at {db_path}")
        # ZipFile cache for sharded data (per-worker)
        # Dictionary mapping path -> ZipFile object
        self.zip_cache = {}
                        
        dataroot = Path(dataroot)
        self.samples_dir = dataroot / "samples"
        self.lidar_dir = self.samples_dir / "LIDAR_TOP"
        self.dataroot = dataroot  # Store for shard loading
        
        # Disk cache for expensive depth projections
        # v4: Store at mid-resolution (448) for aligned cropping with reasonable I/O
        self.cache_res = 448  # Cached depth resolution
        self.cache_dir = Path(__file__).parent / "cache" / f"depth_maps_v4_{self.cache_res}"
        # Ensure cache directory exists (fixes potential cache miss loop)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using cache directory: {self.cache_dir}")
        
        # Load nuScenes metadata
        self._load_metadata(dataroot)
        
        # Build paired samples (camera_path, lidar_path, sample_token)
        self._build_sample_pairs()
        
        # Train/val split
        self._apply_split(split)
        
        # Setup transforms
        self._setup_transforms()
        
        # Phase 2 Memory Optimization: Compact metadata to reduce base RAM
        self._compact_metadata()
        
        # Phase 3: Pre-build shard existence index to eliminate per-sample Path.exists() calls
        self._cache_shard_index = {}  # shard_id -> shard_path (only for existing shards)
        for shard_file in self.cache_dir.glob("shard_*.zip"):
            # Extract shard_id from filename like "shard_42.zip"
            try:
                sid = int(shard_file.stem.split('_')[1])
                self._cache_shard_index[sid] = shard_file
            except (ValueError, IndexError):
                pass
        self.has_shards = len(self._cache_shard_index) > 0
        if self.has_shards:
            print(f"  Detected {len(self._cache_shard_index)} sharded zip caches in {self.cache_dir}")
        
        # Detect raw data shards (images + LiDAR in zip files)
        self.shards_dir = dataroot / "nuscenes_shards"
        self._data_shard_index = {}  # shard_id -> shard_path (only for existing shards)
        for shard_file in self.shards_dir.glob("data_shard_*.zip"):
            try:
                sid = int(shard_file.stem.split('_')[-1])
                self._data_shard_index[sid] = shard_file
            except (ValueError, IndexError):
                pass
        self.has_data_shards = len(self._data_shard_index) > 0
        disable_data_shards_env = str(os.environ.get("MM_DISABLE_DATA_SHARDS", "")).lower()
        if disable_data_shards_env in {"1", "true", "yes", "y"}:
            self.has_data_shards = False
            print("📂 Data shard loading disabled by MM_DISABLE_DATA_SHARDS; using individual files")
        if self.has_data_shards:
            print(f"📦 Using {len(self._data_shard_index)} SHARDED raw data files from {self.shards_dir}")
        else:
            print(f"📂 Using INDIVIDUAL files from {self.samples_dir}")
        
        n_samples = self._n_pairs if hasattr(self, '_n_pairs') else len(self.pairs) if self.pairs else 0
        print(f"MMNuScenesDataset [arch={arch}, lidar_mode={self.lidar_mode}]: {n_samples} samples, "
              f"Global={V} ({global_crops_scale}), Local={local_crops_number} ({local_crops_scale}), "
              f"Total Views={self.total_views}, img_size={img_size}")
    
    def _load_metadata(self, dataroot: Path):
        """
        Phase 3: Optimized Load. 
        Filters heavy JSONs (annotations, ego_pose) DURING initial load to avoid RAM spikes.
        """
        if (dataroot / "v1.0-trainval").exists():
            meta_dir = dataroot / "v1.0-trainval"
            self.version = "v1.0-trainval"
        elif (dataroot / "v1.0-mini").exists():
            meta_dir = dataroot / "v1.0-mini"
            self.version = "v1.0-mini"
        else:
            v1_dirs = list(dataroot.glob("v1.0-*"))
            meta_dir = v1_dirs[0] if v1_dirs else (dataroot / "v1.0-trainval")
            self.version = meta_dir.name
        
        print(f"Loading metadata from: {meta_dir.name} (Split: {self.split})")
        
        # 1. Load small mappings first to determine target samples
        # --------------------------------------------------------
        scene_token_to_name = {}
        with open(meta_dir / "scene.json") as f:
            for s in json.load(f):
                scene_token_to_name[s["token"]] = s["name"]
        self.scene_token_to_name = scene_token_to_name
        
        sample_to_scene_token = {}
        with open(meta_dir / "sample.json") as f:
            for s in json.load(f):
                sample_to_scene_token[s["token"]] = s["scene_token"]
        self.sample_to_scene = sample_to_scene_token
        
        # 2. Determine Target Scenes for this split
        # -----------------------------------------
        target_scenes = None
        if self.split_strategy == "official":
            try:
                from nuscenes.utils.splits import create_splits_scenes
                splits = create_splits_scenes()
                if "mini" in self.version:
                    if self.split in ["trainval", "all"]:
                        target_scenes = set(splits['mini_train']).union(set(splits['mini_val']))
                    else:
                        target_scenes = set(splits['mini_train'] if self.split == 'train' else splits['mini_val'])
                else:
                    if self.split in ["trainval", "all"]:
                        target_scenes = set(splits['train']).union(set(splits['val']))
                    else:
                        target_scenes = set(splits['train'] if self.split == 'train' else splits['val'])
            except ImportError:
                print("⚠️ official split requested but nuscenes-devkit not installed. Cannot pre-filter.")
        
        # Determine used sample tokens
        # We only care about samples that belong to our target scenes
        used_samples = set()
        for stoken, sc_token in sample_to_scene_token.items():
            s_name = scene_token_to_name.get(sc_token, "")
            if target_scenes is None or s_name in target_scenes:
                used_samples.add(stoken)

        # 3. Load heavy metadata WITH filtering
        # --------------------------------------
        
        # Load sample_data (Keyframes only)
        self.sample_data = {}
        used_calib_tokens = set()
        used_ego_tokens = set()
        
        with open(meta_dir / "sample_data.json") as f:
            for entry in json.load(f):
                if entry.get("is_key_frame", False):
                    stoken = entry["sample_token"]
                    if stoken in used_samples:
                        if stoken not in self.sample_data:
                            self.sample_data[stoken] = {"cameras": {}, "lidar": None}
                        
                        calib_token = entry["calibrated_sensor_token"]
                        ego_token = entry["ego_pose_token"]
                        used_calib_tokens.add(calib_token)
                        used_ego_tokens.add(ego_token)
                        
                        filename = Path(entry["filename"]).name
                        if "CAM" in entry["filename"]:
                            cam_name = Path(entry["filename"]).parent.name
                            self.sample_data[stoken]["cameras"][cam_name] = {
                                "path": self.samples_dir / cam_name / filename,
                                "cam_name": cam_name, # Added for convenience
                                "calib_token": calib_token,
                                "ego_token": ego_token,
                            }
                        elif "LIDAR" in entry["filename"]:
                            self.sample_data[stoken]["lidar"] = {
                                "path": self.lidar_dir / filename,
                                "sd_token": entry["token"],
                                "calib_token": calib_token,
                                "ego_token": ego_token,
                            }

        # Load Annotations (THE MASSIVE ONE)
        self.sample_annotations = {}
        with open(meta_dir / "sample_annotation.json") as f:
            for annot in json.load(f):
                stoken = annot["sample_token"]
                if stoken in used_samples:
                    if stoken not in self.sample_annotations:
                        self.sample_annotations[stoken] = []
                    self.sample_annotations[stoken].append(annot)

        # Load Calibrations
        self.calibrations = {}
        with open(meta_dir / "calibrated_sensor.json") as f:
            for calib in json.load(f):
                if calib["token"] in used_calib_tokens:
                    self.calibrations[calib["token"]] = {
                        "translation": np.array(calib["translation"]),
                        "rotation": np.array(calib["rotation"]),
                        "intrinsic": np.array(calib["camera_intrinsic"]) if calib["camera_intrinsic"] else None
                    }
        
        # Load Ego Poses
        self.ego_poses = {}
        with open(meta_dir / "ego_pose.json") as f:
            for pose in json.load(f):
                if pose["token"] in used_ego_tokens:
                    self.ego_poses[pose["token"]] = {
                        "translation": np.array(pose["translation"]),
                        "rotation": np.array(pose["rotation"]),
                    }
        
        # Small remainders
        self.instance_to_category = {}
        with open(meta_dir / "instance.json") as f:
            for inst in json.load(f):
                self.instance_to_category[inst["token"]] = inst["category_token"]
        
        self.category_names = {}
        with open(meta_dir / "category.json") as f:
            for cat in json.load(f):
                self.category_names[cat["token"]] = cat["name"]
        
        gc.collect() # Final cleanup
    
    def _build_sample_pairs(self):
        """Build list of valid camera-LiDAR pairs."""
        self.pairs = []
        self.scene_labels = {}
        self.cam_to_idx = {cam: i for i, cam in enumerate(self.CAMERA_VIEWS)}
        
        for sample_token, data in self.sample_data.items():
            # Check we have both cameras and lidar for this sample
            if data["cameras"] and data["lidar"] is not None:
                lidar_info = data["lidar"]
                lidar_path = lidar_info["path"]
                
                if not lidar_path.exists():
                    continue
                
                # Create one pair per camera (with correct calibration for each)
                for cam_name, cam_info in data["cameras"].items():
                    # Apply camera filter if specified
                    if self.cameras_filter is not None and cam_name not in self.cameras_filter:
                        continue
                        
                    cam_path = cam_info["path"]
                    
                    if not cam_path.exists():
                        continue
                    
                    # Extract scene ID from filename (Log ID)
                    scene_id = cam_path.name.split("__")[0]
                    if scene_id not in self.scene_labels:
                        self.scene_labels[scene_id] = len(self.scene_labels)
                    
                    # Determine official scene name
                    scene_token = self.sample_to_scene.get(sample_token)
                    official_scene_name = self.scene_token_to_name.get(scene_token, "unknown")
                    
                    self.pairs.append({
                        "camera": cam_path,
                        "lidar": lidar_path,
                        "sample_token": sample_token,
                        "scene_id": scene_id, # This is actually Log ID
                        "scene_name": official_scene_name, # This is 'scene-XXXX'
                        "camera_name": cam_name,
                        "cam_calib_token": cam_info["calib_token"],
                        "cam_ego_token": cam_info["ego_token"],
                        "lidar_calib_token": lidar_info["calib_token"],
                        "lidar_ego_token": lidar_info["ego_token"],
                        "lidar_sd_token": lidar_info.get("sd_token", ""),
                    })
        
        self.num_scenes = len(self.scene_labels)
        self.num_cameras = len(self.CAMERA_VIEWS)
        self.num_locations = 2
    
    def _apply_split(self, split: str):
        """Apply train/val split."""
        import random
        rng = random.Random(42)
        
        if self.split_strategy == "random":
            # Legacy robust random split (80/20 of all frames)
            # High correlation data leakage likely
            indices = list(range(len(self.pairs)))
            rng.shuffle(indices)
            
            split_idx = int(0.8 * len(indices))
            if split == "train":
                indices = indices[:split_idx]
            else:
                indices = indices[split_idx:]
            
            self.pairs = [self.pairs[i] for i in indices]
            
        elif self.split_strategy == "scene_based":
            # Robust split: Split WITHIN scenes but with a BUFFER
            # Train: 0-70%, Buffer: 70-80% (Dropped), Val: 80-100%
            # This ensures we see every scene type, but don't train on Frame T and test on Frame T+1
            
            # Group by scene
            scenes = {}
            for pair in self.pairs:
                sid = pair["scene_id"]
                if sid not in scenes:
                    scenes[sid] = []
                scenes[sid].append(pair)
            
            final_pairs = []
            
            for sid in sorted(scenes.keys()):
                # Sort pairs by sample_token to ensure deterministic split
                scene_pairs = sorted(scenes[sid], key=lambda x: x["sample_token"])
                n = len(scene_pairs)
                
                # split points
                idx_train_end = int(0.7 * n)
                idx_val_start = int(0.8 * n) # 10% buffer
                
                if split == "train":
                    selected = scene_pairs[:idx_train_end]
                else:
                    selected = scene_pairs[idx_val_start:]
                    
                final_pairs.extend(selected)
            
            self.pairs = final_pairs

        elif self.split_strategy == "official":
            # Use official NuScenes strat: defined sets of scenes for train vs val
            try:
                from nuscenes.utils.splits import create_splits_scenes
            except ImportError:
                print("⚠️ official split requested but nuscenes-devkit not installed. Falling back to random.")
                # Fallback to random if not installed, but ideally caller ensures it
                self.split_strategy = "random"
                return self._apply_split(split)

            splits = create_splits_scenes()
            # splits is a dict: 'train', 'val', 'test', 'mini_train', 'mini_val'

            # Explicit override for validation split to avoid heuristic ambiguity
            if split == "val" and self.official_val_mode in {"full", "mini"}:
                if self.official_val_mode == "mini":
                    print("  official_val_mode=mini: forcing official mini val split")
                    target_scenes = set(splits['mini_val'])
                else:
                    print("  official_val_mode=full: forcing official full val split")
                    target_scenes = set(splits['val'])
                self.pairs = [p for p in self.pairs if p["scene_name"] in target_scenes]
                return
            
            # Determine which split list to use
            # If we are effectively using mini metadata, mapped to 'mini_train'/'mini_val'
            # If using trainval metadata, mapped to 'train'/'val'
            # We can guess based on the number of scenes available or dataset version detection
            
            # Heuristic: Check if any of our scenes are in 'train'
            # If we are in mini, our scenes will only be in 'mini_train'/'mini_val'
            # If we are in trainval, they will be in 'train'/'val'
            
            # Get list of all available scene IDs in our loaded data
            available_scenes = set(p["scene_name"] for p in self.pairs)
            
            target_list_name = split # 'train' or 'val'
            
            # Check if we should use mini splits
            # Safe check: if we have scenes that are NOT in standard train/val/test, maybe we are in mini?
            # Actually standard train/val covers all trainval scenes.
            # Mini scenes are a subset.
            # If we loaded 'v1.0-mini', we should probably use 'mini_train' and 'mini_val'
            # Check self.version or heuristic
            
            # Check self.version first for definitive answer
            if "mini" in self.version:
                is_mini = True
            elif "trainval" in self.version:
                is_mini = False
            else:
                # Fallback heuristic: Check overlap with official 'train' OR 'val'
                train_scenes_official = set(splits['train'])
                val_scenes_official = set(splits['val'])
                
                overlap_train = available_scenes.intersection(train_scenes_official)
                overlap_val = available_scenes.intersection(val_scenes_official)
                
                # If we share ANY scenes with the full train/val splits, we are likely in full mode
                # (Mini scenes are disjoint from Train/Val usually, or a subset... 
                # actually mini scenes ARE in train/val, but let's rely on version where possible)
                # Correction: mini scenes ARE a subset of trainval. 
                # But if we have scenes that are IN trainval but NOT in mini, then we are definitely full.
                
                mini_scenes = set(splits['mini_train']).union(set(splits['mini_val']))
                non_mini_scenes = available_scenes - mini_scenes
                
                if len(non_mini_scenes) > 0:
                    is_mini = False
                elif len(available_scenes) > 0:
                    # All our scenes are in mini... so we could be mini OR full.
                    # Defaulting to mini here is risky if we just happened to load a subset.
                    # But usually, if we are 'trainval', we have plenty of non-mini scenes.
                    is_mini = True
                else:
                    # No available scenes? (Empty dataset)
                    is_mini = False # Default to full to avoid constraining to mini

            if is_mini:
                # Perhaps we are in mini
                print("  No overlap with official 'train'. Assuming v1.0-mini splits.")
                if split == "train":
                    target_scenes = set(splits['mini_train'])
                elif split == "val":
                    target_scenes = set(splits['mini_val'])
                elif split == "trainval" or split == "all":
                    target_scenes = set(splits['mini_train']).union(set(splits['mini_val']))
            else:
                print(f"  Using full v1.0-trainval splits (version={self.version}).")
                if split == "train":
                    target_scenes = set(splits['train'])
                elif split == "val":
                    target_scenes = set(splits['val'])
                elif split == "trainval" or split == "all":
                    target_scenes = set(splits['train']).union(set(splits['val']))

            # Filter pairs
            self.pairs = [p for p in self.pairs if p["scene_name"] in target_scenes]
    
    def _compact_metadata(self):
        """
        Phase 2 Memory Optimization (v2):
        Convert heavy dict-based metadata to compact numpy arrays.
        This reduces memory by ~5x and enables memory-mapping for workers.
        """
        if not self.pairs:
            return

        # 1. Identify all used tokens
        used_samples = set()
        used_calibs = set()
        used_egos = set()
        
        for p in self.pairs:
            token = p["sample_token"]
            used_samples.add(token)
            
            # Find associated data in sample_data
            if token in self.sample_data:
                data = self.sample_data[token]
                for cam in data["cameras"].values():
                    used_calibs.add(cam["calib_token"])
                    used_egos.add(cam["ego_token"])
                if data["lidar"]:
                    used_calibs.add(data["lidar"]["calib_token"])
                    used_egos.add(data["lidar"]["ego_token"])

        # 2. Filter dictionaries first
        self.sample_data = {t: v for t, v in self.sample_data.items() if t in used_samples}
        calibrations_filtered = {t: v for t, v in self.calibrations.items() if t in used_calibs}
        ego_poses_filtered = {t: v for t, v in self.ego_poses.items() if t in used_egos}
        sample_annotations_filtered = {t: v for t, v in self.sample_annotations.items() if t in used_samples}
        
        # 3. Convert to compact stores
        self.calib_store = CompactCalibrationStore()
        self.calib_store.build_from_dicts(calibrations_filtered, ego_poses_filtered)
        
        self.annotation_store = CompactAnnotationStore()
        self.annotation_store.build_from_dicts(
            sample_annotations_filtered,
            self.instance_to_category,
            self.category_names
        )
        
        # 4. Remove old dict-based storage
        del self.calibrations
        del self.ego_poses
        del self.sample_annotations
        self.calibrations = None
        self.ego_poses = None
        self.sample_annotations = None
        
        # 5. Compact Mappings
        self.sample_to_scene = {t: v for t, v in self.sample_to_scene.items() if t in used_samples}
        used_scenes = set(self.sample_to_scene.values())
        self.scene_token_to_name = {t: v for t, v in self.scene_token_to_name.items() if t in used_scenes}
        
        # 6. Convert pairs list to more compact representation
        # Store paths as strings to avoid Path object overhead (~400 bytes each)
        self._compact_pairs()
        
        # Clean up
        gc.collect()
        print(f"  Metadata compacted to numpy arrays. Calib entries: {len(self.calib_store.token_to_idx)}")
    
    def _compact_pairs(self):
        """
        Convert pairs list from dicts with Path objects to more compact representation.
        This reduces memory for the pairs list from ~500 bytes/pair to ~200 bytes/pair.
        """
        # Store string arrays instead of Path objects
        n = len(self.pairs)
        self.pairs_camera_paths = np.empty(n, dtype=object)
        self.pairs_lidar_paths = np.empty(n, dtype=object)
        self.pairs_sample_tokens = np.empty(n, dtype=object)
        self.pairs_scene_ids = np.empty(n, dtype=object)
        self.pairs_scene_names = np.empty(n, dtype=object)
        self.pairs_camera_names = np.empty(n, dtype=object)
        self.pairs_cam_calib_tokens = np.empty(n, dtype=object)
        self.pairs_cam_ego_tokens = np.empty(n, dtype=object)
        self.pairs_lidar_calib_tokens = np.empty(n, dtype=object)
        self.pairs_lidar_ego_tokens = np.empty(n, dtype=object)
        self.pairs_lidar_sd_tokens = np.empty(n, dtype=object)
        
        for i, p in enumerate(self.pairs):
            self.pairs_camera_paths[i] = str(p["camera"])
            self.pairs_lidar_paths[i] = str(p["lidar"])
            self.pairs_sample_tokens[i] = p["sample_token"]
            self.pairs_scene_ids[i] = p["scene_id"]
            self.pairs_scene_names[i] = p["scene_name"]
            self.pairs_camera_names[i] = p["camera_name"]
            self.pairs_cam_calib_tokens[i] = p["cam_calib_token"]
            self.pairs_cam_ego_tokens[i] = p["cam_ego_token"]
            self.pairs_lidar_calib_tokens[i] = p["lidar_calib_token"]
            self.pairs_lidar_ego_tokens[i] = p["lidar_ego_token"]
            self.pairs_lidar_sd_tokens[i] = p.get("lidar_sd_token", "")
        
        # Keep pairs list for len() but make it lightweight
        self._n_pairs = n
        del self.pairs
        self.pairs = None
    
    def _get_pair(self, idx: int) -> Dict:
        """Reconstruct a pair dict from compact arrays."""
        return {
            "camera": Path(self.pairs_camera_paths[idx]),
            "lidar": Path(self.pairs_lidar_paths[idx]),
            "sample_token": self.pairs_sample_tokens[idx],
            "scene_id": self.pairs_scene_ids[idx],
            "scene_name": self.pairs_scene_names[idx],
            "camera_name": self.pairs_camera_names[idx],
            "cam_calib_token": self.pairs_cam_calib_tokens[idx],
            "cam_ego_token": self.pairs_cam_ego_tokens[idx],
            "lidar_calib_token": self.pairs_lidar_calib_tokens[idx],
            "lidar_ego_token": self.pairs_lidar_ego_tokens[idx],
            "lidar_sd_token": self.pairs_lidar_sd_tokens[idx],
        }
    
    def __len__(self):
        return self._n_pairs if hasattr(self, '_n_pairs') else len(self.pairs)
        
    def __getstate__(self):
        """Custom pickling to handle unpickleable ZipFile objects."""
        state = self.__dict__.copy()
        # Don't pickle open zip handles - let workers create their own
        state['zip_cache'] = {}
        return state

    def __setstate__(self, state):
        """Restore state and initialize empty cache."""
        self.__dict__.update(state)
        # Ensure cache is initialized
        if 'zip_cache' not in self.__dict__:
            self.zip_cache = {}
    
    def _get_zip_handle(self, path: Path) -> zipfile.ZipFile:
        """Get or open a cached ZipFile handle."""
        path_str = str(path)
        if path_str not in self.zip_cache:
            try:
                self.zip_cache[path_str] = zipfile.ZipFile(path, 'r')
            except Exception as e:
                # If opening fails, return None or let caller handle it?
                # Caller expects a handle, so re-raising is probably better,
                # but existing code handles exceptions.
                # Let's clean up if partial failure
                if path_str in self.zip_cache:
                    del self.zip_cache[path_str]
                raise e
        return self.zip_cache[path_str]
    
    def _setup_transforms(self):
        """Setup image transforms."""
        self.normalize_pipeline = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        color_jitter = v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
        ])

        self.dino_global_pipeline_1 = v2.Compose([
            color_jitter,
            v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            self.normalize_pipeline,
        ])
        self.dino_global_pipeline_2 = v2.Compose([
            color_jitter,
            v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.1),
            v2.RandomSolarize(threshold=128, p=0.2),
            self.normalize_pipeline,
        ])
        self.dino_local_pipeline = v2.Compose([
            color_jitter,
            v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            self.normalize_pipeline,
        ])

        # Camera augmentations
        # Camera augmentations - Base components
        self.aug_pipeline = v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Test pipeline checking
        self.test_pipeline = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Define RRC components separately so we can extract params manually
        self.global_rrc = v2.RandomResizedCrop(self.img_size, scale=self.global_crops_scale)
        self.local_rrc = v2.RandomResizedCrop(self.local_img_size, scale=self.local_crops_scale)
        
        # Bind wrappers that handle return_params
        self.cam_global_aug = lambda x, return_params=False, view_index=0: self._apply_aug(
            x,
            self.global_rrc,
            self.aug_pipeline,
            return_params,
            is_global=True,
            view_index=view_index,
        )
        self.cam_local_aug = lambda x, return_params=False, view_index=0: self._apply_aug(
            x,
            self.local_rrc,
            self.aug_pipeline,
            return_params,
            is_global=False,
            view_index=view_index,
        )
        
        # Test aug logic
        self.cam_test = lambda x, return_params=False: self._apply_test_aug(x, return_params)
        
        # Range image augmentations (simpler - just crops and flips)
        # Range images need less aggressive augmentation since they're already different from camera
        self.range_aug = v2.Compose([
            v2.RandomResizedCrop((self.img_size, self.img_size), scale=(0.5, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=False),  # Already normalized
        ])
        
        self.range_test = v2.Compose([
            v2.Resize((self.img_size, self.img_size)),
            v2.ToDtype(torch.float32, scale=False),
        ])
        
        # Range local crops (Small)
        self.range_local_aug = v2.Compose([
             v2.RandomResizedCrop((self.local_img_size, self.local_img_size), scale=(0.5, 1.0)),
             v2.RandomHorizontalFlip(),
             v2.ToDtype(torch.float32, scale=False),
        ])

        # Pre-instantiate deterministic RGB augmentation (for use in synchronized loops)
        # Excludes RandomHorizontalFlip (handled manually) and RandomResizedCrop (handled manually)
        self.rgb_aug_no_flip = self.dino_global_pipeline_1 if self.use_official_dino_augs else v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Clean full-view transform for probing (no jitter, no random crop)
        # Resizes to img_size square (distorting aspect ratio if needed, or use CenterCrop)
        # Using Resize then CenterCrop is safer for preservation
        self.full_view_transform = v2.Compose([
            v2.Resize(self.img_size),  # Resize smallest edge
            v2.CenterCrop(self.img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def _select_rgb_pipeline(self, is_global: bool, view_index: int = 0, synced: bool = False):
        if not self.use_official_dino_augs:
            return self.rgb_aug_no_flip if synced else self.aug_pipeline
        if is_global:
            return self.dino_global_pipeline_1 if (view_index % 2 == 0) else self.dino_global_pipeline_2
        return self.dino_local_pipeline

    def _apply_rgb_post_aug(self, img, *, is_global: bool, view_index: int = 0) -> torch.Tensor:
        return self._select_rgb_pipeline(is_global=is_global, view_index=view_index, synced=True)(img)

    def _apply_aug(self, img, rrc_transform, pipeline, return_params=False, is_global=True, view_index=0):
        """Helper to apply RRC + Pipeline and optionally return crop params."""
        # Get params from RRC without applying it yet
        i, j, h, w = rrc_transform.get_params(img, rrc_transform.scale, rrc_transform.ratio)
        
        # Apply Resized Crop (Crop + Resize to target size)
        # We must use resized_crop to ensure the output tensor always matches rrc_transform.size
        # This fixes the "stack expects equal size" error.
        img_cropped = v2.functional.resized_crop(img, i, j, h, w, rrc_transform.size)
        
        # Apply rest of pipeline
        if self.use_official_dino_augs:
            if np.random.random() < 0.5:
                img_cropped = v2.functional.horizontal_flip(img_cropped)
            pipeline = self._select_rgb_pipeline(is_global=is_global, view_index=view_index, synced=False)
        out = pipeline(img_cropped)
        
        if return_params:
            return out, (i, j, h, w)
        return out

    def _apply_test_aug(self, img, return_params=False):
        """Helper to apply deterministic Test transform (Resize+CenterCrop) and return params."""
        # 1. Resize logic: Scale smallest edge to self.img_size
        orig_w, orig_h = img.size
        scale = self.img_size / min(orig_w, orig_h)
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        
        # 2. CenterCrop logic
        crop_h = self.img_size
        crop_w = self.img_size
        
        # Calculate crop coordinates in the RESIZED image space
        # (top, left, height, width)
        top = (new_h - crop_h) // 2
        left = (new_w - crop_w) // 2
        
        # For spatial labels, we need these coordinates in the ORIGINAL image space (approx)
        # Or we map everything to "crop space".
        # LeJEPA expects params to be (i, j, h, w) in original pixels?
        # Actually standard RRC returns params in original pixels.
        # But Resize+CenterCrop is two steps.
        # Effective crop in original pixels:
        # The center crop corresponds to a region in original image.
        # center_original = (orig_h/2, orig_w/2)
        # crop_size_original_h = crop_h / scale
        # crop_size_original_w = crop_w / scale
        # So:
        h_orig = int(crop_h / scale)
        w_orig = int(crop_w / scale)
        i_orig = (orig_h - h_orig) // 2
        j_orig = (orig_w - w_orig) // 2
        
        # Apply transforms using standard library for correctness on pixels
        # But we use our manually computed crop for params
        # Wait, if we use standard Resize+CenterCrop, we rely on PIL/Torchvision match
        
        # Let's simple apply the pipeline manually to match:    
        img_resized = v2.functional.resize(img, [new_h, new_w])
        img_cropped = v2.functional.center_crop(img_resized, [crop_h, crop_w])
        out = self.test_pipeline(img_cropped)
        
        if return_params:
            # Return params in ORIGINAL image coordinates (i, j, h, w)
            return out, (i_orig, j_orig, h_orig, w_orig)
        return out
    
    def _get_global_labels(self, pair: Dict) -> Dict:
        """Get global labels invariant to cropping."""
        scene_label = self.scene_labels[pair["scene_id"]]
        cam_label = self.cam_to_idx.get(pair["camera_name"], 0)
        loc_prefix = pair["scene_id"].split("-")[0]
        loc_label = self.LOCATION_MAP.get(loc_prefix, 0)
        
        return {
            "scene": scene_label,
            "camera": cam_label,
            "location": loc_label,
        }

    def _get_spatial_labels(self, pair: Dict, crop_rect: Tuple[int, int, int, int] = None) -> Dict:
        """
        Get labels that depend on the specific image crop.
        
        Args:
            pair: Sample metadata
            crop_rect: (i, j, h, w) tuple defining the crop in original image coordinates.
                       If None, assumes full image.
        """
        sample_token = pair["sample_token"]
        # Use augmented annotations if available (from LiDAR augmentation pipeline)
        if '_augmented_annotations' in pair:
            annotations = pair['_augmented_annotations']
        elif hasattr(self, 'annotation_store') and self.annotation_store is not None:
            annotations = self.annotation_store.get_annotations(sample_token)
        else:
            annotations = self.sample_annotations.get(sample_token, [])
        
        num_cars = 0
        num_peds = 0
        box_2d_list = []  # List of (u_min, v_min, u_max, v_max) in crop coordinates
        
        # Get calibration and ego pose for 3D->2D projection (use compact store)
        cam_calib_token = pair.get("cam_calib_token")
        cam_ego_token = pair.get("cam_ego_token")
        if hasattr(self, 'calib_store') and self.calib_store is not None:
            cam_calib = self.calib_store.get_calib(cam_calib_token) if cam_calib_token else {}
            cam_ego = self.calib_store.get_ego(cam_ego_token) if cam_ego_token else {}
        else:
            cam_calib = self.calibrations.get(cam_calib_token, {}) if cam_calib_token else {}
            cam_ego = self.ego_poses.get(cam_ego_token, {}) if cam_ego_token else {}
        intrinsic = cam_calib.get("intrinsic")
        
        # Standard NuScenes image size
        img_w, img_h = 1600, 900
        
        # Crop parameters
        if crop_rect:
            ci, cj, ch, cw = crop_rect
        else:
            ci, cj, ch, cw = 0, 0, img_h, img_w
        
        # Use module-level numpy quaternion conversion (faster than scipy)
        quat_to_rot = quat_to_rot_numpy
        
        # Helper: Get 8 corners of a 3D bounding box
        def get_box_corners(center, size, rotation):
            """
            center: [x, y, z] in world frame
            size: [width, length, height] (x, y, z extents)
            rotation: quaternion [w, x, y, z]
            Returns: 8 corners in world frame
            """
            w, l, h = size[0] / 2, size[1] / 2, size[2] / 2
            
            # Corners in local frame (centered at origin)
            corners_local = np.array([
                [-w, -l, -h], [-w, -l,  h], [-w,  l, -h], [-w,  l,  h],
                [ w, -l, -h], [ w, -l,  h], [ w,  l, -h], [ w,  l,  h],
            ])
            
            # Rotate corners
            rot_matrix = quat_to_rot(rotation)
            corners_world = (rot_matrix @ corners_local.T).T + center
            return corners_world
            
        for annot in annotations:
            inst_token = annot.get("instance_token")
            
            # Need full 3D box info for projection
            if (intrinsic is not None and 
                "translation" in annot and "size" in annot and "rotation" in annot and
                "rotation" in cam_calib and "translation" in cam_calib and
                "rotation" in cam_ego and "translation" in cam_ego):
                try:
                    center = np.array(annot["translation"])
                    size = np.array(annot["size"])
                    rotation = np.array(annot["rotation"])
                    
                    # Get 8 corners in world frame
                    corners_world = get_box_corners(center, size, rotation)
                    
                    # Transform all corners: World -> Ego -> Camera
                    ego_R = quat_to_rot(cam_ego["rotation"])
                    cam_R = quat_to_rot(cam_calib["rotation"])
                    
                    corners_2d = []
                    for corner in corners_world:
                        # World -> Ego
                        point_ego = ego_R.T @ (corner - np.array(cam_ego["translation"]))
                        # Ego -> Camera
                        point_cam = cam_R.T @ (point_ego - np.array(cam_calib["translation"]))
                        
                        # Skip if behind camera
                        if point_cam[2] <= 0:
                            continue
                        
                        # Project to 2D
                        point_2d = intrinsic @ point_cam
                        u = point_2d[0] / point_2d[2]
                        v = point_2d[1] / point_2d[2]
                        corners_2d.append([u, v])
                    
                    if len(corners_2d) < 4:  # Need at least 4 corners visible
                        continue
                    
                    corners_2d = np.array(corners_2d)
                    
                    # Compute 2D bounding box
                    u_min, v_min = corners_2d.min(axis=0)
                    u_max, v_max = corners_2d.max(axis=0)
                    
                    # Clip to crop bounds
                    u_min = max(u_min, cj)
                    v_min = max(v_min, ci)
                    u_max = min(u_max, cj + cw)
                    v_max = min(v_max, ci + ch)
                    
                    # Check if box overlaps with crop
                    if u_max > u_min and v_max > v_min:
                        # Determine object class
                        obj_class = "other"
                        if inst_token:
                            # Use compact annotation store if available
                            if hasattr(self, 'annotation_store') and self.annotation_store is not None:
                                cat_name = self.annotation_store.get_category(inst_token)
                            else:
                                cat_token = self.instance_to_category.get(inst_token, "")
                                cat_name = self.category_names.get(cat_token, "")
                            if "car" in cat_name.lower() or "truck" in cat_name.lower() or "bus" in cat_name.lower():
                                obj_class = "car"
                                num_cars += 1
                            elif "pedestrian" in cat_name.lower() or "person" in cat_name.lower():
                                obj_class = "ped"
                                num_peds += 1
                        
                        # Convert to crop-relative coordinates and store with class
                        box_2d_list.append({
                            "box": [u_min - cj, v_min - ci, u_max - cj, v_max - ci],
                            "class": obj_class
                        })
                                
                except Exception:
                    pass  # Skip problematic annotations
        
        # Compute 8x8 grid occupancy using bounding boxes (matches depth_grid resolution)
        grid_occupancy = np.zeros(64, dtype=np.float32)
        grid_occupancy_car = np.zeros(64, dtype=np.float32)
        grid_occupancy_ped = np.zeros(64, dtype=np.float32)
        cell_w = cw / 8
        cell_h = ch / 8
        
        for box_info in box_2d_list:
            u_min, v_min, u_max, v_max = box_info["box"]
            obj_class = box_info["class"]
            
            # Find all grid cells that this box overlaps
            col_start = max(0, int(u_min / cell_w))
            col_end = min(7, int(u_max / cell_w))
            row_start = max(0, int(v_min / cell_h))
            row_end = min(7, int(v_max / cell_h))
            
            for row in range(row_start, row_end + 1):
                for col in range(col_start, col_end + 1):
                    cell_idx = row * 8 + col
                    grid_occupancy[cell_idx] = 1.0
                    if obj_class == "car":
                        grid_occupancy_car[cell_idx] = 1.0
                    elif obj_class == "ped":
                        grid_occupancy_ped[cell_idx] = 1.0
        
        return {
            "num_cars": num_cars,
            "num_pedestrians": num_peds,
            "num_objects": num_cars + num_peds,
            "depth_grid": np.zeros(64, dtype=np.float32), 
            "depth_grid_mask": np.zeros(64, dtype=np.float32),
            "depth_grid_hr": np.zeros(3136, dtype=np.float32),
            "depth_grid_mask_hr": np.zeros(3136, dtype=np.float32),
            "mean_depth": 0.0,
            "has_boxes": len(box_2d_list) > 0,
            "grid_occupancy": grid_occupancy,
            "grid_occupancy_car": grid_occupancy_car,
            "grid_occupancy_ped": grid_occupancy_ped,
        }
    
    def _get_det_seg_labels(self, pair: Dict, crop_rect=None, lidar_points=None, depth_map=None) -> Dict:
        """Compute detection + segmentation labels for the clean probe view.

        Returns a dict with:
          gt_classes, gt_centers, gt_sizes, gt_orientations, gt_mask,
          gt_centers_2d, seg_map   (all numpy arrays)
        or empty/zero arrays when calibration data is unavailable.
        """
        from src.detection_labels import (
            compute_bbox_labels,
            compute_seg_map,
            compute_center_crop_region,
            load_lidarseg_bin,
            LIDARSEG_TO_SIMPLIFIED,
        )

        max_objects = 50
        empty = {
            "gt_classes":      np.full(max_objects, -1, dtype=np.int64),
            "gt_centers":      np.zeros((max_objects, 3), dtype=np.float32),
            "gt_sizes":        np.zeros((max_objects, 3), dtype=np.float32),
            "gt_orientations": np.zeros((max_objects, 2), dtype=np.float32),
            "gt_mask":         np.zeros(max_objects, dtype=np.float32),
            "gt_centers_2d":   np.zeros((max_objects, 2), dtype=np.float32),
            "seg_map":         np.zeros((224, 224), dtype=np.int64),
        }

        sample_token = pair["sample_token"]

        if self.precomputed_labels_path is not None:
            from src.detection_labels import load_det_seg_labels

            required_keys = [
                "gt_classes",
                "gt_centers",
                "gt_sizes",
                "gt_orientations",
                "gt_mask",
                "gt_centers_2d",
            ] if self.det_seg_label_mode == "bbox_only" else None
            cached_data = load_det_seg_labels(
                sample_token=sample_token,
                cam_name=pair["camera_name"],
                cache_dir=self.precomputed_labels_path,
                has_shards=self.precomputed_labels_has_shards,
                required_keys=required_keys,
            )
            if cached_data is not None:
                seg_map = cached_data.get("seg_map")
                if seg_map is None:
                    seg_map = np.zeros((224, 224), dtype=np.int64)
                return {
                    "gt_classes": cached_data.get("gt_classes", empty["gt_classes"]),
                    "gt_centers": cached_data.get("gt_centers", empty["gt_centers"]),
                    "gt_sizes": cached_data.get("gt_sizes", empty["gt_sizes"]),
                    "gt_orientations": cached_data.get("gt_orientations", empty["gt_orientations"]),
                    "gt_mask": cached_data.get("gt_mask", empty["gt_mask"]),
                    "gt_centers_2d": cached_data.get("gt_centers_2d", empty["gt_centers_2d"]),
                    "seg_map": seg_map,
                    "has_seg_map": bool(np.asarray(seg_map).sum() > 0),
                }

        # ── calibration ─────────────────────────────────────────────
        cam_calib_token  = pair.get("cam_calib_token")
        cam_ego_token    = pair.get("cam_ego_token")
        lidar_calib_token = pair.get("lidar_calib_token")
        lidar_ego_token   = pair.get("lidar_ego_token")

        if hasattr(self, 'calib_store') and self.calib_store is not None:
            cam_calib  = self.calib_store.get_calib(cam_calib_token)  if cam_calib_token  else {}
            cam_ego    = self.calib_store.get_ego(cam_ego_token)      if cam_ego_token    else {}
            lidar_calib = self.calib_store.get_calib(lidar_calib_token) if lidar_calib_token else {}
            lidar_ego   = self.calib_store.get_ego(lidar_ego_token)    if lidar_ego_token   else {}
        else:
            cam_calib   = self.calibrations.get(cam_calib_token, {})   if cam_calib_token  else {}
            cam_ego     = self.ego_poses.get(cam_ego_token, {})       if cam_ego_token    else {}
            lidar_calib = self.calibrations.get(lidar_calib_token, {}) if lidar_calib_token else {}
            lidar_ego   = self.ego_poses.get(lidar_ego_token, {})     if lidar_ego_token   else {}

        intrinsic = cam_calib.get("intrinsic")
        if intrinsic is None:
            return empty

        # ── crop region (probe view = center crop) ──────────────────
        if crop_rect is None:
            crop_rect = compute_center_crop_region(img_hw=(900, 1600), target_size=224)

        # ── annotations ─────────────────────────────────────────────
        # Use augmented annotations if available (from LiDAR augmentation pipeline)
        if '_augmented_annotations' in pair:
            annotations = pair['_augmented_annotations']
            def get_cat_fn(inst_token):
                # For pasted objects, category is stored directly
                for ann in annotations:
                    if ann.get('instance_token') == inst_token:
                        if 'category_name' in ann:
                            return ann['category_name']
                # Fall back to standard lookup
                if hasattr(self, 'annotation_store') and self.annotation_store is not None:
                    return self.annotation_store.get_category(inst_token)
                else:
                    cat_token = self.instance_to_category.get(inst_token, "")
                    return self.category_names.get(cat_token, "")
        elif hasattr(self, 'annotation_store') and self.annotation_store is not None:
            annotations = self.annotation_store.get_annotations(sample_token)
            def get_cat_fn(inst_token):
                return self.annotation_store.get_category(inst_token)
        else:
            annotations = self.sample_annotations.get(sample_token, [])
            def get_cat_fn(inst_token):
                cat_token = self.instance_to_category.get(inst_token, "")
                return self.category_names.get(cat_token, "")

        bbox_labels = compute_bbox_labels(
            annotations=annotations,
            cam_calib=cam_calib,
            cam_ego=cam_ego,
            get_category_fn=get_cat_fn,
            max_objects=max_objects,
            img_hw=(900, 1600),
            crop_region=crop_rect,
        )

        # ── semantic segmentation map ───────────────────────────────
        seg_map = np.zeros((224, 224), dtype=np.int64)

        # For fine-tuning that only needs detection targets, skip expensive seg generation.
        if self.det_seg_label_mode == "bbox_only":
            return {
                "gt_classes":      bbox_labels["gt_classes"],
                "gt_centers":      bbox_labels["gt_centers"],
                "gt_sizes":        bbox_labels["gt_sizes"],
                "gt_orientations": bbox_labels["gt_orientations"],
                "gt_mask":         bbox_labels["gt_mask"],
                "gt_centers_2d":   bbox_labels.get("gt_centers_2d", np.zeros((max_objects, 2), dtype=np.float32)),
                "seg_map":         seg_map,
            }

        lidar_sd_token = pair.get("lidar_sd_token", "")
        lidarseg_labels = None
        if '_augmented_lidarseg_labels' in pair:
            try:
                lidarseg_labels = np.asarray(pair['_augmented_lidarseg_labels'], dtype=np.uint8)
            except Exception:
                lidarseg_labels = None
        if lidarseg_labels is None:
            lidarseg_dir = None
            for candidate in [
                Path(self.dataroot) / "lidarseg" / "v1.0-trainval",
                Path(self.dataroot) / "lidarseg" / "v1.0-mini",
            ]:
                if candidate.exists():
                    lidarseg_dir = candidate
                    break
            if lidarseg_dir and lidar_sd_token:
                lidarseg_labels = load_lidarseg_bin(lidarseg_dir, lidar_sd_token)

        if lidarseg_labels is not None:
            # We need lidar_points + full lidar-to-cam transform
            pts = lidar_points
            if pts is None:
                # Try loading lidar points
                if self.has_data_shards:
                    pts = self._load_lidar_from_shard(pair)
                if pts is None:
                    try:
                        pts = load_lidar_bin(str(pair["lidar"]))
                    except Exception:
                        pts = None

            if pts is not None and len(lidarseg_labels) == len(pts):
                try:
                    # Build lidar-to-camera transform
                    lidar_to_ego = np.eye(4)
                    lidar_to_ego[:3, :3] = quat_to_rot_numpy(lidar_calib["rotation"])
                    lidar_to_ego[:3, 3]  = lidar_calib["translation"]

                    lidar_ego_to_world = np.eye(4)
                    lidar_ego_to_world[:3, :3] = quat_to_rot_numpy(lidar_ego["rotation"])
                    lidar_ego_to_world[:3, 3]  = lidar_ego["translation"]

                    world_to_cam_ego = np.eye(4)
                    cam_ego_rot = quat_to_rot_numpy(cam_ego["rotation"])
                    world_to_cam_ego[:3, :3] = cam_ego_rot.T
                    world_to_cam_ego[:3, 3]  = -cam_ego_rot.T @ np.asarray(cam_ego["translation"])

                    ego_to_cam = np.eye(4)
                    cam_rot = quat_to_rot_numpy(cam_calib["rotation"])
                    ego_to_cam[:3, :3] = cam_rot.T
                    ego_to_cam[:3, 3]  = -cam_rot.T @ np.asarray(cam_calib["translation"])

                    lidar_to_cam = ego_to_cam @ world_to_cam_ego @ lidar_ego_to_world @ lidar_to_ego

                    seg_map = compute_seg_map(
                        lidar_points=pts,
                        lidarseg_labels=lidarseg_labels,
                        intrinsic=np.asarray(intrinsic, dtype=np.float64),
                        lidar_to_cam=lidar_to_cam.astype(np.float64),
                        img_hw=(900, 1600),
                        target_hw=(224, 224),
                        crop_region=crop_rect,
                        fill_max_dist=5,
                    )
                except Exception:
                    pass  # Leave seg_map as zeros

        result = {
            "gt_classes":      bbox_labels["gt_classes"],
            "gt_centers":      bbox_labels["gt_centers"],
            "gt_sizes":        bbox_labels["gt_sizes"],
            "gt_orientations": bbox_labels["gt_orientations"],
            "gt_mask":         bbox_labels["gt_mask"],
            "gt_centers_2d":   bbox_labels.get("gt_centers_2d", np.zeros((max_objects, 2), dtype=np.float32)),
            "seg_map":         seg_map,
            "has_seg_map":     bool(seg_map.sum() > 0),
        }
        return result

    def _resize_label_map(self, label_map: np.ndarray, target_size: int) -> np.ndarray:
        if label_map.shape == (target_size, target_size):
            return label_map.astype(np.int64, copy=False)
        pil_map = Image.fromarray(label_map.astype(np.int32), mode='I')
        pil_map = pil_map.resize((target_size, target_size), Image.NEAREST)
        return np.array(pil_map, dtype=np.int64)

    def _get_multimae_view_label_targets(
        self,
        pair: Dict,
        global_crop_rects: List[Tuple[int, int, int, int]],
        probe_crop_rect: Optional[Tuple[int, int, int, int]],
        lidar_points: Optional[np.ndarray],
    ) -> Dict:
        if self.det_seg_label_mode == "bbox_only":
            return {}

        targets: Dict[str, np.ndarray] = {}
        if global_crop_rects:
            global_seg_maps = []
            global_has_seg = []
            for crop_rect in global_crop_rects:
                seg_info = self._get_det_seg_labels(pair, crop_rect=crop_rect, lidar_points=lidar_points)
                global_seg_maps.append(self._resize_label_map(seg_info["seg_map"], self.img_size))
                global_has_seg.append(bool(seg_info.get("has_seg_map", False)))
            targets["multimae_global_seg_map"] = np.stack(global_seg_maps).astype(np.int64)
            targets["multimae_global_has_seg_map"] = np.asarray(global_has_seg, dtype=np.bool_)

        if probe_crop_rect is not None:
            seg_info = self._get_det_seg_labels(pair, crop_rect=probe_crop_rect, lidar_points=lidar_points)
            targets["multimae_probe_seg_map"] = self._resize_label_map(seg_info["seg_map"], self.img_size)
            targets["multimae_probe_has_seg_map"] = np.bool_(seg_info.get("has_seg_map", False))

        return targets

    def _get_labels(self, pair: Dict) -> Dict:
        """Legacy support wrapper."""
        g = self._get_global_labels(pair)
        s = self._get_spatial_labels(pair, None) # Full image
        g.update(s)
        return g
    
    def _load_image_from_shard(self, pair: Dict) -> Optional[Image.Image]:
        """Load camera image from data shard zip. Returns None if not found."""
        try:
            token = pair['sample_token']
            shard_id = get_shard_id(token)
            shard_path = self._data_shard_index.get(shard_id)
            
            if shard_path is None:
                return None
                
            cam_path = Path(pair['camera'])
            arcname = cam_path.relative_to(self.dataroot)
            
            # Use cached handle
            zf = self._get_zip_handle(shard_path)
            with zf.open(str(arcname)) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                return img
        except Exception:
            return None
    
    def _load_lidar_from_shard(self, pair: Dict) -> Optional[np.ndarray]:
        """Load LiDAR points from data shard zip. Returns None if not found."""
        try:
            token = pair['sample_token']
            shard_id = get_shard_id(token)
            shard_path = self._data_shard_index.get(shard_id)
            
            if shard_path is None:
                return None
                
            lidar_path = Path(pair['lidar'])
            arcname = lidar_path.relative_to(self.dataroot)
            
            # Use cached handle
            zf = self._get_zip_handle(shard_path)
            with zf.open(str(arcname)) as f:
                points = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 5)
                return points
        except Exception:
            return None
    
    def __getitem__(self, idx: int):
        # Use compact pair accessor
        pair = self._get_pair(idx) if hasattr(self, '_n_pairs') else self.pairs[idx]
        labels = self._get_global_labels(pair) # Global labels first, spatial added per-view later
        
        # Load camera image - try shard first, then individual file
        img = None
        if self.has_data_shards:
            img = self._load_image_from_shard(pair)
        
        if img is None:
            # Fallback to individual file
            try:
                img = Image.open(pair["camera"]).convert("RGB")
            except (OSError, Image.UnidentifiedImageError) as e:
                print(f"Warning: SKIPPING Corrupt Source Image: {pair['camera']}. Using black replacement.")
                # Use standard NuScenes resolution 1600x900 black image
                img = Image.new('RGB', (1600, 900), color='black')
        
        # Apply modality dropout during training
        drop_camera = False
        drop_lidar = False
        if self.modality_dropout > 0 and self.V > 1:  # only during training
            if np.random.random() < self.modality_dropout:
                if np.random.random() < 0.5:
                    drop_camera = True
                else:
                    drop_lidar = True
        
        # ---------------------------
        # 1. Process Camera Views
        # ---------------------------
        # NOTE: For lidar_mode="depth" we rebuild synchronized RGB+depth views later
        # in the depth branch. Skipping this generic camera path avoids duplicate
        # augmentations + spatial-label computation.
        if self.lidar_mode != "depth":
            cam_views_global = []
            cam_views_local = []
            spatial_labels_per_view = []

            # Determine views (global vs local)
            if self.V > 1 or self.local_crops_number > 0:
                # Global crops (usually 2)
                for global_idx in range(self.V):
                    # Standard JEPA: Target view (global)
                    view, params = self.cam_global_aug(img, return_params=True, view_index=global_idx)
                    cam_views_global.append(view)
                    if not self.encoder_only_labels:
                        labels_spatial = self._get_spatial_labels(pair, params)
                        spatial_labels_per_view.append(labels_spatial) # Index matches cam_views order

                # Add clean probe view (Global view V+1)
                # This is non-augmented (Resize+CenterCrop) for evaluation
                if self.include_probe_view:
                    view_clean, params_clean = self.cam_test(img, return_params=True)
                    cam_views_global.append(view_clean)
                    if not self.encoder_only_labels:
                        labels_clean = self._get_spatial_labels(pair, params_clean)
                        spatial_labels_per_view.append(labels_clean)

                # Local crops (usually 4-6)
                for local_idx in range(self.local_crops_number):
                    view, params = self.cam_local_aug(img, return_params=True, view_index=local_idx)
                    cam_views_local.append(view)
                    if not self.encoder_only_labels:
                        labels_spatial = self._get_spatial_labels(pair, params)
                        spatial_labels_per_view.append(labels_spatial)

                if cam_views_global:
                    cam_views_global = torch.stack(cam_views_global)
                if cam_views_local:
                    cam_views_local = torch.stack(cam_views_local)
            else:
                # Test time / V=1: Full image Resize + Crop
                view, params = self.cam_test(img, return_params=True)
                cam_views_global = view.unsqueeze(0)
                cam_views_local = torch.empty(0)
                if not self.encoder_only_labels:
                    labels_spatial = self._get_spatial_labels(pair, params)
                    spatial_labels_per_view.append(labels_spatial)

            # Modality dropout for camera
            if drop_camera:
                if isinstance(cam_views_global, torch.Tensor):
                    cam_views_global = torch.zeros_like(cam_views_global)
                if isinstance(cam_views_local, torch.Tensor) and cam_views_local.numel() > 0:
                    cam_views_local = torch.zeros_like(cam_views_local)

            cam_views = {
                'global': cam_views_global,
                'local': cam_views_local
            }
        else:
            # Placeholder; depth branch below creates and returns synchronized views.
            cam_views = {
                'global': torch.empty(0),
                'local': torch.empty(0),
            }
        
        # ---------------------------
        # 2. LiDAR / Depth Processing
        # ---------------------------
        # Check cache
        filename_base = f"{pair['sample_token']}_{pair['camera_name']}"
        cache_path = self.cache_dir / f"{filename_base}.png"
        labels_cache_path = self.cache_dir / f"{filename_base}_labels_v3.npz"
        
        need_computation = True
        depth_map = None
        inputs_need_depth = (self.lidar_mode == "depth")

        # When LiDAR augmentations are active, we MUST recompute depth/labels from
        # the augmented point cloud — cached artifacts reflect un-augmented data.
        _aug_active = (
            self.split == "train" and (
                getattr(self, "lidar_aug_enabled", False) or getattr(self, "copy_paste_enabled", False)
            )
        )

        # Strategy 1: Check Sharded Zip Cache (High Efficiency)
        # ---------------------------------------------------------
        if self.has_shards and not _aug_active:
            shard_id = get_shard_id(pair['sample_token'])
            shard_path = self._cache_shard_index.get(shard_id)
            if shard_path is not None:
                try:
                    # Use cached handle for the cache shard too
                    zf = self._get_zip_handle(shard_path)
                    
                    try:
                        # Load Labels
                        with zf.open(f"{filename_base}_labels_v3.npz") as f:
                            cached_labels = np.load(io.BytesIO(f.read()))
                            labels["mean_depth"] = float(cached_labels["mean_depth"])
                            labels["depth_grid"] = cached_labels["depth_grid"]
                            labels["depth_grid_mask"] = cached_labels["depth_grid_mask"]
                        
                        # Load Depth Map (if needed)
                        if inputs_need_depth:
                            with zf.open(f"{filename_base}.png") as f:
                                depth_int = np.array(Image.open(io.BytesIO(f.read()))) 
                                depth_map = depth_int.astype(np.float32) / 256.0
                        
                        need_computation = False # Success!
                    except KeyError:
                        pass # File not in this zip
                except Exception:
                    pass

        # Strategy 2: Check Individual File Cache (Legacy/Online Fallback)
        # ----------------------------------------------------------------------------
        if need_computation and not _aug_active:
            labels_cached = labels_cache_path.exists()
            depth_cached = cache_path.exists()
            can_skip = labels_cached if not inputs_need_depth else (labels_cached and depth_cached)
            
            if can_skip:
                try:
                    cached_labels = np.load(labels_cache_path)
                    labels["mean_depth"] = float(cached_labels["mean_depth"])
                    labels["depth_grid"] = cached_labels["depth_grid"]
                    labels["depth_grid_mask"] = cached_labels["depth_grid_mask"]
                    
                    if inputs_need_depth:
                        depth_int = np.array(Image.open(cache_path))
                        depth_map = depth_int.astype(np.float32) / 256.0
                    
                    need_computation = False
                except Exception:
                    need_computation = True

        # Phase 3 Lazy Load: ONLY load LiDAR if we actually need to compute or use points
        # ------------------------------------------------------------------------------
        if need_computation or self.lidar_mode != "depth":
            # Try shard first, then individual file
            lidar_points = None
            if self.has_data_shards:
                lidar_points = self._load_lidar_from_shard(pair)
            if lidar_points is None:
                lidar_points = load_lidar_bin(str(pair["lidar"]))
        else:
            lidar_points = None # Will not be used since depth_map is in cache

        # ── LiDAR Augmentations (scene-level + copy-paste) ──────────
        # Applied BEFORE any projection so that depth maps, range images,
        # spatial labels, and det/seg labels all reflect the augmented scene.
        if lidar_points is not None and self.split == "train":
            _aug_annotations = None  # will hold augmented annotations for label recompute
            _aug_lidarseg = None

            # If we may compute seg_map later, keep lidarseg labels aligned with the augmented point cloud.
            if (not self.legacy_mode) and (self.det_seg_label_mode != "bbox_only"):
                try:
                    lidar_sd_token = pair.get("lidar_sd_token", "")
                    if lidar_sd_token:
                        from src.detection_labels import load_lidarseg_bin
                        lidarseg_dir = None
                        for candidate in [
                            Path(self.dataroot) / "lidarseg" / "v1.0-trainval",
                            Path(self.dataroot) / "lidarseg" / "v1.0-mini",
                        ]:
                            if candidate.exists():
                                lidarseg_dir = candidate
                                break
                        if lidarseg_dir is not None:
                            _aug_lidarseg = load_lidarseg_bin(lidarseg_dir, lidar_sd_token)
                            if _aug_lidarseg is not None and lidar_points is not None:
                                if len(_aug_lidarseg) != len(lidar_points):
                                    print(f"DEBUG: lidarseg length {len(_aug_lidarseg)} != lidar_points length {len(lidar_points)}")
                                    _aug_lidarseg = None
                except Exception as e:
                    print(f"FAILED TO LOAD LIDAR_SEG: {e}")
                    _aug_lidarseg = None

            def _quat_to_rot(q):
                q = np.asarray(q, dtype=np.float64)
                w, x, y, z = q
                x2, y2, z2 = x * x, y * y, z * z
                xy, xz, yz = x * y, x * z, y * z
                wx, wy, wz = w * x, w * y, w * z
                return np.array(
                    [
                        [1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)],
                        [2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)],
                        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)],
                    ],
                    dtype=np.float64,
                )

            def _rot_to_quat(R):
                tr = float(R[0, 0] + R[1, 1] + R[2, 2])
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

            def _world_lidar_xforms(pair_local):
                try:
                    if not hasattr(self, 'calib_store'):
                        return None, None
                    lc = self.calib_store.get_calib(pair_local.get('lidar_calib_token'))
                    le = self.calib_store.get_ego(pair_local.get('lidar_ego_token'))

                    lidar_to_ego = np.eye(4, dtype=np.float64)
                    lidar_to_ego[:3, :3] = _quat_to_rot(lc['rotation'])
                    lidar_to_ego[:3, 3] = np.asarray(lc['translation'], dtype=np.float64)

                    ego_to_world = np.eye(4, dtype=np.float64)
                    ego_to_world[:3, :3] = _quat_to_rot(le['rotation'])
                    ego_to_world[:3, 3] = np.asarray(le['translation'], dtype=np.float64)

                    lidar_to_world = ego_to_world @ lidar_to_ego
                    world_to_lidar = np.linalg.inv(lidar_to_world)
                    return lidar_to_world, world_to_lidar
                except Exception:
                    return None, None

            def _anns_world_to_lidar(anns_world, world_to_lidar):
                if world_to_lidar is None:
                    return anns_world
                out = []
                for a in anns_world:
                    a2 = dict(a)
                    tw = np.asarray(a.get('translation', [0, 0, 0]), dtype=np.float64)
                    qw = np.asarray(a.get('rotation', [1, 0, 0, 0]), dtype=np.float64)
                    tl = (world_to_lidar @ np.append(tw, 1.0))[:3]
                    Rw = _quat_to_rot(qw)
                    Rl = world_to_lidar[:3, :3] @ Rw
                    a2['translation'] = tl
                    a2['rotation'] = _rot_to_quat(Rl)
                    out.append(a2)
                return out

            def _anns_lidar_to_world(anns_lidar, lidar_to_world):
                if lidar_to_world is None:
                    return anns_lidar
                out = []
                for a in anns_lidar:
                    a2 = dict(a)
                    tl = np.asarray(a.get('translation', [0, 0, 0]), dtype=np.float64)
                    ql = np.asarray(a.get('rotation', [1, 0, 0, 0]), dtype=np.float64)
                    tw = (lidar_to_world @ np.append(tl, 1.0))[:3]
                    Rl = _quat_to_rot(ql)
                    Rw = lidar_to_world[:3, :3] @ Rl
                    a2['translation'] = tw
                    a2['rotation'] = _rot_to_quat(Rw)
                    out.append(a2)
                return out

            lidar_to_world, world_to_lidar = _world_lidar_xforms(pair)

            # 1) Scene-level augmentation (scaling, rotation, translation, jitter)
            if self.lidar_aug_enabled and self._scene_augmentor is not None:
                # Get annotations in LiDAR frame for consistent augmentation
                sample_token = pair["sample_token"]
                if _aug_annotations is None:
                    if hasattr(self, 'annotation_store') and self.annotation_store is not None:
                        _aug_annotations = self.annotation_store.get_annotations(sample_token)
                    else:
                        _aug_annotations = self.sample_annotations.get(sample_token, [])
                    import copy as _copy
                    _aug_annotations = _copy.deepcopy(_aug_annotations)
                # Alignment is critical for fusion architectures (B, C) and depth/points modes (D, A)
                # Any non-rigid scene augmentation (rotation, flipping) breaks sub-pixel alignment 
                # unless the camera images are also warped/flipped accordingly.
                _alignment_safe = self.lidar_mode in ("depth", "aligned_points", "range")
                _anns_lidar = _anns_world_to_lidar(_aug_annotations, world_to_lidar)
                
                if _aug_lidarseg is not None:
                    lidar_points, _anns_lidar, _aug_lidarseg = self._scene_augmentor(
                        lidar_points, _anns_lidar, alignment_safe=_alignment_safe, lidarseg_labels=_aug_lidarseg)
                else:
                    lidar_points, _anns_lidar = self._scene_augmentor(
                        lidar_points, _anns_lidar, alignment_safe=_alignment_safe)
                        
                _aug_annotations = _anns_lidar_to_world(_anns_lidar, lidar_to_world)

            # 2) Copy-paste augmentation (paste objects from GT database)
            if self.copy_paste_enabled and self._copy_paste_augmentor is not None:
                sample_token = pair["sample_token"]
                if _aug_annotations is None:
                    if hasattr(self, 'annotation_store') and self.annotation_store is not None:
                        _aug_annotations = self.annotation_store.get_annotations(sample_token)
                    else:
                        _aug_annotations = self.sample_annotations.get(sample_token, [])
                    import copy as _copy
                    _aug_annotations = _copy.deepcopy(_aug_annotations)

                # Build lidar_to_cam for image paste (if we have calibration)
                _cp_intrinsic = None
                _cp_lidar_to_cam = None
                try:
                    _cct = pair.get("cam_calib_token")
                    _lct = pair.get("lidar_calib_token")
                    _cet = pair.get("cam_ego_token")
                    _let = pair.get("lidar_ego_token")
                    if _cct and _lct and _cet and _let and hasattr(self, 'calib_store'):
                        _cc = self.calib_store.get_calib(_cct)
                        _lc = self.calib_store.get_calib(_lct)
                        _ce = self.calib_store.get_ego(_cet)
                        _le = self.calib_store.get_ego(_let)
                        _cp_intrinsic = _cc.get("intrinsic")
                        if _cp_intrinsic is not None:
                            from src.detection_labels import compute_lidar_to_cam_transform
                            _cp_lidar_to_cam = compute_lidar_to_cam_transform(
                                _cc, _lc, _ce, _le)
                except Exception:
                    pass

                _anns_lidar = _anns_world_to_lidar(_aug_annotations, world_to_lidar)
                _cp_out = self._copy_paste_augmentor(
                    lidar_points,
                    _anns_lidar,
                    img,
                    _cp_intrinsic,
                    _cp_lidar_to_cam,
                    current_camera_name=pair.get("camera_name"),
                    lidarseg_labels=_aug_lidarseg,
                )
                if isinstance(_cp_out, (tuple, list)) and len(_cp_out) == 4:
                    lidar_points, _anns_lidar, img, _aug_lidarseg = _cp_out
                else:
                    lidar_points, _anns_lidar, img = _cp_out
                _aug_annotations = _anns_lidar_to_world(_anns_lidar, lidar_to_world)

            # Store augmented annotations so downstream label functions can use them
            if _aug_annotations is not None:
                pair = dict(pair)  # shallow copy to not mutate original
                pair['_augmented_annotations'] = _aug_annotations
                if _aug_lidarseg is not None:
                    pair['_augmented_lidarseg_labels'] = _aug_lidarseg

        if need_computation:
            # Slow path: Compute from scratch
            cam_calib_token = pair.get("cam_calib_token")
            lidar_calib_token = pair.get("lidar_calib_token")
            cam_ego_token = pair.get("cam_ego_token")
            lidar_ego_token = pair.get("lidar_ego_token")
            
            orig_w, orig_h = img.size
            # Check all required tokens are available (use compact store)
            tokens_valid = (
                cam_calib_token and cam_calib_token in self.calib_store.token_to_idx and
                lidar_calib_token and lidar_calib_token in self.calib_store.token_to_idx and
                cam_ego_token and cam_ego_token in self.calib_store.token_to_idx and
                lidar_ego_token and lidar_ego_token in self.calib_store.token_to_idx
            )
            
            if tokens_valid:
                cam_calib = self.calib_store.get_calib(cam_calib_token)
                # ... (rest of projection logic) ...
                
                lidar_calib = self.calib_store.get_calib(lidar_calib_token)
                cam_ego = self.calib_store.get_ego(cam_ego_token)
                lidar_ego = self.calib_store.get_ego(lidar_ego_token)
                intrinsic = cam_calib["intrinsic"]
                
                if intrinsic is not None:
                    # 1. LiDAR → Ego (at lidar time)
                    lidar_to_ego = np.eye(4)
                    lidar_to_ego[:3, :3] = quat_to_rot_numpy(lidar_calib["rotation"])
                    lidar_to_ego[:3, 3] = lidar_calib["translation"]
                    
                    # 2. Ego (lidar time) → World
                    lidar_ego_to_world = np.eye(4)
                    lidar_ego_to_world[:3, :3] = quat_to_rot_numpy(lidar_ego["rotation"])
                    lidar_ego_to_world[:3, 3] = lidar_ego["translation"]
                    
                    # 3. World → Ego (cam time)  (inverse transform)
                    world_to_cam_ego = np.eye(4)
                    cam_ego_rot = quat_to_rot_numpy(cam_ego["rotation"])
                    world_to_cam_ego[:3, :3] = cam_ego_rot.T
                    world_to_cam_ego[:3, 3] = -cam_ego_rot.T @ cam_ego["translation"]
                    
                    # 4. Ego (cam time) → Camera (inverse of camera extrinsic)
                    ego_to_cam = np.eye(4)
                    cam_rot = quat_to_rot_numpy(cam_calib["rotation"])
                    ego_to_cam[:3, :3] = cam_rot.T
                    ego_to_cam[:3, 3] = -cam_rot.T @ cam_calib["translation"]
                    
                    # Full transformation: LiDAR → Ego → World → Ego → Camera
                    lidar_to_cam = ego_to_cam @ world_to_cam_ego @ lidar_ego_to_world @ lidar_to_ego
                    
                    depth_map_fullres = lidar_to_depth_map_full(
                        lidar_points,
                        intrinsic=intrinsic,
                        lidar_to_cam_transform=lidar_to_cam.astype(np.float32),
                        img_size=(orig_h, orig_w),
                        max_depth=80.0
                    )
                    
                    # Compute Labels from FULL resolution
                    # Always save if we computed it (ensures consistency)
                    d_tensor_full = torch.from_numpy(depth_map_fullres).unsqueeze(0).unsqueeze(0)
                    valid_mask_full = (d_tensor_full > 0).float()
                    valid_count_full = valid_mask_full.sum()
                    
                    if valid_count_full > 0:
                        mean_depth_val = (d_tensor_full.sum() / valid_count_full).item()
                    else:
                        mean_depth_val = 0.0
                    
                    # 8x8 grid from full resolution
                    grid_sum = torch.nn.functional.adaptive_avg_pool2d(d_tensor_full, (8, 8)) * (d_tensor_full.shape[-2] * d_tensor_full.shape[-1] / 64)
                    grid_count = torch.nn.functional.adaptive_avg_pool2d(valid_mask_full, (8, 8)) * (d_tensor_full.shape[-2] * d_tensor_full.shape[-1] / 64)
                    
                    depth_grid_label = torch.zeros_like(grid_sum)
                    mask_nz = grid_count > 0
                    depth_grid_label[mask_nz] = grid_sum[mask_nz] / grid_count[mask_nz]
                    
                    # Convert from normalized [0,1] to actual Meters (max_depth=80m)
                    labels["mean_depth"] = mean_depth_val * 80.0  # In Meters
                    labels["depth_grid"] = (depth_grid_label * 80.0).flatten().numpy()  # In Meters
                    labels["depth_grid_mask"] = mask_nz.float().flatten().numpy()
                    
                    # Save labels cache (with race condition protection)
                    # Skip writing cache when augmentations are active to avoid polluting
                    if not _aug_active and not labels_cache_path.exists():
                        try:
                            np.savez_compressed(labels_cache_path, 
                                mean_depth=labels["mean_depth"],
                                depth_grid=labels["depth_grid"],
                                depth_grid_mask=labels["depth_grid_mask"])
                        except (PermissionError, OSError) as e:
                            print(f"Warning: Failed to write labels cache: {e}")
                            pass  # Another worker already writing
                            
                    # Downsample to cache resolution using max pooling (preserves sparse points)
                    # IMPORTANT: Preserve aspect ratio to ensure correct crop alignment
                    depth_t = torch.from_numpy(depth_map_fullres).unsqueeze(0).unsqueeze(0)
                    # Scale to cache_res on the shorter dimension, preserving aspect ratio
                    scale_factor = self.cache_res / min(orig_h, orig_w)
                    cache_h = round(orig_h * scale_factor)
                    cache_w = round(orig_w * scale_factor)
                    depth_small = torch.nn.functional.adaptive_max_pool2d(depth_t, (cache_h, cache_w))
                    depth_map = depth_small.squeeze().numpy()
                else:
                    # No valid intrinsic, create aspect-preserving zeros
                    scale_factor = self.cache_res / min(orig_h, orig_w)
                    cache_h = round(orig_h * scale_factor)
                    cache_w = round(orig_w * scale_factor)
                    depth_map = np.zeros((cache_h, cache_w), dtype=np.float32)
            else:
                # No valid tokens, create aspect-preserving zeros using default 900x1600
                scale_factor = self.cache_res / min(900, 1600)
                cache_h = round(900 * scale_factor)
                cache_w = round(1600 * scale_factor)
                depth_map = np.zeros((cache_h, cache_w), dtype=np.float32)
            
            # Save depth to cache (at cache resolution) - with race protection
            # Skip writing cache when augmentations are active
            if depth_map is not None and not _aug_active and not cache_path.exists():
                try:
                    depth_int = (depth_map * 256).astype(np.uint16)
                    Image.fromarray(depth_int).save(cache_path)
                except (PermissionError, OSError) as e:
                    print(f"Warning: Failed to write depth cache: {e}")
                    pass  # Another worker already writing
            elif depth_map is None:
                # Should not happen but handle it - use aspect-preserving zeros
                scale_factor = self.cache_res / min(900, 1600)
                cache_h = round(900 * scale_factor)
                cache_w = round(1600 * scale_factor)
                depth_map = np.zeros((cache_h, cache_w), dtype=np.float32)


        # 2. Process Input Modalities
        # ---------------------------

        # A. Range Image Mode
        if self.lidar_mode == "range":
            # Range images (360° panoramic, NOT aligned with camera)
            range_img = lidar_to_range_image(
                lidar_points, 
                H=self.range_size[0], 
                W=self.range_size[1]
            )
            range_tensor = torch.from_numpy(range_img).permute(2, 0, 1)
            
            range_transform = self.range_aug if self.V > 1 or self.local_crops_number > 0 else self.range_test
            
            range_views_global = []
            range_views_local = []
            
            # Global Views
            for _ in range(self.V):
                 range_views_global.append(range_transform(range_tensor))
                 
            # Add clean probe view (Global view V+1)
            # Use range_test (Resize only, no random crop)
            if self.include_probe_view:
                range_views_global.append(self.range_test(range_tensor))
            
            # Local Views - use smaller resolution for consistency
            range_local_transform = self.range_local_aug if self.V > 1 else range_transform
            for _ in range(self.local_crops_number):
                 range_views_local.append(range_local_transform(range_tensor))
            
            if drop_lidar:
                 range_views_global = [torch.zeros_like(x) for x in range_views_global]
                 range_views_local = [torch.zeros_like(x) for x in range_views_local]
                 
            range_views_global = torch.stack(range_views_global) if range_views_global else torch.empty(0)
            range_views_local = torch.stack(range_views_local) if range_views_local else torch.empty(0)
            
            range_views = {
                'global': range_views_global,
                'local': range_views_local
            }
            
            # For arch D with range mode, concatenate to make RGBD
            if self.arch == "D":
                # Take first channel of range as depth proxy
                # We need to concatenate global with global, local with local
                # Note: cam_views is now a dict
                
                rgbd_global = torch.cat([cam_views['global'], range_views['global'][:, 0:1]], dim=1)
                
                rgbd_local = torch.empty(0)
                if cam_views['local'].numel() > 0:
                     rgbd_local = torch.cat([cam_views['local'], range_views['local'][:, 0:1]], dim=1)
                
                final_views = {'global': rgbd_global, 'local': rgbd_local}
                
                # Fall back to full-image spatial labels
                if not self.encoder_only_labels:
                    sp = self._get_spatial_labels(pair, None)
                    labels.update(sp)
                    labels.update(self._get_det_seg_labels(pair))
                return final_views, torch.zeros(1), labels
            else:
                if not self.encoder_only_labels:
                    sp = self._get_spatial_labels(pair, None)
                    labels.update(sp)
            if not self.legacy_mode and not self.encoder_only_labels:
                labels.update(self._get_det_seg_labels(pair))
            return cam_views, range_views, labels
        
        # B. Depth Mode (Aligned)
        elif self.lidar_mode == "depth":
            # We MUST have depth_map here (computed/loaded in step 1)
            # Get original image dimensions for aspect ratio calculation
            orig_h_img, orig_w_img = img.size[1], img.size[0]  # PIL is (W, H)
            scale_factor = self.cache_res / min(orig_h_img, orig_w_img)
            expected_cache_h = round(orig_h_img * scale_factor)
            expected_cache_w = round(orig_w_img * scale_factor)
            
            if depth_map is None:
                 # Should theoretically not happen if logic is correct, but safety net:
                 depth_map = np.zeros((expected_cache_h, expected_cache_w), dtype=np.float32)

            # Convert depth to PIL for transforms
            depth_pil = Image.fromarray((depth_map * 255).astype(np.uint8))
            
            # Create views with aligned transforms
            depth_views = []
            rgb_views = []  # IMPORTANT: Collect synchronized RGB here, NOT using pre-generated cam_views
            spatial_labels_per_view = []  # RESET: Section 1 already appended labels for unsynced cam views; depth mode creates its own synced views
            rgbd_views = []
            multimae_global_crop_rects = []
            multimae_probe_crop_rect = None
            
            # Total iterations = Global V + Local N
            total_n = self.V + self.local_crops_number
            if self.V == 1 and self.local_crops_number == 0:
                total_n = 1
            
            for v_idx in range(total_n):
                crop_rect = None # Will store (i, j, h, w) for spatial labels
                if total_n > 1:
                    # Decide if this is a Global or Local crop
                    if v_idx < self.V:
                        scale = self.global_crops_scale
                        target_size = self.img_size
                    else:
                        scale = self.local_crops_scale
                        target_size = self.local_img_size
                    
                    # Apply same random crop to both RGB and depth
                    i, j, h, w = v2.RandomResizedCrop.get_params(
                        img, scale=scale, ratio=(0.75, 1.333)
                    )
                    rgb_crop = v2.functional.resized_crop(
                        img, i, j, h, w, (target_size, target_size)
                    )
                    crop_rect = (i, j, h, w)
                    
                    # Apply same crop to depth (scaled to cache resolution)
                    # Use consistent scale factor (same ratio for both dimensions)
                    cache_h, cache_w = depth_map.shape[:2]
                    scale_h = cache_h / orig_h_img
                    scale_w = cache_w / orig_w_img
                    
                    # Scale crop coordinates
                    di = int(i * scale_h)
                    dj = int(j * scale_w)
                    dh = max(1, int(h * scale_h))
                    dw = max(1, int(w * scale_w))
                    
                    # Clamp to valid range
                    di = min(di, cache_h - 1)
                    dj = min(dj, cache_w - 1)
                    dh = min(dh, cache_h - di)
                    dw = min(dw, cache_w - dj)
                    
                    depth_crop = depth_map[di:di+dh, dj:dj+dw]
                    
                    # Max pool to target size
                    depth_t = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0)
                    depth_resized = torch.nn.functional.adaptive_max_pool2d(depth_t, (target_size, target_size))
                    depth_tensor = depth_resized.squeeze(0)
                    
                    # Decide flip ONCE for this view (sync between RGB and depth)
                    do_flip = np.random.random() < 0.5
                    
                    # RGB augmentation (pre-instantiated)
                    rgb_tensor = self._apply_rgb_post_aug(rgb_crop, is_global=(v_idx < self.V), view_index=v_idx)
                    
                    # Apply synchronized flip to both RGB and depth
                    if do_flip:
                        rgb_tensor = torch.flip(rgb_tensor, dims=[-1])  # Flip W dimension
                        depth_tensor = torch.flip(depth_tensor, dims=[-1])
                    
                else:
                    if self.finetune_mode and self.split == 'train':
                        # Finetune mode: Apply augmented random crop + flip (synchronized RGB+depth)
                        # Use large crop scale (default 0.8-1.0) for near-full-image views
                        i, j, h, w = v2.RandomResizedCrop.get_params(
                            img, scale=self.finetune_crop_scale, ratio=(0.75, 1.333)
                        )
                        rgb_crop = v2.functional.resized_crop(
                            img, i, j, h, w, (self.img_size, self.img_size)
                        )
                        crop_rect = (i, j, h, w)
                        
                        # Crop depth using the SAME coordinates (scaled to cached depth resolution)
                        cache_h, cache_w = depth_map.shape[:2]
                        orig_h_img_a, orig_w_img_a = img.size[1], img.size[0]
                        scale_h = cache_h / orig_h_img_a
                        scale_w = cache_w / orig_w_img_a
                        
                        di = int(i * scale_h)
                        dj = int(j * scale_w)
                        dh = max(1, int(h * scale_h))
                        dw = max(1, int(w * scale_w))
                        di = min(di, cache_h - 1)
                        dj = min(dj, cache_w - 1)
                        dh = min(dh, cache_h - di)
                        dw = min(dw, cache_w - dj)
                        
                        depth_crop = depth_map[di:di+dh, dj:dj+dw]
                        depth_t = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0)
                        depth_tensor = torch.nn.functional.adaptive_max_pool2d(depth_t, (self.img_size, self.img_size)).squeeze(0)
                        
                        # Synchronized random flip
                        do_flip = np.random.random() < 0.5
                        
                        # RGB augmentation (color jitter + grayscale)
                        rgb_tensor = self._apply_rgb_post_aug(rgb_crop, is_global=True, view_index=0)
                        
                        if do_flip:
                            rgb_tensor = torch.flip(rgb_tensor, dims=[-1])
                            depth_tensor = torch.flip(depth_tensor, dims=[-1])
                    else:
                        # Validation / non-finetune: Deterministic center crop
                        rgb_tensor = self.cam_test(img)
                    
                        # Compute crop_rect that matches Resize + CenterCrop behavior
                        # v2.Resize(img_size) scales smallest edge to img_size
                        orig_h_img, orig_w_img = img.size[1], img.size[0]  # PIL is (W, H)
                        scale = self.img_size / min(orig_h_img, orig_w_img)
                        new_h = round(orig_h_img * scale)
                        new_w = round(orig_w_img * scale)
                        
                        # CenterCrop extracts center img_size x img_size from resized
                        top_new = (new_h - self.img_size) // 2
                        left_new = (new_w - self.img_size) // 2
                        
                        # Back-project to original image coordinates
                        i = int(top_new / scale)  # top in original
                        j = int(left_new / scale)  # left in original
                        h = int(self.img_size / scale)  # height in original
                        w = int(self.img_size / scale)  # width in original
                        
                        # Clamp to image bounds
                        h = min(h, orig_h_img - i)
                        w = min(w, orig_w_img - j)
                        
                        crop_rect = (i, j, h, w)
                        
                        # Crop depth using the SAME coordinates as RGB (scaled to cached depth resolution)
                        cache_h, cache_w = depth_map.shape[:2]
                        scale_h = cache_h / orig_h_img
                        scale_w = cache_w / orig_w_img
                        
                        di = int(i * scale_h)
                        dj = int(j * scale_w)
                        dh = max(1, int(h * scale_h))
                        dw = max(1, int(w * scale_w))
                        
                        # Clamp to cache bounds
                        di = min(di, cache_h - 1)
                        dj = min(dj, cache_w - 1)
                        dh = min(dh, cache_h - di)
                        dw = min(dw, cache_w - dj)
                        
                        depth_crop = depth_map[di:di+dh, dj:dj+dw]
                        
                        # Resize to model input size
                        depth_t = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0)
                        depth_tensor = torch.nn.functional.adaptive_max_pool2d(depth_t, (self.img_size, self.img_size)).squeeze(0)
                
                # For arch D, create RGBD
                if self.arch == "D":
                    rgbd = torch.cat([rgb_tensor, depth_tensor], dim=0)
                    rgbd_views.append(rgbd)
                else:
                    # For arch B/C - collect synchronized RGB and depth separately
                    rgb_views.append(rgb_tensor)
                    if self.lidar_mode == "depth":
                        # Aligned depth: Keep as 1 channel
                        # No need to repeat to 5 channels anymore, as encoder is aligned-aware
                        depth_views.append(depth_tensor)
                    else:
                        # Fallback (legacy): If we somehow got here with something else
                        # Expand to 5 channels: depth, depth, depth, depth, depth
                        depth_5ch = depth_tensor.repeat(5, 1, 1)
                        depth_views.append(depth_5ch)

                if v_idx < self.V and crop_rect is not None:
                    multimae_global_crop_rects.append(crop_rect)
                
                # Compute crop-aware spatial labels for this view
                sp = self._get_spatial_labels(pair, crop_rect) if not self.encoder_only_labels else None
                
                # Compute depth metrics for the crop
                if (not self.encoder_only_labels) and depth_crop is not None and depth_crop.size > 0:
                    d_crop_t = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0)
                    valid_mask_crop = (d_crop_t > 0).float()
                    valid_count_crop = valid_mask_crop.sum()
                    
                    if valid_count_crop > 0:
                        sp["mean_depth"] = (d_crop_t.sum() / valid_count_crop).item() * 80.0
                    
                    # 8x8 grid on crop
                    n_pix = d_crop_t.shape[-2] * d_crop_t.shape[-1]
                    g_sum = torch.nn.functional.adaptive_avg_pool2d(d_crop_t, (8, 8)) * (n_pix / 64)
                    g_cnt = torch.nn.functional.adaptive_avg_pool2d(valid_mask_crop, (8, 8)) * (n_pix / 64)
                    g_label = torch.zeros_like(g_sum)
                    g_nz = g_cnt > 0
                    g_label[g_nz] = g_sum[g_nz] / g_cnt[g_nz]
                    sp["depth_grid"] = (g_label * 80.0).flatten().numpy()
                    sp["depth_grid_mask"] = g_nz.float().flatten().numpy()

                    # High-res 56×56 depth grid for patch-based depth probe
                    n_pix_hr = d_crop_t.shape[-2] * d_crop_t.shape[-1]
                    g_sum_hr = torch.nn.functional.adaptive_avg_pool2d(d_crop_t, (56, 56)) * (n_pix_hr / 3136)
                    g_cnt_hr = torch.nn.functional.adaptive_avg_pool2d(valid_mask_crop, (56, 56)) * (n_pix_hr / 3136)
                    g_label_hr = torch.zeros_like(g_sum_hr)
                    g_nz_hr = g_cnt_hr > 0
                    g_label_hr[g_nz_hr] = g_sum_hr[g_nz_hr] / g_cnt_hr[g_nz_hr]
                    sp["depth_grid_hr"] = (g_label_hr * 80.0).flatten().numpy()
                    sp["depth_grid_mask_hr"] = g_nz_hr.float().flatten().numpy()
                else:
                    # depth_crop wasn't set (e.g., test mode uses different var)
                    # Try using depth_tensor for test mode if needed
                    pass # Use defaults from _get_spatial_labels
                    
                if sp is not None:
                    spatial_labels_per_view.append(sp)
            
            if self.include_probe_view:
                # --- Append Clean Full View (Deterministically Processed) for PROBING ---
                # Used as the last view for training spatial probes on clean global context
                
                # 1. Clean RGB
                rgb_full = self.full_view_transform(img)
            
                # 2. Clean Depth (Same as test/val logic: center crop of resized)
                # Compute params
                orig_h_img, orig_w_img = img.size[1], img.size[0]
                scale = self.img_size / min(orig_h_img, orig_w_img)
                new_h = round(orig_h_img * scale)
                new_w = round(orig_w_img * scale)
                
                top_new = (new_h - self.img_size) // 2
                left_new = (new_w - self.img_size) // 2
                
                # Back-project to original image coordinates
                i_full = int(top_new / scale)
                j_full = int(left_new / scale)
                h_full = int(self.img_size / scale)
                w_full = int(self.img_size / scale)
                
                # Clamp
                h_full = min(h_full, orig_h_img - i_full)
                w_full = min(w_full, orig_w_img - j_full)
                crop_rect_full = (i_full, j_full, h_full, w_full)
                
                # Crop Depth
                cache_h, cache_w = depth_map.shape[:2]
                scale_h = cache_h / orig_h_img
                scale_w = cache_w / orig_w_img
                
                di = min(int(i_full * scale_h), cache_h - 1)
                dj = min(int(j_full * scale_w), cache_w - 1)
                dh = min(max(1, int(h_full * scale_h)), cache_h - di)
                dw = min(max(1, int(w_full * scale_w)), cache_w - dj)
                
                depth_crop_full = depth_map[di:di+dh, dj:dj+dw]
                
                # Resize
                depth_t = torch.from_numpy(depth_crop_full).unsqueeze(0).unsqueeze(0)
                depth_full = torch.nn.functional.adaptive_max_pool2d(depth_t, (self.img_size, self.img_size)).squeeze(0)
                
                # 3. Add to lists
                if self.arch == "D":
                    rgbd_full = torch.cat([rgb_full, depth_full], dim=0)
                    rgbd_views.append(rgbd_full)
                else:
                    rgb_views.append(rgb_full)
                    if self.lidar_mode == "depth":
                        depth_views.append(depth_full)
                    else:
                         depth_views.append(depth_full.repeat(5, 1, 1))
                
                # 4. Spatial Labels
                sp_full = self._get_spatial_labels(pair, crop_rect_full) if not self.encoder_only_labels else None
                # Add depth metrics manually as _get_spatial_labels handles grid via points usually?
                # Actually _get_spatial_labels relies on points, but for depth mode we compute manually in loop
                # Copy manual computation logic
                if (not self.encoder_only_labels) and depth_crop_full.size > 0:
                    d_crop_t = depth_t # already computed
                    valid_mask_crop = (d_crop_t > 0).float()
                    valid_count_crop = valid_mask_crop.sum()
                    if valid_count_crop > 0:
                         sp_full["mean_depth"] = (d_crop_t.sum() / valid_count_crop).item() * 80.0
                    
                    n_pix = d_crop_t.shape[-2] * d_crop_t.shape[-1]
                    g_sum = torch.nn.functional.adaptive_avg_pool2d(d_crop_t, (8, 8)) * (n_pix / 64)
                    g_cnt = torch.nn.functional.adaptive_avg_pool2d(valid_mask_crop, (8, 8)) * (n_pix / 64)
                    g_label = torch.zeros_like(g_sum)
                    g_nz = g_cnt > 0
                    g_label[g_nz] = g_sum[g_nz] / g_cnt[g_nz]
                    sp_full["depth_grid"] = (g_label * 80.0).flatten().numpy()
                    sp_full["depth_grid_mask"] = g_nz.float().flatten().numpy()

                    # High-res 56×56 depth grid for patch-based depth probe
                    n_pix_hr = d_crop_t.shape[-2] * d_crop_t.shape[-1]
                    g_sum_hr = torch.nn.functional.adaptive_avg_pool2d(d_crop_t, (56, 56)) * (n_pix_hr / 3136)
                    g_cnt_hr = torch.nn.functional.adaptive_avg_pool2d(valid_mask_crop, (56, 56)) * (n_pix_hr / 3136)
                    g_label_hr = torch.zeros_like(g_sum_hr)
                    g_nz_hr = g_cnt_hr > 0
                    g_label_hr[g_nz_hr] = g_sum_hr[g_nz_hr] / g_cnt_hr[g_nz_hr]
                    sp_full["depth_grid_hr"] = (g_label_hr * 80.0).flatten().numpy()
                    sp_full["depth_grid_mask_hr"] = g_nz_hr.float().flatten().numpy()
                
                if sp_full is not None:
                    spatial_labels_per_view.append(sp_full)
                multimae_probe_crop_rect = crop_rect_full
            
            # --- Packing ---
            # Use crop_rect from training view for finetune, or crop_rect_full for non-finetune
            # crop_rect is the last crop_rect from the loop (which is fine for finetune mode)
            # crop_rect_full is defined only if not self.finetune_mode
            label_crop_rect = crop_rect if self.finetune_mode else (crop_rect_full if self.include_probe_view and 'crop_rect_full' in locals() else None)
            
            if self.arch == "D":
                if self.finetune_mode:
                    g_list = rgbd_views[:self.V]
                    l_list = rgbd_views[self.V : self.V + self.local_crops_number]
                elif self.include_probe_view:
                    # Includes the "Clean Full View" as the LAST global view
                    g_list = rgbd_views[:self.V]
                    full_v = rgbd_views[-1]
                    g_list = g_list + [full_v]
                    l_list = rgbd_views[self.V : self.V + self.local_crops_number]
                else:
                    g_list = rgbd_views[:self.V]
                    l_list = rgbd_views[self.V : self.V + self.local_crops_number]
                
                rgbd_dict = {
                    'global': torch.stack(g_list) if g_list else torch.empty(0),
                    'local': torch.stack(l_list) if l_list else torch.empty(0)
                }

                if drop_lidar:
                    rgbd_dict['global'][:, 3, :, :] = 0
                    if len(l_list) > 0:
                        rgbd_dict['local'][:, 3, :, :] = 0
                if drop_camera:
                    rgbd_dict['global'][:, :3, :, :] = 0
                    if len(l_list) > 0:
                        rgbd_dict['local'][:, :3, :, :] = 0
                
                # Aggregate spatial labels
                if spatial_labels_per_view:
                    for k in spatial_labels_per_view[0].keys():
                        vals = [s[k] for s in spatial_labels_per_view]
                        if len(vals) == 1:
                            labels[k] = vals[0]
                        elif isinstance(vals[0], (int, float)):
                            labels[k] = np.array(vals, dtype=np.float32)
                        else:
                            labels[k] = np.stack(vals)
                if self.return_multimae_view_labels and not self.legacy_mode and not self.encoder_only_labels:
                    labels.update(
                        self._get_multimae_view_label_targets(
                            pair,
                            multimae_global_crop_rects,
                            multimae_probe_crop_rect,
                            lidar_points,
                        )
                    )
                if not self.legacy_mode and not self.encoder_only_labels:
                    labels.update(self._get_det_seg_labels(pair, crop_rect=label_crop_rect, lidar_points=lidar_points))
                return rgbd_dict, torch.zeros(1), labels
            else:
                if self.finetune_mode:
                    rgb_g = rgb_views[:self.V]
                    rgb_l = rgb_views[self.V : self.V + self.local_crops_number]
                    d_g = depth_views[:self.V]
                    d_l = depth_views[self.V : self.V + self.local_crops_number]
                elif self.include_probe_view:
                    # Mixed resolution packing for Arch B/C
                    # Includes the "Clean Full View" as the LAST global view
                    rgb_g = rgb_views[:self.V]
                    rgb_full_v = rgb_views[-1]
                    rgb_g = rgb_g + [rgb_full_v]
                    rgb_l = rgb_views[self.V : self.V + self.local_crops_number]
                    d_g = depth_views[:self.V]
                    d_full_v = depth_views[-1]
                    d_g = d_g + [d_full_v]
                    d_l = depth_views[self.V : self.V + self.local_crops_number]
                else:
                    rgb_g = rgb_views[:self.V]
                    rgb_l = rgb_views[self.V : self.V + self.local_crops_number]
                    d_g = depth_views[:self.V]
                    d_l = depth_views[self.V : self.V + self.local_crops_number]
                cam_views = {
                    'global': torch.stack(rgb_g) if rgb_g else torch.empty(0),
                    'local': torch.stack(rgb_l) if rgb_l else torch.empty(0)
                }
                
                mod2_views = {
                    'global': torch.stack(d_g) if d_g else torch.empty(0),
                    'local': torch.stack(d_l) if d_l else torch.empty(0)
                }
                
                if drop_lidar:
                    mod2_views['global'] = torch.zeros_like(mod2_views['global'])
                    if len(d_l) > 0:
                         mod2_views['local'] = torch.zeros_like(mod2_views['local'])
                if drop_camera:
                    cam_views['global'] = torch.zeros_like(cam_views['global'])
                    if len(rgb_l) > 0:
                         cam_views['local'] = torch.zeros_like(cam_views['local'])

                # Aggregate spatial labels
                if spatial_labels_per_view:
                    for k in spatial_labels_per_view[0].keys():
                        vals = [s[k] for s in spatial_labels_per_view]
                        if len(vals) == 1:
                            labels[k] = vals[0]
                        elif isinstance(vals[0], (int, float)):
                            labels[k] = np.array(vals, dtype=np.float32)
                        else:
                            labels[k] = np.stack(vals)
                if self.return_multimae_view_labels and not self.legacy_mode and not self.encoder_only_labels:
                    labels.update(
                        self._get_multimae_view_label_targets(
                            pair,
                            multimae_global_crop_rects,
                            multimae_probe_crop_rect,
                            lidar_points,
                        )
                    )
                if not self.legacy_mode and not self.encoder_only_labels:
                    labels.update(self._get_det_seg_labels(pair, crop_rect=label_crop_rect, lidar_points=lidar_points))
                return cam_views, mod2_views, labels
        
        elif self.lidar_mode == "aligned_points":
            # Aligned points: 3D points visible in camera FOV, with crop support
            # Similar to depth mode but returns 3D points filtered by crop bounds
            
            # Cache path for aligned points
            points_cache_path = self.cache_dir / f"{pair['sample_token']}_{pair['camera_name']}_aligned_pts.npz"
            
            aligned_pts = None
            aligned_uv = None
            orig_w, orig_h = img.size
            
            # Try to load from cache
            if points_cache_path.exists():
                cached = np.load(points_cache_path)
                aligned_pts = cached["points"]  # (M, 5) in camera frame
                aligned_uv = cached["uv"]       # (M, 2) pixel coordinates
            else:
                # Compute from scratch
                cam_calib_token = pair.get("cam_calib_token")
                lidar_calib_token = pair.get("lidar_calib_token")
                cam_ego_token = pair.get("cam_ego_token")
                lidar_ego_token = pair.get("lidar_ego_token")
                
                tokens_valid = (
                    cam_calib_token and cam_calib_token in self.calib_store.token_to_idx and
                    lidar_calib_token and lidar_calib_token in self.calib_store.token_to_idx and
                    cam_ego_token and cam_ego_token in self.calib_store.token_to_idx and
                    lidar_ego_token and lidar_ego_token in self.calib_store.token_to_idx
                )
                
                if tokens_valid:
                    cam_calib = self.calib_store.get_calib(cam_calib_token)
                    lidar_calib = self.calib_store.get_calib(lidar_calib_token)
                    cam_ego = self.calib_store.get_ego(cam_ego_token)
                    lidar_ego = self.calib_store.get_ego(lidar_ego_token)
                    intrinsic = cam_calib.get("intrinsic")
                    
                    if intrinsic is not None:
                        # Use module-level numpy quaternion conversion (faster than scipy)
                        quat_to_rot = quat_to_rot_numpy
                        
                        # Build full lidar-to-camera transform (same as depth projection)
                        lidar_to_ego = np.eye(4)
                        lidar_to_ego[:3, :3] = quat_to_rot(lidar_calib["rotation"])
                        lidar_to_ego[:3, 3] = lidar_calib["translation"]
                        
                        lidar_ego_to_world = np.eye(4)
                        lidar_ego_to_world[:3, :3] = quat_to_rot(lidar_ego["rotation"])
                        lidar_ego_to_world[:3, 3] = lidar_ego["translation"]
                        
                        world_to_cam_ego = np.eye(4)
                        cam_ego_rot = quat_to_rot(cam_ego["rotation"])
                        world_to_cam_ego[:3, :3] = cam_ego_rot.T
                        world_to_cam_ego[:3, 3] = -cam_ego_rot.T @ cam_ego["translation"]
                        
                        ego_to_cam = np.eye(4)
                        cam_rot = quat_to_rot(cam_calib["rotation"])
                        ego_to_cam[:3, :3] = cam_rot.T
                        ego_to_cam[:3, 3] = -cam_rot.T @ cam_calib["translation"]
                        
                        lidar_to_cam = ego_to_cam @ world_to_cam_ego @ lidar_ego_to_world @ lidar_to_ego
                        
                        # Get aligned points in camera frame with UV coordinates
                        aligned_pts, aligned_uv = lidar_to_aligned_points(
                            lidar_points,
                            intrinsic=np.array(intrinsic),
                            lidar_to_cam_transform=lidar_to_cam.astype(np.float32),
                            img_size=(orig_h, orig_w),
                            max_depth=80.0,
                        )
                        
                        # Save to cache (with race condition protection)
                        if not points_cache_path.exists():
                            try:
                                np.savez_compressed(points_cache_path, points=aligned_pts, uv=aligned_uv)
                            except (PermissionError, OSError) as e:
                                print(f"Warning: Failed to write points cache: {e}")
                                pass  # Another worker already writing
            
            # Fallback if no valid points
            if aligned_pts is None or len(aligned_pts) == 0:
                points = subsample_points(lidar_points, self.n_points)
                points = normalize_points(points)
                points_tensor = torch.from_numpy(points).float()
                if drop_lidar:
                    points_tensor = torch.zeros_like(points_tensor)
                sp = self._get_spatial_labels(pair, None)
                labels.update(sp)
                if not self.legacy_mode:
                    labels.update(self._get_det_seg_labels(pair, lidar_points=lidar_points))
                return cam_views, points_tensor, labels
            
            # Generate per-view RGB + points with synchronized crop filtering
            spatial_labels_per_view = []
            points_views = []
            rgb_views = []  # Generate RGB here to sync with points
            
            total_n = self.V + self.local_crops_number
            if self.V == 1 and self.local_crops_number == 0:
                total_n = 1
            
            # RGB augmentation pipeline (pre-instantiated) - NO FLIP, flip handled separately
            # Using self.rgb_aug_no_flip defined in __init__
            
            for v_idx in range(total_n):
                crop_rect = None
                do_flip = False  # Will be set for training mode
                
                if total_n > 1:
                    # Decide crop scale based on global vs local
                    if v_idx < self.V:
                        scale = self.global_crops_scale
                        target_size = self.img_size
                    else:
                        scale = self.local_crops_scale
                        target_size = self.local_img_size
                    
                    # Decide flip ONCE for this view (sync between RGB and points)
                    do_flip = np.random.random() < 0.5
                    
                    # Get random crop parameters - SAME for RGB and points
                    i, j, h, w = v2.RandomResizedCrop.get_params(
                        img, scale=scale, ratio=(0.75, 1.333)
                    )
                    crop_rect = (i, j, h, w)
                    
                    # Apply crop to RGB
                    rgb_crop = v2.functional.resized_crop(
                        img, i, j, h, w, (target_size, target_size)
                    )
                    # RGB augmentation (pre-instantiated)
                    rgb_tensor = self._apply_rgb_post_aug(rgb_crop, is_global=(v_idx < self.V), view_index=v_idx)
                    
                    # Apply synchronized flip to RGB
                    if do_flip:
                        rgb_tensor = torch.flip(rgb_tensor, dims=[-1])
                    
                    # Filter points by UV falling within same crop
                    u_min, v_min = j, i
                    u_max, v_max = j + w, i + h
                    
                    in_crop = (
                        (aligned_uv[:, 0] >= u_min) & (aligned_uv[:, 0] < u_max) &
                        (aligned_uv[:, 1] >= v_min) & (aligned_uv[:, 1] < v_max)
                    )
                    crop_pts = aligned_pts[in_crop].copy()  # Copy to allow modification
                    
                    # Apply synchronized flip to points (mirror X coordinate)
                    if do_flip and len(crop_pts) > 0:
                        crop_pts[:, 0] = -crop_pts[:, 0]  # Mirror X in camera frame
                else:
                    # Test mode: use center crop matching RGB
                    rgb_tensor = self.cam_test(img)
                    
                    scale_factor = self.img_size / min(orig_h, orig_w)
                    new_h = round(orig_h * scale_factor)
                    new_w = round(orig_w * scale_factor)
                    
                    top_new = (new_h - self.img_size) // 2
                    left_new = (new_w - self.img_size) // 2
                    
                    i = int(top_new / scale_factor)
                    j = int(left_new / scale_factor)
                    h = int(self.img_size / scale_factor)
                    w = int(self.img_size / scale_factor)
                    
                    h = min(h, orig_h - i)
                    w = min(w, orig_w - j)
                    crop_rect = (i, j, h, w)
                    
                    # Filter points by center crop bounds
                    u_min, v_min = j, i
                    u_max, v_max = j + w, i + h
                    
                    in_crop = (
                        (aligned_uv[:, 0] >= u_min) & (aligned_uv[:, 0] < u_max) &
                        (aligned_uv[:, 1] >= v_min) & (aligned_uv[:, 1] < v_max)
                    )
                    crop_pts = aligned_pts[in_crop]
                
                rgb_views.append(rgb_tensor)
                
                # Save raw crop points for label generation (before subsampling/normalization)
                crop_pts_raw = crop_pts.copy() if len(crop_pts) > 0 else np.empty((0, 5), dtype=np.float32)
                
                # Subsample/pad to fixed size and normalize
                if len(crop_pts) > 0:
                    crop_pts = subsample_points(crop_pts, self.n_points)
                    crop_pts = normalize_points(crop_pts)
                else:
                    # Empty crop: use zero points
                    crop_pts = np.zeros((self.n_points, 5), dtype=np.float32)
                
                points_views.append(torch.from_numpy(crop_pts).float())
                
                
                # Compute spatial labels for this crop
                sp = self._get_spatial_labels(pair, crop_rect)
                
                # FIX: Calculate proper depth metrics for aligned points
                # (previously this was missing, causing 0 metrics for Arch A)
                if len(crop_pts_raw) > 0:
                    # 1. Mean depth (from Z coordinate in camera frame)
                    # crop_pts_raw is (N, 5) where [2] is Z (depth)
                    sp["mean_depth"] = float(np.mean(crop_pts_raw[:, 2]))
                    
                    # 2. Depth Grid (8x8)
                    # Use UV coordinates to map points to grid cells
                    # We need UVs relative to the crop
                    crop_u = aligned_uv[in_crop, 0] - u_min
                    crop_v = aligned_uv[in_crop, 1] - v_min
                    crop_z = crop_pts_raw[:, 2] # Depth
                    
                    # Grid cell size in crop pixels
                    # Note: crop_rect is (i, j, h, w) -> (v, u, h, w)
                    # u matches w, v matches h
                    cell_w = crop_rect[3] / 8.0
                    cell_h = crop_rect[2] / 8.0
                    
                    # Determine grid indices for each point
                    grid_c = np.clip((crop_u / cell_w).astype(np.int64), 0, 7)
                    grid_r = np.clip((crop_v / cell_h).astype(np.int64), 0, 7)
                    flat_indices = grid_r * 8 + grid_c
                    
                    # Accumulate depth per cell
                    grid_sum = np.zeros(64, dtype=np.float32)
                    grid_cnt = np.zeros(64, dtype=np.float32)
                    
                    # Fast accumulation using np.add.at
                    np.add.at(grid_sum, flat_indices, crop_z.astype(np.float32))
                    np.add.at(grid_cnt, flat_indices, 1)
                    
                    # Compute average
                    grid_label = np.zeros(64, dtype=np.float32)
                    mask_nz = grid_cnt > 0
                    grid_label[mask_nz] = grid_sum[mask_nz] / grid_cnt[mask_nz]
                    
                    sp["depth_grid"] = grid_label
                    sp["depth_grid_mask"] = mask_nz.astype(np.float32)

                    # High-res 56×56 depth grid for patch-based depth probe
                    cell_w_hr = crop_rect[3] / 56.0
                    cell_h_hr = crop_rect[2] / 56.0
                    grid_c_hr = np.clip((crop_u / cell_w_hr).astype(np.int64), 0, 55)
                    grid_r_hr = np.clip((crop_v / cell_h_hr).astype(np.int64), 0, 55)
                    flat_hr = grid_r_hr * 56 + grid_c_hr
                    g_sum_hr = np.zeros(3136, dtype=np.float32)
                    g_cnt_hr = np.zeros(3136, dtype=np.float32)
                    np.add.at(g_sum_hr, flat_hr, crop_z.astype(np.float32))
                    np.add.at(g_cnt_hr, flat_hr, 1)
                    g_lbl_hr = np.zeros(3136, dtype=np.float32)
                    mask_hr = g_cnt_hr > 0
                    g_lbl_hr[mask_hr] = g_sum_hr[mask_hr] / g_cnt_hr[mask_hr]
                    sp["depth_grid_hr"] = g_lbl_hr
                    sp["depth_grid_mask_hr"] = mask_hr.astype(np.float32)
                
                spatial_labels_per_view.append(sp)
            
            # --- Append Clean Full View (Deterministically Processed) for PROBING ---
            
            # 1. Clean RGB
            rgb_full = self.full_view_transform(img)
            
            # 2. Clean Points (Filter to crop)
            orig_h_img, orig_w_img = img.size[1], img.size[0]
            scale = self.img_size / min(orig_h_img, orig_w_img)
            new_h = round(orig_h_img * scale)
            new_w = round(orig_w_img * scale)
            top_new = (new_h - self.img_size) // 2
            left_new = (new_w - self.img_size) // 2
            i_full = int(top_new / scale)
            j_full = int(left_new / scale)
            h_full = int(self.img_size / scale)
            w_full = int(self.img_size / scale)
            h_full = min(h_full, orig_h_img - i_full)
            w_full = min(w_full, orig_w_img - j_full)
            
            crop_rect_full = (i_full, j_full, h_full, w_full)
            
            u_min, v_min = j_full, i_full
            u_max, v_max = j_full + w_full, i_full + h_full
            
            in_crop = (
                (aligned_uv[:, 0] >= u_min) & (aligned_uv[:, 0] < u_max) &
                (aligned_uv[:, 1] >= v_min) & (aligned_uv[:, 1] < v_max)
            )
            crop_pts = aligned_pts[in_crop].copy()
            
            crop_pts_raw = crop_pts.copy() if len(crop_pts) > 0 else np.empty((0, 5), dtype=np.float32)
            
            if len(crop_pts) > 0:
                crop_pts = subsample_points(crop_pts, self.n_points)
                crop_pts = normalize_points(crop_pts)
            else:
                crop_pts = np.zeros((self.n_points, 5), dtype=np.float32)
            
            rgb_views.append(rgb_full)
            points_views.append(torch.from_numpy(crop_pts).float())
            
            # Calculate spatial labels for full view
            sp_full = self._get_spatial_labels(pair, crop_rect_full)
            # Add manual depth metrics (omitted for brevity as standard sp handles it logic?)
            # Actually standard sp only does boxes. We need depth.
            if len(crop_pts_raw) > 0:
                 sp_full["mean_depth"] = float(np.mean(crop_pts_raw[:, 2]))
                 
                 # Grid logic (copy-paste from loop)
                 crop_u = aligned_uv[in_crop, 0] - u_min
                 crop_v = aligned_uv[in_crop, 1] - v_min
                 crop_z = crop_pts_raw[:, 2]
                 cell_w = crop_rect_full[3] / 8.0
                 cell_h = crop_rect_full[2] / 8.0
                 grid_c = np.clip((crop_u / cell_w).astype(np.int64), 0, 7)
                 grid_r = np.clip((crop_v / cell_h).astype(np.int64), 0, 7)
                 flat = grid_r * 8 + grid_c
                 g_sum = np.zeros(64, dtype=np.float32)
                 g_cnt = np.zeros(64, dtype=np.float32)
                 np.add.at(g_sum, flat, crop_z.astype(np.float32))
                 np.add.at(g_cnt, flat, 1)
                 g_lbl = np.zeros(64, dtype=np.float32)
                 mask = g_cnt > 0
                 g_lbl[mask] = g_sum[mask] / g_cnt[mask]
                 sp_full["depth_grid"] = g_lbl
                 sp_full["depth_grid_mask"] = mask.astype(np.float32)

                 # High-res 56×56 depth grid for patch-based depth probe
                 cell_w_hr = crop_rect_full[3] / 56.0
                 cell_h_hr = crop_rect_full[2] / 56.0
                 grid_c_hr = np.clip((crop_u / cell_w_hr).astype(np.int64), 0, 55)
                 grid_r_hr = np.clip((crop_v / cell_h_hr).astype(np.int64), 0, 55)
                 flat_hr = grid_r_hr * 56 + grid_c_hr
                 g_sum_hr = np.zeros(3136, dtype=np.float32)
                 g_cnt_hr = np.zeros(3136, dtype=np.float32)
                 np.add.at(g_sum_hr, flat_hr, crop_z.astype(np.float32))
                 np.add.at(g_cnt_hr, flat_hr, 1)
                 g_lbl_hr = np.zeros(3136, dtype=np.float32)
                 mask_hr = g_cnt_hr > 0
                 g_lbl_hr[mask_hr] = g_sum_hr[mask_hr] / g_cnt_hr[mask_hr]
                 sp_full["depth_grid_hr"] = g_lbl_hr
                 sp_full["depth_grid_mask_hr"] = mask_hr.astype(np.float32)
            
            spatial_labels_per_view.append(sp_full)
                
            
            # Stack views 
            # SEPARATE global vs local points/rgb because of different logical purpose?
            # Actually points are subsampled to fixed N, so they CAN be stacked.
            # RGB CANNOT be stacked (mixed res).
            
            # Separate global/local for RGB
            num_global = self.V
            num_local = self.local_crops_number
            # Total loop was N = V + Local + 1 (full view is usually part of val, wait.)
            # Full view logic above was appended.
            # Usually full view is not part of training views?
            # Standard: V global + N local crops.
            # The "Clean Full View" lines suggest it IS appended as an extra view?
            # Let's check loop `for v_idx in range(total_n):` where total_n = V + local.
            # Then we appended ONE more full view.
            # So total views = V + local + 1.
            
            # We treat the last "full view" as a GLOBAL view usually?
            # Or just separate it.
            
            # Pack into dict
            rgb_global_list = rgb_views[:self.V]
            rgb_local_list = rgb_views[self.V:self.V+self.local_crops_number]
            # Use extra view as extra global or discard? 
            # Standard training usually doesn't use the clean full view for backprop, only probing.
            # But here it is returned.
            # Let's count it as GLOBAL.
            if len(rgb_views) > self.V + self.local_crops_number:
                 rgb_global_list.append(rgb_views[-1])
            
            cam_views = {
                'global': torch.stack(rgb_global_list),
                'local': torch.stack(rgb_local_list) if rgb_local_list else torch.empty(0)
            }
            
            # Points are fixed size (N, 5), so we CAN stack them all if we want, 
            # BUT downstream encoder expects matching view counts.
            # So we should split points too.
            pts_global_list = points_views[:self.V]
            pts_local_list = points_views[self.V:self.V+self.local_crops_number]
            if len(points_views) > self.V + self.local_crops_number:
                 pts_global_list.append(points_views[-1])
                 
            points_tensor = {
                'global': torch.stack(pts_global_list),
                'local': torch.stack(pts_local_list) if pts_local_list else torch.empty(0)
            }
            
            if drop_camera:
                cam_views['global'] = torch.zeros_like(cam_views['global'])
                if cam_views['local'].numel() > 0:
                     cam_views['local'] = torch.zeros_like(cam_views['local'])
                     
            if drop_lidar:
                points_tensor['global'] = torch.zeros_like(points_tensor['global'])
                if points_tensor['local'].numel() > 0:
                     points_tensor['local'] = torch.zeros_like(points_tensor['local'])
            
            # Aggregate labels
            if spatial_labels_per_view:
                for k in spatial_labels_per_view[0].keys():
                    vals = [s[k] for s in spatial_labels_per_view]
                    if len(vals) == 1:
                        labels[k] = vals[0]
                    elif isinstance(vals[0], (int, float)):
                        labels[k] = np.array(vals, dtype=np.float32)
                    else:
                        labels[k] = np.stack(vals)
            
            # Sync: First V are global, middle N are local, LAST is probe (full view)
            # We put the probe into 'global' as the last element
            g_list = rgb_views[:self.V]
            full_v = rgb_views[-1]
            g_list.append(full_v)
            
            l_list = rgb_views[self.V : self.V + self.local_crops_number]
            
            cam_views = {
                'global': torch.stack(g_list) if g_list else torch.empty(0),
                'local': torch.stack(l_list) if l_list else torch.empty(0)
            }
            
            # Points
            pg_list = points_views[:self.V]
            p_full = points_views[-1]
            pg_list.append(p_full)
            
            pl_list = points_views[self.V : self.V + self.local_crops_number]
            
            points_tensor = {
                'global': torch.stack(pg_list) if pg_list else torch.empty(0),
                'local': torch.stack(pl_list) if pl_list else torch.empty(0)
            }
            
            if drop_camera:
                cam_views['global'] = torch.zeros_like(cam_views['global'])
                if len(l_list) > 0: cam_views['local'] = torch.zeros_like(cam_views['local'])
            if drop_lidar:
                points_tensor['global'] = torch.zeros_like(points_tensor['global'])
                if len(pl_list) > 0: points_tensor['local'] = torch.zeros_like(points_tensor['local'])

            if not self.legacy_mode:
                labels.update(self._get_det_seg_labels(pair))
            return cam_views, points_tensor, labels
        
        else:  # lidar_mode == "points"
            # Raw point cloud (full 360°, NOT aligned with camera)
            points = subsample_points(lidar_points, self.n_points)
            points = normalize_points(points)
            points_tensor = torch.from_numpy(points).float()
            if drop_lidar:
                points_tensor = torch.zeros_like(points_tensor)
            
            # Fall back to full-image spatial labels for points mode
            sp = self._get_spatial_labels(pair, None)
            labels.update(sp)
            if not self.legacy_mode:
                labels.update(self._get_det_seg_labels(pair))
            return cam_views, points_tensor, labels
    
    # Note: __len__ is defined earlier using _n_pairs for compact storage


def mm_collate_fn(batch):
    """Custom collate for multi-modal data with mixed resolutions (dicts)."""
    # batch[i][0] is cam_views, which can be dict or tensor
    
    # Process Camera
    example_cam = batch[0][0]
    if isinstance(example_cam, dict):
        cam_views = {}
        for k in example_cam.keys():
            # Check if this key has content (local might be empty)
            if example_cam[k].numel() > 0 or (isinstance(example_cam[k], torch.Tensor) and example_cam[k].ndim > 0):
                cam_views[k] = torch.stack([b[0][k] for b in batch])
            else:
                 cam_views[k] = torch.empty(0)
    else:
        cam_views = torch.stack([b[0] for b in batch])
        
    # Process Modality2
    example_mod2 = batch[0][1]
    if isinstance(example_mod2, dict):
        modality2 = {}
        for k in example_mod2.keys():
            if example_mod2[k].numel() > 0:
                modality2[k] = torch.stack([b[1][k] for b in batch])
            else:
                 modality2[k] = torch.empty(0)
    else:
        # Check if it's points or tensor
        # Points (N, 5) can be stacked usually
        modality2 = torch.stack([b[1] for b in batch])
    
    # helper for stacking list of NumPy or Tensor arrays
    # b[2] is the labels dict
    keys = batch[0][2].keys()
    
    labels = {}
    for k in keys:
        example = batch[0][2][k]
        values = [b[2][k] for b in batch] # list of items from batch
        
        if isinstance(example, (int, float)):
             labels[k] = torch.tensor(values)
        elif isinstance(example, np.ndarray):
             labels[k] = torch.from_numpy(np.stack(values))
        elif isinstance(example, torch.Tensor):
             labels[k] = torch.stack(values)
        else:
             labels[k] = values # default list

    return cam_views, modality2, labels


if __name__ == "__main__":
    # Test dataset
    print("Testing Option B (Range Images)...")
    ds_b = MMNuScenesDataset(
        "/path/to/nuscenes_data",
        split="train",
        arch="B",
        V=2,
    )
    cam, rng, labels = ds_b[0]
    if isinstance(cam, dict):
        print("Camera views (Dict):")
        for k, v in cam.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"Camera views shape: {cam.shape}")
    if isinstance(rng, dict):
        print("Range views (Dict):")
        for k, v in rng.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"Range views shape: {rng.shape}")
    print(f"Labels: scene={labels['scene']}, camera={labels['camera']}")
    print(f"Depth Labels (Arch B/Range): mean={labels['mean_depth']:.4f}, grid_sum={labels['depth_grid'].sum():.4f}")
    
    print("\nTesting Option A (Points)...")
    ds_a = MMNuScenesDataset(
        "/path/to/nuscenes_data",
        split="train", 
        arch="A",
        V=2,
    )
    cam, pts, labels = ds_a[0]
    if isinstance(cam, dict):
        print("Camera views (Dict):")
        for k, v in cam.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"Camera views shape: {cam.shape}")
    if isinstance(pts, dict):
        print("Points (Dict):")
        for k, v in pts.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"Points shape: {pts.shape}")
    print(f"Depth Labels (Arch A/Points): mean={labels['mean_depth']:.4f}, grid_sum={labels['depth_grid'].sum():.4f}")

    print("\n---------------------------------------------------------")
    print("Testing DataLoader (Batch Stacking)...")
    dataloader = DataLoader(
        ds_b, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0, 
        collate_fn=mm_collate_fn
    )
    
    print("Iterating 2 batches...")
    for i, batch in enumerate(dataloader):
        if i >= 2: break
        # batch is (cam_views, modality2, labels)
        cam, mod2, lbl = batch
        
        print(f"Batch {i}:")
        if isinstance(cam, dict):
            print(f"  Cam Global: {cam['global'].shape}")
            print(f"  Cam Local:  {cam['local'].shape if cam['local'].numel() > 0 else 'Empty'}")
        else:
            print(f"  Cam: {cam.shape}")
            
        if isinstance(mod2, dict):
             print(f"  Mod2 Global: {mod2['global'].shape}")
        elif isinstance(mod2, torch.Tensor):
             print(f"  Mod2: {mod2.shape}")
             
        print(f"  Scene Labels: {len(lbl['scene'])}")
    
    print("\nSUCCESS: Dataset and DataLoader are functioning correctly.")
