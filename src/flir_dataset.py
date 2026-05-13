"""
flir_dataset.py

Paired RGB + thermal dataset loader for the Teledyne FLIR ADAS Thermal Dataset v2.

This integrates FLIR into the MM-LeJEPA training path by reusing the existing
"depth-like" 1-channel second-modality interface used by aligned depth inputs.
The dataset provides paired RGB and thermal crops plus lightweight probe labels
that can be derived from 2D COCO annotations.

Important limitation:
- FLIR provides paired RGB/thermal frames and 2D boxes, but not the LiDAR-style
  geometry targets used by the NuScenes/Waymo patch probes.
- Accordingly, this loader returns zero-filled geometry/depth targets for the
    unsupported 3D-style probes, while exposing real 2D box labels for a FLIR-
    specific 2D detection patch probe.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms import v2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Luminance-weighted average of ImageNet stats for 1-channel thermal input
THERMAL_MEAN = (0.449,)
THERMAL_STD = (0.226,)

RGB_ALIGN_LEFT = 0.0844
RGB_ALIGN_TOP = 0.0887
RGB_ALIGN_WIDTH = 0.8875
RGB_ALIGN_HEIGHT = 0.8010

FLIR_PEDESTRIAN_CATEGORY_IDS = {1}
FLIR_VEHICLE_CATEGORY_IDS = {3, 6, 8, 79}
FLIR_2D_DET_CLASS_NAMES = [
    "person",
    "bike",
    "car",
    "motor",
    "bus",
    "truck",
    "light",
    "sign",
]
FLIR_CATEGORY_TO_2D_CLASS = {
    1: 0,   # person
    2: 1,   # bike
    3: 2,   # car
    4: 3,   # motor
    6: 4,   # bus
    8: 5,   # truck
    10: 6,  # light
    12: 7,  # sign
    77: 3,  # scooter -> motor
    79: 2,  # other vehicle -> car
}

FLIR_PROBE_LABEL_MODES = (
    'rgb',
    'thermal',
    'consensus',
    'union',
    'weighted_union',
)

FLIR_WEIGHTED_UNION_RGB_WEIGHT = {
    'person': 0.25,
    'bike': 0.45,
    'car': 0.50,
    'motor': 0.40,
    'bus': 0.50,
    'truck': 0.50,
    'light': 0.80,
    'sign': 0.85,
}

# FLIR currently only trains/evaluates 2D box patch probes. Keep unsupported
# segmentation placeholders minimal so high probe resolutions do not waste host RAM.
FLIR_UNUSED_SEG_PLACEHOLDER_SHAPE = (1, 1)


def _parse_flir_filename(file_name: str) -> Optional[Tuple[str, int, str]]:
    name = Path(file_name).name
    match = re.match(r"video-([^-]+)-frame-(\d+)-([^.]+)\.jpg$", name)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def _extract_rgb_video_map(index_data: Dict) -> Dict[str, str]:
    mapping = {}
    for video in index_data.get("videos", []):
        desc = str(video.get("description", ""))
        match = re.search(r'\{"RGB":\s*"([^"]+)"\}', desc)
        if match is not None:
            mapping[video["id"]] = match.group(1)
    return mapping


def _extract_video_pair_map(map_data: Dict[str, str]) -> Dict[str, str]:
    return {Path(rgb_file).name: Path(thermal_file).name for rgb_file, thermal_file in map_data.items()}


def _pack_flir_annotations(annotations: List[Dict]) -> np.ndarray:
    packed = np.zeros((len(annotations), 5), dtype=np.float32)
    for idx, ann in enumerate(annotations):
        bbox = ann.get("bbox", [0, 0, 0, 0])
        packed[idx, 0] = float(ann.get("category_id", -1))
        packed[idx, 1] = float(bbox[0]) if len(bbox) > 0 else 0.0
        packed[idx, 2] = float(bbox[1]) if len(bbox) > 1 else 0.0
        packed[idx, 3] = float(bbox[2]) if len(bbox) > 2 else 0.0
        packed[idx, 4] = float(bbox[3]) if len(bbox) > 3 else 0.0
    return packed


def _box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


class FlirAdasDataset(Dataset):
    """Paired RGB/thermal FLIR dataset with MM-LeJEPA-compatible outputs."""

    def __init__(
        self,
        dataroot: str,
        split: str = "train",
        arch: str = "C",
        lidar_mode: str = "depth",
        V: int = 2,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 4,
        img_size: int = 224,
        local_img_size: int = 96,
        probe_img_size: Optional[int] = None,
        occupancy_grid_size: int = 28,
        modality_dropout: float = 0.0,
        legacy_mode: bool = False,
        finetune_mode: bool = False,
        include_probe_view: Optional[bool] = None,
        encoder_only_labels: bool = False,
        det_seg_label_mode: str = "bbox_only",
        detection_label_source: str = "rgb",
        dino_aug_mode: str = "default",
        resize_mode: str = "center_crop",
        align_modalities: bool = True,
    ):
        self.package_root = Path(dataroot)
        self.dataroot = self._resolve_dataset_root(self.package_root)
        self.split = str(split).lower()
        self.arch = arch
        self.V = V
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.img_size = img_size
        self.local_img_size = local_img_size
        self.probe_img_size = int(probe_img_size or img_size)
        self.occupancy_grid_size = max(1, int(occupancy_grid_size))
        self.modality_dropout = modality_dropout
        self.legacy_mode = legacy_mode
        self.finetune_mode = finetune_mode
        self.include_probe_view = (not self.finetune_mode) if include_probe_view is None else bool(include_probe_view)
        self.encoder_only_labels = bool(encoder_only_labels)
        self.det_seg_label_mode = str(det_seg_label_mode).lower()
        self.dino_aug_mode = str(dino_aug_mode).lower()
        self.use_official_dino_augs = self.split == "train" and self.dino_aug_mode == "official"
        self.resize_mode = str(resize_mode).lower()
        self.align_modalities = align_modalities
        if self.resize_mode not in {"center_crop", "letterbox"}:
            raise ValueError(
                f"Unsupported FLIR resize_mode={resize_mode!r}. Use 'center_crop' or 'letterbox'."
            )
        self.detection_label_source = str(detection_label_source).lower()
        if self.detection_label_source not in {"rgb", "thermal"}:
            raise ValueError(
                f"Unsupported FLIR detection_label_source={detection_label_source!r}. Use 'rgb' or 'thermal'."
            )

        if lidar_mode != "depth":
            print(f"FLIR only supports depth-style 1-channel modality loading; got lidar_mode={lidar_mode}, falling back to depth")
        self.lidar_mode = "depth"

        self.num_scenes = 1
        self.num_cameras = 1
        self.num_locations = 1
        self.max_objects_2d = 50

        self._load_pairs()
        self._setup_transforms()

        print(
            f"FlirAdasDataset [arch={arch}, modality=thermal]: {len(self.pairs)} paired samples, "
            f"Global={V}, Local={local_crops_number}"
        )

    def _resolve_dataset_root(self, dataroot: Path) -> Path:
        if (dataroot / "images_rgb_train").exists():
            return dataroot
        nested_root = dataroot / "FLIR_ADAS_v2"
        if (nested_root / "images_rgb_train").exists():
            return nested_root
        raise FileNotFoundError(
            f"Could not find FLIR split directories under {dataroot}. Expected images_rgb_train directly or under FLIR_ADAS_v2/."
        )

    def _split_specs(self) -> List[Tuple[str, Path, Path]]:
        if self.split == "train":
            return [("train", self.dataroot / "images_rgb_train", self.dataroot / "images_thermal_train")]
        if self.split == "val":
            return [("val", self.dataroot / "images_rgb_val", self.dataroot / "images_thermal_val")]
        if self.split in {"video_test", "val_video", "video"}:
            return [("video_test", self.dataroot / "video_rgb_test", self.dataroot / "video_thermal_test")]
        if self.split in {"train+video_test", "train_video_test", "train_video", "train+video"}:
            return [
                ("train", self.dataroot / "images_rgb_train", self.dataroot / "images_thermal_train"),
                ("video_test", self.dataroot / "video_rgb_test", self.dataroot / "video_thermal_test"),
            ]
        raise ValueError(
            f"Unsupported FLIR split '{self.split}'. Use one of: train, val, video_test, train+video_test."
        )

    def _video_map_path(self) -> Path:
        candidates = [
            self.package_root / "rgb_to_thermal_vid_map.json",
            self.dataroot / "rgb_to_thermal_vid_map.json",
            self.dataroot.parent / "rgb_to_thermal_vid_map.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError("Could not locate rgb_to_thermal_vid_map.json for FLIR video pairing.")

    def _read_coco_split(self, root: Path) -> Dict[Tuple[str, int], Dict]:
        coco_path = root / "coco.json"
        index_path = root / "index.json"
        data_dir = root / "data"
        if not coco_path.exists():
            raise FileNotFoundError(f"Missing FLIR annotations: {coco_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Missing FLIR metadata: {index_path}")
        if not data_dir.exists():
            raise FileNotFoundError(f"Missing FLIR image directory: {data_dir}")

        with open(coco_path, "r") as f:
            coco = json.load(f)
        with open(index_path, "r") as f:
            index_data = json.load(f)

        anns_by_image = defaultdict(list)
        for ann in coco.get("annotations", []):
            anns_by_image[ann["image_id"]].append(ann)

        frame_meta_by_key = {}
        for frame in index_data.get("frames", []):
            video_meta = frame.get("videoMetadata", {})
            key = (video_meta.get("videoId"), int(video_meta.get("frameIndex", -1)))
            frame_meta_by_key[key] = {
                "dataset_frame_id": frame.get("datasetFrameId"),
                "video_id": video_meta.get("videoId"),
                "frame_index": int(video_meta.get("frameIndex", -1)),
            }

        result = {}
        for img in coco.get("images", []):
            file_name = img.get("file_name")
            if not file_name:
                continue
            parsed = _parse_flir_filename(file_name)
            if parsed is None:
                continue
            video_id, frame_index, dataset_frame_id = parsed
            image_path = root / file_name
            if not image_path.exists():
                continue
            key = (video_id, frame_index)
            frame_meta = frame_meta_by_key.get(key, {})
            result[key] = {
                "path": image_path,
                "width": int(img.get("width", 640)),
                "height": int(img.get("height", 512)),
                "annotations": _pack_flir_annotations(anns_by_image.get(img["id"], [])),
                "image_id": img["id"],
                "file_name": file_name,
                "video_id": video_id,
                "frame_index": frame_index,
                "dataset_frame_id": frame_meta.get("dataset_frame_id", dataset_frame_id),
            }
        return result, index_data

    def _read_coco_by_filename(self, root: Path) -> Dict[str, Dict]:
        coco_path = root / "coco.json"
        index_path = root / "index.json"
        data_dir = root / "data"
        if not coco_path.exists():
            raise FileNotFoundError(f"Missing FLIR annotations: {coco_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Missing FLIR metadata: {index_path}")
        if not data_dir.exists():
            raise FileNotFoundError(f"Missing FLIR image directory: {data_dir}")

        with open(coco_path, "r") as f:
            coco = json.load(f)

        anns_by_image = defaultdict(list)
        for ann in coco.get("annotations", []):
            anns_by_image[ann["image_id"]].append(ann)

        result = {}
        for img in coco.get("images", []):
            file_name = img.get("file_name")
            if not file_name:
                continue
            image_path = root / file_name
            if not image_path.exists():
                continue
            parsed = _parse_flir_filename(file_name)
            if parsed is None:
                continue
            video_id, frame_index, dataset_frame_id = parsed
            result[Path(file_name).name] = {
                "path": image_path,
                "width": int(img.get("width", 640)),
                "height": int(img.get("height", 512)),
                "annotations": _pack_flir_annotations(anns_by_image.get(img["id"], [])),
                "image_id": img["id"],
                "file_name": file_name,
                "video_id": video_id,
                "frame_index": frame_index,
                "dataset_frame_id": dataset_frame_id,
            }
        return result

    def _append_image_pairs(self, split_name: str, rgb_root: Path, thermal_root: Path) -> Tuple[int, int]:
        rgb_items, _ = self._read_coco_split(rgb_root)
        thermal_items, thermal_index = self._read_coco_split(thermal_root)
        thermal_to_rgb_video = _extract_rgb_video_map(thermal_index)

        if not thermal_to_rgb_video:
            raise RuntimeError("No FLIR thermal-to-RGB video mappings found in index.json metadata.")

        added = 0
        unpaired_thermal = 0
        for (thermal_video_id, frame_index), thermal_item in sorted(thermal_items.items()):
            rgb_video_id = thermal_to_rgb_video.get(thermal_video_id)
            if rgb_video_id is None:
                unpaired_thermal += 1
                continue
            rgb_item = rgb_items.get((rgb_video_id, frame_index))
            if rgb_item is None:
                unpaired_thermal += 1
                continue
            self.pairs.append(
                {
                    "source_split": split_name,
                    "pair_key": f"{split_name}:{rgb_video_id}:{thermal_video_id}:{frame_index}",
                    "rgb_path": rgb_item["path"],
                    "thermal_path": thermal_item["path"],
                    "rgb_width": rgb_item["width"],
                    "rgb_height": rgb_item["height"],
                    "thermal_width": thermal_item["width"],
                    "thermal_height": thermal_item["height"],
                    "rgb_annotations": rgb_item["annotations"],
                    "thermal_annotations": thermal_item["annotations"],
                    "rgb_video_id": rgb_video_id,
                    "thermal_video_id": thermal_video_id,
                    "frame_index": frame_index,
                    "rgb_file_name": rgb_item["file_name"],
                    "thermal_file_name": thermal_item["file_name"],
                }
            )
            added += 1
        return added, unpaired_thermal

    def _append_video_pairs(self, split_name: str, rgb_root: Path, thermal_root: Path) -> Tuple[int, int]:
        rgb_items = self._read_coco_by_filename(rgb_root)
        thermal_items = self._read_coco_by_filename(thermal_root)
        with open(self._video_map_path(), "r") as f:
            exact_pair_map = _extract_video_pair_map(json.load(f))

        added = 0
        missing = 0
        for rgb_name, thermal_name in exact_pair_map.items():
            rgb_item = rgb_items.get(rgb_name)
            thermal_item = thermal_items.get(thermal_name)
            if rgb_item is None or thermal_item is None:
                missing += 1
                continue
            self.pairs.append(
                {
                    "source_split": split_name,
                    "pair_key": f"{split_name}:{rgb_name}:{thermal_name}",
                    "rgb_path": rgb_item["path"],
                    "thermal_path": thermal_item["path"],
                    "rgb_width": rgb_item["width"],
                    "rgb_height": rgb_item["height"],
                    "thermal_width": thermal_item["width"],
                    "thermal_height": thermal_item["height"],
                    "rgb_annotations": rgb_item["annotations"],
                    "thermal_annotations": thermal_item["annotations"],
                    "rgb_video_id": rgb_item["video_id"],
                    "thermal_video_id": thermal_item["video_id"],
                    "frame_index": rgb_item["frame_index"],
                    "rgb_file_name": rgb_item["file_name"],
                    "thermal_file_name": thermal_item["file_name"],
                }
            )
            added += 1
        return added, missing

    def _load_pairs(self):
        self.pairs = []
        pairing_stats = []
        for split_name, rgb_root, thermal_root in self._split_specs():
            if split_name == "video_test":
                added, missing = self._append_video_pairs(split_name, rgb_root, thermal_root)
                pairing_stats.append((split_name, added, missing, "video pairs missing exact file matches"))
            else:
                added, unpaired = self._append_image_pairs(split_name, rgb_root, thermal_root)
                pairing_stats.append((split_name, added, unpaired, "thermal frames without a recoverable RGB pair"))

        if not self.pairs:
            raise RuntimeError("No FLIR RGB/thermal frame pairs could be constructed from the downloaded metadata.")

        stats_str = "; ".join(
            f"{split_name}: kept {added}, skipped {skipped} {reason}"
            for split_name, added, skipped, reason in pairing_stats
        )
        print(f"FLIR pairing summary: {stats_str}")

    def _setup_transforms(self):
        color_jitter = v2.Compose(
            [
                v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                v2.RandomGrayscale(p=0.2),
            ]
        )
        self.rgb_post_official_global_1 = v2.Compose(
            [
                color_jitter,
                v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.rgb_post_official_global_2 = v2.Compose(
            [
                color_jitter,
                v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.rgb_post_official_local = v2.Compose(
            [
                color_jitter,
                v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.rgb_post_aug = v2.Compose(
            [
                v2.ColorJitter(0.8, 0.8, 0.8, 0.2),
                v2.RandomGrayscale(p=0.2),
                v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.rgb_post_test = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.thermal_post = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(THERMAL_MEAN, THERMAL_STD),
            ]
        )

    def _apply_rgb_post_aug(self, image: Image.Image, *, is_global: bool, view_index: int = 0) -> torch.Tensor:
        if not self.use_official_dino_augs:
            return self.rgb_post_aug(image)
        if is_global:
            pipeline = self.rgb_post_official_global_1 if (view_index % 2 == 0) else self.rgb_post_official_global_2
        else:
            pipeline = self.rgb_post_official_local
        return pipeline(image)

    def __len__(self):
        return len(self.pairs)

    def _sample_normalized_crop(self, image: Image.Image, scale: Tuple[float, float], ratio: Tuple[float, float]) -> Tuple[float, float, float, float]:
        i, j, h, w = v2.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        img_w, img_h = image.size
        return (
            i / float(img_h),
            j / float(img_w),
            h / float(img_h),
            w / float(img_w),
        )

    def _apply_normalized_crop(
        self,
        image: Image.Image,
        crop_box: Tuple[float, float, float, float],
        output_size: int,
    ) -> Image.Image:
        top_frac, left_frac, height_frac, width_frac = crop_box
        img_w, img_h = image.size
        top = int(round(top_frac * img_h))
        left = int(round(left_frac * img_w))
        height = max(1, int(round(height_frac * img_h)))
        width = max(1, int(round(width_frac * img_w)))
        top = min(max(0, top), max(0, img_h - 1))
        left = min(max(0, left), max(0, img_w - 1))
        height = min(height, img_h - top)
        width = min(width, img_w - left)
        return TF.resized_crop(
            image,
            top,
            left,
            height,
            width,
            (output_size, output_size),
            interpolation=InterpolationMode.BICUBIC,
        )

    def _center_crop_to_square(self, image: Image.Image, output_size: int) -> Image.Image:
        orig_w, orig_h = image.size
        scale = output_size / min(orig_w, orig_h)
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        return TF.center_crop(
            TF.resize(image, (new_h, new_w), interpolation=InterpolationMode.BICUBIC),
            (output_size, output_size),
        )

    def _letterbox_to_square(self, image: Image.Image, output_size: int, fill: int = 0) -> Image.Image:
        orig_w, orig_h = image.size
        scale = min(output_size / max(orig_w, 1), output_size / max(orig_h, 1))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = TF.resize(image, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)
        pad_left = (output_size - new_w) // 2
        pad_right = output_size - new_w - pad_left
        pad_top = (output_size - new_h) // 2
        pad_bottom = output_size - new_h - pad_top
        return TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=fill)

    def _resize_probe_view(self, image: Image.Image, output_size: int, fill: int = 0) -> Image.Image:
        if self.resize_mode == "letterbox":
            return self._letterbox_to_square(image, output_size, fill=fill)
        return self._center_crop_to_square(image, output_size)

    def _make_synced_views(
        self,
        rgb_img: Image.Image,
        thermal_img: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_global_views = []
        rgb_local_views = []
        thermal_global_views = []
        thermal_local_views = []
        rgb_probe = torch.empty(0)
        thermal_probe = torch.empty(0)
        ratio = (3.0 / 4.0, 4.0 / 3.0)

        for global_idx in range(self.V):
            crop_box = self._sample_normalized_crop(rgb_img, self.global_crops_scale, ratio)
            rgb_crop_box = crop_box
            if getattr(self, "align_modalities", False):
                rgb_crop_box = (
                    RGB_ALIGN_TOP + crop_box[0] * RGB_ALIGN_HEIGHT,
                    RGB_ALIGN_LEFT + crop_box[1] * RGB_ALIGN_WIDTH,
                    crop_box[2] * RGB_ALIGN_HEIGHT,
                    crop_box[3] * RGB_ALIGN_WIDTH,
                )
            rgb_crop = self._apply_normalized_crop(rgb_img, rgb_crop_box, self.img_size)
            thermal_crop = self._apply_normalized_crop(thermal_img, crop_box, self.img_size)
            if not self.finetune_mode:
                do_flip = np.random.random() < 0.5
                if do_flip:
                    rgb_crop = TF.hflip(rgb_crop)
                    thermal_crop = TF.hflip(thermal_crop)
            rgb_global_views.append(self._apply_rgb_post_aug(rgb_crop, is_global=True, view_index=global_idx))
            thermal_global_views.append(self.thermal_post(thermal_crop))

        if self.include_probe_view:
            # Use synchronized full-image resize for probe views so both
            # modalities share the same spatial layout.  Independent
            # letterboxing produces different scales/padding for RGB
            # (1800x1600) vs thermal (640x512) — a ~2.8x scale mismatch
            # that breaks position-wise fusion in fusion-token encoders.
            probe_crop = (0.0, 0.0, 1.0, 1.0)
            rgb_probe_crop = (RGB_ALIGN_TOP, RGB_ALIGN_LEFT, RGB_ALIGN_HEIGHT, RGB_ALIGN_WIDTH) if getattr(self, "align_modalities", False) else probe_crop
            rgb_clean = self._apply_normalized_crop(rgb_img, rgb_probe_crop, self.probe_img_size)
            thermal_clean = self._apply_normalized_crop(thermal_img, probe_crop, self.probe_img_size)
            rgb_probe = self.rgb_post_test(rgb_clean).unsqueeze(0)
            thermal_probe = self.thermal_post(thermal_clean).unsqueeze(0)

        for local_idx in range(self.local_crops_number):
            crop_box = self._sample_normalized_crop(rgb_img, self.local_crops_scale, ratio)
            rgb_crop_box = crop_box
            if getattr(self, "align_modalities", False):
                rgb_crop_box = (
                    RGB_ALIGN_TOP + crop_box[0] * RGB_ALIGN_HEIGHT,
                    RGB_ALIGN_LEFT + crop_box[1] * RGB_ALIGN_WIDTH,
                    crop_box[2] * RGB_ALIGN_HEIGHT,
                    crop_box[3] * RGB_ALIGN_WIDTH,
                )
            rgb_crop = self._apply_normalized_crop(rgb_img, rgb_crop_box, self.local_img_size)
            thermal_crop = self._apply_normalized_crop(thermal_img, crop_box, self.local_img_size)
            if not self.finetune_mode:
                do_flip = np.random.random() < 0.5
                if do_flip:
                    rgb_crop = TF.hflip(rgb_crop)
                    thermal_crop = TF.hflip(thermal_crop)
            rgb_local_views.append(self._apply_rgb_post_aug(rgb_crop, is_global=False, view_index=local_idx))
            thermal_local_views.append(self.thermal_post(thermal_crop))

        rgb_global = torch.stack(rgb_global_views) if rgb_global_views else torch.empty(0)
        rgb_local = torch.stack(rgb_local_views) if rgb_local_views else torch.empty(0)
        thermal_global = torch.stack(thermal_global_views) if thermal_global_views else torch.empty(0)
        thermal_local = torch.stack(thermal_local_views) if thermal_local_views else torch.empty(0)
        return rgb_global, rgb_local, thermal_global, thermal_local, rgb_probe, thermal_probe

    def _get_global_labels(self, pair: Dict) -> Dict:
        return {
            "scene": 0,
            "camera": 0,
            "location": 0,
        }

    def _project_annotations_to_probe_objects(
        self,
        annotations: np.ndarray,
        orig_w: int,
        orig_h: int,
        is_rgb_source: bool,
    ) -> List[Dict[str, object]]:
        projected_objects: List[Dict[str, object]] = []
        for ann in annotations:
            category_id = int(ann[0])
            det_cls = FLIR_CATEGORY_TO_2D_CLASS.get(category_id, -1)
            if det_cls < 0:
                continue
            probe_box = self._project_box_to_probe_crop(
                ann[1:5],
                orig_w,
                orig_h,
                is_rgb_source=is_rgb_source,
            )
            if probe_box is None:
                continue
            projected_objects.append(
                {
                    'category_id': category_id,
                    'det_cls': det_cls,
                    'class_name': FLIR_2D_DET_CLASS_NAMES[det_cls],
                    'box_xyxy': np.asarray(probe_box, dtype=np.float32),
                }
            )
        return projected_objects

    def _merge_probe_boxes(self, rgb_box: np.ndarray, thermal_box: np.ndarray, det_cls: int, mode: str) -> np.ndarray:
        if mode == 'weighted_union':
            class_name = FLIR_2D_DET_CLASS_NAMES[det_cls]
            rgb_weight = FLIR_WEIGHTED_UNION_RGB_WEIGHT.get(class_name, 0.5)
            thermal_weight = 1.0 - rgb_weight
            merged = rgb_weight * rgb_box + thermal_weight * thermal_box
        else:
            merged = 0.5 * (rgb_box + thermal_box)
        return merged.astype(np.float32)

    def _build_probe_label_objects(self, pair: Dict, mode: str) -> List[Dict[str, object]]:
        mode = str(mode).lower()
        rgb_objects = self._project_annotations_to_probe_objects(
            pair['rgb_annotations'],
            pair['rgb_width'],
            pair['rgb_height'],
            is_rgb_source=True,
        )
        thermal_objects = self._project_annotations_to_probe_objects(
            pair['thermal_annotations'],
            pair['thermal_width'],
            pair['thermal_height'],
            is_rgb_source=False,
        )

        if mode == 'rgb':
            return rgb_objects
        if mode == 'thermal':
            return thermal_objects

        matched_objects: List[Dict[str, object]] = []
        used_thermal = set()
        unmatched_rgb = []
        for rgb_idx, rgb_obj in enumerate(rgb_objects):
            best_idx = -1
            best_iou = -1.0
            for thermal_idx, thermal_obj in enumerate(thermal_objects):
                if thermal_idx in used_thermal or thermal_obj['det_cls'] != rgb_obj['det_cls']:
                    continue
                current_iou = _box_iou_xyxy(rgb_obj['box_xyxy'], thermal_obj['box_xyxy'])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_idx = thermal_idx
            if best_idx >= 0:
                used_thermal.add(best_idx)
                thermal_obj = thermal_objects[best_idx]
                matched_objects.append(
                    {
                        'category_id': rgb_obj['category_id'],
                        'det_cls': rgb_obj['det_cls'],
                        'class_name': rgb_obj['class_name'],
                        'box_xyxy': self._merge_probe_boxes(
                            rgb_obj['box_xyxy'],
                            thermal_obj['box_xyxy'],
                            rgb_obj['det_cls'],
                            mode=mode,
                        ),
                    }
                )
            else:
                unmatched_rgb.append(rgb_obj)

        unmatched_thermal = [obj for idx, obj in enumerate(thermal_objects) if idx not in used_thermal]
        if mode == 'consensus':
            return matched_objects
        if mode in {'union', 'weighted_union'}:
            return matched_objects + unmatched_rgb + unmatched_thermal
        raise ValueError(f"Unsupported FLIR probe label mode: {mode}")

    def _spatial_labels_from_probe_objects(self, probe_objects: List[Dict[str, object]], key_suffix: str = "") -> Dict:
        num_cars = 0
        num_peds = 0
        num_objects = 0
        grid_occupancy = np.zeros(64, dtype=np.float32)
        grid_occupancy_car = np.zeros(64, dtype=np.float32)
        grid_occupancy_ped = np.zeros(64, dtype=np.float32)
        grid_occupancy_hr = np.zeros(self.occupancy_grid_size * self.occupancy_grid_size, dtype=np.float32)
        grid_occupancy_hr_classes = np.zeros(
            (len(FLIR_2D_DET_CLASS_NAMES), self.occupancy_grid_size * self.occupancy_grid_size),
            dtype=np.float32,
        )
        box_seg_map = np.zeros((self.occupancy_grid_size, self.occupancy_grid_size), dtype=np.int64)
        cell_size = self.probe_img_size / 8.0
        cell_size_hr = self.probe_img_size / float(self.occupancy_grid_size)
        projected_boxes = []

        for obj in probe_objects:
            category_id = int(obj['category_id'])
            det_cls = int(obj['det_cls'])
            num_objects += 1
            obj_class = None
            if category_id in FLIR_PEDESTRIAN_CATEGORY_IDS:
                num_peds += 1
                obj_class = 'ped'
            elif category_id in FLIR_VEHICLE_CATEGORY_IDS:
                num_cars += 1
                obj_class = 'car'

            x1, y1, x2, y2 = obj['box_xyxy'].tolist()
            projected_boxes.append((float((x2 - x1) * (y2 - y1)), det_cls, x1, y1, x2, y2))
            col_start = max(0, int(x1 / cell_size))
            col_end = min(7, int((x2 - 1e-6) / cell_size))
            row_start = max(0, int(y1 / cell_size))
            row_end = min(7, int((y2 - 1e-6) / cell_size))
            for row in range(row_start, row_end + 1):
                for col in range(col_start, col_end + 1):
                    cell_idx = row * 8 + col
                    grid_occupancy[cell_idx] = 1.0
                    if obj_class == 'car':
                        grid_occupancy_car[cell_idx] = 1.0
                    elif obj_class == 'ped':
                        grid_occupancy_ped[cell_idx] = 1.0

            col_start_hr = max(0, int(x1 / cell_size_hr))
            col_end_hr = min(self.occupancy_grid_size - 1, int((x2 - 1e-6) / cell_size_hr))
            row_start_hr = max(0, int(y1 / cell_size_hr))
            row_end_hr = min(self.occupancy_grid_size - 1, int((y2 - 1e-6) / cell_size_hr))
            for row in range(row_start_hr, row_end_hr + 1):
                for col in range(col_start_hr, col_end_hr + 1):
                    cell_idx = row * self.occupancy_grid_size + col
                    grid_occupancy_hr[cell_idx] = 1.0
                    grid_occupancy_hr_classes[det_cls, cell_idx] = 1.0

        for _, det_cls, x1, y1, x2, y2 in sorted(projected_boxes, key=lambda item: item[0], reverse=True):
            col_start_hr = max(0, int(x1 / cell_size_hr))
            col_end_hr = min(self.occupancy_grid_size - 1, int((x2 - 1e-6) / cell_size_hr))
            row_start_hr = max(0, int(y1 / cell_size_hr))
            row_end_hr = min(self.occupancy_grid_size - 1, int((y2 - 1e-6) / cell_size_hr))
            box_seg_map[row_start_hr:row_end_hr + 1, col_start_hr:col_end_hr + 1] = det_cls + 1

        return {
            f"num_cars{key_suffix}": num_cars,
            f"num_pedestrians{key_suffix}": num_peds,
            f"num_objects{key_suffix}": num_objects,
            f"mean_depth{key_suffix}": 0.0,
            f"depth_grid{key_suffix}": np.zeros(64, dtype=np.float32),
            f"depth_grid_mask{key_suffix}": np.zeros(64, dtype=np.float32),
            f"depth_grid_hr{key_suffix}": np.zeros(3136, dtype=np.float32),
            f"depth_grid_mask_hr{key_suffix}": np.zeros(3136, dtype=np.float32),
            f"has_boxes{key_suffix}": num_objects > 0,
            f"grid_occupancy{key_suffix}": grid_occupancy,
            f"grid_occupancy_car{key_suffix}": grid_occupancy_car,
            f"grid_occupancy_ped{key_suffix}": grid_occupancy_ped,
            f"grid_occupancy_hr{key_suffix}": grid_occupancy_hr,
            f"grid_occupancy_hr_classes{key_suffix}": grid_occupancy_hr_classes,
            f"box_seg_map{key_suffix}": box_seg_map,
            f"has_box_seg_map{key_suffix}": True,
        }

    def _detection_2d_labels_from_probe_objects(self, probe_objects: List[Dict[str, object]], key_suffix: str = "") -> Dict:
        gt_boxes_2d = np.zeros((self.max_objects_2d, 4), dtype=np.float32)
        gt_classes_2d = np.full(self.max_objects_2d, -1, dtype=np.int64)
        gt_mask_2d = np.zeros(self.max_objects_2d, dtype=np.float32)
        gt_centers_2d = np.zeros((self.max_objects_2d, 2), dtype=np.float32)

        for obj_idx, obj in enumerate(probe_objects[:self.max_objects_2d]):
            x1, y1, x2, y2 = obj['box_xyxy'].tolist()
            gt_boxes_2d[obj_idx] = np.array(
                [x1 / self.probe_img_size, y1 / self.probe_img_size, x2 / self.probe_img_size, y2 / self.probe_img_size],
                dtype=np.float32,
            )
            gt_centers_2d[obj_idx] = np.array(
                [((x1 + x2) * 0.5) / self.probe_img_size, ((y1 + y2) * 0.5) / self.probe_img_size],
                dtype=np.float32,
            )
            gt_classes_2d[obj_idx] = int(obj['det_cls'])
            gt_mask_2d[obj_idx] = 1.0

        return {
            f"gt_boxes_2d{key_suffix}": gt_boxes_2d,
            f"gt_classes_2d{key_suffix}": gt_classes_2d,
            f"gt_mask_2d{key_suffix}": gt_mask_2d,
            f"gt_centers_2d{key_suffix}": gt_centers_2d,
        }

    def _project_box_to_probe_crop(self, box_xywh: List[float], orig_w: int, orig_h: int, is_rgb_source: bool = False) -> Optional[List[float]]:
        """Project a box from original image coordinates to probe-view coordinates.

        Probe views use a synchronized full-image resize (both modalities
        stretched to ``probe_img_size x probe_img_size``), so the mapping is
        simply a per-axis scale.
        """
        x, y, w, h = [float(v) for v in box_xywh]
        if w <= 1 or h <= 1:
            return None

        if getattr(self, "align_modalities", False) and is_rgb_source:
            crop_px_x = RGB_ALIGN_LEFT * orig_w
            crop_px_y = RGB_ALIGN_TOP * orig_h
            crop_px_w = RGB_ALIGN_WIDTH * orig_w
            crop_px_h = RGB_ALIGN_HEIGHT * orig_h
            x = x - crop_px_x
            y = y - crop_px_y
            orig_w = crop_px_w
            orig_h = crop_px_h

        # Synchronized stretch: full image → (probe_img_size, probe_img_size)
        sx = self.probe_img_size / max(orig_w, 1)
        sy = self.probe_img_size / max(orig_h, 1)

        x1 = x * sx
        y1 = y * sy
        x2 = (x + w) * sx
        y2 = (y + h) * sy

        x1 = max(0.0, min(float(self.probe_img_size), x1))
        y1 = max(0.0, min(float(self.probe_img_size), y1))
        x2 = max(0.0, min(float(self.probe_img_size), x2))
        y2 = max(0.0, min(float(self.probe_img_size), y2))

        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _select_label_source(self, pair: Dict, source: Optional[str] = None) -> Tuple[np.ndarray, int, int, bool]:
        label_source = str(source or self.detection_label_source).lower()
        if label_source == "thermal":
            return pair["thermal_annotations"], pair["thermal_width"], pair["thermal_height"], False
        return pair["rgb_annotations"], pair["rgb_width"], pair["rgb_height"], True

    def _get_spatial_labels(self, pair: Dict, source: Optional[str] = None, key_suffix: str = "") -> Dict:
        annotations, orig_w, orig_h, is_rgb_source = self._select_label_source(pair, source)
        probe_objects = self._project_annotations_to_probe_objects(
            annotations,
            orig_w,
            orig_h,
            is_rgb_source=is_rgb_source,
        )
        return self._spatial_labels_from_probe_objects(probe_objects, key_suffix=key_suffix)

    def _get_detection_2d_labels(self, pair: Dict, source: Optional[str] = None, key_suffix: str = "") -> Dict:
        annotations, orig_w, orig_h, is_rgb_source = self._select_label_source(pair, source)
        probe_objects = self._project_annotations_to_probe_objects(
            annotations,
            orig_w,
            orig_h,
            is_rgb_source=is_rgb_source,
        )
        return self._detection_2d_labels_from_probe_objects(probe_objects, key_suffix=key_suffix)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        labels = self._get_global_labels(pair)
        if not self.encoder_only_labels:
            labels.update(self._get_spatial_labels(pair, source="rgb"))
            labels.update(self._get_spatial_labels(pair, source="thermal", key_suffix="_thermal"))
            labels.update(self._spatial_labels_from_probe_objects(self._build_probe_label_objects(pair, mode='consensus'), key_suffix="_consensus"))
            labels.update(self._spatial_labels_from_probe_objects(self._build_probe_label_objects(pair, mode='union'), key_suffix="_union"))
            labels.update(self._spatial_labels_from_probe_objects(self._build_probe_label_objects(pair, mode='weighted_union'), key_suffix="_weighted_union"))

        try:
            rgb_img = Image.open(pair["rgb_path"]).convert("RGB")
        except Exception:
            rgb_img = Image.new("RGB", (pair["rgb_width"], pair["rgb_height"]), "black")

        try:
            thermal_img = Image.open(pair["thermal_path"]).convert("L")
        except Exception:
            thermal_img = Image.new("L", (pair["thermal_width"], pair["thermal_height"]), 0)

        rgb_global, rgb_local, thermal_global, thermal_local, rgb_probe, thermal_probe = self._make_synced_views(rgb_img, thermal_img)

        cam_views = {
            "global": rgb_global,
            "local": rgb_local,
            "probe": rgb_probe,
        }
        modality2 = {
            "global": thermal_global,
            "local": thermal_local,
            "probe": thermal_probe,
        }

        if not self.legacy_mode and not self.encoder_only_labels:
            labels.update(
                {
                    "gt_classes": np.full(50, -1, dtype=np.int64),
                    "gt_centers": np.zeros((50, 3), dtype=np.float32),
                    "gt_sizes": np.zeros((50, 3), dtype=np.float32),
                    "gt_orientations": np.zeros((50, 2), dtype=np.float32),
                    "gt_mask": np.zeros(50, dtype=np.float32),
                    "seg_map": np.zeros(FLIR_UNUSED_SEG_PLACEHOLDER_SHAPE, dtype=np.int64),
                    "panoptic_seg_map": np.zeros(FLIR_UNUSED_SEG_PLACEHOLDER_SHAPE, dtype=np.int64),
                    "has_seg_map": False,
                    "has_panoptic_seg_map": False,
                }
            )
            labels.update(self._get_detection_2d_labels(pair, source="rgb"))
            labels.update(self._get_detection_2d_labels(pair, source="thermal", key_suffix="_thermal"))
            labels.update(self._detection_2d_labels_from_probe_objects(self._build_probe_label_objects(pair, mode='consensus'), key_suffix="_consensus"))
            labels.update(self._detection_2d_labels_from_probe_objects(self._build_probe_label_objects(pair, mode='union'), key_suffix="_union"))
            labels.update(self._detection_2d_labels_from_probe_objects(self._build_probe_label_objects(pair, mode='weighted_union'), key_suffix="_weighted_union"))

        if self.modality_dropout > 0 and self.split == "train":
            if np.random.random() < self.modality_dropout:
                modality2 = {
                    "global": torch.zeros_like(modality2["global"]),
                    "local": torch.zeros_like(modality2["local"]),
                    "probe": torch.zeros_like(modality2["probe"]),
                }

        return cam_views, modality2, labels


def flir_collate_fn(batch):
    """Custom collate function matching the MM-LeJEPA dict/tensor contract."""
    cam_views_list, modality2_list, labels_list = zip(*batch)

    cam_views = {
        "global": torch.stack([cv["global"] for cv in cam_views_list]),
        "local": (
            torch.stack([cv["local"] for cv in cam_views_list])
            if all(cv["local"].numel() > 0 for cv in cam_views_list)
            else torch.empty(0)
        ),
        "probe": (
            torch.stack([cv["probe"] for cv in cam_views_list])
            if all(cv["probe"].numel() > 0 for cv in cam_views_list)
            else torch.empty(0)
        ),
    }
    modality2 = {
        "global": torch.stack([mv["global"] for mv in modality2_list]),
        "local": (
            torch.stack([mv["local"] for mv in modality2_list])
            if all(mv["local"].numel() > 0 for mv in modality2_list)
            else torch.empty(0)
        ),
        "probe": (
            torch.stack([mv["probe"] for mv in modality2_list])
            if all(mv["probe"].numel() > 0 for mv in modality2_list)
            else torch.empty(0)
        ),
    }

    labels = {}
    for key in labels_list[0].keys():
        values = [lbl[key] for lbl in labels_list]
        if isinstance(values[0], (int, float)):
            labels[key] = torch.tensor(values)
        elif isinstance(values[0], np.ndarray):
            labels[key] = torch.from_numpy(np.stack(values))
        elif isinstance(values[0], torch.Tensor):
            labels[key] = torch.stack(values)
        elif isinstance(values[0], (bool, np.bool_)):
            labels[key] = torch.tensor(values, dtype=torch.bool)
        else:
            labels[key] = values

    return cam_views, modality2, labels