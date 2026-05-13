"""
LiDAR utilities for MM-LeJEPA.

Functions for loading NuScenes LiDAR point clouds and converting them to
various representations (range images, depth maps) for multi-modal learning.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_lidar_bin(path: str, keep_all_features: bool = False) -> np.ndarray:
    """
    Load NuScenes or Waymo LiDAR point cloud from .pcd.bin file.
    
    NuScenes format: N x 5 float32 (x, y, z, intensity, ring_index)
    Waymo format: N x 6 float32 (x, y, z, intensity, ring_index, semantic_label)
    
    Args:
        path: Path to .pcd.bin file
        keep_all_features: If True, do not truncate the 6th semantic column
        
    Returns:
        points: (N, C) array of points
    """
    data = np.fromfile(path, dtype=np.float32)
    
    if data.size % 5 == 0 and data.size % 6 != 0:
        points = data.reshape(-1, 5)
    elif data.size % 6 == 0 and data.size % 5 != 0:
        points = data.reshape(-1, 6)
        if not keep_all_features:
            points = points[:, :5]
    elif data.size % 5 == 0 and data.size % 6 == 0:
        if 'waymo' in path.lower():
            points = data.reshape(-1, 6)
            if not keep_all_features:
                points = points[:, :5]
        else:
            points = data.reshape(-1, 5)
    else:
        points = data.reshape(-1, 5)
        
    return points


def lidar_to_range_image(
    points: np.ndarray,
    H: int = 64,
    W: int = 1024,
    fov_up: float = 10.0,
    fov_down: float = -30.0,
    max_range: float = 80.0,
) -> np.ndarray:
    """
    Convert 3D point cloud to 2D range image using cylindrical projection.
    
    This creates a 2D image where each pixel contains depth and other features,
    making it suitable for processing with standard 2D CNNs or ViTs.
    
    Args:
        points: (N, 5) point cloud [x, y, z, intensity, ring]
        H: Height of range image (number of vertical bins)
        W: Width of range image (number of horizontal bins)
        fov_up: Upper vertical FOV in degrees
        fov_down: Lower vertical FOV in degrees
        max_range: Maximum range to normalize depth values
        
    Returns:
        range_image: (H, W, 5) array with channels:
            [depth, intensity, x, y, z]
    """
    # Extract coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    
    # Compute depth (range)
    depth = np.sqrt(x**2 + y**2 + z**2)
    
    # Filter out invalid points
    valid = depth > 0.1
    x, y, z, intensity, depth = x[valid], y[valid], z[valid], intensity[valid], depth[valid]
    
    # Compute angles
    # Yaw (horizontal angle): atan2(y, x) -> [-pi, pi]
    yaw = np.arctan2(y, x)
    
    # Pitch (vertical angle): asin(z / depth)
    pitch = np.arcsin(z / depth)
    
    # Convert FOV to radians
    fov_up_rad = np.deg2rad(fov_up)
    fov_down_rad = np.deg2rad(fov_down)
    fov_range = fov_up_rad - fov_down_rad
    
    # Map angles to image coordinates
    # Horizontal: yaw [-pi, pi] -> [0, W-1]
    col = ((yaw + np.pi) / (2 * np.pi)) * W
    col = np.clip(col, 0, W - 1).astype(np.int32)
    
    # Vertical: pitch [fov_down, fov_up] -> [H-1, 0]
    row = (1.0 - (pitch - fov_down_rad) / fov_range) * H
    row = np.clip(row, 0, H - 1).astype(np.int32)
    
    # Initialize range image (5 channels: depth, intensity, x, y, z)
    range_image = np.zeros((H, W, 5), dtype=np.float32)
    
    # Fill in range image (later points overwrite earlier - closer gets priority)
    # Sort by depth descending so closer points overwrite farther ones
    order = np.argsort(-depth)
    row, col = row[order], col[order]
    depth, intensity = depth[order], intensity[order]
    x, y, z = x[order], y[order], z[order]
    
    range_image[row, col, 0] = depth / max_range  # normalize depth
    range_image[row, col, 1] = intensity / 255.0  # normalize intensity (approx)
    range_image[row, col, 2] = x / max_range      # normalize coordinates
    range_image[row, col, 3] = y / max_range
    range_image[row, col, 4] = z / max_range
    
    return range_image


def range_image_to_rgb(range_image: np.ndarray) -> np.ndarray:
    """
    Convert range image to RGB for visualization.
    
    Args:
        range_image: (H, W, 5) range image
        
    Returns:
        rgb: (H, W, 3) RGB image (depth=R, intensity=G, height=B)
    """
    rgb = np.zeros((range_image.shape[0], range_image.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = (range_image[:, :, 0] * 255).astype(np.uint8)  # depth -> red
    rgb[:, :, 1] = (range_image[:, :, 1] * 255).astype(np.uint8)  # intensity -> green
    rgb[:, :, 2] = ((range_image[:, :, 4] + 0.5) * 255).astype(np.uint8)  # z -> blue
    return rgb


def lidar_to_depth_map(
    points: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_rotation: np.ndarray,
    extrinsic_translation: np.ndarray,
    img_size: Tuple[int, int] = (900, 1600),
    max_depth: float = 80.0,
) -> np.ndarray:
    """
    Project 3D LiDAR points to camera view to create sparse depth map.
    
    NOTE: This is a simplified 2-step transform (lidar→ego→camera) that ignores
    ego pose differences between LiDAR and camera capture times.
    For proper multi-sweep scenarios, use lidar_to_depth_map_full().
    
    Args:
        points: (N, 5) point cloud in LiDAR frame
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic_rotation: (3, 3) rotation from LiDAR to camera
        extrinsic_translation: (3,) translation from LiDAR to camera
        img_size: (H, W) target image size
        max_depth: Maximum depth for normalization
        
    Returns:
        depth_map: (H, W) sparse depth map (0 = no depth)
    """
    H, W = img_size
    
    # Extract xyz
    xyz = points[:, :3]
    
    # Transform to camera frame
    xyz_cam = (extrinsic_rotation @ xyz.T).T + extrinsic_translation
    
    # Filter points behind camera
    valid = xyz_cam[:, 2] > 0.1
    xyz_cam = xyz_cam[valid]
    
    if len(xyz_cam) == 0:
        return np.zeros((H, W), dtype=np.float32)
    
    # Project to image plane
    uv = (intrinsic @ xyz_cam.T).T
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    depth = xyz_cam[:, 2]
    
    # Filter points outside image
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, depth = u[valid].astype(np.int32), v[valid].astype(np.int32), depth[valid]
    
    # Create depth map
    depth_map = np.zeros((H, W), dtype=np.float32)
    
    # Sort by depth descending so closer points have priority
    order = np.argsort(-depth)
    u, v, depth = u[order], v[order], depth[order]
    
    depth_map[v, u] = depth / max_depth
    
    return depth_map


def lidar_to_depth_map_full(
    points: np.ndarray,
    intrinsic: np.ndarray,
    lidar_to_cam_transform: np.ndarray,
    img_size: Tuple[int, int] = (900, 1600),
    max_depth: float = 80.0,
    min_dist: float = 1.0,
) -> np.ndarray:
    """
    Project 3D LiDAR points to camera view using full 4-step transformation.
    
    This properly accounts for ego pose differences between LiDAR and camera
    capture times by transforming through world coordinates:
        LiDAR → Ego(lidar time) → World → Ego(cam time) → Camera
    
    Args:
        points: (N, 5) point cloud in LiDAR frame [x, y, z, intensity, ring]
        intrinsic: (3, 3) camera intrinsic matrix
        lidar_to_cam_transform: (4, 4) full transformation matrix from LiDAR to camera
                                (should be computed as: ego_to_cam @ world_to_cam_ego @ 
                                 lidar_ego_to_world @ lidar_to_ego)
        img_size: (H, W) target image size
        max_depth: Maximum depth for normalization
        min_dist: Minimum distance from camera to include points
        
    Returns:
        depth_map: (H, W) sparse depth map (0 = no depth, values normalized by max_depth)
    """
    H, W = img_size
    
    # Extract xyz and convert to homogeneous coordinates
    xyz = points[:, :3]
    n_points = xyz.shape[0]
    points_h = np.vstack([xyz.T, np.ones(n_points)])  # (4, N) homogeneous
    
    # Transform points to camera frame using full transform
    points_cam = lidar_to_cam_transform @ points_h  # (4, N)
    
    # Extract depths (Z in camera frame)
    depths = points_cam[2, :]
    
    # Filter points behind the camera
    valid_depth = depths > min_dist
    points_cam = points_cam[:, valid_depth]
    depths = depths[valid_depth]
    
    if len(depths) == 0:
        return np.zeros((H, W), dtype=np.float32)
    
    # Project to image plane
    points_2d_h = intrinsic @ points_cam[:3, :]  # (3, N)
    points_2d = points_2d_h[:2, :] / points_2d_h[2, :]  # (2, N) normalize
    u = points_2d[0, :]
    v = points_2d[1, :]
    
    # Filter points outside image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_bounds].astype(np.int32)
    v = v[in_bounds].astype(np.int32)
    depths = depths[in_bounds]
    
    if len(depths) == 0:
        return np.zeros((H, W), dtype=np.float32)
    
    # Clip depths and normalize
    depths = np.clip(depths, 0, max_depth) / max_depth
    
    # Create depth map - sort by depth descending so closer points have priority
    depth_map = np.zeros((H, W), dtype=np.float32)
    order = np.argsort(-depths)
    u, v, depths = u[order], v[order], depths[order]
    
    depth_map[v, u] = depths
    
    return depth_map


def lidar_to_aligned_points(
    points: np.ndarray,
    intrinsic: np.ndarray,
    lidar_to_cam_transform: np.ndarray,
    img_size: Tuple[int, int] = (900, 1600),
    max_depth: float = 80.0,
    min_dist: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 3D LiDAR points that are visible within the camera's field of view.
    
    Unlike lidar_to_depth_map_full which returns a 2D depth image, this returns
    the actual 3D points in camera frame that project within the image bounds.
    This is useful for Architecture A which uses PointMLP on 3D points.
    
    Args:
        points: (N, 5) point cloud in LiDAR frame [x, y, z, intensity, ring]
        intrinsic: (3, 3) camera intrinsic matrix
        lidar_to_cam_transform: (4, 4) full transformation matrix from LiDAR to camera
        img_size: (H, W) camera image size for FOV filtering
        max_depth: Maximum depth to include
        min_dist: Minimum distance from camera to include points
        
    Returns:
        points_cam: (M, 5) points in camera frame [x, y, z, intensity, ring]
                    Only includes points visible in camera FOV
        uv: (M, 2) corresponding 2D pixel coordinates [u, v]
    """
    H, W = img_size
    
    # Extract xyz and convert to homogeneous coordinates
    xyz = points[:, :3]
    n_points = xyz.shape[0]
    points_h = np.vstack([xyz.T, np.ones(n_points)])  # (4, N) homogeneous
    
    # Transform points to camera frame using full transform
    points_cam_h = lidar_to_cam_transform @ points_h  # (4, N)
    points_cam = points_cam_h[:3, :].T  # (N, 3) [x, y, z] in camera frame
    
    # Extract depths (Z in camera frame)
    depths = points_cam[:, 2]
    
    # Filter by depth range
    valid_depth = (depths > min_dist) & (depths < max_depth)
    
    # Project to image plane for FOV check
    points_2d_h = intrinsic @ points_cam.T  # (3, N)
    u = points_2d_h[0, :] / (points_2d_h[2, :] + 1e-8)
    v = points_2d_h[1, :] / (points_2d_h[2, :] + 1e-8)
    
    # Filter points within image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    # Combine all validity conditions
    valid = valid_depth & in_bounds
    
    # Extract valid points
    points_cam_valid = points_cam[valid]
    intensity_ring = points[valid, 3:5]  # Keep intensity and ring index
    points_out = np.hstack([points_cam_valid, intensity_ring])  # (M, 5)
    
    uv = np.stack([u[valid], v[valid]], axis=1)  # (M, 2)
    
    return points_out, uv


def subsample_points(points: np.ndarray, n_points: int = 16384) -> np.ndarray:
    """
    Subsample point cloud to fixed number of points.
    
    Uses random sampling if too many points, or replication if too few.
    
    Args:
        points: (N, C) point cloud
        n_points: Target number of points
        
    Returns:
        subsampled: (n_points, C) point cloud
    """
    N = points.shape[0]
    
    if N >= n_points:
        # Random subsampling
        indices = np.random.choice(N, n_points, replace=False)
    else:
        # Replicate points to reach target
        indices = np.random.choice(N, n_points, replace=True)
    
    return points[indices]


def normalize_points(points: np.ndarray, center: bool = True) -> np.ndarray:
    """
    Normalize point cloud coordinates.
    
    Args:
        points: (N, C) point cloud where first 3 channels are xyz
        center: Whether to center the point cloud
        
    Returns:
        normalized: (N, C) normalized point cloud
    """
    normalized = points.copy()
    xyz = normalized[:, :3]
    
    if center:
        centroid = xyz.mean(axis=0)
        xyz = xyz - centroid
    
    # Scale to unit sphere
    max_dist = np.max(np.linalg.norm(xyz, axis=1))
    if max_dist > 0:
        xyz = xyz / max_dist
    
    normalized[:, :3] = xyz
    return normalized


if __name__ == "__main__":
    # Quick test
    import glob
    
    dataroot = sys.argv[1] if len(sys.argv) > 1 else "/path/to/nuscenes_data"
    lidar_files = glob.glob(os.path.join(dataroot, "samples/LIDAR_TOP/*.pcd.bin"))
    if lidar_files:
        pts = load_lidar_bin(lidar_files[0])
        print(f"Loaded {pts.shape[0]} points with shape {pts.shape}")
        print(f"XYZ range: x=[{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}], "
              f"y=[{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}], "
              f"z=[{pts[:, 2].min():.1f}, {pts[:, 2].max():.1f}]")
        
        range_img = lidar_to_range_image(pts)
        print(f"Range image shape: {range_img.shape}")
        print(f"Non-zero pixels: {(range_img[:, :, 0] > 0).sum()}")
        
        # Save visualization
        rgb = range_image_to_rgb(range_img)
        from PIL import Image
        Image.fromarray(rgb).save("range_image_test.png")
        print("Saved range image visualization to range_image_test.png")
