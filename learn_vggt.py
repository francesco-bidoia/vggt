# -*- coding: utf-8 -*-
"""A step-by-step version of ``demo_colmap.py``
================================================

This script exposes the same command line interface as ``demo_colmap.py`` but
presents the pipeline in a linear and heavily commented manner. It illustrates
how to use :mod:`vggt` for camera and depth prediction and how to convert the
results to the COLMAP format, optionally running bundle adjustment.

Usage
-----
```
python learn_vggt.py --scene_dir <folder> [--use_ba] [other options]
```
The ``scene_dir`` must contain an ``images`` folder with the input frames.
The resulting reconstruction is saved to ``scene_dir/sparse``.
"""

import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import pycolmap

# VGGT model and utility functions
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)


# -----------------------------------------------------------------------------
# Argument parser (identical options to demo_colmap.py)
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VGGT demo (learn_vggt)")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False,
                        help="Use bundle adjustment for reconstruction")
    # Bundle adjustment related parameters
    parser.add_argument("--max_reproj_error", type=float, default=8.0,
                        help="Maximum reprojection error for reconstruction")
    parser.add_argument("--shared_camera", action="store_true", default=False,
                        help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE",
                        help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2,
                        help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=5,
                        help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=2048,
                        help="Maximum number of query points")
    parser.add_argument("--fine_tracking", action="store_true", default=True,
                        help="Use fine tracking (slower but more accurate)")
    parser.add_argument("--conf_thres_value", type=float, default=5.0,
                        help="Confidence threshold for depth filtering when BA is disabled")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# VGGT forward pass helper
# -----------------------------------------------------------------------------

def run_vggt(model: VGGT, images: torch.Tensor, dtype: torch.dtype,
             resolution: int = 518):
    """Run VGGT on a batch of images to predict cameras and depth."""
    assert images.dim() == 4 and images.shape[1] == 3

    # Resize to the resolution expected by VGGT
    images = F.interpolate(images, size=(resolution, resolution),
                          mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens, ps_idx = model.aggregator(images)

        # Camera and depth prediction heads
        pose_enc = model.camera_head(aggregated_tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc,
                                                            images.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens, images, ps_idx)

    # Convert to numpy arrays and remove batch dimension
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


# -----------------------------------------------------------------------------
# Main demo logic
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Run the full reconstruction pipeline."""

    # 1. Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Using seed: {args.seed}")

    # 2. Configure device and precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Device : {device}")
    print(f"Dtype  : {dtype}")

    # 3. Load the pretrained VGGT model
    print("Loading VGGT model ...")
    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    model.eval()
    model = model.to(device)
    print("Model ready")

    # 4. Load and preprocess input images
    image_dir = os.path.join(args.scene_dir, "images")
    image_paths = glob.glob(os.path.join(image_dir, "*"))
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_names = [os.path.basename(p) for p in image_paths]

    vggt_res = 518          # VGGT internal resolution
    load_res = 1024         # resolution used when loading images
    images, original_coords = load_and_preprocess_images_square(image_paths, load_res)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # 5. Run VGGT to estimate cameras and depth maps
    extrinsic, intrinsic, depth_map, depth_conf = run_vggt(model, images, dtype, vggt_res)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # 6. Optional bundle adjustment
    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = load_res / vggt_res
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
            pred_tracks, pred_vis, pred_confs, points_3d, point_colors = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )
            torch.cuda.empty_cache()

        # Rescale intrinsics from 518 → 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis > args.vis_thresh

        reconstruction, _ = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=point_colors,
        )
        if reconstruction is None:
            raise ValueError("No reconstruction could be built using BA")

        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)
        reconstruction_resolution = load_res
    else:
        # Feed-forward mode (no BA)
        conf_thres = args.conf_thres_value
        max_pts = 100000  # limit number of exported 3D points
        shared_camera = False
        camera_type = "PINHOLE"

        image_size = np.array([vggt_res, vggt_res])
        num_frames, height, width, _ = points_3d.shape

        point_colors = F.interpolate(images, size=(vggt_res, vggt_res),
                                     mode="bilinear", align_corners=False)
        point_colors = (point_colors.cpu().numpy() * 255).astype(np.uint8)
        point_colors = point_colors.transpose(0, 2, 3, 1)

        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        mask = depth_conf >= conf_thres
        mask = randomly_limit_trues(mask, max_pts)

        points_3d = points_3d[mask]
        points_xyf = points_xyf[mask]
        point_colors = point_colors[mask]

        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            point_colors,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )
        reconstruction_resolution = vggt_res

    # 7. Rescale cameras to original resolution and rename images
    reconstruction = rename_and_rescale_cameras(
        reconstruction,
        base_names,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d=True,
        shared_camera=shared_camera,
    )

    # 8. Save COLMAP files and a PLY point cloud
    sparse_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    reconstruction.write(sparse_dir)

    trimesh.PointCloud(points_3d, colors=point_colors).export(
        os.path.join(args.scene_dir, "sparse/points.ply")
    )
    print(f"Reconstruction written to {sparse_dir}")


# -----------------------------------------------------------------------------
# Helper to rename images and rescale camera parameters
# -----------------------------------------------------------------------------

def rename_and_rescale_cameras(reconstruction: pycolmap.Reconstruction,
                               image_paths: list,
                               original_coords: np.ndarray,
                               img_size: int,
                               shift_point2d: bool = False,
                               shared_camera: bool = False):
    """Update image names and adjust camera parameters to match original size."""
    rescale_camera = True
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            params = pycamera.params.copy()
            real_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_size) / img_size
            params = params * resize_ratio
            params[-2:] = real_size / 2  # principal point at image center
            pycamera.params = params
            pycamera.width = real_size[0]
            pycamera.height = real_size[1]

        if shift_point2d:
            top_left = original_coords[pyimageid - 1, :2]
            for p2d in pyimage.points2D:
                p2d.xy = (p2d.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False
    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        main(args)
