import random
import subprocess
import shutil
import numpy as np
import glob
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import trimesh
import pycolmap

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import get_batches_with_overlap
from demo_colmap import (
    parse_args,
    run_VGGT,
    rename_colmap_recons_and_rescale_camera,
    save_debug_sparse,
)
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap

def main(args):
    print("Arguments:", vars(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)

    image_dir = os.path.join(args.scene_dir, "tmp")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    image_path_list = sorted(image_path_list, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    base_image_path_list = [os.path.basename(p) for p in image_path_list]

    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    batches = get_batches_with_overlap(image_path_list, args.batch_size, args.overlap)

    if args.debug:
        debug_dir = os.path.join(args.scene_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(os.path.join(debug_dir, "batches"), exist_ok=True)

    batch_dirs = []
    for batch_idx, batch_paths in enumerate(batches):
        print(f"Processing batch {batch_idx} with {len(batch_paths)} images")
        imgs, coords = load_and_preprocess_images_square(batch_paths, img_load_resolution)
        imgs = imgs.to(device)
        extr_b, intr_b, depth_b, conf_b = run_VGGT(model, imgs, dtype, vggt_fixed_resolution)

        if args.debug:
            batch_debug_dir = os.path.join(debug_dir, "batches", f"batch_{batch_idx}")
            save_debug_sparse(
                extr_b,
                intr_b,
                depth_b,
                conf_b,
                imgs.cpu(),
                coords.numpy(),
                [os.path.basename(p) for p in batch_paths],
                batch_debug_dir,
                args.conf_thres_value,
                vggt_fixed_resolution,
            )

        points_3d = unproject_depth_map_to_point_map(depth_b, extr_b, intr_b)

        image_size = np.array(imgs.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        intr_for_ba = intr_b.copy()
        intr_for_ba[:, :2, :] *= scale

        with torch.cuda.amp.autocast(dtype=dtype):
            t_b, v_b, c_b, p3d_b, color_b = predict_tracks(
                imgs,
                conf=conf_b,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

        track_mask = v_b > args.vis_thresh
        reconstruction_b, _ = batch_np_matrix_to_pycolmap(
            p3d_b,
            extr_b,
            intr_for_ba,
            t_b,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=args.shared_camera,
            camera_type=args.camera_type,
            points_rgb=color_b,
        )
        if reconstruction_b is None:
            raise ValueError(f"No reconstruction can be built for batch {batch_idx}")

        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction_b, ba_options)

        recon_b = rename_colmap_recons_and_rescale_camera(
            reconstruction_b,
            [os.path.basename(p) for p in batch_paths],
            coords.numpy(),
            img_size=img_load_resolution,
            shift_point2d_to_original_res=True,
            shared_camera=args.shared_camera,
            center_pp=False,
        )

        batch_dir = os.path.join(args.scene_dir, "sparse_batches", f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)
        recon_b.write(batch_dir)
        trimesh.PointCloud(p3d_b, colors=color_b).export(os.path.join(batch_dir, "points.ply"))
        batch_dirs.append(batch_dir)

    merged_dir = os.path.join(args.scene_dir, "sparse_merged")
    os.makedirs(merged_dir, exist_ok=True)
    if len(batch_dirs) == 1:
        shutil.copytree(batch_dirs[0], merged_dir, dirs_exist_ok=True)
    else:
        current_dir = batch_dirs[0]
        for midx, nxt in enumerate(batch_dirs[1:], start=1):
            tmp_dir = os.path.join(args.scene_dir, f"tmp_merge_{midx}")
            subprocess.run([
                "colmap",
                "model_merger",
                f"--input_path1={current_dir}",
                f"--input_path2={nxt}",
                f"--output_path={tmp_dir}",
            ], check=True)
            current_dir = tmp_dir
        shutil.move(current_dir, merged_dir)

if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        main(args)
