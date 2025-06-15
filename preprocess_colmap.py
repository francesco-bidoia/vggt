import argparse
import glob
import os
import shutil
import subprocess
from typing import List

import trimesh

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF
from tqdm import tqdm

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import (
    create_pixel_coordinate_grid,
    get_batches_with_overlap,
    randomly_limit_trues,
)
from vggt.utils.load_fn import load_and_preprocess_images_square
from demo_colmap import (
    rename_colmap_recons_and_rescale_camera,
    run_VGGT,
)
from vggt.dependency.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)
from vggt.dependency.track_predict import predict_tracks


def stage_resize(scene_dir: str, target_res: int) -> List[str]:
    """Load images from ``tmp`` and save square resized versions to ``tmp2``."""

    src_dir = os.path.join(scene_dir, "tmp")
    dst_dir = os.path.join(scene_dir, "tmp2")
    os.makedirs(dst_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(src_dir, "*")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )

    coords = []
    names = []
    to_pil = TF.ToPILImage()

    for p in tqdm(image_paths, desc="resize"):
        img, coord = load_and_preprocess_images_square([p], target_res)
        name = os.path.basename(p)
        to_pil(img[0]).save(os.path.join(dst_dir, name))
        coords.append(coord.numpy()[0])
        names.append(name)

    np.save(os.path.join(dst_dir, "coords.npy"), np.array(coords))
    return names


def stage_batches(scene_dir: str, names: List[str], batch_size: int, overlap: int) -> List[List[int]]:
    """Generate batch indices and save them to ``batches.txt``."""

    idxs = [int(os.path.splitext(n)[0]) for n in names]
    batches = get_batches_with_overlap(idxs, batch_size, overlap)
    batch_file = os.path.join(scene_dir, "batches.txt")
    with open(batch_file, "w") as f:
        for b in batches:
            f.write(" ".join(str(i) for i in b) + "\n")
    return batches


def stage_vggt(
    scene_dir: str,
    batches: List[List[int]],
    target_res: int,
    vggt_res: int,
    dtype: torch.dtype,
    device: str,
    conf_thres: float,
) -> None:
    """Run VGGT on each batch and save outputs under ``batches/batch_*``."""

    preprocess_dir = os.path.join(scene_dir, "tmp2")
    coords = np.load(os.path.join(preprocess_dir, "coords.npy"))
    to_tensor = TF.ToTensor()

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval().to(device)

    for b in batches:
        start, end = b[0], b[-1]
        batch_dir = os.path.join(scene_dir, "batches", f"batch_{start}_{end}")
        os.makedirs(batch_dir, exist_ok=True)

        names = [f"{i:08d}.png" for i in b]
        images = torch.stack(
            [to_tensor(Image.open(os.path.join(preprocess_dir, n))) for n in names]
        ).to(device)

        extr, intr, depth, conf = run_VGGT(model, images, dtype, vggt_res)
        np.savez(
            os.path.join(batch_dir, "vggt_outputs.npz"),
            extr=extr,
            intr=intr,
            depth=depth,
            conf=conf,
            indices=b,
        )

        points_3d = unproject_depth_map_to_point_map(depth, extr, intr)
        image_size = np.array([vggt_res, vggt_res])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_res, vggt_res), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = conf >= conf_thres
        conf_mask = randomly_limit_trues(conf_mask, 100000)

        p3d = points_3d[conf_mask]
        pxyf = points_xyf[conf_mask]
        prgb = points_rgb[conf_mask]

        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            p3d, pxyf, prgb, extr, intr, image_size
        )

        reconstruction = rename_colmap_recons_and_rescale_camera(
            reconstruction,
            names,
            coords[b],
            img_size=vggt_res,
            shift_point2d_to_original_res=True,
            shared_camera=False,
            center_pp=False,
        )

        sparse_dir = os.path.join(batch_dir, "sparse_orig")
        os.makedirs(sparse_dir, exist_ok=True)
        reconstruction.write(sparse_dir)
        trimesh.PointCloud(p3d, colors=prgb).export(os.path.join(sparse_dir, "points.ply"))

    del model
    torch.cuda.empty_cache()


def stage_tracking(
    scene_dir: str,
    batches: List[List[int]],
    target_res: int,
    vggt_res: int,
    dtype: torch.dtype,
    device: str,
    vis_thresh: float,
    max_reproj_error: float,
    query_frame_num: int,
    max_query_pts: int,
    fine_tracking: bool,
    shared_camera: bool,
    camera_type: str,
) -> None:
    """Run tracking and BA for each batch."""

    preprocess_dir = os.path.join(scene_dir, "tmp2")
    coords = np.load(os.path.join(preprocess_dir, "coords.npy"))
    to_tensor = TF.ToTensor()

    for b in batches:
        start, end = b[0], b[-1]
        batch_dir = os.path.join(scene_dir, "batches", f"batch_{start}_{end}")
        data = np.load(os.path.join(batch_dir, "vggt_outputs.npz"))
        extr = data["extr"]
        intr = data["intr"]
        depth = data["depth"]
        conf = data["conf"]
        indices = data["indices"]

        names = [f"{i:08d}.png" for i in indices]
        images = torch.stack(
            [to_tensor(Image.open(os.path.join(preprocess_dir, n))) for n in names]
        ).to(device)

        points_3d = unproject_depth_map_to_point_map(depth, extr, intr)

        intr_ba = intr.copy()
        intr_ba[:, :2, :] *= float(target_res) / vggt_res

        with torch.cuda.amp.autocast(dtype=dtype):
            t_b, v_b, c_b, p3d_b, color_b = predict_tracks(
                images,
                conf=conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=max_query_pts,
                query_frame_num=query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=fine_tracking,
            )

        track_mask = v_b > vis_thresh
        reconstruction_b, _ = batch_np_matrix_to_pycolmap(
            p3d_b,
            extr,
            intr_ba,
            t_b,
            np.array([target_res, target_res]),
            masks=track_mask,
            max_reproj_error=max_reproj_error,
            shared_camera=shared_camera,
            camera_type=camera_type,
            points_rgb=color_b,
        )

        if reconstruction_b is None:
            continue

        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction_b, ba_options)

        recon_b = rename_colmap_recons_and_rescale_camera(
            reconstruction_b,
            names,
            coords[indices],
            img_size=target_res,
            shift_point2d_to_original_res=True,
            shared_camera=shared_camera,
            center_pp=False,
        )

        sparse_dir = os.path.join(batch_dir, "sparse_ba")
        os.makedirs(sparse_dir, exist_ok=True)
        recon_b.write(sparse_dir)
        trimesh.PointCloud(p3d_b, colors=color_b).export(os.path.join(sparse_dir, "points.ply"))


def stage_merge_align(scene_dir: str, batches: List[List[int]]) -> None:
    """Merge all batch reconstructions and run COLMAP model_aligner."""

    batch_dirs = [
        os.path.join(scene_dir, "batches", f"batch_{b[0]}_{b[-1]}", "sparse_ba")
        for b in batches
    ]

    merged_dir = os.path.join(scene_dir, "sparse_merged")
    os.makedirs(merged_dir, exist_ok=True)

    if len(batch_dirs) == 1:
        shutil.copytree(batch_dirs[0], merged_dir, dirs_exist_ok=True)
    else:
        current = batch_dirs[0]
        for i, nxt in enumerate(batch_dirs[1:], start=1):
            tmp_dir = os.path.join(scene_dir, f"tmp_merge_{i}")
            subprocess.run(
                [
                    "colmap",
                    "model_merger",
                    f"--input_path1={current}",
                    f"--input_path2={nxt}",
                    f"--output_path={tmp_dir}",
                ],
                check=True,
            )
            current = tmp_dir
        shutil.move(current, merged_dir)

    aligned_dir = os.path.join(scene_dir, "sparse_aligned")
    subprocess.run(
        [
            "colmap",
            "model_aligner",
            f"--input_path={merged_dir}",
            f"--output_path={aligned_dir}",
        ],
        check=True,
    )


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )

    if args.stage in ["preprocess", "all"]:
        names = stage_resize(args.scene_dir, args.target_res)
        batches = stage_batches(args.scene_dir, names, args.batch_size, args.overlap)
    else:
        preprocess_dir = os.path.join(args.scene_dir, "tmp2")
        names = sorted(
            glob.glob(os.path.join(preprocess_dir, "*.png")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        names = [os.path.basename(n) for n in names]
        with open(os.path.join(args.scene_dir, "batches.txt")) as f:
            batches = [list(map(int, line.strip().split())) for line in f if line.strip()]

    if args.stage in ["vggt", "all"]:
        stage_vggt(
            args.scene_dir,
            batches,
            args.target_res,
            518,
            dtype,
            device,
            args.conf_thres_value,
        )

    if args.stage in ["track", "all"]:
        stage_tracking(
            args.scene_dir,
            batches,
            args.target_res,
            518,
            dtype,
            device,
            args.vis_thresh,
            args.max_reproj_error,
            args.query_frame_num,
            args.max_query_pts,
            args.fine_tracking,
            args.shared_camera,
            args.camera_type,
        )

    if args.stage in ["align", "all"]:
        stage_merge_align(args.scene_dir, batches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT COLMAP preprocessing pipeline")
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["preprocess", "vggt", "track", "align", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--target_res", type=int, default=1024)
    parser.add_argument("--max_reproj_error", type=float, default=8.0)
    parser.add_argument("--vis_thresh", type=float, default=0.2)
    parser.add_argument("--query_frame_num", type=int, default=5)
    parser.add_argument("--max_query_pts", type=int, default=2048)
    parser.add_argument("--fine_tracking", action="store_true")
    parser.add_argument("--shared_camera", action="store_true")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE")
    parser.add_argument("--conf_thres_value", type=float, default=5.0)

    with torch.no_grad():
        main(parser.parse_args())

