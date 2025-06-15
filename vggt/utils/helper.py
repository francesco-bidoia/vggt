# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def get_batches_with_overlap(items, batch_size, overlap):
    """Split a list into batches with overlap.

    Args:
        items (list): list of items to split
        batch_size (int): number of items per batch
        overlap (int): number of overlapping items between consecutive batches

    Returns:
        list[list]: list of batches
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if overlap >= batch_size:
        raise ValueError("overlap must be smaller than batch_size")

    batches = []
    start = 0
    n = len(items)
    while start < n:
        end = min(start + batch_size, n)
        batches.append(items[start:end])
        if end == n:
            break
        start = end - overlap
    return batches


def compute_similarity_transform(src, dst):
    """Compute similarity transform (scale, rotation, translation) from src to dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[1] == 3

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = src_centered.T @ dst_centered / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    var_src = (src_centered ** 2).sum() / src.shape[0]
    scale = S.sum() / var_src if var_src > 0 else 1.0

    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def extrinsics_to_centers(extrinsics):
    """Convert extrinsic matrices (3x4, world to cam) to camera centers."""
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    return -(R.transpose(0, 2, 1) @ t[..., None]).squeeze(-1)


def align_extrinsics(extrinsics_new, extrinsics_ref, extrinsics_new_overlap):
    """Align new extrinsics to reference frame using overlap."""
    if len(extrinsics_ref) == 0:
        return extrinsics_new

    centers_ref = extrinsics_to_centers(extrinsics_ref)
    centers_new = extrinsics_to_centers(extrinsics_new_overlap)

    scale, R_align, t_align = compute_similarity_transform(centers_new, centers_ref)

    aligned = []
    for extr in extrinsics_new:
        R = extr[:3, :3]
        t = extr[:3, 3]
        center = -R.T @ t
        center = scale * (R_align @ center) + t_align
        R_new = R_align @ R
        t_new = -R_new @ center
        extr_aligned = np.concatenate([R_new, t_new[:, None]], axis=1)
        aligned.append(extr_aligned)

    return np.stack(aligned, axis=0)


def merge_track_batches(track_batches, num_frames):
    """Merge track predictions from multiple batches."""

    total_tracks = sum(batch["tracks"].shape[1] for batch in track_batches)
    if total_tracks == 0:
        return (
            np.zeros((num_frames, 0, 2), dtype=np.float32),
            np.zeros((num_frames, 0), dtype=np.float32),
            None,
            None,
            None,
        )

    tracks_all = np.zeros((num_frames, total_tracks, 2), dtype=track_batches[0]["tracks"].dtype)
    vis_all = np.zeros((num_frames, total_tracks), dtype=track_batches[0]["vis"].dtype)

    conf_list = []
    points_list = []
    color_list = []

    offset = 0
    for batch in track_batches:
        idx = np.asarray(batch["indices"], dtype=int)
        t = batch["tracks"]
        v = batch["vis"]
        n = t.shape[1]
        tracks_all[idx, offset : offset + n] = t
        vis_all[idx, offset : offset + n] = v
        if batch.get("confs") is not None:
            conf_list.append(batch["confs"])
        if batch.get("points3d") is not None:
            points_list.append(batch["points3d"])
        if batch.get("colors") is not None:
            color_list.append(batch["colors"])
        offset += n

    confs = np.concatenate(conf_list, axis=0) if conf_list else None
    points3d = np.concatenate(points_list, axis=0) if points_list else None
    colors = np.concatenate(color_list, axis=0) if color_list else None

    return tracks_all, vis_all, confs, points3d, colors

def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries
    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    # randomly pick which True positions to keep
    sampled_indices = np.random.choice(true_indices, size=max_trues, replace=False)  # shape = (max_trues,)

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)


def create_pixel_coordinate_grid(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices for all frames.
    Returns:
        tuple: A tuple containing:
            - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                            with x, y coordinates and frame indices
            - y_coords (numpy.ndarray): Array of y coordinates for all frames
            - x_coords (numpy.ndarray): Array of x coordinates for all frames
            - f_coords (numpy.ndarray): Array of frame indices for all frames
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf
