# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import numpy as np
import os


def color_from_xy(x, y, W, H, cmap_name="hsv"):
    """
    Map (x, y) -> color in (R, G, B).
    1) Normalize x,y to [0,1].
    2) Combine them into a single scalar c in [0,1].
    3) Use matplotlib's colormap to convert c -> (R,G,B).

    You can customize step 2, e.g., c = (x + y)/2, or some function of (x, y).
    """
    import matplotlib.cm
    import matplotlib.colors

    x_norm = x / max(W - 1, 1)
    y_norm = y / max(H - 1, 1)
    # Simple combination:
    c = (x_norm + y_norm) / 2.0

    cmap = matplotlib.cm.get_cmap(cmap_name)
    # cmap(c) -> (r,g,b,a) in [0,1]
    rgba = cmap(c)
    r, g, b = rgba[0], rgba[1], rgba[2]
    return (r, g, b)  # in [0,1], RGB order


def get_track_colors_by_position(tracks_b, vis_mask_b=None, image_width=None, image_height=None, cmap_name="hsv"):
    """
    Given all tracks in one sample (b), compute a (N,3) array of RGB color values
    in [0,255]. The color is determined by the (x,y) position in the first
    visible frame for each track.

    Args:
        tracks_b: Tensor of shape (S, N, 2). (x,y) for each track in each frame.
        vis_mask_b: (S, N) boolean mask; if None, assume all are visible.
        image_width, image_height: used for normalizing (x, y).
        cmap_name: for matplotlib (e.g., 'hsv', 'rainbow', 'jet').

    Returns:
        track_colors: np.ndarray of shape (N, 3), each row is (R,G,B) in [0,255].
    """
    S, N, _ = tracks_b.shape
    track_colors = np.zeros((N, 3), dtype=np.uint8)

    if vis_mask_b is None:
        # treat all as visible
        vis_mask_b = torch.ones(S, N, dtype=torch.bool, device=tracks_b.device)

    for i in range(N):
        # Find first visible frame for track i
        visible_frames = torch.where(vis_mask_b[:, i])[0]
        if len(visible_frames) == 0:
            # track is never visible; just assign black or something
            track_colors[i] = (0, 0, 0)
            continue

        first_s = int(visible_frames[0].item())
        # use that frame's (x,y)
        x, y = tracks_b[first_s, i].tolist()

        # map (x,y) -> (R,G,B) in [0,1]
        r, g, b = color_from_xy(x, y, W=image_width, H=image_height, cmap_name=cmap_name)
        # scale to [0,255]
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        track_colors[i] = (r, g, b)

    return track_colors


def visualize_tracks_on_images(
    images,
    tracks,
    track_vis_mask=None,
    out_dir="track_visuals_concat_by_xy",
    image_format="CHW",  # "CHW" or "HWC"
    normalize_mode="[0,1]",
    cmap_name="hsv",  # e.g. "hsv", "rainbow", "jet"
    frames_per_row=4,  # New parameter for grid layout
    save_grid=True,  # Flag to control whether to save the grid image
):
    """
    Visualizes frames in a grid layout with specified frames per row.
    Each track's color is determined by its (x,y) position
    in the first visible frame (or frame 0 if always visible).
    Finally convert the BGR result to RGB before saving.
    Also saves each individual frame as a separate PNG file.

    Args:
        images: torch.Tensor (S, 3, H, W) if CHW or (S, H, W, 3) if HWC.
        tracks: torch.Tensor (S, N, 2), last dim = (x, y).
        track_vis_mask: torch.Tensor (S, N) or None.
        out_dir: folder to save visualizations.
        image_format: "CHW" or "HWC".
        normalize_mode: "[0,1]", "[-1,1]", or None for direct raw -> 0..255
        cmap_name: a matplotlib colormap name for color_from_xy.
        frames_per_row: number of frames to display in each row of the grid.
        save_grid: whether to save all frames in one grid image.

    Returns:
        None (saves images in out_dir).
    """

    if len(tracks.shape) == 4:
        tracks = tracks.squeeze(0)
        images = images.squeeze(0)
        if track_vis_mask is not None:
            track_vis_mask = track_vis_mask.squeeze(0)

    import matplotlib

    matplotlib.use("Agg")  # for non-interactive (optional)

    os.makedirs(out_dir, exist_ok=True)

    S = images.shape[0]
    _, N, _ = tracks.shape  # (S, N, 2)

    # Move to CPU
    images = images.cpu().clone()
    tracks = tracks.cpu().clone()
    if track_vis_mask is not None:
        track_vis_mask = track_vis_mask.cpu().clone()

    # Infer H, W from images shape
    if image_format == "CHW":
        # e.g. images[s].shape = (3, H, W)
        H, W = images.shape[2], images.shape[3]
    else:
        # e.g. images[s].shape = (H, W, 3)
        H, W = images.shape[1], images.shape[2]

    # Pre-compute the color for each track i based on first visible position
    track_colors_rgb = get_track_colors_by_position(
        tracks,  # shape (S, N, 2)
        vis_mask_b=track_vis_mask if track_vis_mask is not None else None,
        image_width=W,
        image_height=H,
        cmap_name=cmap_name,
    )

    # We'll accumulate each frame's drawn image in a list
    frame_images = []

    for s in range(S):
        # shape => either (3, H, W) or (H, W, 3)
        img = images[s]

        # Convert to (H, W, 3)
        if image_format == "CHW":
            img = img.permute(1, 2, 0)  # (H, W, 3)
        # else "HWC", do nothing

        img = img.numpy().astype(np.float32)

        # Scale to [0,255] if needed
        if normalize_mode == "[0,1]":
            img = np.clip(img, 0, 1) * 255.0
        elif normalize_mode == "[-1,1]":
            img = (img + 1.0) * 0.5 * 255.0
            img = np.clip(img, 0, 255.0)
        # else no normalization

        # Convert to uint8
        img = img.astype(np.uint8)

        # For drawing in OpenCV, convert to BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw each visible track
        cur_tracks = tracks[s]  # shape (N, 2)
        if track_vis_mask is not None:
            valid_indices = torch.where(track_vis_mask[s])[0]
        else:
            valid_indices = range(N)

        cur_tracks_np = cur_tracks.numpy()
        for i in valid_indices:
            x, y = cur_tracks_np[i]
            pt = (int(round(x)), int(round(y)))

            # track_colors_rgb[i] is (R,G,B). For OpenCV circle, we need BGR
            R, G, B = track_colors_rgb[i]
            color_bgr = (int(B), int(G), int(R))
            cv2.circle(img_bgr, pt, radius=3, color=color_bgr, thickness=-1)

        # Convert back to RGB for consistent final saving:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Save individual frame
        frame_path = os.path.join(out_dir, f"frame_{s:04d}.png")
        # Convert to BGR for OpenCV imwrite
        frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, frame_bgr)

        frame_images.append(img_rgb)

    # Only create and save the grid image if save_grid is True
    if save_grid:
        # Calculate grid dimensions
        num_rows = (S + frames_per_row - 1) // frames_per_row  # Ceiling division

        # Create a grid of images
        grid_img = None
        for row in range(num_rows):
            start_idx = row * frames_per_row
            end_idx = min(start_idx + frames_per_row, S)

            # Concatenate this row horizontally
            row_img = np.concatenate(frame_images[start_idx:end_idx], axis=1)

            # If this row has fewer than frames_per_row images, pad with black
            if end_idx - start_idx < frames_per_row:
                padding_width = (frames_per_row - (end_idx - start_idx)) * W
                padding = np.zeros((H, padding_width, 3), dtype=np.uint8)
                row_img = np.concatenate([row_img, padding], axis=1)

            # Add this row to the grid
            if grid_img is None:
                grid_img = row_img
            else:
                grid_img = np.concatenate([grid_img, row_img], axis=0)

        out_path = os.path.join(out_dir, "tracks_grid.png")
        # Convert back to BGR for OpenCV imwrite
        grid_img_bgr = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, grid_img_bgr)
        print(f"[INFO] Saved color-by-XY track visualization grid -> {out_path}")

    print(f"[INFO] Saved {S} individual frames to {out_dir}/frame_*.png")


def visualize_tracks_with_projections(
    images,
    tracks,
    projected_tracks,
    track_vis_mask=None,
    out_dir="track_proj_debug",
    image_format="CHW",
    normalize_mode="[0,1]",
    cmap_name="hsv",
    frames_per_row=4,
    save_grid=True,
):
    """Visualize 2D tracks together with their reprojections.

    Parameters
    ----------
    images : torch.Tensor
        Images tensor ``(S,3,H,W)`` or ``(S,H,W,3)``.
    tracks : torch.Tensor or np.ndarray
        Tracked 2D points ``(S,N,2)``.
    projected_tracks : torch.Tensor or np.ndarray
        Reprojected 2D points ``(S,N,2)`` to compare against ``tracks``.
    track_vis_mask : torch.Tensor or np.ndarray, optional
        Visibility mask ``(S,N)``.
    out_dir : str
        Output directory to save visualization images.
    image_format : str
        Image format ("CHW" or "HWC").
    normalize_mode : str
        Image normalization, same as :func:`visualize_tracks_on_images`.
    cmap_name : str
        Colormap name for coloring tracks.
    frames_per_row : int
        Number of frames per row when saving the grid.
    save_grid : bool
        Whether to save the grid image in addition to individual frames.
    """

    if isinstance(tracks, np.ndarray):
        tracks = torch.from_numpy(tracks)
    if isinstance(projected_tracks, np.ndarray):
        projected_tracks = torch.from_numpy(projected_tracks)
    if track_vis_mask is not None and isinstance(track_vis_mask, np.ndarray):
        track_vis_mask = torch.from_numpy(track_vis_mask)

    if len(tracks.shape) == 4:
        tracks = tracks.squeeze(0)
        projected_tracks = projected_tracks.squeeze(0)
        images = images.squeeze(0)
        if track_vis_mask is not None:
            track_vis_mask = track_vis_mask.squeeze(0)

    import matplotlib

    matplotlib.use("Agg")

    os.makedirs(out_dir, exist_ok=True)

    S = images.shape[0]
    _, N, _ = tracks.shape

    images = images.cpu().clone()
    tracks = tracks.cpu().clone()
    projected_tracks = projected_tracks.cpu().clone()
    if track_vis_mask is not None:
        track_vis_mask = track_vis_mask.cpu().clone()

    if image_format == "CHW":
        H, W = images.shape[2], images.shape[3]
    else:
        H, W = images.shape[1], images.shape[2]

    track_colors_rgb = get_track_colors_by_position(
        tracks, vis_mask_b=track_vis_mask, image_width=W, image_height=H, cmap_name=cmap_name
    )

    frame_images = []

    for s in range(S):
        img = images[s]
        if image_format == "CHW":
            img = img.permute(1, 2, 0)
        img = img.numpy().astype(np.float32)
        if normalize_mode == "[0,1]":
            img = np.clip(img, 0, 1) * 255.0
        elif normalize_mode == "[-1,1]":
            img = (img + 1.0) * 0.5 * 255.0
            img = np.clip(img, 0, 255.0)

        img = img.astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cur_tracks = tracks[s]
        cur_proj = projected_tracks[s]
        if track_vis_mask is not None:
            valid_indices = torch.where(track_vis_mask[s])[0]
        else:
            valid_indices = range(N)

        cur_tracks_np = cur_tracks.numpy()
        cur_proj_np = cur_proj.numpy()

        for i in valid_indices:
            x, y = cur_tracks_np[i]
            px, py = cur_proj_np[i]
            pt = (int(round(x)), int(round(y)))
            proj_pt = (int(round(px)), int(round(py)))
            R, G, B = track_colors_rgb[i]
            color_bgr = (int(B), int(G), int(R))
            cv2.circle(img_bgr, pt, radius=3, color=color_bgr, thickness=-1)
            cv2.drawMarker(
                img_bgr,
                proj_pt,
                color=color_bgr,
                markerType=cv2.MARKER_CROSS,
                markerSize=7,
                thickness=1,
            )

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frame_path = os.path.join(out_dir, f"frame_{s:04d}.png")
        frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, frame_bgr)
        frame_images.append(img_rgb)

    if save_grid:
        num_rows = (S + frames_per_row - 1) // frames_per_row
        grid_img = None
        for row in range(num_rows):
            start_idx = row * frames_per_row
            end_idx = min(start_idx + frames_per_row, S)
            row_img = np.concatenate(frame_images[start_idx:end_idx], axis=1)
            if end_idx - start_idx < frames_per_row:
                padding_width = (frames_per_row - (end_idx - start_idx)) * W
                padding = np.zeros((H, padding_width, 3), dtype=np.uint8)
                row_img = np.concatenate([row_img, padding], axis=1)
            if grid_img is None:
                grid_img = row_img
            else:
                grid_img = np.concatenate([grid_img, row_img], axis=0)

        out_path = os.path.join(out_dir, "tracks_proj_grid.png")
        grid_img_bgr = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, grid_img_bgr)
        print(f"[INFO] Saved track/projection visualization grid -> {out_path}")

    print(f"[INFO] Saved {S} individual frames to {out_dir}/frame_*.png")

