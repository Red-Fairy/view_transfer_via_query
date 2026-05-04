"""Online lift-and-render: equirect (RGB + radial depth) → target-perspective video.

Step 1 (lift): equirect pixels at the chosen source frame t0 → 3D world point cloud.
Step 2 (render): project all world points to each target perspective camera, z-buffer
                 to handle occlusion, output rendered RGB + per-pixel visibility mask.

Conventions:
  • Equirect was rendered by UE → pixel→ray uses UE camera frame
    (+X forward, +Y right, +Z up, LHS).
  • Depth is **radial distance** from camera (not perpendicular Z).
  • All c2w matrices are OpenCV convention (RHS, +X right, +Y down, +Z forward), meters.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from .parse_cameras import M_CV_TO_UE_CAM


# UE camera frame → OpenCV camera frame (transpose of M_CV_TO_UE_CAM)
M_UE_TO_CV_CAM: np.ndarray = np.array(M_CV_TO_UE_CAM, dtype=np.float64).T


# ── Depth I/O ───────────────────────────────────────────────────────────────


def load_depth(path: str) -> np.ndarray:
    """Load a single-channel depth file as float32 [H, W].

    Recognizes:
      • .exr   (cv2 with OPENCV_IO_ENABLE_OPENEXR=1, or imageio)
      • .npy   (numpy save)
      • .pt    (torch save)

    NOTE: this function returns RAW values from the file. The caller must
    apply unit conversion (e.g. cm→m for UE renders) and sky-sentinel
    filtering. See `load_depth_ue` for a UE-specific convenience.
    """
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    if path.endswith((".pt", ".pth")):
        import torch as _t
        return _t.load(path, weights_only=True, map_location="cpu").numpy().astype(np.float32)
    # OpenCV needs OPENCV_IO_ENABLE_OPENEXR=1 to decode EXR.  Setting it here only
    # affects subsequent cv2 calls — fine if cv2 hasn't decoded any EXR yet.
    import os
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError("cv2 returned None (OpenEXR may be disabled)")
    except Exception:
        try:
            import imageio.v3 as iio
            depth = iio.imread(path)
        except Exception as e:
            raise IOError(f"Failed to load depth {path}: {e}")
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def load_depth_ue(path: str, sky_threshold_cm: float = 6.5e4) -> np.ndarray:
    """Convenience: load a UE-rendered depth EXR and convert to meters.

    UE outputs radial distance in cm. The float16 max (~65504) is used by
    UE as a "sky" / infinity sentinel — values >= sky_threshold_cm are
    replaced with NaN so downstream lift filters them out.
    """
    raw = load_depth(path)
    raw = np.where(raw >= sky_threshold_cm, np.nan, raw)
    return (raw * 0.01).astype(np.float32)  # cm → m


# Backward compatibility alias
load_depth_exr = load_depth


# ── Equirect → unit ray (UE convention) ─────────────────────────────────────


def equirect_pixel_to_unit_ray_ue(
    He: int, We: int, device, dtype=torch.float32
) -> torch.Tensor:
    """[He, We, 3] unit rays in UE camera frame for each pixel center.

    Pixel-center convention: pixel (i, j) lives at (j+0.5, i+0.5).
    """
    ys = (torch.arange(He, device=device, dtype=dtype) + 0.5) / He
    xs = (torch.arange(We, device=device, dtype=dtype) + 0.5) / We
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    lon = xx * (2.0 * math.pi) - math.pi      # [-π, π]
    lat = math.pi / 2.0 - yy * math.pi        # [+π/2, -π/2]
    cos_lat = torch.cos(lat)
    rays = torch.stack(
        [
            cos_lat * torch.cos(lon),  # ue +X (forward)
            cos_lat * torch.sin(lon),  # ue +Y (right)
            torch.sin(lat),            # ue +Z (up)
        ],
        dim=-1,
    )
    return rays


# ── Step 1: lift equirect → world point cloud ──────────────────────────────


def lift_pano_to_world_pointcloud(
    rgb: torch.Tensor,            # [3, He, We] float in [0, 1]
    depth: torch.Tensor,          # [He, We] radial distance (meters), float
    pano_c2w_cv: torch.Tensor,    # [4, 4] OpenCV
    valid_depth_min: float = 1e-3,
    valid_depth_max: float = 1e6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lift one equirect frame (with radial depth) to a world-space point cloud.

    Returns:
        xyz_world: [N, 3]   world coords (OpenCV convention)
        rgb_flat:  [N, 3]   RGB in [0, 1]
    where N = (#pixels with valid depth).
    """
    assert rgb.shape[0] == 3 and rgb.dim() == 3
    He, We = depth.shape
    device, dtype = rgb.device, rgb.dtype

    # 1. Unit rays in UE camera frame  [He, We, 3]
    rays_ue = equirect_pixel_to_unit_ray_ue(He, We, device, dtype)
    # 2. Convert UE camera → OpenCV camera frame
    M = torch.tensor(M_UE_TO_CV_CAM, device=device, dtype=dtype)
    rays_cv = torch.einsum("ij,hwj->hwi", M, rays_ue)  # [He, We, 3]
    # 3. Radial distance × unit ray = 3D point in OpenCV camera frame
    pts_cam_cv = rays_cv * depth.to(dtype).unsqueeze(-1)  # [He, We, 3]

    # 4. Camera → world via c2w
    R = pano_c2w_cv[:3, :3].to(dtype)
    t = pano_c2w_cv[:3, 3].to(dtype)
    pts_flat = pts_cam_cv.reshape(-1, 3)             # [N_full, 3]
    pts_world = pts_flat @ R.T + t                    # [N_full, 3]
    rgb_flat = rgb.permute(1, 2, 0).reshape(-1, 3)    # [N_full, 3]

    # 5. Filter invalid depth (background, sky, NaN)
    d = depth.reshape(-1).to(dtype)
    valid = (d > valid_depth_min) & (d < valid_depth_max) & torch.isfinite(d)
    return pts_world[valid], rgb_flat[valid]


# ── Step 2: render world point cloud → target perspective video ────────────


def render_pointcloud_to_perspective(
    xyz_world: torch.Tensor,        # [N, 3]
    rgb: torch.Tensor,              # [N, 3] in [0, 1]
    target_c2w: torch.Tensor,       # [T, 4, 4] OpenCV
    intrinsics: torch.Tensor,       # [4]: fx, fy, cx, cy  (per-sample, fixed across t)
    pers_h: int,
    pers_w: int,
    chunk_size: int = 4,
    z_min: float = 1e-3,
    z_eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Z-buffer rasterize a point cloud to T target-perspective frames.

    Returns:
        rendered:    [T, 3, pers_h, pers_w]   float in [0, 1] (zeros where empty)
        visibility:  [T, 1, pers_h, pers_w]   float in {0, 1}
    """
    T = target_c2w.shape[0]
    N = xyz_world.shape[0]
    device = xyz_world.device
    dtype = xyz_world.dtype
    fx, fy, cx, cy = intrinsics.to(device=device, dtype=dtype).unbind()

    rendered = torch.zeros(T, 3, pers_h, pers_w, device=device, dtype=dtype)
    visibility = torch.zeros(T, 1, pers_h, pers_w, device=device, dtype=dtype)
    if N == 0:
        return rendered, visibility

    # Homogeneous coords once
    xyz_h = torch.cat([xyz_world, xyz_world.new_ones(N, 1)], dim=1)  # [N, 4]

    HW = pers_h * pers_w

    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        # World → camera frame for this chunk
        c2w_chunk = target_c2w[s:e].to(device=device, dtype=torch.float32)
        w2c = torch.linalg.inv(c2w_chunk).to(dtype)  # [c, 4, 4]
        # [c, 3, 4] @ [N, 4] -> [c, N, 3]
        pts_cam = torch.einsum("cij,nj->cni", w2c[:, :3, :], xyz_h)

        for i in range(e - s):
            t_idx = s + i
            pts = pts_cam[i]  # [N, 3]
            z = pts[:, 2]
            # In-frustum filter
            in_front = z > z_min
            u_f = fx * pts[:, 0] / z.clamp(min=z_min) + cx
            v_f = fy * pts[:, 1] / z.clamp(min=z_min) + cy
            in_bounds = (
                in_front
                & (u_f >= 0) & (u_f < pers_w)
                & (v_f >= 0) & (v_f < pers_h)
            )

            u_v = u_f[in_bounds].long()
            v_v = v_f[in_bounds].long()
            z_v = z[in_bounds]
            rgb_v = rgb[in_bounds]
            if z_v.numel() == 0:
                continue

            flat_idx = v_v * pers_w + u_v

            # Z-buffer: per-pixel min depth
            depth_buf = torch.full((HW,), float("inf"), device=device, dtype=dtype)
            depth_buf.scatter_reduce_(
                0, flat_idx, z_v, reduce="amin", include_self=True
            )

            # Mark winners (points whose z equals the per-pixel minimum)
            min_at_pt = depth_buf[flat_idx]
            is_winner = z_v <= min_at_pt + z_eps
            u_w = u_v[is_winner]
            v_w = v_v[is_winner]
            rgb_w = rgb_v[is_winner]

            rendered[t_idx, :, v_w, u_w] = rgb_w.t()
            visibility[t_idx, 0, v_w, u_w] = 1.0

    return rendered, visibility


# ── End-to-end ─────────────────────────────────────────────────────────────


def lift_and_render(
    static_rgb: torch.Tensor,         # [3, He, We] in [0, 1]
    static_depth: torch.Tensor,       # [He, We]   radial m
    pano_c2w_at_t0: torch.Tensor,     # [4, 4]     OpenCV
    target_c2w: torch.Tensor,         # [T, 4, 4]  OpenCV  (perspective camera traj)
    intrinsics: torch.Tensor,         # [4]
    pers_h: int = 480,
    pers_w: int = 832,
    chunk_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: lift one source frame and render to a target trajectory."""
    pts_world, rgb_flat = lift_pano_to_world_pointcloud(
        static_rgb, static_depth, pano_c2w_at_t0
    )
    return render_pointcloud_to_perspective(
        pts_world, rgb_flat, target_c2w, intrinsics, pers_h, pers_w, chunk_size=chunk_size
    )
