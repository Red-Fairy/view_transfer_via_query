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


def load_depth_ue(
    path: str,
    sky_threshold_cm: float = 6.5e4,
    infinity_depth_m: float = 1.0e4,
) -> np.ndarray:
    """Load a UE-rendered depth EXR and convert to meters.

    UE outputs radial distance in cm. The float16 max (~65504) is used by
    UE as a "sky" / infinity sentinel — values >= ``sky_threshold_cm`` are
    replaced with the finite stand-in ``infinity_depth_m`` (default 10 km)
    so the lift pipeline keeps these pixels and projects them to the correct
    sky region of the target view (parallax is negligible for sky-distant
    points across realistic camera translations).

    Reference: see /share/ma/scratch/rundong/Unreal_Projects/UE-Render/lift_render/project_depth.py
    """
    raw = load_depth(path)
    sky = raw >= sky_threshold_cm
    meters = (raw * 0.01).astype(np.float32)  # cm → m
    meters[sky] = float(infinity_depth_m)
    return meters


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


# ── Cubemap mesh: topology ──────────────────────────────────────────────────
# Adapted from /share/ma/scratch/rundong/Unreal_Projects/UE-Render/lift_render/project_depth_mesh.py
# Conventions:
#   - Source pano camera is OpenCV (+X right, +Y down, +Z forward).  The reference
#     used a different pano frame (+X forward, +Y left, +Z up); we therefore use a
#     different `_CUBE_FACE_AXES` table consistent with our OpenCV pano frame.
#   - Triangle winding is INWARD-facing (normals point toward the source camera at
#     origin).  This matches our use case: target cameras live inside the depth shell,
#     so standard backface culling correctly keeps front-visible triangles and drops
#     silhouette-flipped ones.

# Per-face axes for OpenCV pano frame.  Each face pixel (i, j) on a unit cube:
#     pos = n_axis + u * u_axis + v * v_axis
# All faces satisfy `(u_axis × v_axis) == n_axis` (outward).  Original triangle
# winding `(tl, tr, bl)` therefore produces OUTWARD normals.  We flip to
# `(tl, bl, tr)` below so normals consistently point INWARD (toward origin),
# which is what we want for backface culling with the camera inside the shell.
_CUBE_FACE_AXES_CV: list = [
    # (u_axis,                  v_axis,                  n_axis)
    (np.array([0., 1., 0.]),   np.array([0.,  0.,  1.]), np.array([ 1., 0., 0.])),   # +X  (u=+Y, v=+Z) → +X
    (np.array([0., 1., 0.]),   np.array([0.,  0., -1.]), np.array([-1., 0., 0.])),   # -X  (u=+Y, v=-Z) → -X
    (np.array([0., 0., 1.]),   np.array([1.,  0.,  0.]), np.array([ 0., 1., 0.])),   # +Y  (u=+Z, v=+X) → +Y
    (np.array([1., 0., 0.]),   np.array([0.,  0.,  1.]), np.array([ 0.,-1., 0.])),   # -Y  (u=+X, v=+Z) → -Y
    (np.array([1., 0., 0.]),   np.array([0.,  1.,  0.]), np.array([ 0., 0., 1.])),   # +Z  (u=+X, v=+Y) → +Z
    (np.array([0., 1., 0.]),   np.array([1.,  0.,  0.]), np.array([ 0., 0.,-1.])),   # -Z  (u=+Y, v=+X) → -Z
]


_CUBEMAP_TOPOLOGY_CACHE: dict = {}


def _build_cubemap_topology(face_res: int, device: torch.device) -> dict:
    """Build (cached) cube-mesh topology with shared-vertex stitching.

    Returns dict with:
      unique_dirs  : [K, 3]  unit direction per unique vertex (OpenCV pano frame)
      equirect_uv  : [K, 2]  equirect (u, v) ∈ [0, 1] per vertex (for sampling)
      faces        : [F, 3]  int triangle indices into unique_dirs (INWARD winding)
    """
    key = (face_res, str(device))
    cached = _CUBEMAP_TOPOLOGY_CACHE.get(key)
    if cached is not None:
        return cached

    N = face_res
    coords = np.linspace(-1.0, 1.0, N, dtype=np.float64)
    vv, uu = np.meshgrid(coords, coords, indexing="ij")  # vv: rows, uu: cols

    all_dirs = np.empty((6 * N * N, 3), dtype=np.float64)
    for f, (u_ax, v_ax, n_ax) in enumerate(_CUBE_FACE_AXES_CV):
        pos = (
            n_ax[None, None, :]
            + uu[:, :, None] * u_ax[None, None, :]
            + vv[:, :, None] * v_ax[None, None, :]
        )  # (N, N, 3) on the unit cube
        norm = np.linalg.norm(pos, axis=-1, keepdims=True)
        all_dirs[f * N * N : (f + 1) * N * N] = (pos / norm).reshape(-1, 3)

    # Dedupe along rounded direction → one shared vertex at every cube edge / corner.
    rounded = np.round(all_dirs, 10)
    _, first_occ, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    unique_dirs = all_dirs[first_occ]
    unique_dirs /= np.linalg.norm(unique_dirs, axis=-1, keepdims=True)

    # Equirect (u, v) for each unique direction (OpenCV pano: x_fwd, y_down, z_? — match our equirect_pixel_to_unit_ray_ue convention).
    # Our equirect was rendered in UE convention: +X forward, +Y right, +Z up.
    # To convert OpenCV pano direction → UE pano direction, apply M_CV_TO_UE_CAM.
    M = np.array(M_CV_TO_UE_CAM, dtype=np.float64)
    dirs_ue = unique_dirs @ M.T   # (K, 3) in UE frame
    x = dirs_ue[:, 0]
    y = dirs_ue[:, 1]
    z = dirs_ue[:, 2]
    lon = np.arctan2(y, x)               # [-π, π]
    lat = np.arctan2(z, np.sqrt(x * x + y * y).clip(min=1e-12))
    u_eq = (lon + np.pi) / (2.0 * np.pi)             # [0, 1]
    v_eq = (np.pi / 2.0 - lat) / np.pi                # [0, 1]
    equirect_uv = np.stack([u_eq, v_eq], axis=-1)

    # Face index generation per face, INWARD winding (swap tr ↔ bl vs. reference outward winding).
    all_tris = []
    for f in range(6):
        base = f * N * N
        idx_grid = np.arange(N * N, dtype=np.int64).reshape(N, N) + base
        tl = idx_grid[:-1, :-1]
        tr = idx_grid[:-1, 1:]
        bl = idx_grid[1:,  :-1]
        br = idx_grid[1:,  1:]
        # INWARD-winding: original outward (tl, tr, bl) → flipped (tl, bl, tr); same for second tri.
        tri_a = np.stack([tl.ravel(), bl.ravel(), tr.ravel()], axis=-1)
        tri_b = np.stack([tr.ravel(), bl.ravel(), br.ravel()], axis=-1)
        all_tris.append(np.concatenate([tri_a, tri_b], axis=0))
    faces_pre_dedup = np.concatenate(all_tris, axis=0)
    faces_global = inverse[faces_pre_dedup].astype(np.int32)

    topology = {
        "unique_dirs": torch.from_numpy(unique_dirs.astype(np.float32)).to(device),
        "equirect_uv": torch.from_numpy(equirect_uv.astype(np.float32)).to(device),
        "faces": torch.from_numpy(faces_global).to(device),
    }
    _CUBEMAP_TOPOLOGY_CACHE[key] = topology
    return topology


def _nearest_sample_equirect(img: torch.Tensor, u_eq: torch.Tensor, v_eq: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour sample with horizontal wrap. img: [H, W] or [C, H, W]."""
    squeezed = img.ndim == 2
    if squeezed:
        img = img.unsqueeze(0)
    C, H, W = img.shape
    u_i = (torch.round(u_eq * (W - 1)).long()) % W
    v_i = torch.round(v_eq * (H - 1)).long().clamp(0, H - 1)
    flat = img.reshape(C, H * W)
    out = flat[:, v_i * W + u_i].transpose(0, 1)  # (K, C)
    return out.squeeze(-1) if squeezed else out


def _bilinear_sample_equirect(img: torch.Tensor, u_eq: torch.Tensor, v_eq: torch.Tensor) -> torch.Tensor:
    """Bilinear sample with horizontal wrap, vertical clamp."""
    squeezed = img.ndim == 2
    if squeezed:
        img = img.unsqueeze(0)
    C, H, W = img.shape
    u_f = u_eq * (W - 1)
    v_f = v_eq * (H - 1)
    u0 = torch.floor(u_f).long()
    v0 = torch.floor(v_f).long()
    au = (u_f - u0.float()).unsqueeze(-1)
    av = (v_f - v0.float()).unsqueeze(-1)
    u1 = (u0 + 1) % W
    u0 = u0 % W
    v1 = (v0 + 1).clamp(0, H - 1)
    v0 = v0.clamp(0, H - 1)
    flat = img.reshape(C, H * W)
    p00 = flat[:, v0 * W + u0].transpose(0, 1)
    p01 = flat[:, v0 * W + u1].transpose(0, 1)
    p10 = flat[:, v1 * W + u0].transpose(0, 1)
    p11 = flat[:, v1 * W + u1].transpose(0, 1)
    out = (1 - au) * (1 - av) * p00 + au * (1 - av) * p01 + (1 - au) * av * p10 + au * av * p11
    return out.squeeze(-1) if squeezed else out


def _filter_faces(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    min_angle_deg: float = 2.5,
    edge_depth_ratio_thresh: float = 0.5,
    edge_aspect_ratio_thresh: float = 20.0,
) -> torch.Tensor:
    """Three-filter mesh cleanup (silhouette + degenerate triangles)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    def _angle_between(a, b):
        cos = (a * b).sum(-1) / (torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1) + 1e-12)
        return torch.acos(cos.clamp(-1.0, 1.0))

    ang_a = _angle_between(v1 - v0, v2 - v0)
    ang_b = _angle_between(v0 - v1, v2 - v1)
    ang_c = _angle_between(v0 - v2, v1 - v2)
    min_ang = torch.minimum(torch.minimum(ang_a, ang_b), ang_c)
    valid_angle = min_ang >= math.radians(min_angle_deg)

    d0 = torch.linalg.norm(v0, dim=-1)
    d1 = torch.linalg.norm(v1, dim=-1)
    d2 = torch.linalg.norm(v2, dim=-1)

    def _ratio(da, db):
        return torch.abs(da - db) / (torch.maximum(da, db) + 1e-8)

    max_ratio = torch.maximum(torch.maximum(_ratio(d0, d1), _ratio(d1, d2)), _ratio(d2, d0))
    valid_depth = max_ratio < edge_depth_ratio_thresh

    e01 = torch.linalg.norm(v1 - v0, dim=-1)
    e12 = torch.linalg.norm(v2 - v1, dim=-1)
    e20 = torch.linalg.norm(v0 - v2, dim=-1)
    edges = torch.stack([e01, e12, e20], dim=-1)
    valid_aspect = (edges.amax(dim=-1) / edges.amin(dim=-1).clamp(min=1e-8)) < edge_aspect_ratio_thresh

    return faces[valid_angle & valid_depth & valid_aspect]


def build_cubemap_mesh_world(
    rgb: torch.Tensor,                # [3, He, We] float in [0, 1]
    depth: torch.Tensor,              # [He, We] radial m  (sky already filled, no NaN)
    pano_c2w_cv: torch.Tensor,        # [4, 4] OpenCV
    face_res: int = 1024,
    min_angle_deg: float = 2.5,
    edge_depth_ratio_thresh: float = 0.5,
    edge_aspect_ratio_thresh: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift equirect → cubemap mesh (vertices in WORLD frame, OpenCV).

    Returns: (vertices_world [K,3], faces [M,3] int32, colors [K,3]).
    """
    device, dtype = rgb.device, rgb.dtype
    He, We = depth.shape
    topo = _build_cubemap_topology(face_res, device)
    unique_dirs_cv = topo["unique_dirs"].to(dtype)   # OpenCV pano frame
    uv = topo["equirect_uv"]
    faces = topo["faces"]

    u_eq = uv[:, 0]
    v_eq = uv[:, 1]

    # Sample depth (nearest, avoid silhouette phantom mid-depth) and RGB (bilinear).
    depth_2d = depth.to(dtype)
    rgb_chw = rgb.to(dtype)
    depth_k = _nearest_sample_equirect(depth_2d, u_eq, v_eq)   # (K,)
    colors_k = _bilinear_sample_equirect(rgb_chw, u_eq, v_eq)   # (K, 3)

    # Vertices in pano camera frame (OpenCV).
    vertices_pano_cv = unique_dirs_cv * depth_k.unsqueeze(-1)  # (K, 3)

    # Pano camera → world.
    R = pano_c2w_cv[:3, :3].to(dtype)
    t = pano_c2w_cv[:3, 3].to(dtype)
    vertices_world = vertices_pano_cv @ R.T + t

    # Cleanup degenerate triangles (uses pano-camera-frame vertices for depth ratios).
    faces = _filter_faces(
        vertices_pano_cv, faces,
        min_angle_deg=min_angle_deg,
        edge_depth_ratio_thresh=edge_depth_ratio_thresh,
        edge_aspect_ratio_thresh=edge_aspect_ratio_thresh,
    )
    return vertices_world.contiguous(), faces.contiguous(), colors_k.contiguous()


# ── Cubemap mesh: rasterization (nvdiffrast) ────────────────────────────────


_NVDR_CTX_CACHE: dict = {}


def _get_nvdr_ctx(device: torch.device):
    """Lazy-init and cache nvdiffrast.RasterizeCudaContext per device."""
    try:
        import nvdiffrast.torch as dr
    except ImportError as e:
        raise RuntimeError(
            "Mesh rendering requires nvdiffrast.  Install with: "
            "`pip install --extra-index-url https://download.nvidia.com/compute/redist nvdiffrast`"
        ) from e
    key = str(device)
    ctx = _NVDR_CTX_CACHE.get(key)
    if ctx is None:
        ctx = dr.RasterizeCudaContext(device=device)
        _NVDR_CTX_CACHE[key] = ctx
    return ctx, dr


def _opencv_intrinsics_to_opengl_proj(
    fx: float, fy: float, cx: float, cy: float, W: int, H: int,
    near: float, far: float, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """OpenCV intrinsics → OpenGL projection matrix (eye → clip).

    Assumes the eye is in OpenGL convention (+X right, +Y up, looking down −Z).
    Caller is responsible for converting from OpenCV eye frame via diag(1, −1, −1, 1).
    """
    P = torch.zeros(4, 4, device=device, dtype=dtype)
    P[0, 0] = 2.0 * fx / W
    P[1, 1] = 2.0 * fy / H
    P[0, 2] = (W - 2.0 * cx) / W
    P[1, 2] = (2.0 * cy - H) / H
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


@torch.no_grad()
def render_mesh_to_perspective(
    vertices_world: torch.Tensor,    # [K, 3]
    faces: torch.Tensor,             # [M, 3] int32
    colors: torch.Tensor,            # [K, 3] in [0, 1]
    target_c2w: torch.Tensor,        # [T, 4, 4] OpenCV
    intrinsics: torch.Tensor,        # [4]: fx, fy, cx, cy
    pers_h: int,
    pers_w: int,
    backface_cull: bool = True,
    near: float = 1.0e-3,
    far: float = 1.0e7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rasterize the mesh to T target perspective frames using nvdiffrast.

    Returns:
        rendered:    [T, 3, pers_h, pers_w] float in [0, 1]
        visibility:  [T, 1, pers_h, pers_w] float in {0, 1}
    """
    device = vertices_world.device
    dtype = vertices_world.dtype
    ctx, dr = _get_nvdr_ctx(device)
    T = target_c2w.shape[0]

    fx, fy, cx, cy = intrinsics.to(device=device, dtype=dtype).unbind()
    fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)

    P = _opencv_intrinsics_to_opengl_proj(fx, fy, cx, cy, pers_w, pers_h, near, far, device, dtype)
    cv_to_gl = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device, dtype=dtype))

    K = vertices_world.shape[0]
    verts_h = torch.cat([vertices_world.detach(), vertices_world.new_ones(K, 1)], dim=1)  # [K, 4]
    colors_d = colors.detach().contiguous()
    faces_int = faces.int().contiguous()

    rendered = torch.zeros(T, 3, pers_h, pers_w, device=device, dtype=dtype)
    visibility = torch.zeros(T, 1, pers_h, pers_w, device=device, dtype=dtype)

    for t in range(T):
        c2w_t = target_c2w[t].to(device=device, dtype=dtype)
        V = cv_to_gl @ torch.linalg.inv(c2w_t)        # world → OpenGL eye
        MVP = P @ V                                    # world → clip

        pos_clip = (verts_h @ MVP.T).contiguous()     # [K, 4]

        if backface_cull:
            # Signed area in NDC (post perspective-divide x, y)
            w = pos_clip[:, 3:4].clamp(min=1e-12)
            ndc_xy = pos_clip[:, :2] / w               # [K, 2]
            f0 = faces_int[:, 0].long()
            f1 = faces_int[:, 1].long()
            f2 = faces_int[:, 2].long()
            v0 = ndc_xy[f0]; v1 = ndc_xy[f1]; v2 = ndc_xy[f2]
            e1 = v1 - v0
            e2 = v2 - v0
            signed_area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            # Inward-facing winding + Y-flip in projection → front faces have positive signed area.
            # If the wrong sign cuts everything, flip below.  Fall back to "no cull" if mask kills it all.
            front_mask = signed_area > 0
            if front_mask.any():
                faces_use = faces_int[front_mask].contiguous()
            else:
                faces_use = faces_int  # degenerate: render everything rather than nothing
        else:
            faces_use = faces_int

        rast_out, _ = dr.rasterize(
            ctx, pos_clip.unsqueeze(0), faces_use, resolution=[pers_h, pers_w],
        )  # [1, H, W, 4]
        color, _ = dr.interpolate(colors_d.unsqueeze(0), rast_out, faces_use)
        color = dr.antialias(color, rast_out, pos_clip.unsqueeze(0), faces_use)
        # nvdiffrast outputs with origin at bottom-left (OpenGL); flip vertically
        # so row 0 corresponds to image top (our convention).
        color_img = torch.flip(color.squeeze(0), dims=[0])              # [H, W, 3+α (we passed 3)]
        hit_img = torch.flip((rast_out[..., 3:4] > 0).squeeze(0).float(), dims=[0])  # [H, W, 1]

        rendered[t] = color_img.permute(2, 0, 1)[:3].clamp(0.0, 1.0)
        visibility[t] = hit_img.permute(2, 0, 1)

    return rendered, visibility


# ── End-to-end ─────────────────────────────────────────────────────────────


def lift_and_render(
    static_rgb: torch.Tensor,         # [3, He, We] in [0, 1]
    static_depth: torch.Tensor,       # [He, We]   radial m  (sky filled, no NaN)
    pano_c2w_at_t0: torch.Tensor,     # [4, 4]     OpenCV
    target_c2w: torch.Tensor,         # [T, 4, 4]  OpenCV  (perspective camera traj)
    intrinsics: torch.Tensor,         # [4]
    pers_h: int = 480,
    pers_w: int = 832,
    chunk_size: int = 4,
    use_mesh: bool = False,
    mesh_face_res: int = 1024,
    mesh_backface_cull: bool = True,
    mesh_min_angle_deg: float = 2.5,
    mesh_edge_depth_ratio_thresh: float = 0.5,
    mesh_edge_aspect_ratio_thresh: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lift one source frame and render to a target trajectory.

    Two modes:
      • Point cloud (default, ``use_mesh=False``): per-pixel scatter with z-buffer.
      • Cubemap mesh (``use_mesh=True``): equirect → cubemap mesh with shared-vertex
        stitching, filtered for silhouette / degenerate triangles, rasterized via
        nvdiffrast (CUDA only).  Optional backface culling (default on) drops
        triangles whose orientation flipped at silhouettes.
    """
    if use_mesh:
        verts, faces, colors = build_cubemap_mesh_world(
            static_rgb, static_depth, pano_c2w_at_t0,
            face_res=mesh_face_res,
            min_angle_deg=mesh_min_angle_deg,
            edge_depth_ratio_thresh=mesh_edge_depth_ratio_thresh,
            edge_aspect_ratio_thresh=mesh_edge_aspect_ratio_thresh,
        )
        return render_mesh_to_perspective(
            verts, faces, colors, target_c2w, intrinsics,
            pers_h=pers_h, pers_w=pers_w,
            backface_cull=mesh_backface_cull,
        )

    pts_world, rgb_flat = lift_pano_to_world_pointcloud(
        static_rgb, static_depth, pano_c2w_at_t0
    )
    return render_pointcloud_to_perspective(
        pts_world, rgb_flat, target_c2w, intrinsics, pers_h, pers_w, chunk_size=chunk_size
    )
