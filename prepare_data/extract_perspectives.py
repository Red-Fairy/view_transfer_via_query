"""Extract perspective videos from 360 equirectangular videos via differentiable warping.

Pipeline per frame:
  1. Build pers-camera ray grid (OpenCV camera frame: +X right, +Y down, +Z forward)
  2. Rotate rays to pano-camera frame via R_crop (pers → pano, OpenCV)
  3. Convert pano rays to UE-camera frame (the equirect was rendered in UE convention)
  4. Compute (lon, lat) → (u, v) equirect pixel coords
  5. F.grid_sample bilinear from the equirect

Trajectory sampling: one set of (FOV, yaw, pitch, roll) per frame, with smooth jitter.
Outputs the perspective video plus the OpenCV c2w trajectory and intrinsics for that view.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from .parse_cameras import M_CV_TO_UE_CAM


# ── Trajectory sampling ─────────────────────────────────────────────────────


def _sample_drift_array(
    num_frames: int,
    drift_max_deg: float,
    drift_prob: float,
    round_trip_prob: float,
    t_turn_frac: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-axis drift component of length `num_frames`. Three regimes:

      - **none** (prob 1 - drift_prob)             → all zeros
      - **one-way** (prob drift_prob*(1-round_trip_prob))
            constant velocity for the whole clip; end displacement ~ U(-D, +D)
      - **round-trip** (prob drift_prob*round_trip_prob)
            constant |v| out for `t_turn` frames, then constant -|v| back; |v| sized
            so neither the peak nor the end displacement leave [-D, +D]; start ≠ end
            in general (start = end only when t_turn = (N-1)/2)
    """
    N = num_frames
    if N <= 1 or drift_max_deg <= 0 or rng.random() >= drift_prob:
        return np.zeros(N, dtype=np.float32)

    if rng.random() >= round_trip_prob:
        # one-way: peak at frame N-1; |v| automatically ≤ D / (N-1)
        peak = float(rng.uniform(-drift_max_deg, drift_max_deg))
        return (peak * np.arange(N) / max(N - 1, 1)).astype(np.float32)

    # round-trip
    t_turn_lo = max(1, int(round(t_turn_frac[0] * (N - 1))))
    t_turn_hi = min(N - 2, int(round(t_turn_frac[1] * (N - 1))))
    t_turn_hi = max(t_turn_hi, t_turn_lo)  # guard for very short clips
    t_turn = int(rng.integers(t_turn_lo, t_turn_hi + 1))
    end_arm = abs(2 * t_turn - (N - 1))  # |2*t_turn - (N-1)|
    bound = max(t_turn, end_arm)
    v_max = drift_max_deg / max(bound, 1)
    v = float(rng.uniform(-v_max, v_max))
    t = np.arange(N)
    drift = np.where(t <= t_turn, v * t, v * (2 * t_turn - t)).astype(np.float32)
    return drift


def sample_perspective_trajectory(
    num_frames: int,
    fov_range: Tuple[float, float] = (60.0, 90.0),
    base_yaw_range: Tuple[float, float] = (-180.0, 180.0),
    base_pitch_range: Tuple[float, float] = (-10.0, 10.0),
    base_roll_range: Tuple[float, float] = (-3.0, 3.0),
    jitter_std_yaw: float = 1.0,
    jitter_std_pitch: float = 0.6,
    jitter_std_roll: float = 0.3,
    smooth_kernel: int = 21,
    drift_prob_yaw: float = 0.75,
    drift_prob_pitch: float = 0.5,
    round_trip_prob: float = 0.5,
    drift_max_yaw_deg: float = 90.0,
    drift_max_pitch_deg: float = 5.0,
    t_turn_frac: Tuple[float, float] = (0.25, 0.75),
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Sample one smooth random perspective-camera trajectory.

    Per-axis composition:
      yaw[t]   = base_yaw   + box_smooth(N(0, σ_y), k) + drift_yaw(t)
      pitch[t] = base_pitch + box_smooth(N(0, σ_p), k) + drift_pitch(t)
      roll[t]  = base_roll  + box_smooth(N(0, σ_r), k)              # roll: no drift

    Drift on yaw and pitch is sampled independently per axis (independent kind +
    independent t_turn + independent v); see `_sample_drift_array` for shape.

    Returns dict with keys:
      fov_h_deg: float
      yaw_deg, pitch_deg, roll_deg: each [num_frames] float32
    """
    if rng is None:
        rng = np.random.default_rng()

    fov = float(rng.uniform(*fov_range))
    base_yaw = float(rng.uniform(*base_yaw_range))
    base_pitch = float(rng.uniform(*base_pitch_range))
    base_roll = float(rng.uniform(*base_roll_range))

    def _smooth_jitter(std: float) -> np.ndarray:
        raw = rng.normal(0.0, std, num_frames)
        k = min(smooth_kernel, num_frames)
        if k > 1:
            kernel = np.ones(k) / k
            raw = np.convolve(raw, kernel, mode="same")
        return raw

    drift_yaw = _sample_drift_array(
        num_frames, drift_max_yaw_deg, drift_prob_yaw, round_trip_prob, t_turn_frac, rng,
    )
    drift_pitch = _sample_drift_array(
        num_frames, drift_max_pitch_deg, drift_prob_pitch, round_trip_prob, t_turn_frac, rng,
    )

    yaw = base_yaw + _smooth_jitter(jitter_std_yaw) + drift_yaw
    pitch = base_pitch + _smooth_jitter(jitter_std_pitch) + drift_pitch
    roll = base_roll + _smooth_jitter(jitter_std_roll)

    return {
        "fov_h_deg": fov,
        "yaw_deg": yaw.astype(np.float32),
        "pitch_deg": pitch.astype(np.float32),
        "roll_deg": roll.astype(np.float32),
    }


# ── Rotation utilities ──────────────────────────────────────────────────────


def yaw_pitch_roll_to_R(
    yaw_deg: torch.Tensor, pitch_deg: torch.Tensor, roll_deg: torch.Tensor
) -> torch.Tensor:
    """Build [T, 3, 3] rotations in OpenCV convention.

    Convention: R = R_yaw(Y) @ R_pitch(X) @ R_roll(Z),
    where:
      • yaw rotates around +Y (down) axis    — positive yaw   = look right
      • pitch rotates around +X (right) axis — positive pitch = look up   (camera forward → −Y)
      • roll rotates around +Z (forward) axis — positive roll = tilt right (camera +X → +Y)
    R maps perspective-camera rays into pano-camera frame.
    """
    y = torch.deg2rad(yaw_deg)
    p = torch.deg2rad(pitch_deg)
    r = torch.deg2rad(roll_deg)

    cy, sy = torch.cos(y), torch.sin(y)
    cp, sp = torch.cos(p), torch.sin(p)
    cr, sr = torch.cos(r), torch.sin(r)
    zeros = torch.zeros_like(y)
    ones = torch.ones_like(y)

    Ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )
    # R_x(+pitch): positive pitch = look UP (forward → −Y in OpenCV pano frame)
    Rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cp, -sp], dim=-1),
            torch.stack([zeros, sp, cp], dim=-1),
        ],
        dim=-2,
    )
    Rz = torch.stack(
        [
            torch.stack([cr, -sr, zeros], dim=-1),
            torch.stack([sr, cr, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return Ry @ Rx @ Rz


# ── Equi → Perspective projection ───────────────────────────────────────────


def fov_to_intrinsics(fov_h_deg: float, pers_h: int, pers_w: int) -> Tuple[float, float, float, float]:
    """Compute pinhole intrinsics from horizontal FOV. Square pixels (fy=fx)."""
    fx = pers_w / (2.0 * math.tan(math.radians(fov_h_deg) / 2.0))
    fy = fx
    cx = pers_w / 2.0
    cy = pers_h / 2.0
    return fx, fy, cx, cy


def _build_pers_rays(
    pers_h: int, pers_w: int, fov_h_deg: float, device, dtype=torch.float32
) -> torch.Tensor:
    """Per-pixel unit ray directions in OpenCV pers-camera frame [pers_h, pers_w, 3]."""
    fx, fy, cx, cy = fov_to_intrinsics(fov_h_deg, pers_h, pers_w)
    ys = torch.arange(pers_h, device=device, dtype=dtype) + 0.5
    xs = torch.arange(pers_w, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x_cam = (xx - cx) / fx
    y_cam = (yy - cy) / fy
    z_cam = torch.ones_like(x_cam)
    rays = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    rays = rays / rays.norm(dim=-1, keepdim=True)
    return rays  # [H, W, 3]


def _rays_to_equi_grid(
    rays_pers: torch.Tensor,    # [T, H, W, 3]   OpenCV pers frame
    R_crop_cv: torch.Tensor,    # [T, 3, 3]      pers → pano (OpenCV)
    equi_h: int, equi_w: int,
) -> torch.Tensor:
    """Map pers-frame rays to normalized equirect sample coords for grid_sample.

    Returns grid [T, H, W, 2] in [-1, 1] (x=u, y=v) suitable for F.grid_sample.
    """
    T, H, W, _ = rays_pers.shape
    M = torch.tensor(M_CV_TO_UE_CAM, device=rays_pers.device, dtype=rays_pers.dtype)

    # Step 1: pers → pano (OpenCV)
    rays_pano_cv = torch.einsum("tij,thwj->thwi", R_crop_cv, rays_pers)

    # Step 2: OpenCV pano → UE pano frame (M @ d_cv = d_ue)
    rays_pano_ue = torch.einsum("ij,thwj->thwi", M, rays_pano_cv)

    # Step 3: UE camera convention is +X forward, +Y right, +Z up
    #         lon = atan2(y, x);  lat = atan2(z, sqrt(x^2 + y^2))
    x = rays_pano_ue[..., 0]
    y = rays_pano_ue[..., 1]
    z = rays_pano_ue[..., 2]
    lon = torch.atan2(y, x)
    lat = torch.atan2(z, torch.sqrt(x * x + y * y).clamp(min=1e-12))

    # Step 4: equirect pixel coords (u in [0, W], v in [0, H])
    u = (lon + math.pi) / (2.0 * math.pi) * equi_w
    v = (math.pi / 2.0 - lat) / math.pi * equi_h

    # Normalize to [-1, 1] for grid_sample (align_corners=False uses pixel-center alignment)
    grid_x = (u / equi_w) * 2.0 - 1.0
    grid_y = (v / equi_h) * 2.0 - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def equi_to_perspective_video(
    equi_video: torch.Tensor,   # [T, 3, H_e, W_e]   float in [0,1] or [-1,1]
    R_crop_cv: torch.Tensor,    # [T, 3, 3]          per-frame pers → pano (OpenCV)
    fov_h_deg: float,
    pers_h: int = 480,
    pers_w: int = 832,
    sample_mode: str = "bilinear",
) -> torch.Tensor:
    """Project an equirectangular video to perspective using F.grid_sample.

    Returns [T, 3, pers_h, pers_w] in the same value range as input.
    """
    T, C, He, We = equi_video.shape
    assert R_crop_cv.shape == (T, 3, 3), f"R_crop shape {R_crop_cv.shape}"
    device, dtype = equi_video.device, equi_video.dtype

    rays_pers = _build_pers_rays(pers_h, pers_w, fov_h_deg, device=device, dtype=dtype)
    rays_pers = rays_pers.unsqueeze(0).expand(T, -1, -1, -1)  # [T, H, W, 3]
    grid = _rays_to_equi_grid(rays_pers, R_crop_cv.to(dtype), He, We)
    out = F.grid_sample(
        equi_video, grid, mode=sample_mode, padding_mode="border", align_corners=False
    )
    return out


# ── Compose perspective c2w from pano c2w + crop rotation ──────────────────


def compose_perspective_c2w(
    pano_c2w_cv: torch.Tensor,  # [..., 4, 4]   OpenCV (any leading batch dims)
    R_crop_cv: torch.Tensor,    # [..., 3, 3]   pers → pano (OpenCV)
) -> torch.Tensor:
    """pers_c2w = pano_c2w @ R_crop_4x4. Translation unchanged.

    Handles arbitrary leading batch dims, e.g. [T, 4, 4] or [B, T, 4, 4].
    """
    leading = pano_c2w_cv.shape[:-2]
    eye = torch.eye(4, device=pano_c2w_cv.device, dtype=pano_c2w_cv.dtype)
    R_crop_4x4 = eye.expand(*leading, 4, 4).contiguous().clone()
    R_crop_4x4[..., :3, :3] = R_crop_cv
    return pano_c2w_cv @ R_crop_4x4


# ── End-to-end ──────────────────────────────────────────────────────────────


def extract_perspective_from_equi(
    equi_video: torch.Tensor,            # [T, 3, He, We]
    pano_c2w_cv: torch.Tensor,           # [T, 4, 4] OpenCV
    trajectory: Dict[str, np.ndarray],   # output of sample_perspective_trajectory
    pers_h: int = 480,
    pers_w: int = 832,
) -> Dict[str, torch.Tensor]:
    """Run the full extraction. Returns:
        perspective_video [T, 3, pers_h, pers_w]
        pers_c2w          [T, 4, 4]
        intrinsics        [4]  (fx, fy, cx, cy)
        R_crop            [T, 3, 3]  for debugging
    """
    device = equi_video.device
    dtype = equi_video.dtype
    T = equi_video.shape[0]

    yaw = torch.from_numpy(trajectory["yaw_deg"]).to(device=device, dtype=dtype)
    pitch = torch.from_numpy(trajectory["pitch_deg"]).to(device=device, dtype=dtype)
    roll = torch.from_numpy(trajectory["roll_deg"]).to(device=device, dtype=dtype)
    R_crop = yaw_pitch_roll_to_R(yaw, pitch, roll)  # [T, 3, 3]

    pers_video = equi_to_perspective_video(
        equi_video, R_crop, trajectory["fov_h_deg"], pers_h=pers_h, pers_w=pers_w
    )
    pers_c2w = compose_perspective_c2w(pano_c2w_cv, R_crop)
    fx, fy, cx, cy = fov_to_intrinsics(trajectory["fov_h_deg"], pers_h, pers_w)
    intrinsics = torch.tensor([fx, fy, cx, cy], dtype=dtype, device=device)

    return {
        "perspective_video": pers_video,
        "pers_c2w": pers_c2w,
        "intrinsics": intrinsics,
        "R_crop": R_crop,
    }
