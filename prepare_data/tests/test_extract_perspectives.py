"""Unit tests for extract_perspectives.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import math
import numpy as np
import torch
import pytest

from view_transfer_via_query.prepare_data.extract_perspectives import (
    sample_perspective_trajectory,
    yaw_pitch_roll_to_R,
    fov_to_intrinsics,
    equi_to_perspective_video,
    compose_perspective_c2w,
    extract_perspective_from_equi,
    _build_pers_rays,
    _rays_to_equi_grid,
)
from view_transfer_via_query.prepare_data.parse_cameras import (
    M_CV_TO_UE_CAM,
    F_UE_TO_CV_WORLD,
)


# ── Trajectory sampling ──

def test_sample_trajectory_shape():
    traj = sample_perspective_trajectory(num_frames=81, rng=np.random.default_rng(0))
    for k in ["yaw_deg", "pitch_deg", "roll_deg"]:
        assert traj[k].shape == (81,)
    assert 60.0 <= traj["fov_h_deg"] <= 90.0


def test_trajectory_smoothness():
    """Smoothing kernel should keep frame-to-frame yaw deltas small relative to jitter std."""
    traj = sample_perspective_trajectory(
        num_frames=81, jitter_std_yaw=10.0, smooth_kernel=11,
        # Disable drift so this test isolates the smoothing of the noise term.
        drift_prob_yaw=0.0, drift_prob_pitch=0.0,
        rng=np.random.default_rng(0),
    )
    diffs = np.abs(np.diff(traj["yaw_deg"]))
    # With smoothing, frame-to-frame diff should be MUCH smaller than 10° std
    assert diffs.mean() < 5.0


def test_drift_round_trip_within_bounds():
    """Force round-trip drift on yaw; peak and end displacement must respect drift_max."""
    base_yaw = 0.0
    drift_max = 30.0
    # Repeat across many seeds so we exercise t_turn variety.
    for seed in range(50):
        traj = sample_perspective_trajectory(
            num_frames=81,
            jitter_std_yaw=0.0, jitter_std_pitch=0.0, jitter_std_roll=0.0,
            base_yaw_range=(0.0, 0.0),  # collapse base so drift dominates
            base_pitch_range=(0.0, 0.0), base_roll_range=(0.0, 0.0),
            drift_prob_yaw=1.0, drift_prob_pitch=0.0,
            round_trip_prob=1.0,
            drift_max_yaw_deg=drift_max,
            rng=np.random.default_rng(seed),
        )
        yaw = traj["yaw_deg"]
        # Both the in-clip extremum and the endpoints must be within ±drift_max.
        assert abs(yaw.min()) <= drift_max + 1e-3
        assert abs(yaw.max()) <= drift_max + 1e-3
        assert abs(yaw[0])  <= 1e-3       # starts at base (0 here)
        assert abs(yaw[-1]) <= drift_max + 1e-3


def test_drift_axes_independent():
    """yaw and pitch should be able to land in different drift kinds in the same call."""
    seen_kinds = set()
    for seed in range(80):
        traj = sample_perspective_trajectory(
            num_frames=49,
            jitter_std_yaw=0.0, jitter_std_pitch=0.0, jitter_std_roll=0.0,
            base_yaw_range=(0.0, 0.0),
            base_pitch_range=(0.0, 0.0), base_roll_range=(0.0, 0.0),
            drift_prob_yaw=0.5, drift_prob_pitch=0.5,
            round_trip_prob=0.5,
            rng=np.random.default_rng(seed),
        )
        # Classify each axis by its zero-flatness vs end-displacement signature
        def classify(arr):
            arr = np.asarray(arr)
            if np.allclose(arr, 0.0, atol=1e-6):
                return "none"
            # round-trip has at least one sign flip in the per-frame difference
            d = np.diff(arr)
            if np.sum(np.diff(np.sign(d)) != 0) >= 1:
                return "round_trip"
            return "one_way"
        seen_kinds.add((classify(traj["yaw_deg"]), classify(traj["pitch_deg"])))
        if len({k[0] for k in seen_kinds}) > 1 and len({k[1] for k in seen_kinds}) > 1:
            return  # observed both axes taking multiple kinds — independence confirmed
    raise AssertionError(
        f"Did not observe independent kinds across yaw and pitch in 80 seeds. "
        f"Got: {seen_kinds}"
    )


# ── Rotation construction ──

def test_yaw_pitch_roll_identity():
    R = yaw_pitch_roll_to_R(
        torch.zeros(1), torch.zeros(1), torch.zeros(1)
    )
    assert torch.allclose(R[0], torch.eye(3), atol=1e-6)


def test_yaw_only_rotation():
    """Yaw=90° (look right) in OpenCV: pers +Z (forward) → pano +X (right)."""
    R = yaw_pitch_roll_to_R(
        torch.tensor([90.0]), torch.zeros(1), torch.zeros(1),
    )[0]
    pers_forward = torch.tensor([0.0, 0.0, 1.0])
    pano_dir = R @ pers_forward
    assert torch.allclose(pano_dir, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6), (
        f"yaw=90° forward → got {pano_dir}, expected (1,0,0)"
    )


def test_pitch_only_rotation():
    """Pitch=90° (look down) in OpenCV: pers +Z → pano +Y (down)."""
    R = yaw_pitch_roll_to_R(
        torch.zeros(1), torch.tensor([90.0]), torch.zeros(1),
    )[0]
    pers_forward = torch.tensor([0.0, 0.0, 1.0])
    pano_dir = R @ pers_forward
    assert torch.allclose(pano_dir, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_orthonormal_rotations():
    rng = np.random.default_rng(1)
    yaw = torch.from_numpy(rng.uniform(-180, 180, 5)).float()
    pitch = torch.from_numpy(rng.uniform(-30, 30, 5)).float()
    roll = torch.from_numpy(rng.uniform(-15, 15, 5)).float()
    R = yaw_pitch_roll_to_R(yaw, pitch, roll)
    RtR = R.transpose(-1, -2) @ R
    assert torch.allclose(RtR, torch.eye(3).expand_as(RtR), atol=1e-5)


# ── Intrinsics ──

def test_fov_to_intrinsics_90deg():
    fx, fy, cx, cy = fov_to_intrinsics(90.0, pers_h=480, pers_w=832)
    # tan(45°) = 1, so fx = 832 / 2 = 416
    assert math.isclose(fx, 416.0, abs_tol=1e-4)
    assert math.isclose(fy, 416.0, abs_tol=1e-4)
    assert cx == 416.0 and cy == 240.0


# ── Equi → perspective: identity case ──

def _mark_band(equi, ch, h_lo, h_hi, w_lo, w_hi):
    """Helper: mark a rectangular band in an equirect image."""
    equi[0, ch, h_lo:h_hi, w_lo:w_hi] = 1.0


def test_identity_extraction_returns_center():
    """With yaw=pitch=roll=0 and a small FOV, perspective should sample the equirect's
    forward (UE +X) direction at u=W/2, v=H/2."""
    He, We = 256, 512
    equi = torch.zeros(1, 3, He, We)
    # Mark a 32x32 patch around center
    _mark_band(equi, 0, He // 2 - 16, He // 2 + 16, We // 2 - 16, We // 2 + 16)
    R_crop = yaw_pitch_roll_to_R(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    pers = equi_to_perspective_video(equi, R_crop, fov_h_deg=10.0, pers_h=64, pers_w=128)
    # The pers center 16x16 should be saturated
    assert pers[0, 0, 24:40, 56:72].mean().item() > 0.95


def test_yaw_90_extraction_samples_right():
    """With yaw=90°, the pers center should sample equirect at u=3W/4 (right)."""
    He, We = 256, 512
    equi = torch.zeros(1, 3, He, We)
    u_center = 3 * We // 4
    _mark_band(equi, 0, He // 2 - 16, He // 2 + 16, u_center - 16, u_center + 16)
    R_crop = yaw_pitch_roll_to_R(
        torch.tensor([90.0]), torch.zeros(1), torch.zeros(1)
    )
    pers = equi_to_perspective_video(equi, R_crop, fov_h_deg=10.0, pers_h=64, pers_w=128)
    assert pers[0, 0, 24:40, 56:72].mean().item() > 0.95


def test_pitch_90_extraction_samples_bottom():
    """With pitch=90° (look down), pers center should sample equirect's bottom row."""
    He, We = 256, 512
    equi = torch.zeros(1, 3, He, We)
    # Mark a wide horizontal band at the bottom (latitude near -π/2 = down)
    _mark_band(equi, 0, He - 16, He, 0, We)
    R_crop = yaw_pitch_roll_to_R(
        torch.zeros(1), torch.tensor([90.0]), torch.zeros(1)
    )
    pers = equi_to_perspective_video(equi, R_crop, fov_h_deg=10.0, pers_h=64, pers_w=128)
    # With small FOV, all pers pixels should land near the bottom band
    assert pers[0, 0].mean().item() > 0.8


# ── Perspective c2w composition ──

def test_compose_pers_c2w_identity():
    """If R_crop = I, pers_c2w = pano_c2w."""
    pano_c2w = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
    pano_c2w[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0]] * 3)
    R_crop = torch.eye(3).unsqueeze(0).repeat(3, 1, 1)
    pers_c2w = compose_perspective_c2w(pano_c2w, R_crop)
    assert torch.allclose(pers_c2w, pano_c2w)


def test_compose_pers_c2w_yaw():
    """Pers camera with yaw=90° relative to pano: pers_c2w[:3,2] (forward) should
    be the pano's right direction in world."""
    # Pano is identity: looks +Z (forward) in world (OpenCV identity)
    pano_c2w = torch.eye(4).unsqueeze(0)
    R_crop = yaw_pitch_roll_to_R(
        torch.tensor([90.0]), torch.zeros(1), torch.zeros(1)
    )
    pers_c2w = compose_perspective_c2w(pano_c2w, R_crop)
    pers_forward_world = pers_c2w[0, :3, 2]
    # Pano +Z in world is (0,0,1). Pers at yaw=90° looks at pano +X direction.
    # Pano +X in world is (1,0,0).
    assert torch.allclose(pers_forward_world, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


# ── End-to-end ──

def test_extract_perspective_e2e_shapes():
    T = 5
    He, We = 64, 128
    equi = torch.rand(T, 3, He, We)
    pano_c2w = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
    traj = sample_perspective_trajectory(num_frames=T, rng=np.random.default_rng(0))
    out = extract_perspective_from_equi(equi, pano_c2w, traj, pers_h=24, pers_w=40)
    assert out["perspective_video"].shape == (T, 3, 24, 40)
    assert out["pers_c2w"].shape == (T, 4, 4)
    assert out["intrinsics"].shape == (4,)
    assert out["R_crop"].shape == (T, 3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
