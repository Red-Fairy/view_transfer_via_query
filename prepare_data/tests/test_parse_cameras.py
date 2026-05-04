"""Unit tests for parse_cameras.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import json
import numpy as np
import pytest
import tempfile
import torch

from view_transfer_via_query.prepare_data.parse_cameras import (
    ue_c2w_to_opencv,
    parse_camera_params,
    save_camera,
    M_CV_TO_UE_CAM,
    F_UE_TO_CV_WORLD,
)


# ── Identity rotation: camera at origin, looking +X in UE world ──

def test_identity_rotation_position_only():
    """UE c2w with identity rotation and (0, 100, 0) cm position."""
    c2w_ue = np.eye(4)
    c2w_ue[:3, 3] = [0.0, 100.0, 0.0]  # 100 cm = 1 m along +Y
    c2w_cv = ue_c2w_to_opencv(c2w_ue)

    # Position: cm → m, then F flips Y
    assert np.allclose(c2w_cv[:3, 3], [0.0, -1.0, 0.0])

    # Rotation: F @ I @ M = F @ M
    expected_R = F_UE_TO_CV_WORLD @ np.eye(3) @ M_CV_TO_UE_CAM
    assert np.allclose(c2w_cv[:3, :3], expected_R)


def test_identity_rotation_forward_direction():
    """For identity UE rotation (camera looks +X in UE world), OpenCV camera +Z (forward)
    should map to world +X."""
    c2w_ue = np.eye(4)
    c2w_cv = ue_c2w_to_opencv(c2w_ue)
    # OpenCV camera forward = column 2 of R
    cam_forward_world = c2w_cv[:3, 2]
    # In our converted world, UE +X stays +X (only Y is flipped)
    expected = np.array([1.0, 0.0, 0.0])
    assert np.allclose(cam_forward_world, expected), (
        f"Expected forward {expected}, got {cam_forward_world}"
    )


def test_identity_rotation_right_direction():
    """OpenCV camera +X (right) should map to UE camera +Y (right) → world,
    after F flip becomes -world_Y."""
    c2w_ue = np.eye(4)
    c2w_cv = ue_c2w_to_opencv(c2w_ue)
    cam_right_world = c2w_cv[:3, 0]
    # UE +Y after F flip is -Y in our world frame
    expected = np.array([0.0, -1.0, 0.0])
    assert np.allclose(cam_right_world, expected)


def test_identity_rotation_down_direction():
    """OpenCV camera +Y (down) should map to UE camera -Z (down) → world -Z (still -Z after F)."""
    c2w_ue = np.eye(4)
    c2w_cv = ue_c2w_to_opencv(c2w_ue)
    cam_down_world = c2w_cv[:3, 1]
    expected = np.array([0.0, 0.0, -1.0])
    assert np.allclose(cam_down_world, expected)


def test_orthonormal_preservation():
    """Conversion must preserve rotation orthonormality."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Random rotation
        from scipy.spatial.transform import Rotation as R
        R_ue = R.random(random_state=rng).as_matrix()
        c2w_ue = np.eye(4)
        c2w_ue[:3, :3] = R_ue
        c2w_ue[:3, 3] = rng.uniform(-1000, 1000, 3)

        c2w_cv = ue_c2w_to_opencv(c2w_ue)
        R_cv = c2w_cv[:3, :3]
        # R^T R = I and det = +1
        assert np.allclose(R_cv.T @ R_cv, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R_cv), 1.0, atol=1e-9)


def test_position_units_convert():
    """Position should be scaled by 1/100 (cm → m) regardless of rotation."""
    c2w_ue = np.eye(4)
    c2w_ue[:3, 3] = [12345.0, -6789.0, 100.0]  # cm
    c2w_cv = ue_c2w_to_opencv(c2w_ue)
    # F flips Y
    assert np.allclose(c2w_cv[:3, 3], [123.45, 67.89, 1.0])


# ── parse_camera_params ──

def test_parse_camera_params_minimal():
    """Build a minimal JSON in UE format and round-trip parse."""
    fake_json = {
        "coordinate_system": "unreal_engine_lhs_z_up",
        "units": "centimeters",
        "num_frames": 2,
        "cameras": {
            "PanoCam_00": {
                "intrinsics": {
                    "projection": "equirectangular",
                    "width": 4096,
                    "height": 2048,
                    "h_fov_deg": 360.0,
                    "v_fov_deg": 180.0,
                },
                "extrinsics_per_frame": [
                    {
                        "frame": 0,
                        "camera_to_world_4x4": np.eye(4).tolist(),
                    },
                    {
                        "frame": 1,
                        "camera_to_world_4x4": [
                            [1.0, 0.0, 0.0, 100.0],
                            [0.0, 1.0, 0.0, 200.0],
                            [0.0, 0.0, 1.0, 300.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                ],
            },
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(fake_json, f)
        f.flush()
        parsed = parse_camera_params(f.name)

    assert parsed["num_frames"] == 2
    assert "PanoCam_00" in parsed["cameras"]
    c2w = parsed["cameras"]["PanoCam_00"]["c2w"]
    assert c2w.shape == (2, 4, 4)
    # Frame 1 position should be (1, -2, 3) m after conversion (Y flipped, cm→m)
    assert np.allclose(c2w[1, :3, 3], [1.0, -2.0, 3.0])


def test_save_camera(tmp_path):
    """Verify save_camera writes c2w.pt and equirect_intrinsics.json."""
    fake_json = {
        "coordinate_system": "unreal_engine_lhs_z_up",
        "units": "centimeters",
        "num_frames": 1,
        "cameras": {
            "PanoCam_00": {
                "intrinsics": {"projection": "equirectangular", "width": 4096, "height": 2048},
                "extrinsics_per_frame": [{"frame": 0, "camera_to_world_4x4": np.eye(4).tolist()}],
            },
        },
    }
    json_path = tmp_path / "camera.json"
    with open(json_path, "w") as f:
        json.dump(fake_json, f)
    parsed = parse_camera_params(str(json_path))
    out_dir = tmp_path / "out"
    save_camera(parsed, "PanoCam_00", str(out_dir))

    assert (out_dir / "c2w.pt").exists()
    assert (out_dir / "equirect_intrinsics.json").exists()
    c2w = torch.load(out_dir / "c2w.pt", weights_only=True)
    assert c2w.shape == (1, 4, 4)


# ── Real data smoke test ──

REAL_JSON = (
    "/share/ma/scratch/rundong/Unreal_Projects/outputs_non_arranged_cars/"
    "Desert_0car_2-8pp_task200/x-13387_y53630_s1200_m3_v0_n2_p2_p2/camera_params.json"
)


@pytest.mark.skipif(not os.path.exists(REAL_JSON), reason="Real data not available")
def test_parse_real_camera_json():
    parsed = parse_camera_params(REAL_JSON)
    assert parsed["num_frames"] == 240
    assert set(parsed["cameras"].keys()) == {"PanoCam_00", "PanoCam_01"}
    for cam_name, cam in parsed["cameras"].items():
        assert cam["c2w"].shape == (240, 4, 4)
        # Last row of every c2w should be [0, 0, 0, 1]
        assert np.allclose(cam["c2w"][:, 3, :], [0.0, 0.0, 0.0, 1.0])
        # Rotation submatrices should be orthonormal
        R = cam["c2w"][:, :3, :3]
        RtR = np.einsum("tij,tkj->tik", R, R)
        assert np.allclose(RtR, np.eye(3)[None], atol=1e-6)
    # Cameras at different positions (PanoCam_00 vs _01)
    p0 = parsed["cameras"]["PanoCam_00"]["c2w"][0, :3, 3]
    p1 = parsed["cameras"]["PanoCam_01"]["c2w"][0, :3, 3]
    assert np.linalg.norm(p0 - p1) > 0.5  # > 0.5 m apart


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
