"""Convert Unreal Engine camera_params.json → OpenCV-convention c2w tensors.

UE convention:     LHS, +X forward, +Y right, +Z up;       centimeters
OpenCV convention: RHS, +X right,   +Y down,  +Z forward;  meters

Conversion logic:
  • Camera frame: permute axes (UE forward/right/up → OpenCV right/down/forward).
    M @ v_opencv = v_ue, so c2w_opencv_R = F_world @ R_ue @ M.
  • World handedness: flip Y axis to swap LHS → RHS.
  • Position: cm → m, then apply world flip.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict


# OpenCV camera (right, down, forward) → UE camera (forward, right, up)
# Mapping:  ue_x = cv_z,  ue_y = cv_x,  ue_z = -cv_y
M_CV_TO_UE_CAM: np.ndarray = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

# UE world (LHS) → OpenCV-style world (RHS) by flipping Y axis
F_UE_TO_CV_WORLD: np.ndarray = np.diag([1.0, -1.0, 1.0]).astype(np.float64)

UE_TO_M = 1.0 / 100.0  # centimeters → meters


def ue_c2w_to_opencv(c2w_ue: np.ndarray) -> np.ndarray:
    """Convert one UE c2w 4x4 matrix to OpenCV convention (units: meters)."""
    R_ue = c2w_ue[:3, :3]
    t_ue = c2w_ue[:3, 3] * UE_TO_M

    R_cv = F_UE_TO_CV_WORLD @ R_ue @ M_CV_TO_UE_CAM
    t_cv = F_UE_TO_CV_WORLD @ t_ue

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_cv
    out[:3, 3] = t_cv
    return out


def parse_camera_params(json_path: str) -> Dict:
    """Parse a UE camera_params.json file.

    Returns:
        {
          'num_frames': int,
          'cameras': {
            'PanoCam_00': {
              'intrinsics': dict,           # equirectangular metadata, kept verbatim
              'c2w': np.ndarray [T, 4, 4],  # OpenCV convention, meters
            },
            ...
          }
        }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    assert data["coordinate_system"] == "unreal_engine_lhs_z_up", (
        f"Unexpected coord system: {data['coordinate_system']}"
    )
    assert data["units"] == "centimeters", f"Unexpected units: {data['units']}"

    T = int(data["num_frames"])
    cameras: Dict = {}
    for cam_name, cam in data["cameras"].items():
        c2w_arr = np.zeros((T, 4, 4), dtype=np.float64)
        for i, ext in enumerate(cam["extrinsics_per_frame"]):
            assert ext["frame"] == i, f"Frame index mismatch at {i} ({cam_name})"
            c2w_ue = np.asarray(ext["camera_to_world_4x4"], dtype=np.float64)
            c2w_arr[i] = ue_c2w_to_opencv(c2w_ue)

        cameras[cam_name] = {
            "intrinsics": cam["intrinsics"],
            "c2w": c2w_arr,
        }

    return {"num_frames": T, "cameras": cameras}


def save_camera(parsed: Dict, cam_name: str, out_dir: str) -> None:
    """Save one camera's c2w + equirectangular intrinsics under out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cam = parsed["cameras"][cam_name]
    torch.save(torch.from_numpy(cam["c2w"]).float(), out / "c2w.pt")
    with open(out / "equirect_intrinsics.json", "w") as f:
        json.dump(cam["intrinsics"], f, indent=2)
