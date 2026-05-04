"""Video I/O for the prep pipeline.

Loads PNG frame sequences (the primary format on disk) or MP4 files and returns
[T, 3, H, W] uint8 / float tensors.
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Optional


def list_png_frames(dir_path: str) -> List[str]:
    """Sorted list of PNG paths in a directory."""
    files = [f for f in os.listdir(dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    return [os.path.join(dir_path, f) for f in files]


def load_png_sequence(
    dir_path: str,
    start: int = 0,
    num_frames: Optional[int] = None,
    stride: int = 1,
    return_dtype=torch.float32,
) -> torch.Tensor:
    """Load a window of PNG frames → [T, 3, H, W].

    Values normalized to [0, 1] if float dtype, else uint8 [0, 255].
    """
    paths = list_png_frames(dir_path)
    if num_frames is None:
        num_frames = (len(paths) - start) // stride
    indices = [start + i * stride for i in range(num_frames)]
    if max(indices) >= len(paths):
        raise IndexError(
            f"Need frame {max(indices)} but dir has only {len(paths)}: {dir_path}"
        )

    frames = []
    for i in indices:
        bgr = cv2.imread(paths[i], cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Failed to read {paths[i]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

    arr = np.stack(frames, axis=0)  # [T, H, W, 3] uint8
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [T, 3, H, W]
    if return_dtype.is_floating_point:
        tensor = tensor.to(return_dtype) / 255.0
    return tensor


def load_mp4(
    path: str,
    start: int = 0,
    num_frames: Optional[int] = None,
    stride: int = 1,
    return_dtype=torch.float32,
) -> torch.Tensor:
    """Load a window from an MP4 → [T, 3, H, W]. Uses decord (fast random access)."""
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(path)
    total = len(vr)
    if num_frames is None:
        num_frames = (total - start) // stride
    indices = [start + i * stride for i in range(num_frames)]
    if max(indices) >= total:
        raise IndexError(f"Need frame {max(indices)} but mp4 has only {total}: {path}")
    frames = vr.get_batch(indices)  # [T, H, W, 3] uint8 (decord returns RGB)
    tensor = frames.permute(0, 3, 1, 2).contiguous()
    if return_dtype.is_floating_point:
        tensor = tensor.to(return_dtype) / 255.0
    return tensor


def load_video_auto(
    path_or_dir: str,
    start: int = 0,
    num_frames: Optional[int] = None,
    stride: int = 1,
    return_dtype=torch.float32,
) -> torch.Tensor:
    """Detect PNG-dir vs MP4 and dispatch."""
    if os.path.isdir(path_or_dir):
        return load_png_sequence(path_or_dir, start, num_frames, stride, return_dtype)
    return load_mp4(path_or_dir, start, num_frames, stride, return_dtype)
