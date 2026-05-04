"""Plucker ray utilities for view_transfer_via_query.

Reuses ray_condition() from srcpano2tgtpers/plucker.py and adds helpers for
computing plucker at VAE-aligned latent timestamps.
"""

import torch
from typing import Optional
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from srcpano2tgtpers.plucker import ray_condition  # noqa: F401 — re-export


def compute_plucker_at_latent_timestamps(
    K: torch.Tensor,
    c2w: torch.Tensor,
    H: int,
    W: int,
    vae_temporal_factor: int = 4,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-pixel 6D Plucker rays at the timestamps that align with VAE latent frames.

    Wan2.1 causal VAE maps T_video frames to T_lat = (T_video + 3) // 4 latent frames.
    Latent frame i corresponds roughly to raw frame i * 4 (with index 0 being frame 0).
    We sample plucker at raw frames [0, 4, 8, ..., 4*(T_lat-1)] (clamped to valid range).

    Args:
        K:    [B, 4] (constant intrinsics) or [B, T_video, 4] / [B, T_video, 3, 3]
        c2w:  [B, T_video, 4, 4]  camera-to-world (OpenCV convention)
        H, W: video pixel resolution
        vae_temporal_factor: temporal compression factor (default 4)
        normalize: unit-length ray direction (default True)

    Returns:
        plucker: [B, T_lat, H, W, 6]
    """
    B, T_video = c2w.shape[:2]
    T_lat = (T_video + vae_temporal_factor - 1) // vae_temporal_factor
    indices = torch.arange(T_lat, device=c2w.device) * vae_temporal_factor
    indices = indices.clamp(max=T_video - 1)

    c2w_sampled = c2w[:, indices]  # [B, T_lat, 4, 4]

    if K.dim() == 2 and K.shape[-1] == 4:
        # Constant intrinsics — broadcast to T_lat
        K_sampled = K.unsqueeze(1).expand(B, T_lat, 4)
    elif K.dim() == 3 and K.shape[-1] == 4:
        K_sampled = K[:, indices]
    elif K.dim() == 4 and K.shape[-2:] == (3, 3):
        K_sampled = K[:, indices]
    else:
        raise ValueError(f"Unexpected K shape {tuple(K.shape)}")

    return ray_condition(K_sampled, c2w_sampled, H, W, normalize=normalize)


def plucker_to_channels(plucker: torch.Tensor) -> torch.Tensor:
    """Rearrange plucker from [B, T, H, W, 6] → [B, 6, T, H, W] for model input."""
    return plucker.permute(0, 4, 1, 2, 3).contiguous()
