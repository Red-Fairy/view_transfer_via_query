"""Mask packing: raw video-resolution binary mask → latent-aligned 4-channel tensor."""

import torch
import torch.nn.functional as F


def pack_mask(
    mask: torch.Tensor,
    vae_spatial_factor: int = 8,
    vae_temporal_factor: int = 4,
) -> torch.Tensor:
    """Pack a binary video mask to VAE-latent resolution with temporal folding.

    Wan2.1-I2V packs 4 raw mask frames into 4 channels per latent frame.
    Spatial downsampling uses max-pool (conservative: pixel is masked if ANY
    sub-pixel in the 8x8 block is masked).

    Args:
        mask: [B, 1, T_video, H_video, W_video]  float in {0, 1}
        vae_spatial_factor: spatial compression (default 8)
        vae_temporal_factor: temporal compression (default 4)

    Returns:
        packed: [B, vae_temporal_factor, T_lat, H_lat, W_lat]
                where T_lat = (T_video + vae_temporal_factor - 1) // vae_temporal_factor
    """
    B, C, T, H, W = mask.shape
    assert C == 1

    sf = vae_spatial_factor
    tf = vae_temporal_factor
    T_lat = (T + tf - 1) // tf

    # Spatial max-pool (per-frame, treat T as batch)
    m = mask.reshape(B * T, 1, H, W)
    m = F.max_pool2d(m, kernel_size=sf, stride=sf)  # [B*T, 1, H_lat, W_lat]
    H_lat, W_lat = m.shape[2], m.shape[3]
    m = m.reshape(B, T, 1, H_lat, W_lat)

    # Temporal fold: pad T to T_lat * tf, then group into tf channels
    T_padded = T_lat * tf
    if T < T_padded:
        pad = m[:, -1:].expand(B, T_padded - T, 1, H_lat, W_lat)
        m = torch.cat([m, pad], dim=1)

    # [B, T_lat, tf, 1, H_lat, W_lat] → [B, tf, T_lat, H_lat, W_lat]
    m = m.reshape(B, T_lat, tf, 1, H_lat, W_lat)
    m = m.permute(0, 2, 1, 3, 4, 5).reshape(B, tf, T_lat, H_lat, W_lat)
    return m
