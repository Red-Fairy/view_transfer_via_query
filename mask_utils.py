"""Mask packing: raw video-resolution binary mask → latent-aligned 4-channel tensor."""

import torch
import torch.nn.functional as F


def pack_mask(
    mask: torch.Tensor,
    vae_spatial_factor: int = 8,
    vae_temporal_factor: int = 4,
) -> torch.Tensor:
    """Pack a binary video mask to VAE-latent resolution with temporal folding,
    aligned to the Wan2.1 VAE's *causal* temporal compression.

    The Wan2.1 VAE temporal mapping is causal, NOT a uniform stride-tf grouping
    (see `WanVideoVAE.encode`: `iter_ = 1 + (T-1)//tf`):

        latent 0      ← raw frame {0}                       (a single frame)
        latent j (≥1) ← raw frames {tf·(j-1)+1 … tf·j}

    So the tf mask channels co-located with content latent j must describe the
    SAME raw frames that produced content latent j. We achieve this by prepending
    (tf-1) copies of frame 0 before the stride-tf fold — exactly the Wan-I2V mask
    construction — which collapses the first latent group to frame 0 and aligns
    every subsequent group of tf raw frames to one latent frame.

    The previous implementation grouped uniformly from frame 0 and padded the
    TAIL, leaving the mask shifted ~(tf-1) frames *later* than the rendered /
    source / blob latents it is channel-concatenated with (worst at latent 0:
    content = 1 real frame, mask = OR of frames 0..tf-1).

    Spatial downsampling uses max-pool (conservative: pixel is masked if ANY
    sub-pixel in the sf×sf block is masked).

    Args:
        mask: [B, 1, T_video, H_video, W_video]  float in {0, 1}
        vae_spatial_factor: spatial compression (default 8)
        vae_temporal_factor: temporal compression (default 4)

    Returns:
        packed: [B, vae_temporal_factor, T_lat, H_lat, W_lat]
                where T_lat = 1 + (T_video - 1) // vae_temporal_factor
                (== ceil(T_video / vae_temporal_factor); identical to the prior
                formula for the canonical Wan T = tf·k + 1 video lengths).
    """
    B, C, T, H, W = mask.shape
    assert C == 1

    sf = vae_spatial_factor
    tf = vae_temporal_factor
    # Mirrors WanVideoVAE.encode's `iter_ = 1 + (T-1)//tf`. Mathematically equal
    # to the old `(T + tf - 1)//tf` (= ceil(T/tf)); kept in the causal form so it
    # reads as "1 single-frame latent + one latent per tf-frame group".
    T_lat = 1 + (T - 1) // tf

    # Spatial max-pool (per-frame, treat T as batch)
    m = mask.reshape(B * T, 1, H, W)
    m = F.max_pool2d(m, kernel_size=sf, stride=sf)  # [B*T, 1, H_lat, W_lat]
    H_lat, W_lat = m.shape[2], m.shape[3]
    m = m.reshape(B, T, 1, H_lat, W_lat)

    # Causal temporal fold. Prepend (tf-1) copies of frame 0 so the first fold
    # group is {f0,f0,f0,f0} (→ latent 0, matching the VAE's single-frame latent
    # 0) and group j≥1 is {f[tf(j-1)+1] … f[tf·j]} (matching VAE latent j).
    m = torch.cat([m[:, :1].expand(B, tf - 1, 1, H_lat, W_lat), m], dim=1)

    # Defensive tail handling for non-canonical T (Wan uses T = tf·k + 1, where
    # this is exactly T_padded with no tail work). Repeat the last frame to fill
    # a short final group, or trim a long one.
    T_padded = T_lat * tf
    if m.shape[1] < T_padded:
        pad = m[:, -1:].expand(B, T_padded - m.shape[1], 1, H_lat, W_lat)
        m = torch.cat([m, pad], dim=1)
    elif m.shape[1] > T_padded:
        m = m[:, :T_padded]

    # [B, T_lat, tf, 1, H_lat, W_lat] → [B, tf, T_lat, H_lat, W_lat]
    m = m.reshape(B, T_lat, tf, 1, H_lat, W_lat)
    m = m.permute(0, 2, 1, 3, 4, 5).reshape(B, tf, T_lat, H_lat, W_lat)
    return m
