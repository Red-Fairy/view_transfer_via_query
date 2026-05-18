"""Wan2.1 VAE encoding for the prep pipeline.

Loads the Wan2.1 VAE once, encodes perspective videos [T, 3, H, W] in [0, 1]
to latent tensors [16, T_lat, H_lat, W_lat]. Output is in the VAE's normalized
latent space (mean-shifted and std-divided per-channel) — directly usable by the DiT.
"""

import torch
import torch.nn.functional as F
from typing import Iterable, List, Optional
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from diffsynth.models.wan_video_vae import WanVideoVAE, WanVideoVAEStateDictConverter


def load_wan_vae(
    checkpoint_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> WanVideoVAE:
    """Load a WanVideoVAE from a checkpoint.

    Args:
        checkpoint_path: path to Wan2.1_VAE.pth (raw upstream format) or a converted .pt
        device: target device for VAE weights
        dtype: dtype to cast weights to (float32 default for stability)
    """
    vae = WanVideoVAE(z_dim=16)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Auto-detect raw upstream checkpoint and convert if needed
    if any(k.startswith("encoder.") or k.startswith("decoder.") for k in state_dict):
        # Already in DiffSynth-Studio format
        vae.model.load_state_dict(state_dict, strict=False)
    else:
        try:
            converter = WanVideoVAEStateDictConverter()
            converted = converter.from_civitai(state_dict)
            vae.model.load_state_dict(converted, strict=False)
        except Exception:
            # Fall back to direct load if converter signature differs
            vae.model.load_state_dict(state_dict, strict=False)

    vae.eval().requires_grad_(False)
    vae.to(device=device, dtype=dtype)
    return vae


@torch.no_grad()
def encode_video_to_latent(
    vae: WanVideoVAE,
    video: torch.Tensor,
    device: str = "cuda",
    tiled: bool = False,
    keep_on_device: bool = False,
) -> torch.Tensor:
    """Encode one perspective video → latent.

    Args:
        video: [T, 3, H, W] in [0, 1] (float / bf16 / fp16). Cast to the VAE's own dtype
               so callers can run a bf16 VAE without manually matching dtypes.
        keep_on_device: when True, return the latent on `device` instead of CPU. The
            online training/inference path sets this to avoid a D2H+H2D round trip
            per stream. Offline encoders leave the default (False) so on-disk
            shards are written from CPU as before.
    Returns:
        latent: [16, T_lat, H_lat, W_lat] tensor (VAE-normalized space). On CPU when
                keep_on_device is False (default), on `device` otherwise.
    """
    assert video.dim() == 4 and video.shape[1] == 3, f"Expected [T,3,H,W], got {video.shape}"
    vae_dtype = next(vae.model.parameters()).dtype
    # [T, 3, H, W] → [3, T, H, W]; normalize to [-1, 1]; match VAE dtype
    x = video.permute(1, 0, 2, 3).contiguous().to(dtype=vae_dtype)
    x = x * 2.0 - 1.0
    latents = vae.encode([x], device=device, tiled=tiled)  # [1, 16, T_lat, H_lat, W_lat]
    out = latents[0]
    # WanVideoVAE.tiled_encode hardcodes data_device="cpu" — it returns CPU
    # tensors even when the VAE itself lives on `device`. Force the move on
    # the keep_on_device branch so the online path (training + low-VRAM
    # inference) sees a GPU latent regardless of `tiled`.
    return out.to(device) if keep_on_device else out.cpu()


@torch.no_grad()
def encode_videos_to_latents(
    vae: WanVideoVAE,
    videos: Iterable[torch.Tensor],
    device: str = "cuda",
    tiled: bool = False,
) -> List[torch.Tensor]:
    """Encode a list/iterable of [T, 3, H, W] videos. Memory-conscious: one at a time."""
    return [encode_video_to_latent(vae, v, device=device, tiled=tiled) for v in videos]
