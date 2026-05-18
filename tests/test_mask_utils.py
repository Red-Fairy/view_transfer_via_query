"""Tests for mask_utils.pack_mask — focus on Wan2.1 *causal* VAE temporal alignment.

The Wan2.1 VAE compresses time causally (see WanVideoVAE.encode):
    latent 0      ← raw frame {0}
    latent j (≥1) ← raw frames {tf·(j-1)+1 … tf·j}
pack_mask must place the tf mask channels of latent j over exactly those raw
frames, so the visibility mask is registered with the content latents it is
channel-concatenated with in the model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch

from view_transfer_via_query.mask_utils import pack_mask


TF = 4
SF = 8
H = W = 8                 # → H_lat = W_lat = 1
T = TF * 20 + 1           # 81, the canonical Wan video length
T_LAT = 1 + (T - 1) // TF  # 21


def _latent_channel_for_frame(f: int):
    """(latent_idx, channel_idx) the Wan-causal fold must map raw frame f to."""
    if f == 0:
        return 0, None  # frame 0 fills the whole first latent group
    idx = (TF - 1) + f   # position after prepending (tf-1) copies of frame 0
    return idx // TF, idx % TF


def test_shape_and_range():
    mask = torch.ones(2, 1, T, H, W)
    packed = pack_mask(mask, vae_spatial_factor=SF, vae_temporal_factor=TF)
    assert packed.shape == (2, TF, T_LAT, H // SF, W // SF)
    assert packed.min() >= 0 and packed.max() <= 1


def test_first_frame_only_fills_latent0_only():
    """Canonical Wan-I2V case: only frame 0 known.

    Correct causal alignment ⇒ ALL tf channels of latent 0 are 1 and every
    other latent is 0. (The pre-fix uniform-from-0 fold gave latent0 =
    [1,0,0,0] and would fail this.)
    """
    mask = torch.zeros(1, 1, T, H, W)
    mask[:, :, 0] = 1.0
    packed = pack_mask(mask, vae_spatial_factor=SF, vae_temporal_factor=TF)

    assert packed[:, :, 0].min() == 1.0, "latent 0 must be fully set by frame 0"
    assert packed[:, :, 1:].max() == 0.0, "no frame leaks past latent 0"


@pytest.mark.parametrize("f", [0, 1, 2, 3, 4, 5, 8, 9, 40, 77, 78, 79, 80])
def test_single_frame_lands_in_expected_latent(f):
    """A mask set at exactly raw frame f must activate only the latent/channel
    the Wan causal VAE assigns frame f to."""
    mask = torch.zeros(1, 1, T, H, W)
    mask[:, :, f] = 1.0
    packed = pack_mask(mask, vae_spatial_factor=SF, vae_temporal_factor=TF)

    lat, ch = _latent_channel_for_frame(f)
    if ch is None:  # f == 0 fills the whole first latent group
        assert packed[0, :, 0].min() == 1.0
        assert packed[0, :, 1:].max() == 0.0
        return

    assert packed[0, ch, lat].item() == 1.0, (
        f"frame {f} should set latent {lat} channel {ch}"
    )
    # Everything else is zero (mask out the one cell we expect, then check).
    p = packed.clone()
    p[0, ch, lat] = 0.0
    assert p.max() == 0.0, f"frame {f} leaked outside latent {lat} channel {ch}"


def test_frame_tf_is_in_latent1_not_latent0():
    """Regression guard for the original bug: frame == tf (=4) belongs to the
    VAE's latent 1 (frames {1..4}), never latent 0."""
    mask = torch.zeros(1, 1, T, H, W)
    mask[:, :, TF] = 1.0
    packed = pack_mask(mask, vae_spatial_factor=SF, vae_temporal_factor=TF)
    assert packed[0, :, 0].max() == 0.0, "frame tf must NOT touch latent 0"
    assert packed[0, TF - 1, 1].item() == 1.0, "frame tf is the last of latent 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
