"""Tests for the inference pipeline.

Uses a tiny ViewTransferDiT (CPU-friendly) with a stub VAE that simulates the real
VAE's encode/decode contract (16-ch latent, 8x spatial / 4x temporal compression).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Force CPU-friendly attention (FlashAttention is CUDA-only)
import diffsynth.models.wan_video_dit as _wvd
_wvd.FLASH_ATTN_3_AVAILABLE = False
_wvd.FLASH_ATTN_2_AVAILABLE = False
_wvd.SAGE_ATTN_AVAILABLE = False

import torch
import pytest

from diffsynth.diffusion.flow_match import FlowMatchScheduler
from view_transfer_via_query.model import ViewTransferConfig, ViewTransferDiT
from view_transfer_via_query.pipeline import ViewTransferPipeline


# Tiny shapes
B = 1
T_VIDEO = 17           # → T_LAT=5  (since (17+3)//4 = 5)
T_LAT = 5
He, We = 32, 64        # 360 source resolution
PERS_H, PERS_W = 16, 32   # latent: 2x4
H_LAT, W_LAT = PERS_H // 8, PERS_W // 8


# ── Stub VAE ──

class _StubVAE:
    """Mimics WanVideoVAE.encode / decode contracts with the right shapes."""

    def __init__(self):
        # gpu_preprocess / encode_video_to_latent read `vae.model.parameters()`
        # for the VAE dtype; expose a trivial module so the contract holds.
        self.model = torch.nn.Linear(1, 1)

    def parameters(self):
        # pipeline.generate's decode path reads `self.vae.parameters()` dtype.
        return self.model.parameters()

    def encode(self, videos, device, tiled=False):
        # videos: list of [3, T, H, W] in [-1, 1]
        out = []
        for v in videos:
            T = v.shape[1]
            T_lat_v = (T + 3) // 4
            H_lat_v = v.shape[2] // 8
            W_lat_v = v.shape[3] // 8
            out.append(torch.randn(16, T_lat_v, H_lat_v, W_lat_v, device=device))
        return torch.stack(out, dim=0)

    def decode(self, hidden_states, device, tiled=False):
        # hidden_states: list of [16, T_lat, H_lat, W_lat]
        out = []
        for h in hidden_states:
            T = h.shape[1] * 4
            H = h.shape[2] * 8
            W = h.shape[3] * 8
            out.append(torch.randn(3, T, H, W, device=device).clamp(-1, 1))
        return torch.stack(out, dim=0)


# ── Fixtures ──

@pytest.fixture
def cfg():
    return ViewTransferConfig.test()


@pytest.fixture
def model(cfg):
    return ViewTransferDiT(cfg).eval()


@pytest.fixture
def pipeline(model):
    scheduler = FlowMatchScheduler(template="Wan")
    return ViewTransferPipeline(
        model=model, vae=_StubVAE(), scheduler=scheduler,
        device=torch.device("cpu"),
        pers_h=PERS_H, pers_w=PERS_W,
    )


def _make_cpu_batch(cfg) -> dict:
    """Build one cpu_batch matching the dataset's collated output (no rgb_tgt_360)."""
    return {
        "rgb_src_360": torch.randint(0, 255, (B, 3, T_VIDEO, He, We), dtype=torch.uint8),
        "blob_360": torch.randint(0, 255, (B, 3, T_VIDEO, He, We), dtype=torch.uint8),
        "static_rgb_t0": torch.randint(0, 255, (B, 3, He, We), dtype=torch.uint8),
        "static_depth_t0": torch.full((B, He, We), 5.0, dtype=torch.float32),
        "pano_c2w_src": torch.eye(4).expand(B, T_VIDEO, 4, 4).clone(),
        "pano_c2w_tgt": torch.eye(4).expand(B, T_VIDEO, 4, 4).clone(),
        "src_c2w_at_t0": torch.eye(4).expand(B, 4, 4).clone(),
        "text_emb": torch.randn(B, 4, cfg.text_dim),
        "src_fov_h_deg": torch.tensor([90.0]),
        "src_yaw_deg": torch.zeros(B, T_VIDEO),
        "src_pitch_deg": torch.zeros(B, T_VIDEO),
        "src_roll_deg": torch.zeros(B, T_VIDEO),
        "tgt_fov_h_deg": torch.tensor([90.0]),
        "tgt_yaw_deg": torch.zeros(B, T_VIDEO),
        "tgt_pitch_deg": torch.zeros(B, T_VIDEO),
        "tgt_roll_deg": torch.zeros(B, T_VIDEO),
        "t_offset": torch.tensor([0]),
    }


# ── Tests ──

def test_build_uncond_zeros_all():
    cond = {
        "source_latent": torch.randn(2, 16, 5, 8, 8),
        "text_emb": torch.randn(2, 4, 256),
    }
    uncond = ViewTransferPipeline._build_uncond(cond)
    for k, v in uncond.items():
        assert v.shape == cond[k].shape
        assert v.abs().sum() == 0


def test_generate_returns_latent_shape(pipeline, cfg):
    batch = _make_cpu_batch(cfg)
    z = pipeline.generate(
        batch, num_inference_steps=2, guidance_scale=1.0, return_latent=True,
    )
    assert z.shape == (B, 16, T_LAT, H_LAT, W_LAT)


def test_generate_with_cfg_runs(pipeline, cfg):
    batch = _make_cpu_batch(cfg)
    z = pipeline.generate(
        batch, num_inference_steps=2, guidance_scale=5.0, return_latent=True,
    )
    assert z.shape == (B, 16, T_LAT, H_LAT, W_LAT)


def test_generate_decoded_video_shape(pipeline, cfg):
    batch = _make_cpu_batch(cfg)
    videos = pipeline.generate(
        batch, num_inference_steps=2, guidance_scale=1.0, return_latent=False,
    )
    assert videos.dtype == torch.uint8
    assert videos.shape == (B, 3, T_LAT * 4, H_LAT * 8, W_LAT * 8)


def test_generate_ignores_target_if_provided(pipeline, cfg):
    """If user accidentally passes rgb_tgt_360 it should be silently ignored."""
    batch = _make_cpu_batch(cfg)
    batch["rgb_tgt_360"] = torch.randint(0, 255, (B, 3, T_VIDEO, He, We), dtype=torch.uint8)
    z = pipeline.generate(batch, num_inference_steps=2, return_latent=True)
    assert z.shape == (B, 16, T_LAT, H_LAT, W_LAT)


def test_cfg_scale_changes_output(pipeline, cfg):
    """Different guidance scales should produce different latents (deterministic seed)."""
    batch = _make_cpu_batch(cfg)
    torch.manual_seed(0)
    z1 = pipeline.generate(batch, num_inference_steps=3, guidance_scale=1.0, return_latent=True)
    torch.manual_seed(0)
    z2 = pipeline.generate(batch, num_inference_steps=3, guidance_scale=5.0, return_latent=True)
    assert not torch.allclose(z1, z2, atol=1e-4)


def test_subset_cond_zeros_inactive_only():
    cond = {
        "source_latent": torch.randn(2, 16, 5, 8, 8),
        "plucker_src": torch.randn(2, 6, 5, 8, 8),
        "text_emb": torch.randn(2, 4, 256),
    }
    sub = ViewTransferPipeline._subset_cond(cond, ("source_latent",))
    assert torch.equal(sub["source_latent"], cond["source_latent"])
    assert sub["plucker_src"].abs().sum() == 0
    assert sub["text_emb"].abs().sum() == 0
    for k in cond:  # shapes/dtypes preserved
        assert sub[k].shape == cond[k].shape


def test_grouped_guidance_requires_all_three(pipeline, cfg):
    batch = _make_cpu_batch(cfg)
    with pytest.raises(ValueError, match="all three"):
        pipeline.generate(
            batch, num_inference_steps=2, return_latent=True,
            guidance_geom=3.0, guidance_src=1.5,  # text missing
        )


def test_grouped_guidance_runs_and_shape(pipeline, cfg):
    batch = _make_cpu_batch(cfg)
    z = pipeline.generate(
        batch, num_inference_steps=2, return_latent=True,
        guidance_geom=3.0, guidance_src=1.5, guidance_text=2.0,
    )
    assert z.shape == (B, 16, T_LAT, H_LAT, W_LAT)


def test_grouped_differs_from_monolithic(pipeline, cfg):
    """Grouped chained guidance should not coincide with a single-scale CFG run."""
    batch = _make_cpu_batch(cfg)
    torch.manual_seed(0)
    z_mono = pipeline.generate(
        batch, num_inference_steps=3, guidance_scale=5.0, return_latent=True,
    )
    torch.manual_seed(0)
    z_grp = pipeline.generate(
        batch, num_inference_steps=3, return_latent=True,
        guidance_geom=4.0, guidance_src=2.0, guidance_text=1.5,
    )
    assert not torch.allclose(z_mono, z_grp, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
