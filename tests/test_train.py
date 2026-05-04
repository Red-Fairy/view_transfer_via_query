"""Unit tests for the training loop components."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import diffsynth.models.wan_video_dit as _wvd
_wvd.FLASH_ATTN_3_AVAILABLE = False
_wvd.FLASH_ATTN_2_AVAILABLE = False
_wvd.SAGE_ATTN_AVAILABLE = False

import torch
import tempfile
import pytest

from diffsynth.diffusion.flow_match import FlowMatchScheduler
from view_transfer_via_query.model import ViewTransferConfig, ViewTransferDiT, apply_lora
from view_transfer_via_query.train import apply_cfg_dropout, training_step


# ── Fixtures ──

B, T_LAT, H_LAT, W_LAT = 1, 5, 8, 8
H_VID, W_VID = 64, 64
L_TEXT = 4


@pytest.fixture
def cfg():
    return ViewTransferConfig.test()


@pytest.fixture
def model(cfg):
    return ViewTransferDiT(cfg)


def _make_batch(cfg, device="cpu"):
    return {
        "target_latent": torch.randn(B, 16, T_LAT, H_LAT, W_LAT, device=device),
        "source_latent": torch.randn(B, 16, T_LAT, H_LAT, W_LAT, device=device),
        "rendered_latent": torch.randn(B, 16, T_LAT, H_LAT, W_LAT, device=device),
        "blob_latent": torch.randn(B, 16, T_LAT, H_LAT, W_LAT, device=device),
        "mask_packed": torch.randn(B, 4, T_LAT, H_LAT, W_LAT, device=device),
        "plucker_src": torch.randn(B, 6, T_LAT, H_VID, W_VID, device=device),
        "plucker_tgt": torch.randn(B, 6, T_LAT, H_VID, W_VID, device=device),
        "text_emb": torch.randn(B, L_TEXT, cfg.text_dim, device=device),
    }


# ── CFG Dropout ──

def test_cfg_dropout_shapes(cfg):
    batch = _make_batch(cfg)
    batch_out = apply_cfg_dropout(batch, drop_prob=0.5)
    for k, v in batch_out.items():
        assert v.shape == batch[k].shape, f"{k} shape mismatch"


def test_cfg_dropout_zeros_some(cfg):
    torch.manual_seed(0)
    batch = _make_batch(cfg)
    # With p=1.0, everything should be zeroed
    batch_out = apply_cfg_dropout(batch, drop_prob=1.0)
    assert batch_out["source_latent"].abs().sum() == 0
    assert batch_out["rendered_latent"].abs().sum() == 0
    assert batch_out["blob_latent"].abs().sum() == 0
    assert batch_out["plucker_tgt"].abs().sum() == 0
    assert batch_out["text_emb"].abs().sum() == 0


# ── Training Step ──

def test_training_step_returns_scalar(model, cfg):
    scheduler = FlowMatchScheduler(template="Wan")
    scheduler.set_timesteps(num_inference_steps=1000, training=True)
    batch = _make_batch(cfg)
    loss = training_step(model, batch, scheduler)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_training_step_backward(model, cfg):
    model.freeze_base()
    apply_lora(model, rank=4, alpha=4.0)
    scheduler = FlowMatchScheduler(template="Wan")
    scheduler.set_timesteps(num_inference_steps=1000, training=True)
    batch = _make_batch(cfg)
    loss = training_step(model, batch, scheduler)
    loss.backward()
    # Check LoRA grads exist
    block0_q = model.blocks[0].self_attn.q
    assert block0_q.lora_A.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
