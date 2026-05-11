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
from view_transfer_via_query.train import apply_cfg_dropout, training_step, save_trainable


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


# ── CFG Dropout (v2: per-stream + joint) ──

def test_cfg_dropout_shapes(cfg):
    batch = _make_batch(cfg)
    batch_out = apply_cfg_dropout(batch, per_stream_prob=0.25, joint_prob=0.25)
    for k, v in batch_out.items():
        assert v.shape == batch[k].shape, f"{k} shape mismatch"


def test_cfg_dropout_zeros_when_joint_prob_one(cfg):
    """joint_prob=1.0 forces every stream to be zeroed in every batch element."""
    torch.manual_seed(0)
    batch = _make_batch(cfg)
    batch_out = apply_cfg_dropout(batch, per_stream_prob=0.0, joint_prob=1.0)
    assert batch_out["source_latent"].abs().sum() == 0
    assert batch_out["rendered_latent"].abs().sum() == 0
    assert batch_out["blob_latent"].abs().sum() == 0
    assert batch_out["plucker_src"].abs().sum() == 0
    assert batch_out["plucker_tgt"].abs().sum() == 0
    assert batch_out["mask_packed"].abs().sum() == 0
    assert batch_out["text_emb"].abs().sum() == 0


def test_cfg_dropout_per_stream_only(cfg):
    """per_stream_prob=1.0 + joint_prob=0.0 still zeros every stream (every stream
    independently rolls True with p=1.0)."""
    torch.manual_seed(0)
    batch = _make_batch(cfg)
    batch_out = apply_cfg_dropout(batch, per_stream_prob=1.0, joint_prob=0.0)
    assert batch_out["source_latent"].abs().sum() == 0
    assert batch_out["text_emb"].abs().sum() == 0


def test_cfg_dropout_no_drop_preserves_input(cfg):
    """All probs zero → batch unchanged."""
    torch.manual_seed(0)
    batch = _make_batch(cfg)
    orig = {k: v.clone() for k, v in batch.items()}
    batch_out = apply_cfg_dropout(batch, per_stream_prob=0.0, joint_prob=0.0)
    for k in orig:
        assert torch.equal(batch_out[k], orig[k]), f"{k} changed despite zero drop probs"


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


# ── save_trainable ──

def test_save_trainable_lora_filters_to_prefixes(model, tmp_path):
    """LoRA mode (default) keeps only LoRA + new-module keys."""
    model.freeze_base()
    apply_lora(model, rank=4, alpha=4.0)
    out = tmp_path / "trainable.pt"
    save_trainable(model, str(out), full=False)
    sd = torch.load(str(out), map_location="cpu", weights_only=True)
    assert len(sd) > 0
    # No base/frozen Wan keys leak through.
    for k in sd:
        assert (
            "lora_" in k
            or "plucker_encoder" in k
            or "patch_embed_source" in k
            or "geoada_" in k
            or "cross_attn_src" in k
        ), f"unexpected key in LoRA save: {k}"


def test_save_trainable_full_dumps_entire_state_dict(model, tmp_path):
    """full=True must persist every parameter so a full-finetune run can be reloaded."""
    out = tmp_path / "trainable_full.pt"
    save_trainable(model, str(out), full=True)
    sd = torch.load(str(out), map_location="cpu", weights_only=True)
    full_keys = set(model.state_dict().keys())
    assert set(sd.keys()) == full_keys, (
        f"full save dropped keys: missing={full_keys - set(sd.keys())}  "
        f"extra={set(sd.keys()) - full_keys}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
