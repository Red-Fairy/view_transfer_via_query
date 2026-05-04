"""Unit tests for ViewTransferDiT (tiny config, CPU)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Force CPU-compatible attention fallback (FlashAttention is CUDA-only)
import diffsynth.models.wan_video_dit as _wvd
_wvd.FLASH_ATTN_3_AVAILABLE = False
_wvd.FLASH_ATTN_2_AVAILABLE = False
_wvd.SAGE_ATTN_AVAILABLE = False

import torch
import pytest
from view_transfer_via_query.model import (
    ViewTransferConfig,
    ViewTransferDiT,
    ViewTransferSelfAttention,
    ViewTransferDiTBlock,
    LoRALinear,
    apply_lora,
)
from view_transfer_via_query.mask_utils import pack_mask


# ── Fixtures ──

B, T_LAT, H_LAT, W_LAT = 1, 5, 8, 8   # tiny latent grid
H_VID, W_VID = H_LAT * 8, W_LAT * 8     # 64 x 64 video resolution
T_VID = 17                                # → T_LAT = ceil(17/4) = 5
L_TEXT = 4


@pytest.fixture
def cfg():
    return ViewTransferConfig.test()


@pytest.fixture
def model(cfg):
    return ViewTransferDiT(cfg)


def _make_inputs(cfg, device="cpu"):
    """Build random inputs with correct shapes for the tiny config."""
    return dict(
        noisy_latent=torch.randn(B, 16, T_LAT, H_LAT, W_LAT, device=device),
        rendered_latent=torch.randn(B, 16, T_LAT, H_LAT, W_LAT, device=device),
        mask_packed=torch.randn(B, 4, T_LAT, H_LAT, W_LAT, device=device),
        blob_latent=torch.randn(B, 16, T_LAT, H_LAT, W_LAT, device=device),
        source_latent=torch.randn(B, 16, T_LAT, H_LAT, W_LAT, device=device),
        plucker_src=torch.randn(B, 6, T_LAT, H_VID, W_VID, device=device),
        plucker_tgt=torch.randn(B, 6, T_LAT, H_VID, W_VID, device=device),
        timestep=torch.tensor([500.0], device=device),
        text_emb=torch.randn(B, L_TEXT, cfg.text_dim, device=device),
    )


# ── Shape tests ──

def test_forward_shape(model, cfg):
    inputs = _make_inputs(cfg)
    out = model(**inputs)
    assert out.shape == (B, cfg.out_dim, T_LAT, H_LAT, W_LAT), f"Got {out.shape}"


def test_prepare_plucker_shape(model):
    plucker = torch.randn(B, 6, T_LAT, H_VID, W_VID)
    tokens = model.prepare_plucker(plucker)
    ph, pw = model.config.patch_size[1], model.config.patch_size[2]
    expected_T_tok = T_LAT * (H_LAT // ph) * (W_LAT // pw)
    assert tokens.shape == (B, expected_T_tok, model.config.plucker_token_dim)


# ── Gradient tests ──

def test_gradient_flows_through_new_modules(model, cfg):
    """Plucker encoder and both patch embeds should receive gradients."""
    inputs = _make_inputs(cfg)
    out = model(**inputs)
    loss = out.sum()
    loss.backward()

    assert model.patch_embed_target.weight.grad is not None
    assert model.patch_embed_source.weight.grad is not None
    for i, block in enumerate(model.blocks):
        assert block.plucker_encoder.weight.grad is not None, f"Block {i} plucker grad missing"


def test_freeze_base(model, cfg):
    model.freeze_base()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("patch_embed_target" in n for n in trainable)
    assert any("patch_embed_source" in n for n in trainable)
    assert any("plucker_encoder" in n for n in trainable)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert any("text_embedding" in n for n in frozen)
    assert any("time_embedding" in n for n in frozen)


# ── LoRA tests ──

def test_lora_application(model, cfg):
    model.freeze_base()
    apply_lora(model, rank=8, alpha=8.0)
    # LoRA adapters should be trainable
    lora_params = [n for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    assert len(lora_params) > 0, "No LoRA params found"
    # Base attn linears should be frozen
    for block in model.blocks:
        assert isinstance(block.self_attn.q, LoRALinear)
        assert not block.self_attn.q.base.weight.requires_grad


def test_lora_forward_shape(model, cfg):
    apply_lora(model, rank=8, alpha=8.0)
    inputs = _make_inputs(cfg)
    out = model(**inputs)
    assert out.shape == (B, cfg.out_dim, T_LAT, H_LAT, W_LAT)


def test_lora_gradient(model, cfg):
    model.freeze_base()
    apply_lora(model, rank=8, alpha=8.0)
    inputs = _make_inputs(cfg)
    out = model(**inputs)
    out.sum().backward()
    for block in model.blocks:
        q_lora = block.self_attn.q
        assert q_lora.lora_A.weight.grad is not None
        assert q_lora.lora_B.weight.grad is not None


# ── Mask packing test ──

def test_pack_mask():
    mask = torch.ones(2, 1, T_VID, H_VID, W_VID)
    packed = pack_mask(mask, vae_spatial_factor=8, vae_temporal_factor=4)
    assert packed.shape == (2, 4, T_LAT, H_VID // 8, W_VID // 8)
    assert packed.min() >= 0 and packed.max() <= 1


# ── Self-attention KQ-bias test ──

def test_self_attn_kq_bias():
    dim, heads = 96, 4  # head_dim=24, divisible by 6 for 3D RoPE
    sa = ViewTransferSelfAttention(dim, heads)
    from diffsynth.models.wan_video_dit import precompute_freqs_cis_3d
    freqs_tuple = precompute_freqs_cis_3d(dim // heads)
    f, h, w = 2, 2, 2
    freqs = torch.cat([
        freqs_tuple[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_tuple[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_tuple[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1)
    L = f * h * w
    x = torch.randn(1, L, dim)

    # Without bias
    out_no_bias = sa(x, freqs)
    # With zero bias (should be same)
    out_zero = sa(x, freqs, kq_bias=torch.zeros_like(x))
    assert torch.allclose(out_no_bias, out_zero, atol=1e-5)

    # With nonzero bias (should differ)
    out_biased = sa(x, freqs, kq_bias=torch.randn_like(x))
    assert not torch.allclose(out_no_bias, out_biased, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
