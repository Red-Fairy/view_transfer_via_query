"""
View Transfer DiT — Modified Wan2.1-T2V-14B for camera-controlled video reshooting (Stage 2).

Target stream (ch-concat 52ch): noisy(16) + rendered(16) + mask(4) + blob(16) → Conv3d → tokens
Source stream (seq-concat 16ch): source perspective video → Conv3d → tokens
Plucker (KQ-bias):  6D rays → pixel-unshuffle 8x → patchify (1,2,2) → per-block Linear → Q/K add
Text (cross-attn):  pre-computed T5 [B,L,4096]
Joint self-attn over [source, target] tokens; 3D RoPE with temporal offset.
Output: predicted velocity on target tokens only.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint
from dataclasses import dataclass
from typing import Tuple, Optional
from einops import rearrange

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diffsynth.models.wan_video_dit import (
    RMSNorm,
    AttentionModule,
    CrossAttention,
    GateModule,
    Head,
    modulate,
    sinusoidal_embedding_1d,
    precompute_freqs_cis_3d,
    rope_apply,
)


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class ViewTransferConfig:
    dim: int = 5120
    in_dim_target: int = 52  # 16 noisy + 16 rendered + 4 mask + 16 blob
    in_dim_source: int = 16
    out_dim: int = 16
    ffn_dim: int = 13824
    freq_dim: int = 256
    text_dim: int = 4096
    num_heads: int = 40
    num_layers: int = 40
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    plucker_token_dim: int = 1536  # 6 * 8^2 * 2 * 2  (pixel-unshuffle 8x then patchify 2x2)
    eps: float = 1e-6
    vae_spatial_factor: int = 8

    @classmethod
    def test(cls):
        # head_dim must be divisible by 6 for 3D RoPE to split evenly
        return cls(
            dim=192, ffn_dim=512, freq_dim=64, text_dim=192,
            num_heads=4, num_layers=2,
        )

    @classmethod
    def wan_t2v_14B(cls):
        # Matches the default ViewTransferConfig(); explicit for symmetry.
        return cls()

    @classmethod
    def wan_t2v_1B3(cls):
        return cls(
            dim=1536, ffn_dim=8960,
            num_heads=12, num_layers=30,
            plucker_token_dim=1536,
        )


# Registry for CLI / pipeline selection by string name.
MODEL_CONFIGS = {
    "1.3B": ViewTransferConfig.wan_t2v_1B3,
    "14B":  ViewTransferConfig.wan_t2v_14B,
}


# ── Self-Attention with KQ-bias ─────────────────────────────────────────────


class ViewTransferSelfAttention(nn.Module):
    """SelfAttention with optional Plucker KQ-bias (Lyra2-style pre-projection add)."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(num_heads)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        kq_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_qk = x + kq_bias if kq_bias is not None else x
        q = self.norm_q(self.q(x_qk))
        k = self.norm_k(self.k(x_qk))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        return self.o(self.attn(q, k, v))


# ── DiT Block ────────────────────────────────────────────────────────────────


class ViewTransferDiTBlock(nn.Module):
    """DiT block with per-block Plucker encoder → KQ-bias injection.
    No CLIP cross-attn (has_image_input=False)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        plucker_dim: int = 1536,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.self_attn = ViewTransferSelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=False)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.plucker_encoder = nn.Linear(plucker_dim, dim, bias=False)
        nn.init.zeros_(self.plucker_encoder.weight)

    def forward(self, x, context, t_mod, freqs, plucker_tokens=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        kq_bias = None
        if plucker_tokens is not None:
            kq_bias = self.plucker_encoder(plucker_tokens)

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, kq_bias=kq_bias))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


# ── Full Model ───────────────────────────────────────────────────────────────


class ViewTransferDiT(nn.Module):
    """Modified Wan2.1-T2V-14B for view transfer.

    Forward inputs:
        noisy_latent     [B, 16, T, Hl, Wl]    target noisy latent
        rendered_latent  [B, 16, T, Hl, Wl]    warped panorama latent
        mask_packed      [B,  4, T, Hl, Wl]    visibility mask (4-frame packing)
        blob_latent      [B, 16, T, Hl, Wl]    blob-video latent
        source_latent    [B, 16, T, Hl, Wl]    source perspective video latent
        plucker_src      [B,  6, T, Hv, Wv]    source camera plucker at latent timestamps
        plucker_tgt      [B,  6, T, Hv, Wv]    target camera plucker
        timestep         [B]                    diffusion timestep
        text_emb         [B, L, text_dim]       pre-computed T5 text embedding

    Returns:
        velocity_pred    [B, 16, T, Hl, Wl]    predicted velocity (target)
    """

    def __init__(self, config: ViewTransferConfig = ViewTransferConfig()):
        super().__init__()
        c = self.config = config

        self.patch_embed_target = nn.Conv3d(
            c.in_dim_target, c.dim, kernel_size=c.patch_size, stride=c.patch_size
        )
        self.patch_embed_source = nn.Conv3d(
            c.in_dim_source, c.dim, kernel_size=c.patch_size, stride=c.patch_size
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(c.text_dim, c.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(c.dim, c.dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(c.freq_dim, c.dim), nn.SiLU(), nn.Linear(c.dim, c.dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(c.dim, c.dim * 6))

        self.blocks = nn.ModuleList(
            [
                ViewTransferDiTBlock(c.dim, c.num_heads, c.ffn_dim, c.plucker_token_dim, c.eps)
                for _ in range(c.num_layers)
            ]
        )

        self.head = Head(c.dim, c.out_dim, c.patch_size, c.eps)

        head_dim = c.dim // c.num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

    # ─ Plucker helpers ─

    def prepare_plucker(self, plucker: torch.Tensor) -> torch.Tensor:
        """Raw plucker → patchified tokens.

        [B, 6, T_lat, H_vid, W_vid] → [B, T_tok, 1536]

        1. Pixel-unshuffle 8× spatially: 6 → 6·64 = 384 ch, H/8, W/8
        2. Patchify (1,2,2): 384 → 384·4 = 1536 per token
        """
        sf = self.config.vae_spatial_factor
        ph, pw = self.config.patch_size[1], self.config.patch_size[2]
        x = rearrange(plucker, "b c t (h fh) (w fw) -> b (c fh fw) t h w", fh=sf, fw=sf)
        x = rearrange(x, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=ph, pw=pw)
        return x

    # ─ RoPE helpers ─

    def build_freqs(self, f: int, h: int, w: int, t_offset: int, device) -> torch.Tensor:
        """3D RoPE frequencies for one stream.  Returns [f*h*w, 1, d_complex]."""
        return (
            torch.cat(
                [
                    self.freqs[0][t_offset : t_offset + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(device)
        )

    # ─ Forward ─

    def forward(
        self,
        noisy_latent: torch.Tensor,
        rendered_latent: torch.Tensor,
        mask_packed: torch.Tensor,
        blob_latent: torch.Tensor,
        source_latent: torch.Tensor,
        plucker_src: torch.Tensor,
        plucker_tgt: torch.Tensor,
        timestep: torch.Tensor,
        text_emb: torch.Tensor,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        c = self.config

        # 1. Target: channel-concat → patchify
        x_tgt = self.patch_embed_target(
            torch.cat([noisy_latent, rendered_latent, mask_packed, blob_latent], dim=1)
        )
        f, h, w = x_tgt.shape[2], x_tgt.shape[3], x_tgt.shape[4]
        x_tgt = rearrange(x_tgt, "b d t h w -> b (t h w) d")

        # 2. Source: patchify
        x_src = self.patch_embed_source(source_latent)
        x_src = rearrange(x_src, "b d t h w -> b (t h w) d")
        T_src = x_src.shape[1]

        # 3. Plucker tokens  [B, 2*T_tok, 1536]
        plucker_tokens = torch.cat(
            [self.prepare_plucker(plucker_src), self.prepare_plucker(plucker_tgt)], dim=1
        )

        # 4. Sequence concat  [B, 2*T_tok, dim]
        x = torch.cat([x_src, x_tgt], dim=1)

        # 5. 3D RoPE: source t=[0,f), target t=[f, 2f)
        freqs = torch.cat(
            [
                self.build_freqs(f, h, w, t_offset=0, device=x.device),
                self.build_freqs(f, h, w, t_offset=f, device=x.device),
            ],
            dim=0,
        )

        # 6. Time & text
        t_emb = self.time_embedding(
            sinusoidal_embedding_1d(c.freq_dim, timestep).to(x.dtype)
        )
        t_mod = self.time_projection(t_emb).unflatten(1, (6, c.dim))
        context = self.text_embedding(text_emb)

        # 7. Transformer blocks
        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, context, t_mod, freqs, plucker_tokens,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs, plucker_tokens)

        # 8. Target slice → head → unpatchify
        x_out = self.head(x[:, T_src:, :], t_emb)
        return rearrange(
            x_out,
            "b (f h w) (pt ph pw d) -> b d (f pt) (h ph) (w pw)",
            f=f, h=h, w=w,
            pt=c.patch_size[0], ph=c.patch_size[1], pw=c.patch_size[2],
            d=c.out_dim,
        )

    # ─ Weight loading ─

    @classmethod
    def from_pretrained(cls, state_dict: dict, config: ViewTransferConfig = ViewTransferConfig()):
        model = cls(config)
        model.load_pretrained_weights(state_dict)
        return model

    def load_pretrained_weights(self, state_dict: dict):
        """Load Wan2.1-T2V state dict.  Handles patch_embedding → patch_embed_target
        (zero-pad extra channels) and copies patch_embedding → patch_embed_source."""
        own = self.state_dict()
        loaded = set()

        for key, param in state_dict.items():
            mapped = key
            if key.startswith("patch_embedding."):
                suffix = key[len("patch_embedding."):]

                # → patch_embed_target (zero-pad extra input channels)
                tgt_key = f"patch_embed_target.{suffix}"
                if tgt_key in own:
                    if suffix == "weight" and param.shape != own[tgt_key].shape:
                        own[tgt_key].zero_()
                        own[tgt_key][:, : param.shape[1]] = param
                    else:
                        own[tgt_key].copy_(param)
                    loaded.add(tgt_key)

                # → patch_embed_source (same 16-ch shape, direct copy)
                src_key = f"patch_embed_source.{suffix}"
                if src_key in own and param.shape == own[src_key].shape:
                    own[src_key].copy_(param)
                    loaded.add(src_key)
                continue

            # Skip keys that don't exist in our model (img_emb, control_adapter, wantodance, etc.)
            if mapped in own and param.shape == own[mapped].shape:
                own[mapped].copy_(param)
                loaded.add(mapped)

        self.load_state_dict(own)
        new_keys = sorted(set(own.keys()) - loaded)
        print(f"Loaded {len(loaded)}/{len(own)} params.  "
              f"New ({len(new_keys)}): {new_keys[:8]}{'...' if len(new_keys) > 8 else ''}")

    # ─ Freeze / train modes ─

    def freeze_base(self):
        """Freeze pretrained weights.  Keep new modules trainable."""
        for p in self.parameters():
            p.requires_grad = False
        trainable_prefixes = ("patch_embed_target", "patch_embed_source", "plucker_encoder")
        for name, p in self.named_parameters():
            if any(name.startswith(pf) or f".{pf}" in name for pf in trainable_prefixes):
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── LoRA ─────────────────────────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a frozen base + low-rank trainable delta."""

    def __init__(self, base: nn.Linear, rank: int = 64, alpha: float = 64.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def apply_lora(
    model: ViewTransferDiT,
    rank: int = 64,
    alpha: float = 64.0,
    target_modules: Tuple[str, ...] = (
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    ),
) -> ViewTransferDiT:
    """Wrap targeted Linear layers inside DiT blocks with LoRA adapters."""
    for block in model.blocks:
        for target in target_modules:
            parts = target.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            base_linear = getattr(parent, attr)
            if isinstance(base_linear, nn.Linear):
                setattr(parent, attr, LoRALinear(base_linear, rank, alpha))
    return model
