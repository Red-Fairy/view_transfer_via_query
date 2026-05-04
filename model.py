"""
View Transfer DiT — Modified Wan2.1-T2V-14B for camera-controlled video reshooting (Stage 2).

Architecture v2 (2026-05-04). See project_files/PLAN.md §10 for full spec.

  - Noisy target latent → patch_embedding (frozen Wan, 16ch)
  - Rendered + blob + mask → channel-concat (36ch) → geoada_patch_embedding → adapter branch
  - Source perspective latent → patch_embed_source → cross_attn_src (M=10 sites in main DiT)
  - Plücker_tgt → KQ-bias on main self-attention
  - Plücker_src → K-side bias on cross_attn_src
  - Text → standard text cross-attn (frozen Wan)

Adapter branch (VerseCrafter / VACE style): N adapter DiT blocks process the conditioning
sequence; each emits a hint via zero-init `after_proj`; first adapter block's `before_proj`
(also zero-init) starts the conditioning stream from the main DiT's first hidden state. Hints
are added to main DiT blocks at `geoada_layers`.

Source cross-attention (IP-Adapter style): zero-init output projection means each
`cross_attn_src` contributes exactly 0 at step 0, regardless of inputs.

Step-0 invariant: with all the zero-inits in place, this model's forward equals
pretrained Wan applied to (noisy_latent, timestep, text_emb). See `assert_step0_invariant`.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint
from dataclasses import dataclass
from typing import Tuple, List
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
    in_dim: int = 16              # noisy target latent (Wan2.1 VAE 16-ch)
    in_dim_source: int = 16       # source perspective latent
    in_dim_geoada: int = 36       # 16 rendered + 16 blob + 4 mask
    out_dim: int = 16
    ffn_dim: int = 13824
    freq_dim: int = 256
    text_dim: int = 4096
    num_heads: int = 40
    num_layers: int = 40
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    plucker_token_dim: int = 1536  # 6 * 8^2 * 2 * 2
    eps: float = 1e-6
    vae_spatial_factor: int = 8

    # v2: adapter branch — N adapter blocks injected at these main-block indices
    geoada_layers: Tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36)
    # v2: cross-attention sites — M main blocks get a cross_attn_src module (interleaved)
    cross_attn_src_layers: Tuple[int, ...] = (2, 6, 10, 14, 18, 22, 26, 30, 34, 38)

    @classmethod
    def test(cls):
        # head_dim must be divisible by 6 for 3D RoPE to split evenly
        return cls(
            dim=192, ffn_dim=512, freq_dim=64, text_dim=192,
            num_heads=4, num_layers=2,
            geoada_layers=(0,),
            cross_attn_src_layers=(1,),
        )

    @classmethod
    def wan_t2v_14B(cls):
        return cls()

    @classmethod
    def wan_t2v_1B3(cls):
        return cls(
            dim=1536, ffn_dim=8960,
            num_heads=12, num_layers=30,
            plucker_token_dim=1536,
            # k=4 → adapter at every 4th of 30 layers
            geoada_layers=(0, 4, 8, 12, 16, 20, 24, 28),
            # interleaved between adapter sites
            cross_attn_src_layers=(2, 6, 10, 14, 18, 22, 26),
        )


# Registry for CLI / pipeline selection by string name.
MODEL_CONFIGS = {
    "1.3B": ViewTransferConfig.wan_t2v_1B3,
    "14B":  ViewTransferConfig.wan_t2v_14B,
}


# ── Self-Attention with KQ-bias (plücker_tgt on Q/K, V untouched) ───────────


class ViewTransferSelfAttention(nn.Module):
    """Wan-style self-attention with optional Q/K additive bias from plücker tokens."""

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

    def forward(self, x, freqs, kq_bias=None):
        x_qk = x + kq_bias if kq_bias is not None else x
        q = self.norm_q(self.q(x_qk))
        k = self.norm_k(self.k(x_qk))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        return self.o(self.attn(q, k, v))


# ── Cross-Attention from target queries to source K/V (zero-init output) ────


class ViewTransferCrossAttention(nn.Module):
    """Target-query → source-K/V cross-attention with zero-init output projection.

    Step-0 invariant: o.weight=0, o.bias=0 ⇒ output is exactly 0 regardless of inputs.
    Optional plücker_src additive bias on K-side only (V is left untouched, preserving
    pretrained content statistics — same idiom as ViewTransferSelfAttention's KQ-bias).
    """

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
        # Zero-init output projection — keeps step-0 contribution exactly 0
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x_tgt, x_src, freqs_tgt, freqs_src, k_kq_bias=None):
        x_src_qk = x_src + k_kq_bias if k_kq_bias is not None else x_src
        q = rope_apply(self.norm_q(self.q(x_tgt)),     freqs_tgt, self.num_heads)
        k = rope_apply(self.norm_k(self.k(x_src_qk)),  freqs_src, self.num_heads)
        v = self.v(x_src)
        return self.o(self.attn(q, k, v))


# ── Main DiT Block ──────────────────────────────────────────────────────────


class ViewTransferDiTBlock(nn.Module):
    """Main DiT block. Has plücker_tgt KQ-bias on self-attn always.
    Optionally has cross_attn_src + plucker_encoder_src when has_cross_src=True.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        plucker_dim: int = 1536,
        eps: float = 1e-6,
        has_cross_src: bool = False,
    ):
        super().__init__()
        self.has_cross_src = has_cross_src
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
        # Plücker_tgt KQ-bias for the self-attn path (zero-init: identity at step 0)
        self.plucker_encoder = nn.Linear(plucker_dim, dim, bias=False)
        nn.init.zeros_(self.plucker_encoder.weight)
        # Optional source cross-attention path (zero-init: identity at step 0)
        if has_cross_src:
            self.norm_src_q = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
            self.norm_src_k = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
            self.cross_attn_src = ViewTransferCrossAttention(dim, num_heads, eps)
            self.plucker_encoder_src = nn.Linear(plucker_dim, dim, bias=False)
            nn.init.zeros_(self.plucker_encoder_src.weight)

    def forward(
        self,
        x,
        x_src,
        context,
        t_mod,
        freqs_tgt,
        freqs_src,
        plucker_tgt_tokens=None,
        plucker_src_tokens=None,
        hint=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        # 1. Self-attn (target only) with plücker_tgt KQ-bias
        kq_bias_tgt = (
            self.plucker_encoder(plucker_tgt_tokens)
            if plucker_tgt_tokens is not None else None
        )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs_tgt, kq_bias=kq_bias_tgt))

        # 2. Optional source cross-attention (zero-init output ⇒ no step-0 contribution)
        if self.has_cross_src and x_src is not None:
            k_bias_src = (
                self.plucker_encoder_src(plucker_src_tokens)
                if plucker_src_tokens is not None else None
            )
            x = x + self.cross_attn_src(
                self.norm_src_q(x), self.norm_src_k(x_src),
                freqs_tgt, freqs_src, k_kq_bias=k_bias_src,
            )

        # 3. Pretrained text cross-attention
        x = x + self.cross_attn(self.norm3(x), context)

        # 4. FFN
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))

        # 5. Optional adapter hint injection (zero-init after_proj ⇒ 0 at step 0)
        if hint is not None:
            x = x + hint

        return x


# ── Adapter DiT Block (VerseCrafter / VACE style) ───────────────────────────


class AdapterDiTBlock(nn.Module):
    """Adapter block: runs in parallel to the main DiT on the rendered+blob+mask conditioning.

    Shares the structural recipe with `ViewTransferDiTBlock` (self-attn / text-cross-attn /
    FFN / modulation / gate) so that warm-start by copying weights from a main DiT block
    is a straight tensor copy on the matching keys.

    No plücker (the conditioning streams already encode target-camera geometry implicitly
    via the rendering/projection pipeline that produced them).
    No source cross-attn (only main DiT blocks attend to source).

    First adapter block: `c = before_proj(c) + x_main` with `before_proj` zero-init, so
    step 0 has c = x_main. Every block also produces `hint = after_proj(c)` with
    `after_proj` zero-init, so step-0 hint = 0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        is_first: bool = False,
    ):
        super().__init__()
        self.is_first = is_first
        self.self_attn = ViewTransferSelfAttention(dim, num_heads, eps)  # KQ-bias unused
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
        # Zero-init injection projections
        if is_first:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        else:
            self.before_proj = None
        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x_main, context, t_mod, freqs):
        """Returns (c_next, hint).
        c_next: adapter hidden state passed to the next adapter block (or discarded after the last).
        hint:   after_proj(c_next), added to the corresponding main DiT block.
        """
        if self.before_proj is not None:
            c = self.before_proj(c) + x_main

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        input_c = modulate(self.norm1(c), shift_msa, scale_msa)
        c = self.gate(c, gate_msa, self.self_attn(input_c, freqs, kq_bias=None))
        c = c + self.cross_attn(self.norm3(c), context)
        input_c = modulate(self.norm2(c), shift_mlp, scale_mlp)
        c = self.gate(c, gate_mlp, self.ffn(input_c))

        hint = self.after_proj(c)
        return c, hint


# ── Full Model ──────────────────────────────────────────────────────────────


class ViewTransferDiT(nn.Module):
    """ViewTransferDiT v2.

    Forward inputs:
        noisy_latent     [B, 16, T, Hl, Wl]    target noisy latent
        rendered_latent  [B, 16, T, Hl, Wl]    static-warped video latent
        mask_packed      [B,  4, T, Hl, Wl]    visibility mask (4-frame packing)
        blob_latent      [B, 16, T, Hl, Wl]    dynamic agents
        source_latent    [B, 16, T, Hl, Wl]    source perspective video latent
        plucker_src      [B,  6, T, Hv, Wv]    source camera plücker
        plucker_tgt      [B,  6, T, Hv, Wv]    target camera plücker
        timestep         [B]                    diffusion timestep
        text_emb         [B, L, text_dim]       T5 text embedding

    Returns:
        velocity_pred    [B, 16, T, Hl, Wl]    predicted velocity (target)
    """

    def __init__(self, config: ViewTransferConfig = ViewTransferConfig()):
        super().__init__()
        self.config = c = config
        self.geoada_layers = tuple(c.geoada_layers)
        self.cross_attn_src_layers = tuple(c.cross_attn_src_layers)
        assert max(self.geoada_layers) < c.num_layers, (
            f"geoada_layers {self.geoada_layers} contains an index >= num_layers={c.num_layers}"
        )
        assert max(self.cross_attn_src_layers) < c.num_layers, (
            f"cross_attn_src_layers {self.cross_attn_src_layers} contains an index >= num_layers={c.num_layers}"
        )
        # Map main-block index → adapter-block index
        self.geoada_layers_mapping = {idx: i for i, idx in enumerate(self.geoada_layers)}
        cross_set = set(self.cross_attn_src_layers)

        # Patch embedders
        self.patch_embedding = nn.Conv3d(
            c.in_dim, c.dim, kernel_size=c.patch_size, stride=c.patch_size,
        )
        self.patch_embed_source = nn.Conv3d(
            c.in_dim_source, c.dim, kernel_size=c.patch_size, stride=c.patch_size,
        )
        self.geoada_patch_embedding = nn.Conv3d(
            c.in_dim_geoada, c.dim, kernel_size=c.patch_size, stride=c.patch_size,
        )

        # Time / text
        self.text_embedding = nn.Sequential(
            nn.Linear(c.text_dim, c.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(c.dim, c.dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(c.freq_dim, c.dim), nn.SiLU(), nn.Linear(c.dim, c.dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(c.dim, c.dim * 6),
        )

        # Main DiT blocks
        self.blocks = nn.ModuleList([
            ViewTransferDiTBlock(
                c.dim, c.num_heads, c.ffn_dim, c.plucker_token_dim, c.eps,
                has_cross_src=(i in cross_set),
            )
            for i in range(c.num_layers)
        ])

        # Adapter blocks (N = len(geoada_layers))
        self.geoada_blocks = nn.ModuleList([
            AdapterDiTBlock(
                c.dim, c.num_heads, c.ffn_dim, c.eps,
                is_first=(i == 0),
            )
            for i in range(len(self.geoada_layers))
        ])

        self.head = Head(c.dim, c.out_dim, c.patch_size, c.eps)
        head_dim = c.dim // c.num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

    # ─ Plücker helpers ─

    def prepare_plucker(self, plucker: torch.Tensor) -> torch.Tensor:
        """Raw plücker → patchified tokens.
        [B, 6, T_lat, H_vid, W_vid] → [B, T_tok, plucker_token_dim]
        """
        sf = self.config.vae_spatial_factor
        ph, pw = self.config.patch_size[1], self.config.patch_size[2]
        x = rearrange(plucker, "b c t (h fh) (w fw) -> b (c fh fw) t h w", fh=sf, fw=sf)
        x = rearrange(x, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=ph, pw=pw)
        return x

    # ─ RoPE helpers ─

    def build_freqs(self, f, h, w, t_offset, device):
        """3D RoPE freqs at temporal positions [t_offset, t_offset+f). [f*h*w, 1, d_complex]."""
        return (
            torch.cat([
                self.freqs[0][t_offset : t_offset + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1)
            .reshape(f * h * w, 1, -1)
            .to(device)
        )

    # ─ Adapter sequential forward ─

    def forward_geoada(
        self, x_main, cond, context, t_mod, freqs, use_gradient_checkpointing,
    ) -> List[torch.Tensor]:
        """Run all N adapter blocks sequentially, returning list of N hint tensors."""
        hints: List[torch.Tensor] = []
        c = cond
        for block in self.geoada_blocks:
            if self.training and use_gradient_checkpointing:
                c, hint = torch.utils.checkpoint.checkpoint(
                    block, c, x_main, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                c, hint = block(c, x_main, context, t_mod, freqs)
            hints.append(hint)
        return hints

    # ─ Main forward ─

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

        # 1. Target tokens (noisy latent only, plain Wan patch_embedding)
        x = self.patch_embedding(noisy_latent)
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = rearrange(x, "b d t h w -> b (t h w) d")

        # 2. Source tokens
        x_src = self.patch_embed_source(source_latent)
        x_src = rearrange(x_src, "b d t h w -> b (t h w) d")

        # 3. Adapter conditioning input (channel-concat 36ch: rendered + blob + mask)
        cond_input = torch.cat([rendered_latent, blob_latent, mask_packed], dim=1)
        cond = self.geoada_patch_embedding(cond_input)
        cond = rearrange(cond, "b d t h w -> b (t h w) d")

        # 4. Plücker tokens
        plucker_tgt_tokens = self.prepare_plucker(plucker_tgt)
        plucker_src_tokens = self.prepare_plucker(plucker_src)

        # 5. RoPE (target and source both at temporal positions [0, f))
        freqs = self.build_freqs(f, h, w, t_offset=0, device=x.device)

        # 6. Time + text
        t_emb = self.time_embedding(
            sinusoidal_embedding_1d(c.freq_dim, timestep).to(x.dtype)
        )
        t_mod = self.time_projection(t_emb).unflatten(1, (6, c.dim))
        context = self.text_embedding(text_emb)

        # 7. Adapter forward (sequential, accumulates hints)
        hints = self.forward_geoada(x, cond, context, t_mod, freqs, use_gradient_checkpointing)

        # 8. Main DiT forward (interleave hint injection at geoada_layers)
        for i, block in enumerate(self.blocks):
            hint = hints[self.geoada_layers_mapping[i]] if i in self.geoada_layers_mapping else None
            if self.training and use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, x_src, context, t_mod, freqs, freqs,
                    plucker_tgt_tokens, plucker_src_tokens, hint,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x, x_src, context, t_mod, freqs, freqs,
                    plucker_tgt_tokens, plucker_src_tokens, hint,
                )

        # 9. Head + unpatchify
        x_out = self.head(x, t_emb)
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
        """Load Wan2.1-T2V state dict and warm-start the new modules.

        Three passes:
          1. Direct copy of pretrained Wan keys (patch_embedding, blocks.*, head, etc.).
             `patch_embed_source` is also populated from pretrained `patch_embedding`.
          2. Adapter blocks `geoada_blocks.{i}` are warm-started from main `blocks.{geoada_layers[i]}`
             (only matching sub-keys; plücker / cross_attn_src / before/after_proj have no
             counterpart in main blocks and are skipped).
          3. `cross_attn_src.{q,k,v,norm_q,norm_k}` in each cross-attn-equipped main block
             is warm-started from that block's `self_attn.{q,k,v,norm_q,norm_k}`.

        Then `_apply_zero_init` re-zeros all 7 step-0-invariant modules so that any
        accidental non-zero values from above are scrubbed.
        """
        own = self.state_dict()
        loaded = set()

        # 1. Direct copy of pretrained Wan keys
        for key, param in state_dict.items():
            if key.startswith("patch_embedding."):
                suffix = key[len("patch_embedding."):]

                tgt_key = f"patch_embedding.{suffix}"
                if tgt_key in own and param.shape == own[tgt_key].shape:
                    own[tgt_key].copy_(param)
                    loaded.add(tgt_key)

                src_key = f"patch_embed_source.{suffix}"
                if src_key in own and param.shape == own[src_key].shape:
                    own[src_key].copy_(param)
                    loaded.add(src_key)
                continue

            if key in own and param.shape == own[key].shape:
                own[key].copy_(param)
                loaded.add(key)

        # 2. Warm-start adapter blocks from corresponding main blocks
        for adapter_idx, main_idx in enumerate(self.geoada_layers):
            main_prefix = f"blocks.{main_idx}."
            adapter_prefix = f"geoada_blocks.{adapter_idx}."
            for k in list(own.keys()):
                if not k.startswith(main_prefix):
                    continue
                suffix = k[len(main_prefix):]
                adapter_key = adapter_prefix + suffix
                if adapter_key in own and own[k].shape == own[adapter_key].shape:
                    own[adapter_key].copy_(own[k])
                    loaded.add(adapter_key)

        # 3. Warm-start cross_attn_src.{q,k,v,norm_q,norm_k} from main self_attn
        for layer_idx in self.cross_attn_src_layers:
            for sub in ("q", "k", "v"):
                for param_name in ("weight", "bias"):
                    src_key = f"blocks.{layer_idx}.self_attn.{sub}.{param_name}"
                    tgt_key = f"blocks.{layer_idx}.cross_attn_src.{sub}.{param_name}"
                    if src_key in own and tgt_key in own and own[src_key].shape == own[tgt_key].shape:
                        own[tgt_key].copy_(own[src_key])
                        loaded.add(tgt_key)
            for sub_norm in ("norm_q", "norm_k"):
                src_key = f"blocks.{layer_idx}.self_attn.{sub_norm}.weight"
                tgt_key = f"blocks.{layer_idx}.cross_attn_src.{sub_norm}.weight"
                if src_key in own and tgt_key in own and own[src_key].shape == own[tgt_key].shape:
                    own[tgt_key].copy_(own[src_key])
                    loaded.add(tgt_key)

        self.load_state_dict(own)

        # 4. Re-apply zero-init for the 7 step-0-invariant modules
        self._apply_zero_init()

        new_keys = sorted(set(own.keys()) - loaded)
        print(
            f"Loaded {len(loaded)}/{len(own)} params. "
            f"Newly initialised ({len(new_keys)}): {new_keys[:8]}"
            f"{'...' if len(new_keys) > 8 else ''}"
        )

    def _apply_zero_init(self):
        """Re-apply zero-init for every module the step-0 invariant depends on.
        Called from `load_pretrained_weights` after the pretrained-copy passes.
        """
        # Adapter projections
        for block in self.geoada_blocks:
            if block.before_proj is not None:
                nn.init.zeros_(block.before_proj.weight)
                nn.init.zeros_(block.before_proj.bias)
            nn.init.zeros_(block.after_proj.weight)
            nn.init.zeros_(block.after_proj.bias)
        # geoada_patch_embedding bias (Xavier init kept on weight)
        if self.geoada_patch_embedding.bias is not None:
            nn.init.zeros_(self.geoada_patch_embedding.bias)
        # Per-block plücker encoders + cross_attn_src.o
        for block in self.blocks:
            nn.init.zeros_(block.plucker_encoder.weight)
            if block.has_cross_src:
                nn.init.zeros_(block.plucker_encoder_src.weight)
                nn.init.zeros_(block.cross_attn_src.o.weight)
                nn.init.zeros_(block.cross_attn_src.o.bias)

    def assert_step0_invariant(self):
        """Assert that every zero-init condition needed for step 0 ≡ pretrained Wan holds.
        See PLAN.md §10.6 for the full list. Raises AssertionError on first violation.
        """
        # 1. Adapter before_proj / after_proj
        for i, block in enumerate(self.geoada_blocks):
            if block.before_proj is not None:
                assert (block.before_proj.weight == 0).all(), \
                    f"geoada_blocks[{i}].before_proj.weight is not all-zero"
                assert (block.before_proj.bias == 0).all(), \
                    f"geoada_blocks[{i}].before_proj.bias is not all-zero"
            assert (block.after_proj.weight == 0).all(), \
                f"geoada_blocks[{i}].after_proj.weight is not all-zero"
            assert (block.after_proj.bias == 0).all(), \
                f"geoada_blocks[{i}].after_proj.bias is not all-zero"

        # 2. geoada_patch_embedding bias
        if self.geoada_patch_embedding.bias is not None:
            assert (self.geoada_patch_embedding.bias == 0).all(), \
                "geoada_patch_embedding.bias is not all-zero"

        # 3. Per-block plücker encoders + cross_attn_src.o
        for i, block in enumerate(self.blocks):
            assert (block.plucker_encoder.weight == 0).all(), \
                f"blocks[{i}].plucker_encoder.weight is not all-zero"
            if block.has_cross_src:
                assert (block.plucker_encoder_src.weight == 0).all(), \
                    f"blocks[{i}].plucker_encoder_src.weight is not all-zero"
                assert (block.cross_attn_src.o.weight == 0).all(), \
                    f"blocks[{i}].cross_attn_src.o.weight is not all-zero"
                assert (block.cross_attn_src.o.bias == 0).all(), \
                    f"blocks[{i}].cross_attn_src.o.bias is not all-zero"

        # 4. LoRA-B in any LoRALinear (uses the runtime class — no import-order concerns)
        for name, module in self.named_modules():
            if module.__class__.__name__ == "LoRALinear":
                assert (module.lora_B.weight == 0).all(), \
                    f"{name}.lora_B.weight is not all-zero"

    # ─ Freeze / train modes ─

    def freeze_base(self):
        """Freeze the pretrained Wan backbone; keep v2 modules trainable.
        Trainable substrings (substring-match against `name` from named_parameters):
          - patch_embed_source                full
          - geoada_                           adapter branch (incl. before/after_proj, modulation, attn, ffn, norms)
          - plucker_encoder                   matches both plucker_encoder and plucker_encoder_src
          - cross_attn_src                    full
        """
        for p in self.parameters():
            p.requires_grad = False
        trainable_substrings = (
            "patch_embed_source",
            "geoada_",
            "plucker_encoder",
            "cross_attn_src",
        )
        for name, p in self.named_parameters():
            if any(s in name for s in trainable_substrings):
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
    """Wrap targeted Linear layers inside MAIN DiT blocks with LoRA adapters.
    Adapter blocks (`geoada_blocks`) and `cross_attn_src` are NOT wrapped — they're
    full-trained per the v2 spec.
    """
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
