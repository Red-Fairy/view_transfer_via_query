"""Inference pipeline for ViewTransferDiT (Stage 2).

Takes a "raw" cpu_batch (same shape as `ViewTransferDataset.__getitem__` plus a `collate`,
**without** `rgb_tgt_360`), runs online preprocessing (equi2pers + lift+render + VAE
encode + plücker + mask), denoises with classifier-free guidance, and VAE-decodes the
predicted target video.

Usage:
    pipe = ViewTransferPipeline.from_pretrained(
        dit_ckpt="/path/to/wan2.1-t2v-14b.safetensors",
        vae_ckpt="/path/to/Wan2.1_VAE.pth",
        lora_ckpt="/path/to/checkpoint-XXXX/trainable_params.pt",  # optional
        device="cuda",
    )
    videos = pipe.generate(cpu_batch, num_inference_steps=50, guidance_scale=5.0)
"""

from __future__ import annotations

import os
import sys
import torch
from typing import Dict, Optional, Tuple
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diffsynth.diffusion.flow_match import FlowMatchScheduler

from .model import ViewTransferConfig, ViewTransferDiT, apply_lora
from .gpu_preprocess import gpu_preprocess
from .prepare_data.encode_latents import load_wan_vae


def _vram_advance_modules(model: torch.nn.Module, target: str):
    """Walk every AutoWrapped* layer in `model` and call its `target` method
    (e.g. "onload" or "offload"). No-op on plain modules.

    Mirrors `BasePipeline.load_models_to_device` from diffsynth — needed because
    we don't subclass that base class but still want the same staged state
    transitions: state 0 (offloaded) → 1 (onloaded) so per-forward `vram_limit`
    pinning to GPU can fire.
    """
    for m in model.modules():
        fn = getattr(m, target, None)
        if callable(fn) and hasattr(m, "state"):
            fn()


# ── Pipeline ────────────────────────────────────────────────────────────────


class ViewTransferPipeline:
    """Inference pipeline. Encapsulates DiT + VAE + flow-matching scheduler."""

    def __init__(
        self,
        model: ViewTransferDiT,
        vae,
        scheduler: FlowMatchScheduler,
        device: torch.device,
        pers_h: int = 480,
        pers_w: int = 832,
        use_mesh: bool = False,
        mesh_face_res: int = 1024,
        low_vram: bool = False,
    ):
        self.model = model.eval()
        self.vae = vae
        self.scheduler = scheduler
        self.device = device
        self.pers_h = pers_h
        self.pers_w = pers_w
        self.use_mesh = use_mesh
        self.mesh_face_res = mesh_face_res
        self.low_vram = low_vram

    @classmethod
    def from_pretrained(
        cls,
        dit_ckpt: str,
        vae_ckpt: str,
        lora_ckpt: Optional[str] = None,
        lora_rank: int = 64,
        lora_alpha: float = 64.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[ViewTransferConfig] = None,
    ) -> "ViewTransferPipeline":
        device = torch.device(device)
        config = config or ViewTransferConfig()

        state_dict = torch.load(dit_ckpt, map_location="cpu", weights_only=True)
        model = ViewTransferDiT.from_pretrained(state_dict, config)
        del state_dict

        if lora_ckpt is not None:
            apply_lora(model, rank=lora_rank, alpha=lora_alpha)
            trainable_state = torch.load(lora_ckpt, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(trainable_state, strict=False)
            if unexpected:
                print(f"WARNING: unexpected keys in LoRA checkpoint: {unexpected[:5]}")

        model.to(device=device, dtype=dtype)
        vae = load_wan_vae(vae_ckpt, device=device, dtype=torch.float32)
        scheduler = FlowMatchScheduler(template="Wan")

        return cls(model=model, vae=vae, scheduler=scheduler, device=device)

    # Grouped-guidance stream membership. The model is trained with independent
    # per-stream CFG dropout (train.apply_cfg_dropout), so each group can carry
    # its own guidance weight at inference. Levels are nested: geom ⊂ geom+src ⊂ full.
    _GEOM_STREAMS = ("plucker_tgt", "rendered_latent", "mask_packed", "blob_latent")
    _SRC_STREAMS = ("source_latent", "plucker_src")
    # text_emb is the only remaining stream; it's whatever `cond` has that the
    # two groups above don't claim, so we don't need to enumerate it.

    @staticmethod
    def _build_uncond(cond: Dict) -> Dict:
        """Zero-out every conditioning stream for the unconditional pass."""
        return {k: torch.zeros_like(v) for k, v in cond.items()}

    @staticmethod
    def _subset_cond(cond: Dict, active_keys) -> Dict:
        """Keep `active_keys` as-is; zero every other stream (same shape/dtype)."""
        active = set(active_keys)
        return {
            k: (v if k in active else torch.zeros_like(v))
            for k, v in cond.items()
        }

    @torch.no_grad()
    def generate(
        self,
        cpu_batch: Dict,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_geom: Optional[float] = None,
        guidance_src: Optional[float] = None,
        guidance_text: Optional[float] = None,
        return_latent: bool = False,
        return_cond_videos: bool = False,
        verbose: bool = False,
        progress: bool = True,
        progress_desc: Optional[str] = None,
    ) -> "torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]":
        """Generate target perspective video.

        Args:
            cpu_batch: dict from `collate_view_transfer` WITHOUT `rgb_tgt_360` (or with;
                       it'll just be ignored). Required keys:
                       rgb_src_360, blob_360, static_rgb_t0, static_depth_t0,
                       pano_c2w_src, pano_c2w_tgt, src_c2w_at_t0, text_emb,
                       src_*_deg, tgt_*_deg.
            num_inference_steps: number of denoising steps (typically 30-100).
            guidance_scale: monolithic CFG scale; 1.0 = no guidance, 5.0 = standard.
                            Used only when grouped guidance is not requested.
            guidance_geom/guidance_src/guidance_text: grouped (chained) guidance
                weights. Pass ALL THREE to enable grouped mode; leave all None to
                use monolithic `guidance_scale`. With nested levels
                uncond ⊂ geom ⊂ geom+src ⊂ full, the velocity is
                  v = v_uncond
                      + w_geom·(v_geom    − v_uncond)
                      + w_src ·(v_geomsrc − v_geom)
                      + w_text·(v_full    − v_geomsrc)
                geom = {plucker_tgt, rendered_latent, mask_packed, blob_latent},
                src  = {source_latent, plucker_src}, text = {text_emb}.
                Costs 4 model forwards/step vs 2 for monolithic CFG.
            return_latent: if True, return predicted latent instead of decoded video.
            return_cond_videos: if True, also return the exact pre-VAE conditioning
                streams the model was built from, so callers can save faithful
                inspection videos without re-rendering.

        Returns:
            videos: [B, 3, T_video, pers_h, pers_w] uint8 in [0, 255], OR
                    if return_latent=True, [B, 16, T_lat, H_lat, W_lat] float32.
            If return_cond_videos=True, returns a tuple
            (videos_or_latent, {"rendered": [B,3,T,H,W] uint8,
                                "mask_vis": [B,1,T,H,W] uint8}) — both cpu.
        """
        # Pop target if accidentally provided — inference doesn't use it
        cpu_batch = {k: v for k, v in cpu_batch.items() if k != "rgb_tgt_360"}

        if self.low_vram:
            # Bring VAE to onload state (DiT to offloaded) so the encode runs
            # without DiT weights pinned on GPU.
            _vram_advance_modules(self.model, "offload")
            _vram_advance_modules(self.vae, "onload")
            torch.cuda.empty_cache()

        # 1. Preprocess (skips target encoding because rgb_tgt_360 absent)
        pp = gpu_preprocess(
            cpu_batch, vae=self.vae, device=self.device,
            pers_h=self.pers_h, pers_w=self.pers_w,
            use_mesh=self.use_mesh, mesh_face_res=self.mesh_face_res,
            tiled_vae=self.low_vram,
            return_videos=return_cond_videos,
        )

        cond_videos: Optional[Dict[str, torch.Tensor]] = None
        if return_cond_videos:
            # The exact pre-VAE conditioning pixels the model was built from —
            # single source of truth, no separate re-render. Move to CPU now so
            # they don't pin GPU memory through the denoise loop (matters under
            # --low_vram); uint8, so cheap to hold.
            cond_videos = {
                "rendered": pp["videos_rendered"].cpu(),
                "mask_vis": pp["mask_vis"].cpu(),
            }

        # Keep only the keys the DiT forward consumes (return_videos and the
        # non-inference target_latent must not leak into the **cond splat).
        _MODEL_COND_KEYS = (
            "source_latent", "rendered_latent", "blob_latent", "mask_packed",
            "plucker_src", "plucker_tgt", "text_emb",
        )
        # Cast all floating tensors to model dtype (gpu_preprocess outputs fp32)
        model_dtype = next(self.model.parameters()).dtype
        cond = {
            k: (pp[k].to(model_dtype) if torch.is_floating_point(pp[k]) else pp[k])
            for k in _MODEL_COND_KEYS
        }

        # 2. Init noise at target latent shape (matches source_latent shape)
        ref = cond["source_latent"]
        B = ref.shape[0]
        z = torch.randn_like(ref)

        if self.low_vram:
            # VAE is done; bring DiT in for the denoising loop.
            _vram_advance_modules(self.vae, "offload")
            _vram_advance_modules(self.model, "onload")
            torch.cuda.empty_cache()

        # 3. Scheduler timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, training=False)
        timesteps = self.scheduler.timesteps.to(self.device)

        # 4. Build the guidance levels.
        grouped = any(g is not None for g in (guidance_geom, guidance_src, guidance_text))
        if grouped and any(g is None for g in (guidance_geom, guidance_src, guidance_text)):
            raise ValueError(
                "grouped guidance needs all three of guidance_geom/src/text set "
                f"(got geom={guidance_geom}, src={guidance_src}, text={guidance_text})"
            )

        uncond = self._build_uncond(cond)
        if grouped:
            # Nested levels: uncond ⊂ geom ⊂ geom+src ⊂ full(=cond).
            cond_geom = self._subset_cond(cond, self._GEOM_STREAMS)
            cond_geomsrc = self._subset_cond(
                cond, self._GEOM_STREAMS + self._SRC_STREAMS
            )

        # 5. Sampling loop with CFG
        model = self.model
        pbar = tqdm(
            timesteps, desc=progress_desc or "denoising",
            disable=not progress, dynamic_ncols=True, leave=False,
        )
        for i, t in enumerate(pbar):
            t_batch = t.float().unsqueeze(0).expand(B)
            if grouped:
                v_uncond = model(noisy_latent=z, timestep=t_batch, **uncond)
                v_geom = model(noisy_latent=z, timestep=t_batch, **cond_geom)
                v_geomsrc = model(noisy_latent=z, timestep=t_batch, **cond_geomsrc)
                v_full = model(noisy_latent=z, timestep=t_batch, **cond)
                v = (
                    v_uncond
                    + guidance_geom * (v_geom - v_uncond)
                    + guidance_src * (v_geomsrc - v_geom)
                    + guidance_text * (v_full - v_geomsrc)
                )
            else:
                v_cond = model(noisy_latent=z, timestep=t_batch, **cond)
                if guidance_scale != 1.0:
                    v_uncond = model(noisy_latent=z, timestep=t_batch, **uncond)
                    v = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v = v_cond
            # scheduler.step internally promotes via float32 sigma; restore model dtype
            z = self.scheduler.step(v, t, z).to(model_dtype)
            sigma = float(self.scheduler.sigmas[i])
            pbar.set_postfix(sigma=f"{sigma:.3f}")
            if verbose and (i % max(1, num_inference_steps // 10) == 0):
                tqdm.write(f"step {i}/{num_inference_steps}  sigma={sigma:.4f}")

        if return_latent:
            return (z, cond_videos) if return_cond_videos else z

        if self.low_vram:
            # Swap DiT off, VAE on for decode.
            _vram_advance_modules(self.model, "offload")
            _vram_advance_modules(self.vae, "onload")
            torch.cuda.empty_cache()

        # 6. VAE decode → [B, 3, T_video, H, W] in [-1, 1]
        # `vae_dtype` matches whatever the VAE was loaded as: fp32 in the default
        # path, bf16 in the low-VRAM path. WanVideoVAE.decode copies its input
        # to CPU internally regardless, so fuse the device move and dtype cast
        # CPU-side: this copies bf16 across PCIe and upcasts on CPU instead of
        # promoting to fp32 on GPU first (half the bytes transferred). `tiled`
        # is only needed in the low-VRAM path to cap decode peak memory.
        vae_dtype = next(self.vae.parameters()).dtype
        latents = [z[b].detach().to(device="cpu", dtype=vae_dtype) for b in range(B)]
        videos = self.vae.decode(latents, device=self.device, tiled=self.low_vram)
        videos = ((videos + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)

        if self.low_vram:
            # Swap DiT back in for the next sample's denoise loop.
            _vram_advance_modules(self.vae, "offload")
            _vram_advance_modules(self.model, "onload")
            torch.cuda.empty_cache()
        out = videos.cpu()
        return (out, cond_videos) if return_cond_videos else out
