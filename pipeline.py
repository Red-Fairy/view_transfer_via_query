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
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diffsynth.diffusion.flow_match import FlowMatchScheduler

from .model import ViewTransferConfig, ViewTransferDiT, apply_lora
from .gpu_preprocess import gpu_preprocess
from .prepare_data.encode_latents import load_wan_vae


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
    ):
        self.model = model.eval()
        self.vae = vae
        self.scheduler = scheduler
        self.device = device
        self.pers_h = pers_h
        self.pers_w = pers_w
        self.use_mesh = use_mesh
        self.mesh_face_res = mesh_face_res

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

    @staticmethod
    def _build_uncond(cond: Dict) -> Dict:
        """Zero-out every conditioning stream for the unconditional pass."""
        return {k: torch.zeros_like(v) for k, v in cond.items()}

    @torch.no_grad()
    def generate(
        self,
        cpu_batch: Dict,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        return_latent: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Generate target perspective video.

        Args:
            cpu_batch: dict from `collate_view_transfer` WITHOUT `rgb_tgt_360` (or with;
                       it'll just be ignored). Required keys:
                       rgb_src_360, blob_360, static_rgb_t0, static_depth_t0,
                       pano_c2w_src, pano_c2w_tgt, src_c2w_at_t0, text_emb,
                       src_*_deg, tgt_*_deg.
            num_inference_steps: number of denoising steps (typically 30-100).
            guidance_scale: CFG scale; 1.0 = no guidance, 5.0 = standard.
            return_latent: if True, return predicted latent instead of decoded video.

        Returns:
            videos: [B, 3, T_video, pers_h, pers_w] uint8 in [0, 255], OR
                    if return_latent=True, [B, 16, T_lat, H_lat, W_lat] float32.
        """
        # Pop target if accidentally provided — inference doesn't use it
        cpu_batch = {k: v for k, v in cpu_batch.items() if k != "rgb_tgt_360"}

        # 1. Preprocess (skips target encoding because rgb_tgt_360 absent)
        cond = gpu_preprocess(
            cpu_batch, vae=self.vae, device=self.device,
            pers_h=self.pers_h, pers_w=self.pers_w,
            use_mesh=self.use_mesh, mesh_face_res=self.mesh_face_res,
        )
        if "target_latent" in cond:
            cond.pop("target_latent")

        # Cast all floating tensors to model dtype (gpu_preprocess outputs fp32)
        model_dtype = next(self.model.parameters()).dtype
        cond = {
            k: (v.to(model_dtype) if torch.is_floating_point(v) else v)
            for k, v in cond.items()
        }

        # 2. Init noise at target latent shape (matches source_latent shape)
        ref = cond["source_latent"]
        B = ref.shape[0]
        z = torch.randn_like(ref)

        # 3. Scheduler timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, training=False)
        timesteps = self.scheduler.timesteps.to(self.device)

        # 4. Unconditional batch (zeros)
        uncond = self._build_uncond(cond)

        # 5. Sampling loop with CFG
        model = self.model
        for i, t in enumerate(timesteps):
            t_batch = t.float().unsqueeze(0).expand(B)
            v_cond = model(noisy_latent=z, timestep=t_batch, **cond)
            if guidance_scale != 1.0:
                v_uncond = model(noisy_latent=z, timestep=t_batch, **uncond)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            # scheduler.step internally promotes via float32 sigma; restore model dtype
            z = self.scheduler.step(v, t, z).to(model_dtype)
            if verbose and (i % max(1, num_inference_steps // 10) == 0):
                print(f"step {i}/{num_inference_steps}  sigma={self.scheduler.sigmas[i].item():.4f}")

        if return_latent:
            return z

        # 6. VAE decode → [B, 3, T_video, H, W] in [-1, 1]
        latents = [z[b].float() for b in range(B)]
        videos = self.vae.decode(latents, device=self.device)
        videos = ((videos + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        return videos.cpu()
