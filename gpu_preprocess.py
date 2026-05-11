"""GPU-side online preprocessing.

Per sample, on GPU:
    1. equi2pers     for source-RGB / target-RGB / blob   →  perspective videos
    2. lift+render   from static-source RGB+depth at t0   →  rendered + visibility
                     (rendered directly in target perspective; no 360 detour)
    3. VAE encode    source / target / rendered / blob
    4. plücker rays  at latent-aligned timestamps for source & target perspectives
    5. mask packing  visibility (perspective-resolution) → 4-ch latent-aligned

Designed to run on a side CUDA stream (see prefetcher.py) so it overlaps with the
model's previous-step fwd/bwd.
"""

from __future__ import annotations

import torch
from torch.profiler import record_function   # no-op when profiler isn't active
from typing import Dict

from .prepare_data.extract_perspectives import (
    yaw_pitch_roll_to_R,
    equi_to_perspective_video,
    compose_perspective_c2w,
    fov_to_intrinsics,
)
from .prepare_data.encode_latents import encode_video_to_latent
from .prepare_data.lift_and_render import lift_and_render
from .mask_utils import pack_mask
from .plucker import compute_plucker_at_latent_timestamps, plucker_to_channels


def _equi2pers_uint8(
    equi_uint8: torch.Tensor,    # [B, C, T, H, W] uint8
    R_crop: torch.Tensor,        # [B, T, 3, 3]
    fov_h_deg: torch.Tensor,     # [B] float
    pers_h: int, pers_w: int,
) -> torch.Tensor:
    """equi2pers per sample (FOV varies → loop over batch).
    Returns float [B, C, T, pers_h, pers_w] in [0, 1]."""
    B = equi_uint8.shape[0]
    outs = []
    for b in range(B):
        equi_f = equi_uint8[b].permute(1, 0, 2, 3).contiguous().float() / 255.0  # [T, C, H, W]
        proj = equi_to_perspective_video(
            equi_f, R_crop[b], fov_h_deg=float(fov_h_deg[b].item()),
            pers_h=pers_h, pers_w=pers_w,
        )  # [T, C, pers_h, pers_w]
        outs.append(proj.permute(1, 0, 2, 3).contiguous())
    return torch.stack(outs, dim=0)


@torch.no_grad()
def gpu_preprocess(
    cpu_batch: Dict,
    *,
    vae,
    device: torch.device,
    pers_h: int = 480,
    pers_w: int = 832,
    vae_temporal_factor: int = 4,
    vae_spatial_factor: int = 8,
    lift_render_chunk: int = 4,
    return_videos: bool = False,
    use_mesh: bool = False,
    mesh_face_res: int = 1024,
    tiled_vae: bool = False,
) -> Dict:
    """One step of online preprocessing. Returns a dict ready for the model.

    When `return_videos=True`, also includes uint8 perspective videos and the
    visibility mask under keys `videos_src/tgt/rendered/blob` and `mask_vis`,
    plus pass-through trajectory metadata for offline inspection.
    """
    B = cpu_batch["rgb_src_360"].shape[0]
    pano_c2w_src = cpu_batch["pano_c2w_src"].to(device, non_blocking=True)  # [B, T, 4, 4]
    pano_c2w_tgt = cpu_batch["pano_c2w_tgt"].to(device, non_blocking=True)
    src_c2w_at_t0 = cpu_batch["src_c2w_at_t0"].to(device, non_blocking=True)  # [B, 4, 4]

    # ── 1. R_crop and equi2pers for the 3 streams that come from 360 sources ──
    src_R = yaw_pitch_roll_to_R(
        cpu_batch["src_yaw_deg"].to(device, non_blocking=True),
        cpu_batch["src_pitch_deg"].to(device, non_blocking=True),
        cpu_batch["src_roll_deg"].to(device, non_blocking=True),
    )  # [B, T, 3, 3]
    tgt_R = yaw_pitch_roll_to_R(
        cpu_batch["tgt_yaw_deg"].to(device, non_blocking=True),
        cpu_batch["tgt_pitch_deg"].to(device, non_blocking=True),
        cpu_batch["tgt_roll_deg"].to(device, non_blocking=True),
    )

    rgb_src = cpu_batch["rgb_src_360"].to(device, non_blocking=True)
    blob = cpu_batch["blob_360"].to(device, non_blocking=True)
    has_target = "rgb_tgt_360" in cpu_batch

    with record_function("preprocess.equi2pers"):
        src_pers = _equi2pers_uint8(rgb_src, src_R, cpu_batch["src_fov_h_deg"], pers_h, pers_w)
        if has_target:
            rgb_tgt = cpu_batch["rgb_tgt_360"].to(device, non_blocking=True)
            tgt_pers = _equi2pers_uint8(rgb_tgt, tgt_R, cpu_batch["tgt_fov_h_deg"], pers_h, pers_w)
        blob_pers = _equi2pers_uint8(blob, tgt_R, cpu_batch["tgt_fov_h_deg"], pers_h, pers_w)

    # ── 2. Lift-and-render: rendered + visibility, directly in target perspective ──
    static_rgb = cpu_batch["static_rgb_t0"].to(device, non_blocking=True)        # [B, 3, He, We] uint8
    static_depth = cpu_batch["static_depth_t0"].to(device, non_blocking=True)    # [B, He, We] fp32

    # Compose target perspective c2w + intrinsics
    tgt_pers_c2w = compose_perspective_c2w(pano_c2w_tgt, tgt_R)                  # [B, T, 4, 4]
    src_pers_c2w = compose_perspective_c2w(pano_c2w_src, src_R)                  # [B, T, 4, 4]

    K_tgt = torch.empty(B, 4, device=device, dtype=torch.float32)
    K_src = torch.empty(B, 4, device=device, dtype=torch.float32)
    for b in range(B):
        fx, fy, cx, cy = fov_to_intrinsics(float(cpu_batch["tgt_fov_h_deg"][b]), pers_h, pers_w)
        K_tgt[b] = torch.tensor([fx, fy, cx, cy])
        fx, fy, cx, cy = fov_to_intrinsics(float(cpu_batch["src_fov_h_deg"][b]), pers_h, pers_w)
        K_src[b] = torch.tensor([fx, fy, cx, cy])

    with record_function("preprocess.lift_and_render"):
        rendered_pers_list = []
        visibility_pers_list = []
        for b in range(B):
            rgb_b = static_rgb[b].float() / 255.0  # [3, He, We]
            depth_b = static_depth[b].float()      # [He, We]
            rendered_b, vis_b = lift_and_render(
                static_rgb=rgb_b,
                static_depth=depth_b,
                pano_c2w_at_t0=src_c2w_at_t0[b],
                target_c2w=tgt_pers_c2w[b],
                intrinsics=K_tgt[b],
                pers_h=pers_h,
                pers_w=pers_w,
                chunk_size=lift_render_chunk,
                use_mesh=use_mesh,
                mesh_face_res=mesh_face_res,
            )
            # rendered_b: [T, 3, pers_h, pers_w]; vis_b: [T, 1, pers_h, pers_w]
            rendered_pers_list.append(rendered_b.permute(1, 0, 2, 3).contiguous())   # [3, T, ...]
            visibility_pers_list.append(vis_b.permute(1, 0, 2, 3).contiguous())      # [1, T, ...]
        rendered_pers = torch.stack(rendered_pers_list, dim=0)  # [B, 3, T, H, W]
        visibility_pers = torch.stack(visibility_pers_list, dim=0)

    # ── 3. VAE encode the 4 RGB streams ──
    # Match input dtype to the VAE's parameter dtype (e.g., bf16 VAE → bf16 input)
    # so the conv ops don't trip on a dtype mismatch.
    vae_dtype = next(vae.model.parameters()).dtype
    def _encode_batch(pers_videos: torch.Tensor) -> torch.Tensor:
        # keep_on_device=True skips encode_video_to_latent's `.cpu()` and our
        # trailing `.to(device)`. Saves one D2H+H2D round-trip per stream per batch.
        # (WanVideoVAE.encode still copies its *input* to CPU internally — out of scope.)
        latents = []
        for b in range(B):
            video_btchw = pers_videos[b].permute(1, 0, 2, 3).contiguous().to(dtype=vae_dtype)
            latents.append(encode_video_to_latent(
                vae, video_btchw, device=device, tiled=tiled_vae, keep_on_device=True,
            ))
        return torch.stack(latents, dim=0)

    with record_function("preprocess.vae_encode"):
        source_latent = _encode_batch(src_pers)
        rendered_latent = _encode_batch(rendered_pers)
        blob_latent = _encode_batch(blob_pers)
        target_latent = _encode_batch(tgt_pers) if has_target else None

    with record_function("preprocess.mask_and_plucker"):
        # visibility_pers is at perspective resolution: [B, 1, T_video, pers_h, pers_w]
        mask_packed = pack_mask(
            visibility_pers,
            vae_spatial_factor=vae_spatial_factor,
            vae_temporal_factor=vae_temporal_factor,
        )  # [B, 4, T_lat, H_lat, W_lat]

        plucker_src = compute_plucker_at_latent_timestamps(
            K_src, src_pers_c2w, pers_h, pers_w, vae_temporal_factor=vae_temporal_factor,
        )
        plucker_tgt = compute_plucker_at_latent_timestamps(
            K_tgt, tgt_pers_c2w, pers_h, pers_w, vae_temporal_factor=vae_temporal_factor,
        )
        plucker_src = plucker_to_channels(plucker_src)  # [B, 6, T_lat, pers_h, pers_w]
        plucker_tgt = plucker_to_channels(plucker_tgt)

    text_emb = cpu_batch["text_emb"].to(device, non_blocking=True)

    out = {
        "source_latent": source_latent,
        "rendered_latent": rendered_latent,
        "blob_latent": blob_latent,
        "mask_packed": mask_packed,
        "plucker_src": plucker_src,
        "plucker_tgt": plucker_tgt,
        "text_emb": text_emb,
    }
    if has_target:
        out["target_latent"] = target_latent

    if return_videos:
        def _to_u8(x):  # x: float in [0, 1]; output uint8
            return (x.clamp(0, 1) * 255).round().to(torch.uint8)

        out["videos_src"] = _to_u8(src_pers)
        out["videos_blob"] = _to_u8(blob_pers)
        out["videos_rendered"] = _to_u8(rendered_pers)
        out["mask_vis"] = _to_u8(visibility_pers)
        if has_target:
            out["videos_tgt"] = _to_u8(tgt_pers)

        # Pass through trajectory metadata + provenance (cheap; needed only for logging).
        for k in (
            "src_yaw_deg", "src_pitch_deg", "src_roll_deg", "src_fov_h_deg",
            "tgt_yaw_deg", "tgt_pitch_deg", "tgt_roll_deg", "tgt_fov_h_deg",
            "t_offset", "location_dir", "src_idx", "tgt_idx",
        ):
            if k in cpu_batch:
                out[k] = cpu_batch[k]

    return out
