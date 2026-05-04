"""Batched CLI inference for ViewTransferDiT.

Reads a locations file (one prepared location dir per line), and for each location
samples (t0, src trajectory, tgt trajectory) — exactly as training does — runs the
trained pipeline, and writes prediction + GT + every conditioning stream into a
per-sample output folder.

With --dual_projection enabled, also runs the reverse direction (src↔tgt swapped,
trajectories swapped) so the source perspective in direction B is identical to
the target perspective in direction A.

Usage:
    python -m view_transfer_via_query.infer \\
        --locations_file /share/.../split_files/scene_locations.txt \\
        --out_dir   ./infer_out/14B-debut-step1600 \\
        --num_per_location 1 \\
        --src_idx 00 --tgt_idx 01 [--dual_projection] \\
        --dit_ckpt  /path/to/Wan2.1-T2V-14B \\
        --vae_ckpt  /path/to/Wan2.1_VAE.pth \\
        --lora_ckpt /path/to/runs/.../checkpoint-XXXX/trainable_params.pt \\
        --model_size 14B --lora_rank 64 \\
        --num_inference_steps 50 --guidance_scale 5.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import imageio.v2 as imageio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from safetensors.torch import load_file as load_safetensors

from view_transfer_via_query.model import MODEL_CONFIGS, ViewTransferDiT, apply_lora
from view_transfer_via_query.dataset import SampleEntry, load_locations_file, collate_view_transfer
from view_transfer_via_query.pipeline import ViewTransferPipeline
from view_transfer_via_query.prepare_data.encode_latents import load_wan_vae
from view_transfer_via_query.prepare_data.video_io import load_png_sequence
from view_transfer_via_query.prepare_data.extract_perspectives import (
    sample_perspective_trajectory,
    yaw_pitch_roll_to_R,
    equi_to_perspective_video,
    compose_perspective_c2w,
    fov_to_intrinsics,
)
from view_transfer_via_query.prepare_data.lift_and_render import (
    load_depth, load_depth_ue, lift_and_render,
)

from diffsynth.diffusion.flow_match import FlowMatchScheduler


# ── DiT loader (sharded HF dir or single file) ───────────────────────────────


def load_dit_state_dict(ckpt_path: str) -> dict:
    import glob
    if os.path.isdir(ckpt_path):
        files = sorted(glob.glob(os.path.join(ckpt_path, "diffusion_pytorch_model*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No diffusion_pytorch_model*.safetensors in {ckpt_path}")
        sd = {}
        for f in files:
            sd.update(load_safetensors(f))
        return sd
    if ckpt_path.endswith(".safetensors"):
        return load_safetensors(ckpt_path)
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)


# ── Sample loading (parameterized: explicit pairing + explicit trajectories) ─


def _list_files(dir_path: str, exts: tuple) -> List[str]:
    fs = sorted(f for f in os.listdir(dir_path) if f.lower().endswith(exts))
    return [os.path.join(dir_path, f) for f in fs]


def _load_window(d: str, t0: int, T: int) -> torch.Tensor:
    """[3, T, H, W] uint8 from a PNG dir."""
    return load_png_sequence(d, start=t0, num_frames=T, return_dtype=torch.uint8) \
        .permute(1, 0, 2, 3).contiguous()


def build_sample(
    entry: SampleEntry,
    t0: int,
    num_video_frames: int,
    traj_src: dict,
    traj_tgt: dict,
) -> Dict:
    """Reproduces ViewTransferDataset.__getitem__ but with externally supplied
    pairing/t0/trajectories — needed so dual-projection can swap them deterministically.
    """
    pano_c2w_src = torch.load(entry.c2w_src_path, map_location="cpu", weights_only=True).float()
    pano_c2w_tgt = torch.load(entry.c2w_tgt_path, map_location="cpu", weights_only=True).float()
    T_full = pano_c2w_src.shape[0]
    assert pano_c2w_tgt.shape[0] == T_full
    assert t0 + num_video_frames <= T_full, f"t0={t0} + T={num_video_frames} > T_full={T_full}"
    t_slice = slice(t0, t0 + num_video_frames)

    rgb_src = _load_window(entry.rgb_src_dir, t0, num_video_frames)
    rgb_tgt = _load_window(entry.rgb_tgt_dir, t0, num_video_frames)
    blob = _load_window(entry.blob_dir, t0, num_video_frames)

    static_rgb_t0 = load_png_sequence(
        entry.static_rgb_dir, start=t0, num_frames=1, return_dtype=torch.uint8,
    )[0]  # [3, He, We]

    depth_files = _list_files(entry.static_depth_dir, exts=(".exr", ".npy", ".pt", ".pth"))
    if t0 >= len(depth_files):
        raise IndexError(f"t0={t0} but only {len(depth_files)} depth frames")
    if depth_files[t0].endswith(".exr"):
        depth_np = load_depth_ue(depth_files[t0])
    else:
        depth_np = load_depth(depth_files[t0])
    static_depth_t0 = torch.from_numpy(depth_np)

    text_emb = torch.load(entry.text_emb_path, map_location="cpu", weights_only=True).float()

    pano_c2w_src_win = pano_c2w_src[t_slice].clone()
    pano_c2w_tgt_win = pano_c2w_tgt[t_slice].clone()

    return {
        "rgb_src_360": rgb_src,
        "rgb_tgt_360": rgb_tgt,
        "blob_360": blob,
        "static_rgb_t0": static_rgb_t0,
        "static_depth_t0": static_depth_t0,
        "pano_c2w_src": pano_c2w_src_win,
        "pano_c2w_tgt": pano_c2w_tgt_win,
        "src_c2w_at_t0": pano_c2w_src_win[0].clone(),
        "text_emb": text_emb,
        "src_fov_h_deg": float(traj_src["fov_h_deg"]),
        "src_yaw_deg": torch.from_numpy(traj_src["yaw_deg"]),
        "src_pitch_deg": torch.from_numpy(traj_src["pitch_deg"]),
        "src_roll_deg": torch.from_numpy(traj_src["roll_deg"]),
        "tgt_fov_h_deg": float(traj_tgt["fov_h_deg"]),
        "tgt_yaw_deg": torch.from_numpy(traj_tgt["yaw_deg"]),
        "tgt_pitch_deg": torch.from_numpy(traj_tgt["pitch_deg"]),
        "tgt_roll_deg": torch.from_numpy(traj_tgt["roll_deg"]),
        "t_offset": t0,
        "location_dir": entry.location_dir,
        "src_idx": entry.src_idx,
        "tgt_idx": entry.tgt_idx,
    }


# ── Pixel-space re-extraction (no VAE) for inspection ───────────────────────


def _to_uint8_video(pers_f: torch.Tensor) -> torch.Tensor:
    """[T, 3, H, W] float in [0, 1] → uint8 cpu."""
    return (pers_f * 255.0).clamp(0, 255).to(torch.uint8).cpu()


def extract_pers_pixel(
    rgb_360_uint8: torch.Tensor,    # [3, T, He, We]
    yaw_deg: torch.Tensor, pitch_deg: torch.Tensor, roll_deg: torch.Tensor,
    fov_h_deg: float, pers_h: int, pers_w: int, device: torch.device,
) -> torch.Tensor:
    R = yaw_pitch_roll_to_R(yaw_deg, pitch_deg, roll_deg).to(device)
    equi_f = (rgb_360_uint8.to(device).float() / 255.0).permute(1, 0, 2, 3).contiguous()
    pers_f = equi_to_perspective_video(equi_f, R, fov_h_deg, pers_h=pers_h, pers_w=pers_w)
    return _to_uint8_video(pers_f)


def render_warp_visibility(
    static_rgb_uint8: torch.Tensor,  # [3, He, We]
    static_depth: torch.Tensor,      # [He, We] meters (NaN = sky)
    src_c2w_at_t0: torch.Tensor,
    pano_c2w_tgt: torch.Tensor,
    tgt_yaw_deg: torch.Tensor, tgt_pitch_deg: torch.Tensor, tgt_roll_deg: torch.Tensor,
    fov_h_deg: float, pers_h: int, pers_w: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (rendered_uint8 [T,3,h,w], visibility_uint8 [T,3,h,w])."""
    rgb_b = static_rgb_uint8.to(device).float() / 255.0
    depth_b = static_depth.to(device).float()
    tgt_R = yaw_pitch_roll_to_R(tgt_yaw_deg, tgt_pitch_deg, tgt_roll_deg).to(device)
    tgt_pers_c2w = compose_perspective_c2w(pano_c2w_tgt.to(device), tgt_R)
    fx, fy, cx, cy = fov_to_intrinsics(fov_h_deg, pers_h, pers_w)
    K = torch.tensor([fx, fy, cx, cy], device=device, dtype=torch.float32)
    rendered, vis = lift_and_render(
        static_rgb=rgb_b, static_depth=depth_b,
        pano_c2w_at_t0=src_c2w_at_t0.to(device),
        target_c2w=tgt_pers_c2w, intrinsics=K,
        pers_h=pers_h, pers_w=pers_w,
    )
    rendered_u8 = _to_uint8_video(rendered)
    vis_u8 = (vis.expand(-1, 3, -1, -1) * 255.0).clamp(0, 255).to(torch.uint8).cpu()
    return rendered_u8, vis_u8


# ── Saving ───────────────────────────────────────────────────────────────────


def save_video_uint8(path: Path, video_THWC_or_TCHW: torch.Tensor, fps: int):
    """Accepts [T,3,H,W] or [T,H,W,3] uint8."""
    v = video_THWC_or_TCHW
    if v.dim() == 4 and v.shape[1] == 3:
        v = v.permute(0, 2, 3, 1)
    arr = v.contiguous().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), arr, fps=fps, macro_block_size=2)


def save_info_json(path: Path, sample: Dict, args: argparse.Namespace):
    info = {
        "location_dir": sample["location_dir"],
        "src_idx": sample["src_idx"],
        "tgt_idx": sample["tgt_idx"],
        "t_offset": int(sample["t_offset"]),
        "src_fov_h_deg": float(sample["src_fov_h_deg"]),
        "src_yaw_deg": sample["src_yaw_deg"].tolist(),
        "src_pitch_deg": sample["src_pitch_deg"].tolist(),
        "src_roll_deg": sample["src_roll_deg"].tolist(),
        "tgt_fov_h_deg": float(sample["tgt_fov_h_deg"]),
        "tgt_yaw_deg": sample["tgt_yaw_deg"].tolist(),
        "tgt_pitch_deg": sample["tgt_pitch_deg"].tolist(),
        "tgt_roll_deg": sample["tgt_roll_deg"].tolist(),
        "src_c2w_at_t0": sample["src_c2w_at_t0"].tolist(),
        "pano_c2w_src_first": sample["pano_c2w_src"][0].tolist(),
        "pano_c2w_tgt_first": sample["pano_c2w_tgt"][0].tolist(),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))


def folder_name(sample: Dict) -> str:
    """<scene>__<location>__src{X}tgt{Y}_t{t0}"""
    loc = Path(sample["location_dir"])
    return f"{loc.parent.name}__{loc.name}__src{sample['src_idx']}tgt{sample['tgt_idx']}_t{int(sample['t_offset'])}"


def render_and_save(
    sample: Dict, pipe: ViewTransferPipeline, out_dir: Path,
    pers_h: int, pers_w: int, fps: int, num_inference_steps: int, guidance_scale: float,
    device: torch.device, args: argparse.Namespace,
):
    sample_dir = out_dir / folder_name(sample)
    if sample_dir.exists() and (sample_dir / "pred.mp4").exists():
        print(f"  [skip] {sample_dir.name} (pred.mp4 exists)")
        return

    # 1. Model prediction
    batch = collate_view_transfer([sample])
    batch_no_gt = {k: v for k, v in batch.items() if k != "rgb_tgt_360"}
    pred = pipe.generate(
        batch_no_gt, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, verbose=False,
    )[0]  # [3, T, H, W] uint8 cpu

    # 2. Pixel-space conditions (no VAE round-trip)
    src_pers = extract_pers_pixel(
        sample["rgb_src_360"], sample["src_yaw_deg"], sample["src_pitch_deg"], sample["src_roll_deg"],
        sample["src_fov_h_deg"], pers_h, pers_w, device,
    )
    tgt_pers = extract_pers_pixel(
        sample["rgb_tgt_360"], sample["tgt_yaw_deg"], sample["tgt_pitch_deg"], sample["tgt_roll_deg"],
        sample["tgt_fov_h_deg"], pers_h, pers_w, device,
    )
    blob_pers = extract_pers_pixel(
        sample["blob_360"], sample["tgt_yaw_deg"], sample["tgt_pitch_deg"], sample["tgt_roll_deg"],
        sample["tgt_fov_h_deg"], pers_h, pers_w, device,
    )
    rendered_pers, visibility_pers = render_warp_visibility(
        sample["static_rgb_t0"], sample["static_depth_t0"],
        sample["src_c2w_at_t0"], sample["pano_c2w_tgt"],
        sample["tgt_yaw_deg"], sample["tgt_pitch_deg"], sample["tgt_roll_deg"],
        sample["tgt_fov_h_deg"], pers_h, pers_w, device,
    )

    # 3. Save
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_video_uint8(sample_dir / "pred.mp4", pred.permute(1, 2, 3, 0), fps)
    save_video_uint8(sample_dir / "tgt_gt.mp4", tgt_pers, fps)
    save_video_uint8(sample_dir / "src.mp4", src_pers, fps)
    save_video_uint8(sample_dir / "rendered.mp4", rendered_pers, fps)
    save_video_uint8(sample_dir / "visibility.mp4", visibility_pers, fps)
    save_video_uint8(sample_dir / "blob.mp4", blob_pers, fps)
    save_info_json(sample_dir / "info.json", sample, args)
    print(f"  [done] {sample_dir.name}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="ViewTransferDiT batched inference over a locations file")
    p.add_argument("--locations_file", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_per_location", type=int, default=1)
    p.add_argument("--src_idx", type=str, default="00")
    p.add_argument("--tgt_idx", type=str, default="01")
    p.add_argument("--dual_projection", action="store_true",
                   help="also run the reverse direction (src↔tgt swapped, trajectories swapped) "
                        "per sample, so source/target videos mirror across the two outputs")

    p.add_argument("--dit_ckpt", type=str, required=True)
    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--lora_ckpt", type=str, default=None)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=float, default=64.0)
    p.add_argument("--model_size", type=str, default="14B", choices=list(MODEL_CONFIGS.keys()))

    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--num_video_frames", type=int, default=81)
    p.add_argument("--pers_h", type=int, default=480)
    p.add_argument("--pers_w", type=int, default=832)

    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    return p.parse_args()


def build_pipeline(args, device: torch.device) -> ViewTransferPipeline:
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    print(f"[infer] loading DiT: {args.dit_ckpt}")
    state_dict = load_dit_state_dict(args.dit_ckpt)
    config = MODEL_CONFIGS[args.model_size]()
    model = ViewTransferDiT.from_pretrained(state_dict, config)
    del state_dict

    if args.lora_ckpt is not None:
        print(f"[infer] applying LoRA rank={args.lora_rank} from {args.lora_ckpt}")
        apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        trainable_state = torch.load(args.lora_ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(trainable_state, strict=False)
        n_ckpt = len(trainable_state)
        n_loaded = n_ckpt - len(unexpected)
        print(f"  loaded {n_loaded}/{n_ckpt} trainable keys")
        if unexpected:
            print(f"  WARN: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    model.to(device=device, dtype=dtype).eval()
    print(f"[infer] loading VAE: {args.vae_ckpt}")
    vae = load_wan_vae(args.vae_ckpt, device=str(device), dtype=torch.float32)
    scheduler = FlowMatchScheduler(template="Wan")
    return ViewTransferPipeline(
        model=model, vae=vae, scheduler=scheduler, device=device,
        pers_h=args.pers_h, pers_w=args.pers_w,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    locations = load_locations_file(args.locations_file)
    print(f"[infer] {len(locations)} locations from {args.locations_file}")

    pipe = build_pipeline(args, device)
    out_dir = Path(args.out_dir)

    n_done = n_skipped = 0
    for loc in locations:
        # Direction A entry (src=A, tgt=B)
        entry_a = SampleEntry(location_dir=loc, src_idx=args.src_idx, tgt_idx=args.tgt_idx)
        if not entry_a.is_complete():
            print(f"[skip] incomplete: {loc}  src={args.src_idx} tgt={args.tgt_idx}")
            n_skipped += 1
            continue
        if args.dual_projection:
            entry_b = SampleEntry(location_dir=loc, src_idx=args.tgt_idx, tgt_idx=args.src_idx)
            if not entry_b.is_complete():
                print(f"[skip] dual incomplete: {loc}  src={args.tgt_idx} tgt={args.src_idx}")
                n_skipped += 1
                continue

        # Pano frame count from the c2w tensor
        pano_c2w_src = torch.load(entry_a.c2w_src_path, map_location="cpu", weights_only=True)
        T_full = pano_c2w_src.shape[0]
        del pano_c2w_src
        if T_full < args.num_video_frames:
            print(f"[skip] T_full={T_full} < num_video_frames={args.num_video_frames}: {loc}")
            n_skipped += 1
            continue
        t_max = T_full - args.num_video_frames

        for k in range(args.num_per_location):
            # System-entropy randomness — no fixed seed
            rng = np.random.default_rng()
            t0 = int(rng.integers(0, t_max + 1))
            traj_a = sample_perspective_trajectory(args.num_video_frames, rng=rng)
            traj_b = sample_perspective_trajectory(args.num_video_frames, rng=rng)

            sample_a = build_sample(entry_a, t0, args.num_video_frames, traj_a, traj_b)
            print(f"[gen] {Path(loc).name}  src={entry_a.src_idx} tgt={entry_a.tgt_idx}  t0={t0}  ({k+1}/{args.num_per_location})")
            render_and_save(
                sample_a, pipe, out_dir, args.pers_h, args.pers_w, args.fps,
                args.num_inference_steps, args.guidance_scale, device, args,
            )
            n_done += 1

            if args.dual_projection:
                # Same t0; swap pairing AND swap trajectories so the source perspective
                # in B equals the target perspective in A (and vice versa).
                sample_b = build_sample(entry_b, t0, args.num_video_frames, traj_b, traj_a)
                print(f"[gen] {Path(loc).name}  src={entry_b.src_idx} tgt={entry_b.tgt_idx}  t0={t0}  (dual)")
                render_and_save(
                    sample_b, pipe, out_dir, args.pers_h, args.pers_w, args.fps,
                    args.num_inference_steps, args.guidance_scale, device, args,
                )
                n_done += 1

    print(f"\n[infer] done.  generated={n_done}  skipped_locations={n_skipped}  out_dir={out_dir}")


if __name__ == "__main__":
    main()
