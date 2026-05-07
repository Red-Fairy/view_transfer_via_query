"""Visualize sampled perspective trajectories on 360 videos.

Samples N pairs of source/target trajectories (same-video or different-video
pairings) and produces side-by-side mp4s with the perspective footprint drawn
as a red boundary box on each 360 video, plus a per-frame text strip showing
fov/yaw/pitch/roll.

Usage:
    python -m view_transfer_via_query.tools.viz_trajectories \\
        --data_root /share/.../outputs_non_arranged_cars_v2 \\
        --out_dir   ./viz_trajectories \\
        --num_same 50 --num_diff 50 \\
        --num_frames 81 --equi_h 384 --equi_w 768
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio.v2 as imageio
from tqdm import tqdm

# 360VideoGeneration's generate_mask_batch (boundary-mask helper).
# Import the module file directly to avoid src/__init__.py's heavy deps (prdc, etc.).
import importlib.util
_pers2equi_path = "/share/ma/scratch/rundong/360VideoGeneration/src/pers2equi.py"
_spec = importlib.util.spec_from_file_location("_p2e", _pers2equi_path)
_p2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p2e)
generate_mask_batch = _p2e.generate_mask_batch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from view_transfer_via_query.prepare_data.extract_perspectives import (
    sample_perspective_trajectory,
    sample_trajectory_pair,
    yaw_pitch_roll_to_R,
    equi_to_perspective_video,
)


# ── Geometry & rendering helpers ─────────────────────────────────────────────


def boundary_red_overlay(
    frames_uint8: torch.Tensor,    # [T, 3, H, W] uint8
    yaw_deg: np.ndarray,            # [T]  degrees
    pitch_deg: np.ndarray,          # [T]
    roll_deg: np.ndarray,           # [T]
    fov_h_deg: float,
    pers_hw_ratio: float = 480 / 832,
    box_width: int = 3,
    device: str = "cuda",
) -> torch.Tensor:
    """Return frames with the perspective footprint drawn as a red boundary."""
    H, W = frames_uint8.shape[-2:]
    yaw_r = torch.from_numpy(np.deg2rad(yaw_deg)).float().to(device)
    pitch_r = torch.from_numpy(np.deg2rad(pitch_deg)).float().to(device)
    roll_r = torch.from_numpy(np.deg2rad(roll_deg)).float().to(device)

    # generate_mask_batch returns [T, 1, H, W] in {0, 1}
    mask = generate_mask_batch(
        fov_x=fov_h_deg, roll=roll_r, yaw=yaw_r, pitch=pitch_r,
        height=H, width=W, hw_ratio=pers_hw_ratio, device=device,
    ).float()

    # Edge detection (same kernel as draw_box.py:add_box_to_video)
    kernel = torch.tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]], device=device
    ).view(1, 1, 3, 3)
    boundary = F.conv2d(mask, kernel, padding=1)
    boundary = (boundary != 0).float()
    # Suppress the image-frame edges (they shouldn't count as box edges)
    boundary[:, :, 0, :] = 0
    boundary[:, :, -1, :] = 0
    boundary[:, :, :, 0] = 0
    boundary[:, :, :, -1] = 0
    bw = max(1, (box_width // 2) * 2 + 1)
    boundary = F.max_pool2d(boundary, bw, stride=1, padding=bw // 2)

    # Apply red on boundary pixels
    frames_f = frames_uint8.float().to(device) / 255.0  # [T, 3, H, W]
    red = torch.zeros_like(frames_f)
    red[:, 0] = 1.0  # R
    out = torch.where(boundary > 0, red, frames_f)
    return (out * 255).clamp(0, 255).byte().cpu()


def render_perspective_uint8(
    equi_uint8: torch.Tensor,    # [T, 3, H, W] uint8 (RGB)
    yaw_deg: np.ndarray,
    pitch_deg: np.ndarray,
    roll_deg: np.ndarray,
    fov_h_deg: float,
    pers_h: int,
    pers_w: int,
    border_w: int = 3,
    device: str = "cuda",
) -> torch.Tensor:
    """Project equirect → perspective with a red border. Returns uint8 [T,3,pers_h,pers_w]."""
    # yaw_pitch_roll_to_R expects DEGREES (it does deg2rad internally) — pass raw.
    yaw_t = torch.from_numpy(np.asarray(yaw_deg, dtype=np.float32))
    pitch_t = torch.from_numpy(np.asarray(pitch_deg, dtype=np.float32))
    roll_t = torch.from_numpy(np.asarray(roll_deg, dtype=np.float32))
    R_crop = yaw_pitch_roll_to_R(yaw_t, pitch_t, roll_t).to(device)  # [T, 3, 3]

    equi_f = equi_uint8.to(device).float() / 255.0
    pers_f = equi_to_perspective_video(equi_f, R_crop, fov_h_deg, pers_h=pers_h, pers_w=pers_w)

    if border_w > 0:
        pers_f[:, 0, :border_w, :] = 1.0
        pers_f[:, 1:, :border_w, :] = 0.0
        pers_f[:, 0, -border_w:, :] = 1.0
        pers_f[:, 1:, -border_w:, :] = 0.0
        pers_f[:, 0, :, :border_w] = 1.0
        pers_f[:, 1:, :, :border_w] = 0.0
        pers_f[:, 0, :, -border_w:] = 1.0
        pers_f[:, 1:, :, -border_w:] = 0.0

    return (pers_f * 255).clamp(0, 255).byte().cpu()


def stack_pers_above_pano(pers_uint8: torch.Tensor, pano_uint8: torch.Tensor,
                          panel_width: int) -> torch.Tensor:
    """Pad pers horizontally with white to `panel_width`, then concat above pano."""
    T_dim, _, pers_h, pers_w = pers_uint8.shape
    if pers_w == panel_width:
        pers_padded = pers_uint8
    elif pers_w < panel_width:
        pad_total = panel_width - pers_w
        pad_l, pad_r = pad_total // 2, pad_total - pad_total // 2
        pad_left = torch.full((T_dim, 3, pers_h, pad_l), 255, dtype=torch.uint8)
        pad_right = torch.full((T_dim, 3, pers_h, pad_r), 255, dtype=torch.uint8)
        pers_padded = torch.cat([pad_left, pers_uint8, pad_right], dim=-1)
    else:
        excess = pers_w - panel_width
        l = excess // 2
        pers_padded = pers_uint8[..., l:l + panel_width]
    return torch.cat([pers_padded, pano_uint8], dim=-2)


def draw_text_strip(total_width: int, panel_width: int, sep_w: int,
                    text_left: str, text_right: str,
                    font_scale: float = 0.6, height: int = 40) -> np.ndarray:
    """White strip with `text_left` centered in the left panel, `text_right` in the right.

    `panel_width` is the width of each video panel; `sep_w` is the separator between
    them; `total_width` should equal `2 * panel_width + sep_w`.
    """
    strip = np.full((height, total_width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    def put_centered(text: str, panel_x0: int):
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = panel_x0 + (panel_width - tw) // 2
        y = (height + th) // 2
        cv2.putText(strip, text, (x, y), font, font_scale, (0, 0, 0),
                    thickness=thickness, lineType=cv2.LINE_AA)

    put_centered(text_left, 0)
    put_centered(text_right, panel_width + sep_w)
    return strip


def fmt_traj(label: str, traj: dict, frame_idx: int) -> str:
    # Hershey fonts are ASCII-only; using "deg" instead of "°" to avoid "??".
    return (f"{label}: fov={traj['fov_h_deg']:6.1f}  "
            f"yaw={traj['yaw_deg'][frame_idx]:+7.1f}  "
            f"pitch={traj['pitch_deg'][frame_idx]:+6.1f}  "
            f"roll={traj['roll_deg'][frame_idx]:+5.1f}")


# ── Video loading ────────────────────────────────────────────────────────────


def load_equi_window(pano_dir: Path, t0: int, T: int, H: int, W: int) -> torch.Tensor:
    """Load `T` frames starting at `t0`, resized to (H, W). Returns uint8 [T,3,H,W].

    Prefers `pano_dir/rgb.mp4` if present AND has enough frames; falls back to PNGs.
    """
    mp4 = pano_dir / "rgb.mp4"
    if mp4.is_file():
        try:
            import decord
            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(str(mp4), width=W, height=H)
            indices = list(range(t0, t0 + T))
            if max(indices) < len(vr):
                frames = vr.get_batch(indices).asnumpy()
                return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        except Exception:
            pass  # fall through to PNG path

    rgb_dir = pano_dir / "rgb"
    paths = sorted(rgb_dir.glob("*.png"))
    if max(t0 + T, t0) > len(paths):
        raise IndexError(f"Need frame {t0+T-1} but dir has only {len(paths)}: {rgb_dir}")
    frames = []
    for i in range(t0, t0 + T):
        bgr = cv2.imread(str(paths[i]), cv2.IMREAD_COLOR)
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    arr = np.stack(frames, axis=0)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


# ── Discovery & sampling ────────────────────────────────────────────────────


def _has_rgb(pano_dir: Path) -> bool:
    return (pano_dir / "rgb.mp4").is_file() or (pano_dir / "rgb").is_dir()


def discover_locations(data_root: Path) -> List[Path]:
    """Find locations with both Pano_00 and Pano_01 RGB sources (mp4 OR PNG dir)."""
    out = []
    for scene in sorted(data_root.iterdir()):
        if not scene.is_dir():
            continue
        for loc in sorted(scene.iterdir()):
            if not loc.is_dir():
                continue
            if _has_rgb(loc / "Pano_00") and _has_rgb(loc / "Pano_01"):
                out.append(loc)
    return out


def sample_pairings(locations: List[Path], num_same: int, num_diff: int,
                    rng: np.random.Generator) -> List[Tuple[Path, str, str]]:
    """Return list of (loc, src_idx, tgt_idx) triples."""
    plan = []
    for _ in range(num_same):
        loc = locations[rng.integers(len(locations))]
        p = "00" if rng.integers(2) == 0 else "01"
        plan.append((loc, p, p))
    for _ in range(num_diff):
        loc = locations[rng.integers(len(locations))]
        if rng.integers(2) == 0:
            plan.append((loc, "00", "01"))
        else:
            plan.append((loc, "01", "00"))
    return plan


# ── Per-sample render ────────────────────────────────────────────────────────


def _downsample_video(equi_uint8: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize [T,3,H,W] uint8 → [T,3,h,w] uint8 via area-average pooling."""
    if equi_uint8.shape[-2:] == (h, w):
        return equi_uint8
    out = F.interpolate(equi_uint8.float(), size=(h, w), mode="area")
    return out.clamp(0, 255).byte()


def render_one(
    loc: Path, src_idx: str, tgt_idx: str, t0: int, num_frames: int,
    equi_h: int, equi_w: int, equi_load_h: int, equi_load_w: int,
    traj_src: dict, traj_tgt: dict,
    pers_h: int, pers_w: int,
    fps: int, device: str, out_path: Path,
):
    # Load at high resolution so the perspective extraction is sharp
    src_full = load_equi_window(loc / f"Pano_{src_idx}", t0, num_frames, equi_load_h, equi_load_w)
    tgt_full = load_equi_window(loc / f"Pano_{tgt_idx}", t0, num_frames, equi_load_h, equi_load_w)

    # Perspective renders sampled from full-res equirect
    src_pers = render_perspective_uint8(
        src_full, traj_src["yaw_deg"], traj_src["pitch_deg"], traj_src["roll_deg"],
        traj_src["fov_h_deg"], pers_h, pers_w, device=device,
    )
    tgt_pers = render_perspective_uint8(
        tgt_full, traj_tgt["yaw_deg"], traj_tgt["pitch_deg"], traj_tgt["roll_deg"],
        traj_tgt["fov_h_deg"], pers_h, pers_w, device=device,
    )

    # Downsample for panorama display (free full-res afterwards)
    src_video = _downsample_video(src_full, equi_h, equi_w); del src_full
    tgt_video = _downsample_video(tgt_full, equi_h, equi_w); del tgt_full

    # Panoramas with red boundary box marking the perspective footprint
    src_boxed = boundary_red_overlay(src_video, traj_src["yaw_deg"], traj_src["pitch_deg"],
                                     traj_src["roll_deg"], traj_src["fov_h_deg"],
                                     device=device)
    tgt_boxed = boundary_red_overlay(tgt_video, traj_tgt["yaw_deg"], traj_tgt["pitch_deg"],
                                     traj_tgt["roll_deg"], traj_tgt["fov_h_deg"],
                                     device=device)

    src_panel = stack_pers_above_pano(src_pers, src_boxed, panel_width=equi_w)
    tgt_panel = stack_pers_above_pano(tgt_pers, tgt_boxed, panel_width=equi_w)

    # Side-by-side with a small white separator
    sep_w = 6
    T_dim, _, H_dim, W_dim = src_panel.shape
    sep = torch.full((T_dim, 3, H_dim, sep_w), 255, dtype=torch.uint8)
    sxs = torch.cat([src_panel, sep, tgt_panel], dim=-1)
    sxs_np = sxs.permute(0, 2, 3, 1).numpy()  # [T, H_total, 2W+sep_w, 3] uint8

    # Per-frame text strip (src centered in left panel, tgt centered in right panel)
    strip_h = 40
    panel_w = W_dim
    total_w = sxs_np.shape[2]
    out_frames = []
    for t in range(num_frames):
        strip = draw_text_strip(
            total_width=total_w, panel_width=panel_w, sep_w=sep_w,
            text_left=fmt_traj("src", traj_src, t),
            text_right=fmt_traj("tgt", traj_tgt, t),
            height=strip_h,
        )
        out_frames.append(np.concatenate([sxs_np[t], strip], axis=0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), out_frames, fps=fps, macro_block_size=2)


# ── Main ────────────────────────────────────────────────────────────────────


def _load_locations_from_file(path: str) -> List[Path]:
    """Read a newline-delimited locations file (same format as dataset.py's)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if _has_rgb(p / "Pano_00") and _has_rgb(p / "Pano_01"):
                out.append(p)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data_root", type=str, default=None,
                     help="walk <data_root>/<scene>/<location>/")
    src.add_argument("--locations_file", type=str, default=None,
                     help="newline-delimited list of location dirs (same as dataset.py)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_same", type=int, default=50)
    p.add_argument("--num_diff", type=int, default=50)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--equi_h", type=int, default=512, help="display height for the panorama panel")
    p.add_argument("--equi_w", type=int, default=1024, help="display width for the panorama panel")
    p.add_argument("--equi_load_h", type=int, default=2048,
                   help="load resolution H for the equirect (used to extract perspective; downsampled to equi_h for display)")
    p.add_argument("--equi_load_w", type=int, default=4096,
                   help="load resolution W (default = native UE 4096)")
    p.add_argument("--pers_h", type=int, default=480, help="perspective render height")
    p.add_argument("--pers_w", type=int, default=832, help="perspective render width")
    p.add_argument("--total_video_frames", type=int, default=240,
                   help="number of frames in each Pano mp4 (used for t0 sampling)")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--min_overlap", type=float, default=0.25,
                   help="min first-frame frustum overlap between src/tgt (0 = unconstrained)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0,
                   help="this process renders only entries where idx %% num_shards == shard_idx")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    if args.locations_file:
        locs = _load_locations_from_file(args.locations_file)
        source_desc = args.locations_file
    else:
        locs = discover_locations(Path(args.data_root))
        source_desc = args.data_root
    if not locs:
        sys.exit(f"No locations with Pano_{{00,01}} RGB from {source_desc}")
    print(f"Found {len(locs)} locations")

    plan = sample_pairings(locs, args.num_same, args.num_diff, rng)
    print(f"Sampled {len(plan)} pairings ({args.num_same} same + {args.num_diff} diff)")

    t_max = args.total_video_frames - args.num_frames
    if t_max < 0:
        sys.exit(f"num_frames {args.num_frames} > total_video_frames {args.total_video_frames}")

    # Precompute per-entry randomness using the shared RNG so all shards see the same plan.
    entries = []
    for i, (loc, src, tgt) in enumerate(plan):
        t0 = int(rng.integers(0, t_max + 1))
        pairing = "same" if src == tgt else "diff"
        # For diff pairings, load c2w + static depth for overlap-direction matching
        c2w_src_t0 = c2w_tgt_t0 = None
        depth_t0 = None
        if pairing == "diff" and args.min_overlap > 0:
            try:
                c2w_src_t0 = torch.load(loc / f"c2w_PanoCam_{src}.pt", map_location="cpu", weights_only=True).float()[t0]
                c2w_tgt_t0 = torch.load(loc / f"c2w_PanoCam_{tgt}.pt", map_location="cpu", weights_only=True).float()[t0]
                # Load static depth at t0 for accurate scene-depth estimation
                from view_transfer_via_query.prepare_data.lift_and_render import load_depth_ue, load_depth
                depth_dir = loc / f"Pano_{src}_static" / "depth"
                if depth_dir.is_dir():
                    dfiles = sorted(f for f in depth_dir.iterdir()
                                    if f.suffix.lower() in (".exr", ".npy", ".pt", ".pth"))
                    if t0 < len(dfiles):
                        dp = str(dfiles[t0])
                        depth_t0 = torch.from_numpy(
                            load_depth_ue(dp) if dp.endswith(".exr") else load_depth(dp))
            except Exception:
                pass  # fall back to heuristic if files missing
        traj_src, traj_tgt = sample_trajectory_pair(
            args.num_frames, pairing=pairing, min_overlap=args.min_overlap,
            pano_c2w_src_at_t0=c2w_src_t0, pano_c2w_tgt_at_t0=c2w_tgt_t0,
            depth_equirect=depth_t0,
            rng=rng,
        )
        kind = "same" if src == tgt else "diff"
        out_name = f"{kind}_{i:03d}_{loc.parent.name}_{loc.name}_{src}{tgt}_t{t0}.mp4"
        entries.append((i, loc, src, tgt, t0, traj_src, traj_tgt, kind, out_name))

    # Render only this shard's entries
    my_entries = [e for e in entries if e[0] % args.num_shards == args.shard_idx]
    print(f"shard {args.shard_idx}/{args.num_shards}: rendering {len(my_entries)} of {len(entries)}")

    for (i, loc, src, tgt, t0, traj_src, traj_tgt, kind, out_name) in tqdm(my_entries):
        out_path = out_dir / out_name
        render_one(
            loc, src, tgt, t0, args.num_frames,
            args.equi_h, args.equi_w, args.equi_load_h, args.equi_load_w,
            traj_src, traj_tgt, args.pers_h, args.pers_w,
            args.fps, args.device, out_path,
        )

    print(f"\n[SHARD {args.shard_idx} DONE] rendered {len(my_entries)} videos")


if __name__ == "__main__":
    main()
