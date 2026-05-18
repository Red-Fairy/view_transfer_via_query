"""Visualize the cubemap-mesh lift+render condition vs. ground-truth components.

Same 2x2 layout as `viz_trajectories.py`, only the **top-right** cell changes:
instead of extracting the target perspective from the dynamic target 360,
we rasterize a cubemap mesh built from the SOURCE static (no-agent) panorama
at frame t0, anchored at the source pano camera position at t0, into the
target perspective trajectory.

Layout per frame:
  [ src perspective (extracted)         ]  [ tgt perspective (mesh-rendered)  ]
  [ src 360 + red bbox                   ]  [ tgt 360 + red bbox                ]
  [ text strip:  src fov/y/p/r  |  tgt fov/y/p/r (mesh)                        ]

Requires `nvdiffrast` and a CUDA GPU (the mesh rasterizer is GPU-only).

Usage:
    python -m view_transfer_via_query.tools.viz_mesh_render \\
        --data_root /share/.../outputs_non_arranged_24fps \\
        --out_dir   ./viz_mesh_render \\
        --num_same 50 --num_diff 50 \\
        --num_frames 81 \\
        --equi_h 512 --equi_w 1024 \\
        --equi_load_h 2048 --equi_load_w 4096 \\
        --pers_h 480 --pers_w 832 \\
        --mesh_face_res 1024 \\
        --src_idx 00 --tgt_idx 01 \\
        --seed 0

Sharded across 4 GPUs (same pattern as viz_4gpu.sh):
    --num_shards 4 --shard_idx {0..3}
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import imageio.v2 as imageio
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from view_transfer_via_query.prepare_data.extract_perspectives import (
    sample_trajectory_pair,
    yaw_pitch_roll_to_R,
    compose_perspective_c2w,
    fov_to_intrinsics,
)
from view_transfer_via_query.prepare_data.lift_and_render import (
    load_depth_ue,
    load_depth,
    build_cubemap_mesh_world,
    render_mesh_to_perspective,
)
# Reuse helpers from viz_trajectories so the unchanged 3 cells look identical.
from view_transfer_via_query.tools.viz_trajectories import (
    boundary_red_overlay,
    render_perspective_uint8,
    stack_pers_above_pano,
    draw_text_strip,
    fmt_traj,
    load_equi_window,
    _downsample_video,
    _has_rgb,
    discover_locations,
    sample_pairings,
    _load_locations_from_file,
)


# ── Static frame loaders ────────────────────────────────────────────────────


def load_static_rgb_t0(
    pano_static_dir: Path, t0: int, equi_load_h: int, equi_load_w: int,
) -> torch.Tensor:
    """[3, H, W] uint8 — single static-RGB frame at index t0, resized to (H, W)."""
    rgb_dir = pano_static_dir / "rgb"
    paths = sorted(rgb_dir.glob("*.png"))
    if t0 >= len(paths):
        raise IndexError(f"t0={t0} but {len(paths)} static RGB frames in {rgb_dir}")
    bgr = cv2.imread(str(paths[t0]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Failed to read {paths[t0]}")
    bgr = cv2.resize(bgr, (equi_load_w, equi_load_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def load_static_depth_t0(
    pano_static_dir: Path, t0: int, equi_load_h: int, equi_load_w: int,
) -> torch.Tensor:
    """[H, W] float32 — single static-depth frame at index t0 in meters (sky filled)."""
    depth_dir = pano_static_dir / "depth"
    files = sorted(f for f in depth_dir.iterdir() if f.suffix.lower() in (".exr", ".npy", ".pt", ".pth"))
    if t0 >= len(files):
        raise IndexError(f"t0={t0} but {len(files)} static depth frames in {depth_dir}")
    path = str(files[t0])
    if path.endswith(".exr"):
        depth_np = load_depth_ue(path)  # already meters; sky filled with 1e4
    else:
        from view_transfer_via_query.prepare_data.lift_and_render import load_depth as _ld
        depth_np = _ld(path)
    if depth_np.shape != (equi_load_h, equi_load_w):
        depth_np = cv2.resize(depth_np, (equi_load_w, equi_load_h), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(depth_np)


# ── Mesh-rendered target perspective ────────────────────────────────────────


def _add_red_border(frames_uint8: torch.Tensor, border_w: int = 3) -> torch.Tensor:
    """[T, 3, H, W] uint8 → same shape with a red 1-pixel-wide-or-more border."""
    out = frames_uint8.clone()
    out[:, 0, :border_w, :] = 255
    out[:, 1:, :border_w, :] = 0
    out[:, 0, -border_w:, :] = 255
    out[:, 1:, -border_w:, :] = 0
    out[:, 0, :, :border_w] = 255
    out[:, 1:, :, :border_w] = 0
    out[:, 0, :, -border_w:] = 255
    out[:, 1:, :, -border_w:] = 0
    return out


def render_target_pers_from_mesh(
    static_rgb_uint8: torch.Tensor,    # [3, He, We] uint8
    static_depth: torch.Tensor,        # [He, We] float32 meters
    pano_c2w_src_at_t0: torch.Tensor,  # [4, 4] OpenCV (mesh anchor)
    pano_c2w_tgt_window: torch.Tensor, # [T, 4, 4] OpenCV
    traj_tgt: dict,
    pers_h: int, pers_w: int,
    mesh_face_res: int,
    device: str,
) -> torch.Tensor:
    """Build a cubemap mesh from the source-static frame and rasterize to the
    target perspective trajectory.  Returns [T, 3, pers_h, pers_w] uint8 (with
    a red 3-pixel border so the cell visually reads as 'a perspective view')."""
    static_rgb_f = static_rgb_uint8.to(device).float() / 255.0
    static_depth_f = static_depth.to(device).float()
    src_c2w_at_t0_d = pano_c2w_src_at_t0.to(device).float()
    pano_c2w_tgt_d = pano_c2w_tgt_window.to(device).float()

    # Build mesh in WORLD frame.
    verts, faces, colors = build_cubemap_mesh_world(
        static_rgb_f, static_depth_f, src_c2w_at_t0_d,
        face_res=mesh_face_res,
    )

    # Compose target perspective c2w trajectory + intrinsics.
    tgt_yaw_t = torch.from_numpy(np.asarray(traj_tgt["yaw_deg"], dtype=np.float32))
    tgt_pitch_t = torch.from_numpy(np.asarray(traj_tgt["pitch_deg"], dtype=np.float32))
    tgt_roll_t = torch.from_numpy(np.asarray(traj_tgt["roll_deg"], dtype=np.float32))
    tgt_R = yaw_pitch_roll_to_R(tgt_yaw_t, tgt_pitch_t, tgt_roll_t).to(device)  # [T, 3, 3]
    tgt_pers_c2w = compose_perspective_c2w(pano_c2w_tgt_d, tgt_R)               # [T, 4, 4]
    fx, fy, cx, cy = fov_to_intrinsics(float(traj_tgt["fov_h_deg"]), pers_h, pers_w)
    K = torch.tensor([fx, fy, cx, cy], device=device, dtype=torch.float32)

    rendered, _ = render_mesh_to_perspective(
        verts, faces, colors, tgt_pers_c2w, K,
        pers_h=pers_h, pers_w=pers_w, backface_cull=True,
    )  # [T, 3, pers_h, pers_w] float in [0, 1]

    out = (rendered * 255.0).clamp(0, 255).to(torch.uint8).cpu()
    return _add_red_border(out, border_w=3)


# ── Per-sample render ────────────────────────────────────────────────────────


def render_one_mesh(
    loc: Path, src_idx: str, tgt_idx: str, t0: int, num_frames: int,
    equi_h: int, equi_w: int, equi_load_h: int, equi_load_w: int,
    traj_src: dict, traj_tgt: dict,
    pers_h: int, pers_w: int, mesh_face_res: int,
    fps: int, device: str, out_path: Path,
):
    # 1. 360 RGB windows (source for top-left perspective + bbox; target for tgt bbox)
    src_video = load_equi_window(loc / f"Pano_{src_idx}", t0, num_frames, equi_load_h, equi_load_w)
    tgt_video = load_equi_window(loc / f"Pano_{tgt_idx}", t0, num_frames, equi_load_h, equi_load_w)

    # 2. Source static at t0 (mesh source)
    static_rgb = load_static_rgb_t0(loc / f"Pano_{src_idx}_static", t0, equi_load_h, equi_load_w)
    static_depth = load_static_depth_t0(loc / f"Pano_{src_idx}_static", t0, equi_load_h, equi_load_w)

    # 3. Camera c2w windows
    pano_c2w_src = torch.load(loc / f"c2w_PanoCam_{src_idx}.pt", map_location="cpu", weights_only=True).float()
    pano_c2w_tgt = torch.load(loc / f"c2w_PanoCam_{tgt_idx}.pt", map_location="cpu", weights_only=True).float()
    pano_c2w_src_t0 = pano_c2w_src[t0]
    pano_c2w_tgt_window = pano_c2w_tgt[t0:t0 + num_frames]

    # 4. Top-right: mesh-rendered target perspective (NEW)
    mesh_pers = render_target_pers_from_mesh(
        static_rgb, static_depth, pano_c2w_src_t0, pano_c2w_tgt_window,
        traj_tgt, pers_h, pers_w, mesh_face_res, device,
    )

    # 5. Top-left: src perspective extracted from dynamic source 360 (unchanged)
    src_pers = render_perspective_uint8(
        src_video,
        traj_src["yaw_deg"], traj_src["pitch_deg"], traj_src["roll_deg"],
        traj_src["fov_h_deg"], pers_h, pers_w, device=device,
    )

    # 6. Downsample 360s for bbox display, then add red bbox overlays
    src_video_disp = _downsample_video(src_video, equi_h, equi_w); del src_video
    tgt_video_disp = _downsample_video(tgt_video, equi_h, equi_w); del tgt_video

    src_boxed = boundary_red_overlay(
        src_video_disp,
        traj_src["yaw_deg"], traj_src["pitch_deg"], traj_src["roll_deg"],
        traj_src["fov_h_deg"], device=device,
    )
    tgt_boxed = boundary_red_overlay(
        tgt_video_disp,
        traj_tgt["yaw_deg"], traj_tgt["pitch_deg"], traj_tgt["roll_deg"],
        traj_tgt["fov_h_deg"], device=device,
    )

    # 7. Compose 2-column panel (top = perspective, bottom = panorama-with-bbox)
    src_panel = stack_pers_above_pano(src_pers, src_boxed, panel_width=equi_w)
    tgt_panel = stack_pers_above_pano(mesh_pers, tgt_boxed, panel_width=equi_w)

    # Side-by-side with separator
    sep_w = 6
    T_dim, _, H_dim, W_dim = src_panel.shape
    sep = torch.full((T_dim, 3, H_dim, sep_w), 255, dtype=torch.uint8)
    sxs = torch.cat([src_panel, sep, tgt_panel], dim=-1)
    sxs_np = sxs.permute(0, 2, 3, 1).numpy()

    # 8. Per-frame text strip (label tgt as MESH-RENDERED)
    strip_h = 40
    panel_w = W_dim
    total_w = sxs_np.shape[2]
    out_frames = []
    for t in range(num_frames):
        strip = draw_text_strip(
            total_width=total_w, panel_width=panel_w, sep_w=sep_w,
            text_left=fmt_traj("src",          traj_src, t),
            text_right=fmt_traj("tgt (mesh)",  traj_tgt, t),
            height=strip_h,
        )
        out_frames.append(np.concatenate([sxs_np[t], strip], axis=0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), out_frames, fps=fps, macro_block_size=2)


# ── Main ────────────────────────────────────────────────────────────────────


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
                   help="load resolution H for the equirect (used for perspective extraction + mesh)")
    p.add_argument("--equi_load_w", type=int, default=4096)
    p.add_argument("--pers_h", type=int, default=480)
    p.add_argument("--pers_w", type=int, default=832)
    p.add_argument("--mesh_face_res", type=int, default=1024,
                   help="cube face resolution for the lift mesh (default 1024)")

    p.add_argument("--src_idx", type=str, default="00")
    p.add_argument("--tgt_idx", type=str, default="01")
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

    # Filter locations to those that ALSO have static + c2w pt files (needed for mesh).
    def _ok(loc: Path) -> bool:
        return (
            (loc / f"Pano_{args.src_idx}_static" / "rgb").is_dir()
            and (loc / f"Pano_{args.src_idx}_static" / "depth").is_dir()
            and (loc / f"c2w_PanoCam_{args.src_idx}.pt").is_file()
            and (loc / f"c2w_PanoCam_{args.tgt_idx}.pt").is_file()
        )
    locs_ok = [p for p in locs if _ok(p)]
    if len(locs_ok) < len(locs):
        print(f"  → {len(locs_ok)} have src-static + c2w_*.pt; skipping {len(locs) - len(locs_ok)} incomplete")

    plan = sample_pairings(locs_ok, args.num_same, args.num_diff, rng)
    print(f"Sampled {len(plan)} pairings ({args.num_same} same + {args.num_diff} diff)")

    # Cache per-location actual frame count from c2w_PanoCam_{src}.pt so we never
    # sample t0 past the shortest video (some panos are < 240 frames).
    _t_full_cache: dict = {}
    def _t_full_for(loc: Path, src_idx: str) -> int:
        key = (str(loc), src_idx)
        if key not in _t_full_cache:
            c2w = torch.load(loc / f"c2w_PanoCam_{src_idx}.pt", map_location="cpu", weights_only=True)
            _t_full_cache[key] = int(c2w.shape[0])
        return _t_full_cache[key]

    # Precompute per-entry randomness so all shards see the same plan
    entries = []
    for i, (loc, src_orig, tgt_orig) in enumerate(plan):
        # If user overrode src/tgt_idx, ignore the sampled pairing's src/tgt and use args.
        # (sample_pairings always picks from {("00","00"), ("00","01"), ("01","00"), ("01","01")}
        #  but the mesh viz really only uses (args.src_idx → args.tgt_idx).  Keep the
        #  same/diff distinction but enforce src/tgt indexing via the user's defaults.)
        if src_orig == tgt_orig:
            src, tgt = args.src_idx, args.src_idx          # same-pano variant of args.src_idx
        else:
            src, tgt = args.src_idx, args.tgt_idx          # cross-pano with user-chosen direction
        t_full = min(_t_full_for(loc, src), _t_full_for(loc, tgt))
        if t_full < args.num_frames:
            print(f"  [skip-entry {i}] {loc.name}: only {t_full} frames < num_frames={args.num_frames}")
            continue
        t0 = int(rng.integers(0, t_full - args.num_frames + 1))
        pairing = "same" if src == tgt else "diff"
        # For diff pairings, load c2w + static depth at t0 for overlap-direction matching.
        c2w_src_t0 = c2w_tgt_t0 = None
        depth_t0 = None
        if pairing == "diff" and args.min_overlap > 0:
            try:
                c2w_src_t0 = torch.load(loc / f"c2w_PanoCam_{src}.pt",
                                        map_location="cpu", weights_only=True).float()[t0]
                c2w_tgt_t0 = torch.load(loc / f"c2w_PanoCam_{tgt}.pt",
                                        map_location="cpu", weights_only=True).float()[t0]
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
        kind = pairing
        out_name = f"{kind}_{i:03d}_{loc.parent.name}_{loc.name}_{src}{tgt}_t{t0}.mp4"
        entries.append((i, loc, src, tgt, t0, traj_src, traj_tgt, kind, out_name))

    my_entries = [e for e in entries if e[0] % args.num_shards == args.shard_idx]
    print(f"shard {args.shard_idx}/{args.num_shards}: rendering {len(my_entries)} of {len(entries)}")

    for (i, loc, src, tgt, t0, traj_src, traj_tgt, kind, out_name) in tqdm(my_entries):
        out_path = out_dir / out_name
        render_one_mesh(
            loc, src, tgt, t0, args.num_frames,
            args.equi_h, args.equi_w, args.equi_load_h, args.equi_load_w,
            traj_src, traj_tgt, args.pers_h, args.pers_w, args.mesh_face_res,
            args.fps, args.device, out_path,
        )

    print(f"\n[SHARD {args.shard_idx} DONE] rendered {len(my_entries)} videos")


if __name__ == "__main__":
    main()
