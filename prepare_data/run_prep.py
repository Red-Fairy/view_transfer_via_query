"""Offline data preparation orchestrator (slim version).

Now that perspective extraction + lift-and-render + VAE encoding all run online inside
gpu_preprocess, this script only handles the truly-offline pieces:

  1. Camera parsing  : UE camera_params.json → c2w_PanoCam_NN.pt (OpenCV, meters)
  2. Text encoding   : prompt string → text_emb.pt  (UMT5-XXL embedding)
  3. (Stub) manifest : verify presence of user-provided blob videos
                       so training fails fast on missing data.

The user separately produces gaussian-agent blob videos in equirect, stored under each
target pano's directory:
  • Pano_00/blobs/*.png   — blobs rendered at Pano_00's camera positions
  • Pano_01/blobs/*.png   — blobs rendered at Pano_01's camera positions

(There is no source dependence — a blob video is determined by the target trajectory.)

Per-location output layout after this script:

    location_dir/
        c2w_PanoCam_00.pt       [T_video, 4, 4]   OpenCV
        c2w_PanoCam_01.pt
        text_emb.pt             [L, 4096]
        Pano_00/, Pano_00_static/, Pano_01/, Pano_01_static/    (already exists)
        Pano_00/blobs/, Pano_01/blobs/                          (user-supplied)

Usage:
    # Per-location
    python -m view_transfer_via_query.prepare_data.run_prep cameras \\
        --location_dir /path/to/scene/.../x_y_s_..._n_p_p

    # Per-scene (encode T5 once and reuse for all its locations)
    python -m view_transfer_via_query.prepare_data.run_prep text \\
        --location_dir /path/to/scene/.../x_y_s_..._n_p_p \\
        --prompt "Desert landscape with sand dunes and rocks" \\
        --t5_ckpt /path/to/models_t5_umt5-xxl-enc-bf16.pth \\
        --t5_tokenizer google/umt5-xxl

    # Verify a location is training-ready
    python -m view_transfer_via_query.prepare_data.run_prep verify \\
        --location_dir /path/to/.../x_y_s_..._n_p_p
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

from .parse_cameras import parse_camera_params, save_camera


# ── Subcommand: cameras ─────────────────────────────────────────────────────


def cmd_cameras(args):
    cam_json = os.path.join(args.location_dir, "camera_params.json")
    parsed = parse_camera_params(cam_json)
    out_dir = Path(args.location_dir)
    for cam_name, cam in parsed["cameras"].items():
        # Save c2w + intrinsics next to the panorama directories
        torch.save(
            torch.from_numpy(cam["c2w"]).float().contiguous(),
            out_dir / f"c2w_{cam_name}.pt",
        )
    print(f"Saved c2w_*.pt for {list(parsed['cameras'].keys())} → {out_dir}")


# ── Subcommand: text ────────────────────────────────────────────────────────


def cmd_text(args):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from view_transfer_via_query.prepare_data.encode_text import (
        load_wan_text_encoder,
        encode_text,
    )

    encoder, tokenizer = load_wan_text_encoder(
        encoder_ckpt=args.t5_ckpt,
        tokenizer_name_or_path=args.t5_tokenizer,
        device=args.device,
        dtype=torch.bfloat16,
    )
    emb = encode_text(encoder, tokenizer, args.prompt, device=args.device)
    out_path = os.path.join(args.location_dir, "text_emb.pt")
    torch.save(emb.contiguous(), out_path)
    print(f"Saved text_emb.pt (shape {tuple(emb.shape)}) → {out_path}")


# ── Subcommand: verify ──────────────────────────────────────────────────────


REQUIRED_FILES = ["c2w_PanoCam_00.pt", "c2w_PanoCam_01.pt", "text_emb.pt"]


def cmd_verify(args):
    """Use dataset's own discovery to count training entries this location yields."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from view_transfer_via_query.dataset import SampleEntry

    loc = Path(args.location_dir)
    print(f"Verifying {loc}:")

    # Top-level prep files
    missing_files = [f for f in REQUIRED_FILES if not (loc / f).is_file()]
    if missing_files:
        print("  Missing prep files:")
        for f in missing_files:
            print(f"    {f}")

    # Try every (src, tgt) pairing and report which are complete
    print("  Pairings:")
    valid = 0
    for s in ("00", "01"):
        for t in ("00", "01"):
            e = SampleEntry(location_dir=str(loc), src_idx=s, tgt_idx=t)
            ok = e.is_complete()
            print(f"    src={s} tgt={t}: {'OK' if ok else 'incomplete'}")
            if not ok:
                # Show why
                checks = {
                    "rgb_src": e.rgb_src_dir,
                    "rgb_tgt": e.rgb_tgt_dir,
                    "static_rgb": e.static_rgb_dir,
                    "static_depth": e.static_depth_dir,
                    "blob": e.blob_dir,
                    "c2w_src": e.c2w_src_path,
                    "c2w_tgt": e.c2w_tgt_path,
                    "text_emb": e.text_emb_path,
                }
                for name, path in checks.items():
                    if not (os.path.isdir(path) or os.path.exists(path)):
                        print(f"        missing {name}: {path}")
            else:
                valid += 1

    print(f"  → {valid} valid pairing(s)")
    if missing_files or valid == 0:
        sys.exit(1)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_cam = sub.add_parser("cameras", help="Parse UE camera_params.json")
    p_cam.add_argument("--location_dir", required=True)
    p_cam.set_defaults(fn=cmd_cameras)

    p_txt = sub.add_parser("text", help="Encode a prompt with UMT5-XXL → text_emb.pt")
    p_txt.add_argument("--location_dir", required=True)
    p_txt.add_argument("--prompt", required=True)
    p_txt.add_argument("--t5_ckpt", required=True)
    p_txt.add_argument("--t5_tokenizer", default="google/umt5-xxl")
    p_txt.add_argument("--device", default="cuda")
    p_txt.set_defaults(fn=cmd_text)

    p_ver = sub.add_parser("verify", help="Check a location has all required files")
    p_ver.add_argument("--location_dir", required=True)
    p_ver.set_defaults(fn=cmd_verify)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
