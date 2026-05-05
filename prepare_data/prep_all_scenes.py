"""Batch data prep across all scenes under a root.

Walks `data_root` (e.g. `outputs_non_arranged_cars_v2/`), matches each scene folder
against keys in `prompts_json` (longest-prefix match, e.g. `Desert_0car_2-8pp_task200`
→ `Desert`), and runs `run_prep.py` operations for every location:

  1. cameras  : parse camera_params.json → c2w_PanoCam_NN.pt (no GPU)
  2. text     : load UMT5-XXL once, encode each unique prompt, write text_emb.pt
                to all locations of that scene

Usage:
    python -m view_transfer_via_query.prepare_data.prep_all_scenes \\
        --data_root /share/.../outputs_non_arranged_cars_v2 \\
        --prompts_json /share/.../scene_prompts.json \\
        --t5_ckpt /home/.../models_t5_umt5-xxl-enc-bf16.pth \\
        --t5_tokenizer google/umt5-xxl

Skip a stage with --skip_cameras / --skip_text. Force re-run with --overwrite.
"""

from __future__ import annotations

# Bootstrap: lets `python prepare_data/prep_all_scenes.py …` work in addition to
# the canonical `python -m view_transfer_via_query.prepare_data.prep_all_scenes …`.
if __name__ == "__main__" and __package__ in (None, ""):
    import os as _os, sys as _sys
    _diffsynth_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if _diffsynth_root not in _sys.path:
        _sys.path.insert(0, _diffsynth_root)
    __package__ = "view_transfer_via_query.prepare_data"

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .parse_cameras import parse_camera_params


# ── Helpers ─────────────────────────────────────────────────────────────────


def find_scene_prompt(folder_name: str, prompts: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Match `folder_name` to a prompt key. Returns (key, prompt) or None.

    Match rule: longest key K such that folder_name == K or folder_name starts with K + "_".
    Sorting by length-desc ensures e.g. `RuralAustralia_Wood_...` matches `RuralAustralia`
    rather than (hypothetical) `Rural`.
    """
    for k in sorted(prompts.keys(), key=len, reverse=True):
        if folder_name == k or folder_name.startswith(k + "_"):
            return k, prompts[k]
    return None


def list_locations(scene_dir: Path) -> List[Path]:
    """A location is any immediate subdir of a scene that has camera_params.json."""
    return sorted(
        p for p in scene_dir.iterdir()
        if p.is_dir() and (p / "camera_params.json").is_file()
    )


# ── Stages ──────────────────────────────────────────────────────────────────


def run_cameras(location_dir: Path, overwrite: bool) -> str:
    out_files = [location_dir / f"c2w_PanoCam_{i:02d}.pt" for i in (0, 1)]
    if not overwrite and all(p.exists() for p in out_files):
        return "skip"
    parsed = parse_camera_params(str(location_dir / "camera_params.json"))
    for cam_name, cam in parsed["cameras"].items():
        torch.save(
            torch.from_numpy(cam["c2w"]).float().contiguous(),
            location_dir / f"c2w_{cam_name}.pt",
        )
    return "wrote"


def run_text_for_scene(
    encoder, tokenizer, prompt: str, locations: List[Path],
    device: str, overwrite: bool, prompt_cache: Dict[str, torch.Tensor],
) -> Tuple[torch.Size, int]:
    """Encode prompt once (cached), write text_emb.pt to all `locations`."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from view_transfer_via_query.prepare_data.encode_text import encode_text

    if prompt not in prompt_cache:
        prompt_cache[prompt] = encode_text(encoder, tokenizer, prompt, device=device).contiguous()
    emb = prompt_cache[prompt]

    written = 0
    for loc in locations:
        out = loc / "text_emb.pt"
        if out.exists() and not overwrite:
            continue
        torch.save(emb, out)
        written += 1
    return emb.shape, written


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="root containing scene folders (e.g. outputs_non_arranged_cars_v2)")
    p.add_argument("--prompts_json", required=True,
                   help="JSON dict {scene_name: prompt_string}")
    p.add_argument("--t5_ckpt", default='../models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.safetensors',
                   help="UMT5-XXL checkpoint path (required unless --skip_text)")
    p.add_argument("--t5_tokenizer", default="google/umt5-xxl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_cameras", action="store_true")
    p.add_argument("--skip_text", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite existing c2w_*.pt / text_emb.pt files")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        sys.exit(f"data_root not a directory: {data_root}")

    prompts: Dict[str, str] = json.loads(Path(args.prompts_json).read_text())

    # 1. Discover scenes + prompt match + locations
    print(f"Scanning {data_root}...")
    scenes = sorted(p for p in data_root.iterdir() if p.is_dir())

    plan: List[Tuple[Path, str, str, List[Path]]] = []
    for scene in scenes:
        match = find_scene_prompt(scene.name, prompts)
        if match is None:
            print(f"  [SKIP] {scene.name}: no prompt key matches")
            continue
        key, prompt = match
        locs = list_locations(scene)
        if not locs:
            print(f"  [SKIP] {scene.name} (matched '{key}'): no locations with camera_params.json")
            continue
        plan.append((scene, key, prompt, locs))
        print(f"  [PLAN] {scene.name} → '{key}' ({len(locs)} locs)")

    if not plan:
        sys.exit("\nNothing to prep.")

    total_locs = sum(len(locs) for _, _, _, locs in plan)
    print(f"\n{len(plan)} scenes, {total_locs} locations total")

    # 2. Cameras pass — fast, no GPU
    if not args.skip_cameras:
        print("\n=== Stage 1: cameras ===")
        wrote, skipped = 0, 0
        for scene, key, _, locs in plan:
            for loc in locs:
                try:
                    status = run_cameras(loc, overwrite=args.overwrite)
                except Exception as e:
                    print(f"  [ERR ] {scene.name}/{loc.name}: {e}")
                    continue
                if status == "wrote":
                    wrote += 1
                else:
                    skipped += 1
        print(f"  → {wrote} written, {skipped} already up-to-date")

    # 3. Text pass — load UMT5 once
    if not args.skip_text:
        if args.t5_ckpt is None:
            sys.exit("--t5_ckpt is required unless --skip_text")
        print("\n=== Stage 2: text (UMT5-XXL) ===")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from view_transfer_via_query.prepare_data.encode_text import load_wan_text_encoder

        print(f"  Loading T5 from {args.t5_ckpt}...")
        encoder, tokenizer = load_wan_text_encoder(
            encoder_ckpt=args.t5_ckpt,
            tokenizer_name_or_path=args.t5_tokenizer,
            device=args.device,
            dtype=torch.bfloat16,
        )

        prompt_cache: Dict[str, torch.Tensor] = {}
        for scene, key, prompt, locs in plan:
            shape, written = run_text_for_scene(
                encoder, tokenizer, prompt, locs,
                device=args.device, overwrite=args.overwrite, prompt_cache=prompt_cache,
            )
            print(f"  {scene.name} → '{key}' shape={tuple(shape)}  "
                  f"({written}/{len(locs)} written)")

        # Free GPU memory
        del encoder, tokenizer
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print(f"  → encoded {len(prompt_cache)} unique prompts")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
