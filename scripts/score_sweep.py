#!/usr/bin/env python3
"""Rank guidance-sweep configs by pred-vs-GT fidelity.

For every config dir under a phase dir, pairs each sample's pred.mp4 with its
tgt_gt.mp4, computes per-frame PSNR / SSIM / LPIPS (torchmetrics), averages
over frames then over samples, and writes a leaderboard sorted by LPIPS
(lower = better). Run on a GPU node for speed; CPU works but is slow.

    python scripts/score_sweep.py \
        --phase-dir infer_out/14B_4gpu_640P_0507/sweep1_4loc \
        --device cuda
"""
import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def load_video(path: Path) -> torch.Tensor:
    """mp4 → [T, 3, H, W] float in [0, 1]."""
    r = imageio.get_reader(str(path))
    frames = np.stack([f for f in r], axis=0)  # [T, H, W, 3] uint8
    r.close()
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0


@torch.no_grad()
def score_pair(pred: torch.Tensor, gt: torch.Tensor, metrics, device) -> dict:
    T = min(pred.shape[0], gt.shape[0])
    pred, gt = pred[:T].to(device), gt[:T].to(device)
    psnr, ssim, lpips = metrics
    psnr.reset(); ssim.reset(); lpips.reset()
    psnr.update(pred, gt)
    ssim.update(pred, gt)
    lpips.update(pred, gt)  # normalize=True → expects [0,1]
    return {"psnr": float(psnr.compute()),
            "ssim": float(ssim.compute()),
            "lpips": float(lpips.compute())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    device = torch.device(a.device)
    metrics = (
        PeakSignalNoiseRatio(data_range=1.0).to(device),
        StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device),
    )

    phase = Path(a.phase_dir)
    rows = []
    for cfg_dir in sorted(d for d in phase.iterdir() if d.is_dir() and d.name.startswith("geom")):
        samples = [s for s in sorted(cfg_dir.iterdir())
                   if (s / "pred.mp4").exists() and (s / "tgt_gt.mp4").exists()]
        if not samples:
            print(f"  [skip] {cfg_dir.name}: no scorable samples")
            continue
        per = []
        for s in samples:
            try:
                m = score_pair(load_video(s / "pred.mp4"),
                               load_video(s / "tgt_gt.mp4"), metrics, device)
                per.append(m)
            except Exception as e:
                print(f"  [warn] {cfg_dir.name}/{s.name}: {e}")
        if not per:
            continue
        agg = {k: float(np.mean([d[k] for d in per])) for k in ("psnr", "ssim", "lpips")}
        agg.update(config=cfg_dir.name, n=len(per))
        rows.append(agg)
        print(f"  {cfg_dir.name:32s} n={len(per)}  "
              f"PSNR={agg['psnr']:.2f}  SSIM={agg['ssim']:.3f}  LPIPS={agg['lpips']:.4f}")

    rows.sort(key=lambda r: r["lpips"])  # lower LPIPS = better
    out_csv = phase / "leaderboard.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "n", "lpips", "psnr", "ssim"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    (phase / "leaderboard.json").write_text(json.dumps(rows, indent=2))
    print(f"\nLeaderboard ({len(rows)} configs) → {out_csv}")
    for i, r in enumerate(rows[:5]):
        print(f"  #{i+1} {r['config']}  LPIPS={r['lpips']:.4f}  "
              f"PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.3f}")


if __name__ == "__main__":
    main()
