"""Stitch the 6 mp4s in an infer_out folder into a 2x3 grid video.

Layout:
    row 1:  src       rendered   tgt_gt
    row 2:  blob      visibility  pred

Usage:
    python -m view_transfer_via_query.tools.grid_infer_output \\
        /path/to/infer_out/<run>/<sample_dir>  [<sample_dir2> ...] \\
        [--out grid.mp4]   # default: <sample_dir>/grid.mp4 per input
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import imageio.v2 as imageio


GRID_LAYOUT = [
    [("src",        "src.mp4"),       ("rendered",   "rendered.mp4"), ("tgt_gt", "tgt_gt.mp4")],
    [("blob",       "blob.mp4"),      ("visibility", "visibility.mp4"), ("pred",  "pred.mp4")],
]


def _row_label_strip(labels: List[str], cell_w: int, height: int = 30,
                     font_scale: float = 0.6) -> np.ndarray:
    strip = np.full((height, cell_w * len(labels), 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, label in enumerate(labels):
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        x = i * cell_w + (cell_w - tw) // 2
        y = (height + th) // 2
        cv2.putText(strip, label, (x, y), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return strip


def make_grid(sample_dir: Path, out_path: Path, fps: int):
    videos: Dict[str, np.ndarray] = {}
    for row in GRID_LAYOUT:
        for _, fname in row:
            path = sample_dir / fname
            if not path.is_file():
                raise FileNotFoundError(f"Missing {fname} in {sample_dir}")
            r = imageio.get_reader(str(path))
            videos[fname] = np.stack([f for f in r], axis=0)
            r.close()

    shapes = {n: v.shape for n, v in videos.items()}
    Ts = {s[0] for s in shapes.values()}
    Hs = {s[1] for s in shapes.values()}
    Ws = {s[2] for s in shapes.values()}
    if not (len(Ts) == len(Hs) == len(Ws) == 1):
        raise ValueError(f"shape mismatch across mp4s in {sample_dir}: {shapes}")
    T, H, W = Ts.pop(), Hs.pop(), Ws.pop()

    label_h = 30
    label_top = _row_label_strip([l[0] for l in GRID_LAYOUT[0]], cell_w=W, height=label_h)
    label_bot = _row_label_strip([l[0] for l in GRID_LAYOUT[1]], cell_w=W, height=label_h)

    out_frames = []
    for t in range(T):
        row1 = np.concatenate([videos[fname][t] for _, fname in GRID_LAYOUT[0]], axis=1)
        row2 = np.concatenate([videos[fname][t] for _, fname in GRID_LAYOUT[1]], axis=1)
        out_frames.append(np.concatenate([label_top, row1, label_bot, row2], axis=0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), out_frames, fps=fps, macro_block_size=2)
    print(f"[grid] {sample_dir.name} → {out_path}  ({T} frames, {out_frames[0].shape[1]}x{out_frames[0].shape[0]})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_dirs", nargs="+", help="one or more sample folders containing the 6 mp4s")
    p.add_argument("--out", type=str, default=None,
                   help="output path. If single input, this is the file path (default: <input_dir>/grid.mp4). "
                        "If multiple inputs, each gets `<input_dir>/grid.mp4` (--out is ignored).")
    p.add_argument("--fps", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    inputs = [Path(d) for d in args.input_dirs]
    for d in inputs:
        if not d.is_dir():
            raise NotADirectoryError(d)

    if len(inputs) == 1 and args.out is not None:
        make_grid(inputs[0], Path(args.out), fps=args.fps)
    else:
        for d in inputs:
            make_grid(d, d / "grid.mp4", fps=args.fps)


if __name__ == "__main__":
    main()
