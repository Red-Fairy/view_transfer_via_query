#!/usr/bin/env python3
"""Emit a sweep-spec JSON for guidance_sweep_manager.py.

phase1 : geom×src grid (text fixed=3) + monolithic diagonal (k,k,k), on the
         fixed 4-location subset, seed=0.
phase2 : explicit --configs on the full 12-location test split (finalists +
         text ablation), seed=0 — same seed ⇒ identical samples as phase1.

    python scripts/make_sweep_spec.py phase1 -o scripts/sweep_spec_phase1.json
    python scripts/make_sweep_spec.py phase2 --configs 5:2:3 6:2:3 5:2:1 5:2:5 \
        -o scripts/sweep_spec_phase2.json
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST12 = "/share/ma/scratch/rundong/Unreal_Projects/split_files/view_transfer_0507/test.txt"
LOC4 = str(HERE / "sweep_loc4.txt")

GEOM = [2, 3, 4, 5, 6, 7, 8]
SRC = [1, 1.5, 2, 3, 4]
TEXT_FIXED = 3
DIAG = [1, 2, 3, 4, 5]   # (k,k,k) ≡ monolithic CFG at scale k


def fmt(x):  # 3.0 -> "3", 1.5 -> "1.5" (must match infer.sh OUT_DIR tag)
    return str(int(x)) if float(x).is_integer() else str(x)


def build_phase1():
    seen, configs = set(), []
    for g in GEOM:
        for s in SRC:
            key = (fmt(g), fmt(s), fmt(TEXT_FIXED))
            if key not in seen:
                seen.add(key)
                configs.append({"geom": key[0], "src": key[1], "text": key[2]})
    for k in DIAG:
        key = (fmt(k), fmt(k), fmt(k))
        if key not in seen:
            seen.add(key)
            configs.append({"geom": key[0], "src": key[1], "text": key[2]})
    return {"phase_tag": "sweep1_4loc", "locations_file": LOC4,
            "seed": 0, "n_expected": 4, "configs": configs}


def build_phase2(triplets):
    configs, seen = [], set()
    for trip in triplets:
        g, s, t = trip.split(":")
        key = (fmt(float(g)), fmt(float(s)), fmt(float(t)))
        if key not in seen:
            seen.add(key)
            configs.append({"geom": key[0], "src": key[1], "text": key[2]})
    return {"phase_tag": "sweep2_full12", "locations_file": TEST12,
            "seed": 0, "n_expected": 12, "configs": configs}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["phase1", "phase2"])
    p.add_argument("--configs", nargs="+", default=[],
                   help="phase2 only: GEOM:SRC:TEXT triplets")
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    spec = build_phase1() if a.phase == "phase1" else build_phase2(a.configs)
    Path(a.out).write_text(json.dumps(spec, indent=2))
    print(f"{a.phase}: {len(spec['configs'])} configs -> {a.out} "
          f"(phase_tag={spec['phase_tag']}, n_expected={spec['n_expected']})")
