"""Gather all location dirs under one or more data roots that pass `SampleEntry.is_complete()`,
and write a shuffled train/test split.

For each `<data_root>/<scene>/<location>/`, checks every (src,tgt) pano pairing, keeps
locations whose required dirs/files all exist. Duplicate location dirs across roots are
deduplicated. The deduplicated list is shuffled with `--seed` and split at `--train_ratio`,
writing `train.txt` and `test.txt` into `--output` (a directory).

Split is per-location, not per-scene; the same scene may appear in both train and test.

Usage:
    python -m view_transfer_via_query.prepare_data.gather_locations \
        --data_roots  /path/to/outputs_non_arranged_24fps /path/to/another_root \
        --output      /path/to/data/                      \
        --train_ratio 0.95
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from view_transfer_via_query.dataset import SampleEntry, discover_entries


def _scene_breakdown(locations: list[str]) -> dict[str, int]:
    by_scene: dict[str, int] = defaultdict(int)
    for loc in locations:
        scene = os.path.basename(os.path.dirname(loc))
        by_scene[scene] += 1
    return by_scene


def _print_breakdown(label: str, locs: list[str]) -> None:
    print(f"{label}: {len(locs)} locations")
    for scene, n in sorted(_scene_breakdown(locs).items()):
        print(f"  {scene}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_roots", nargs="+", required=True)
    ap.add_argument("--output", required=True,
                    help="Output directory; train.txt + test.txt are written inside.")
    ap.add_argument("--train_ratio", type=float, default=0.95,
                    help="Fraction of locations assigned to train; rest go to test.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the shuffle (so the split is reproducible).")
    args = ap.parse_args()

    if not (0.0 < args.train_ratio <= 1.0):
        sys.exit(f"--train_ratio must be in (0, 1], got {args.train_ratio}")

    # 1. Gather + dedupe
    entries: list[SampleEntry] = []
    for root in args.data_roots:
        entries.extend(discover_entries(root, require_complete=True))

    by_loc: dict[str, list[SampleEntry]] = defaultdict(list)
    for e in entries:
        by_loc[e.location_dir].append(e)
    locations = sorted(by_loc.keys())

    # 2. Shuffle deterministically and split
    rng = random.Random(args.seed)
    shuffled = locations[:]
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * args.train_ratio))
    # Always leave at least one in test if any held-out is requested.
    if args.train_ratio < 1.0 and n_train >= len(shuffled):
        n_train = len(shuffled) - 1
    train_locs = sorted(shuffled[:n_train])
    test_locs  = sorted(shuffled[n_train:])

    # 3. Write files
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.txt")
    test_path  = os.path.join(out_dir, "test.txt")
    with open(train_path, "w") as f:
        for loc in train_locs:
            f.write(loc + "\n")
    with open(test_path, "w") as f:
        for loc in test_locs:
            f.write(loc + "\n")

    # 4. Report
    print(f"Total complete locations: {len(locations)}")
    print(f"Total (src,tgt) pairings: {len(entries)}")
    print(f"Train ratio: {args.train_ratio}  (seed={args.seed})")
    print(f"  → {train_path}  ({len(train_locs)} locations)")
    print(f"  → {test_path}   ({len(test_locs)} locations)")
    print()
    _print_breakdown("Train", train_locs)
    print()
    _print_breakdown("Test", test_locs)


if __name__ == "__main__":
    main()
