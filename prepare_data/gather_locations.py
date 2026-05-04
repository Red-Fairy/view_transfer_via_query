"""Gather all location dirs under one or more data roots that pass `SampleEntry.is_complete()`.

For each `<data_root>/<scene>/<location>/`, checks every (src,tgt) pano pairing, and
writes the deduplicated list of complete location dirs (one per line) to `--output`.
Multiple data roots are merged; duplicate location dirs are kept once.

Usage:
    python -m view_transfer_via_query.prepare_data.gather_locations \
        --data_roots /path/to/outputs_non_arranged_24fps /path/to/another_root \
        --output     view_transfer_via_query/scripts/locations_24fps.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from view_transfer_via_query.dataset import SampleEntry, discover_entries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_roots", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    entries: list[SampleEntry] = []
    for root in args.data_roots:
        entries.extend(discover_entries(root, require_complete=True))

    by_loc: dict[str, list[SampleEntry]] = defaultdict(list)
    for e in entries:
        by_loc[e.location_dir].append(e)

    locations = sorted(by_loc.keys())
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for loc in locations:
            f.write(loc + "\n")

    by_scene: dict[str, int] = defaultdict(int)
    for loc in locations:
        scene = os.path.basename(os.path.dirname(loc))
        by_scene[scene] += 1
    print(f"Total complete locations: {len(locations)}")
    print(f"Total (src,tgt) pairings: {len(entries)}")
    print(f"Output: {args.output}")
    print()
    print("Per-scene breakdown:")
    for scene in sorted(by_scene):
        print(f"  {scene}: {by_scene[scene]} locations")


if __name__ == "__main__":
    main()
