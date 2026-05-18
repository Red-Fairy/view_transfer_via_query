"""Tests for video_io: natural frame ordering + empty-window guard (audit #5).

The old `str.sort()` ordered `frame_10.png` before `frame_2.png`, silently
shuffling a non-zero-padded frame sequence; an empty window raised an opaque
`max() arg is an empty sequence`. Both are covered here.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np
import pytest
import torch

from view_transfer_via_query.prepare_data.video_io import (
    natural_sort_key, list_png_frames, load_png_sequence,
)


def test_natural_sort_key_orders_non_padded_numerically():
    names = ["frame_10.png", "frame_2.png", "frame_1.png", "frame_100.png"]
    assert sorted(names, key=natural_sort_key) == [
        "frame_1.png", "frame_2.png", "frame_10.png", "frame_100.png"
    ]
    # Lexicographic would have given the wrong order — sanity-check that.
    assert sorted(names) != sorted(names, key=natural_sort_key)


def test_natural_sort_key_noop_for_zero_padded():
    names = ["00000.png", "00001.png", "00010.png", "00002.png"]
    assert sorted(names, key=natural_sort_key) == sorted(names)  # identical


def _write_frame(path: str, value: int):
    """2x2 BGR PNG whose pixels all equal `value` (frame fingerprint)."""
    cv2.imwrite(path, np.full((2, 2, 3), value, dtype=np.uint8))


def test_list_png_frames_natural_order(tmp_path):
    for i in (1, 2, 10, 11, 100):
        _write_frame(str(tmp_path / f"frame_{i}.png"), i)
    got = [os.path.basename(p) for p in list_png_frames(str(tmp_path))]
    assert got == ["frame_1.png", "frame_2.png", "frame_10.png",
                   "frame_11.png", "frame_100.png"]


def test_load_png_sequence_reads_in_natural_order(tmp_path):
    # Non-padded names; pixel value == frame index so we can verify order.
    for i in range(12):
        _write_frame(str(tmp_path / f"f{i}.png"), i)
    seq = load_png_sequence(str(tmp_path), start=0, num_frames=12,
                            return_dtype=torch.uint8)  # [T,3,H,W]
    order = [int(seq[t, 0, 0, 0]) for t in range(12)]
    assert order == list(range(12)), f"frames out of order: {order}"


def test_load_png_sequence_empty_window_raises_valueerror(tmp_path):
    for i in range(3):
        _write_frame(str(tmp_path / f"{i:05d}.png"), i)
    with pytest.raises(ValueError, match="Empty frame window"):
        load_png_sequence(str(tmp_path), start=0, num_frames=0,
                          return_dtype=torch.uint8)


def test_load_png_sequence_out_of_range_still_indexerror(tmp_path):
    for i in range(3):
        _write_frame(str(tmp_path / f"{i:05d}.png"), i)
    with pytest.raises(IndexError):
        load_png_sequence(str(tmp_path), start=0, num_frames=5,
                          return_dtype=torch.uint8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
