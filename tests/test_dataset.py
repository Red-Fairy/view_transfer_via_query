"""Unit tests for the online dataset (raw 360 windows + sampled trajectories +
single-frame static RGB+depth at t0)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tempfile
import numpy as np
import torch
import pytest
import cv2

from view_transfer_via_query.dataset import (
    ViewTransferDataset,
    SampleEntry,
    discover_entries,
    collate_view_transfer,
)


# Tiny pano sizes for fast tests
T_FULL = 12
T_WIN = 6
He, We = 32, 64


def _write_png_seq(dir_path: str, num_frames: int, color=(128, 128, 128)):
    os.makedirs(dir_path, exist_ok=True)
    for i in range(num_frames):
        img = np.full((He, We, 3), color, dtype=np.uint8)
        cv2.imwrite(os.path.join(dir_path, f"{i:04d}.png"), img)


def _write_depth_seq(dir_path: str, num_frames: int, value: float = 5.0):
    """Write fake depth files as .npy (avoids EXR codec dependency).

    The dataset accepts .exr / .npy / .pt — production uses EXR; tests use .npy.
    """
    os.makedirs(dir_path, exist_ok=True)
    for i in range(num_frames):
        depth = np.full((He, We), value, dtype=np.float32)
        np.save(os.path.join(dir_path, f"{i:04d}.npy"), depth)


def _make_fake_location(loc_dir: str):
    """Build one location with all required dirs/files for both pairings."""
    for pano in ["00", "01"]:
        # Dynamic pano RGB
        _write_png_seq(os.path.join(loc_dir, f"Pano_{pano}", "rgb"), T_FULL)
        # Static RGB + depth (source for lift-render)
        _write_png_seq(os.path.join(loc_dir, f"Pano_{pano}_static", "rgb"), T_FULL)
        _write_depth_seq(os.path.join(loc_dir, f"Pano_{pano}_static", "depth"), T_FULL)
    # Blob videos: rendered at each pano's camera, stored under that pano
    for pano in ["00", "01"]:
        _write_png_seq(os.path.join(loc_dir, f"Pano_{pano}", "blobs"), T_FULL)
    # Camera + text
    c2w = torch.eye(4).unsqueeze(0).expand(T_FULL, 4, 4).clone()
    torch.save(c2w, os.path.join(loc_dir, "c2w_PanoCam_00.pt"))
    torch.save(c2w, os.path.join(loc_dir, "c2w_PanoCam_01.pt"))
    torch.save(torch.randn(4, 4096), os.path.join(loc_dir, "text_emb.pt"))


# ── discover_entries ──

def test_discover_entries_finds_all_pairings():
    """All 4 (src, tgt) pairings are valid as long as both panos have static + blobs."""
    with tempfile.TemporaryDirectory() as tmp:
        scene_dir = os.path.join(tmp, "TestScene")
        loc_dir = os.path.join(scene_dir, "loc_001")
        _make_fake_location(loc_dir)
        entries = discover_entries(tmp)
        assert len(entries) == 4
        pairings = {(e.src_idx, e.tgt_idx) for e in entries}
        assert pairings == {("00", "00"), ("00", "01"), ("01", "00"), ("01", "01")}


def test_discover_skips_incomplete_locations():
    with tempfile.TemporaryDirectory() as tmp:
        scene_dir = os.path.join(tmp, "S")
        loc = os.path.join(scene_dir, "loc")
        _make_fake_location(loc)
        # Remove static depth for Pano_00: pairings with src=00 fail (lift uses src-side static)
        import shutil
        shutil.rmtree(os.path.join(loc, "Pano_00_static", "depth"))
        entries = discover_entries(tmp)
        pairings = {(e.src_idx, e.tgt_idx) for e in entries}
        # Only src=01 pairings survive
        assert pairings == {("01", "00"), ("01", "01")}


# ── __getitem__ ──

def test_getitem_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        loc = os.path.join(tmp, "S", "loc")
        _make_fake_location(loc)
        ds = ViewTransferDataset(tmp, num_video_frames=T_WIN, seed=0)
        s = ds[0]
        # 360 windows
        assert s["rgb_src_360"].shape == (3, T_WIN, He, We)
        assert s["rgb_tgt_360"].shape == (3, T_WIN, He, We)
        assert s["blob_360"].shape == (3, T_WIN, He, We)
        # Single-frame static
        assert s["static_rgb_t0"].shape == (3, He, We)
        assert s["static_depth_t0"].shape == (He, We)
        # Cameras + text
        assert s["pano_c2w_src"].shape == (T_WIN, 4, 4)
        assert s["pano_c2w_tgt"].shape == (T_WIN, 4, 4)
        assert s["src_c2w_at_t0"].shape == (4, 4)
        # Trajectories
        assert s["src_yaw_deg"].shape == (T_WIN,)
        assert s["tgt_pitch_deg"].shape == (T_WIN,)
        assert isinstance(s["src_fov_h_deg"], float)
        # dtype
        assert s["rgb_src_360"].dtype == torch.uint8
        assert s["static_rgb_t0"].dtype == torch.uint8
        assert s["static_depth_t0"].dtype == torch.float32


def test_static_depth_value():
    """Our fake EXR files write constant 5.0 — verify the loader gets it back."""
    with tempfile.TemporaryDirectory() as tmp:
        loc = os.path.join(tmp, "S", "loc")
        _make_fake_location(loc)
        ds = ViewTransferDataset(tmp, num_video_frames=T_WIN, seed=0)
        s = ds[0]
        assert torch.allclose(
            s["static_depth_t0"], torch.full_like(s["static_depth_t0"], 5.0), atol=1e-3
        )


def test_src_c2w_at_t0_matches_window_first_frame():
    """src_c2w_at_t0 should equal pano_c2w_src[0] for the chosen window."""
    with tempfile.TemporaryDirectory() as tmp:
        loc = os.path.join(tmp, "S", "loc")
        _make_fake_location(loc)
        ds = ViewTransferDataset(tmp, num_video_frames=T_WIN, seed=0)
        s = ds[0]
        assert torch.allclose(s["src_c2w_at_t0"], s["pano_c2w_src"][0])


def test_temporal_window_varies():
    with tempfile.TemporaryDirectory() as tmp:
        loc = os.path.join(tmp, "S", "loc")
        _make_fake_location(loc)
        ds = ViewTransferDataset(tmp, num_video_frames=T_WIN, seed=0)
        offsets = {ds[i % len(ds)]["t_offset"] for i in range(20)}
        assert len(offsets) >= 2


# ── collate ──

def test_collate_stacks_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        loc = os.path.join(tmp, "S", "loc")
        _make_fake_location(loc)
        ds = ViewTransferDataset(tmp, num_video_frames=T_WIN, seed=0)
        batch = collate_view_transfer([ds[0], ds[0]])
        assert batch["rgb_src_360"].shape == (2, 3, T_WIN, He, We)
        assert batch["static_rgb_t0"].shape == (2, 3, He, We)
        assert batch["static_depth_t0"].shape == (2, He, We)
        assert batch["pano_c2w_src"].shape == (2, T_WIN, 4, 4)
        assert batch["src_c2w_at_t0"].shape == (2, 4, 4)
        assert batch["src_yaw_deg"].shape == (2, T_WIN)
        assert batch["src_fov_h_deg"].shape == (2,)
        assert batch["t_offset"].shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
