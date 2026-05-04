"""Unit tests for lift_and_render."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import math
import torch
import pytest

from view_transfer_via_query.prepare_data.lift_and_render import (
    equirect_pixel_to_unit_ray_ue,
    lift_pano_to_world_pointcloud,
    render_pointcloud_to_perspective,
    lift_and_render,
)
from view_transfer_via_query.prepare_data.parse_cameras import M_CV_TO_UE_CAM


# ── equirect_pixel_to_unit_ray_ue ──

def test_rays_are_unit_length():
    rays = equirect_pixel_to_unit_ray_ue(32, 64, device="cpu")
    norms = rays.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_center_pixel_is_forward_X():
    """Equirect center (W/2, H/2) → UE +X (forward)."""
    He, We = 32, 64
    rays = equirect_pixel_to_unit_ray_ue(He, We, device="cpu")
    center = rays[He // 2, We // 2]
    # Approximately (1, 0, 0) — exactly correct only at H/W → infinity since pixel center is not exactly at lon=0,lat=0
    assert center[0] > 0.99
    assert abs(center[1]) < 0.05
    assert abs(center[2]) < 0.05


def test_top_row_is_up_Z():
    """Equirect top row → UE +Z (up)."""
    He, We = 32, 64
    rays = equirect_pixel_to_unit_ray_ue(He, We, device="cpu")
    top = rays[0, We // 2]
    # First row latitude ~ +π/2 → ray ≈ (0, 0, 1)
    assert top[2] > 0.95


def test_quarter_width_is_right_Y():
    """At u = 3W/4, longitude ≈ π/2 → UE +Y (right)."""
    He, We = 32, 64
    rays = equirect_pixel_to_unit_ray_ue(He, We, device="cpu")
    right = rays[He // 2, 3 * We // 4]
    assert right[1] > 0.99


# ── lift_pano_to_world_pointcloud ──

def test_lift_at_origin_identity_pose():
    """Pano at world origin with identity rotation: world point = ray * depth in OpenCV cam frame."""
    He, We = 16, 32
    rgb = torch.rand(3, He, We)
    depth = torch.full((He, We), 5.0)
    c2w = torch.eye(4)
    pts, _ = lift_pano_to_world_pointcloud(rgb, depth, c2w)
    assert pts.shape == (He * We, 3)
    # All points should have radial distance 5 from origin
    radial = pts.norm(dim=-1)
    assert torch.allclose(radial, torch.full_like(radial, 5.0), atol=1e-3)


def test_lift_translation_only():
    """Pano translated to (10, 0, 0) in world: all world points should be near (10, 0, 0)."""
    He, We = 16, 32
    rgb = torch.rand(3, He, We)
    depth = torch.full((He, We), 1.0)
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([10.0, 0.0, 0.0])
    pts, _ = lift_pano_to_world_pointcloud(rgb, depth, c2w)
    centroid = pts.mean(dim=0)
    # Centroid should be close to camera position (since uniform distribution on sphere)
    assert torch.allclose(centroid, torch.tensor([10.0, 0.0, 0.0]), atol=0.2)


def test_lift_rgb_preserved():
    He, We = 8, 16
    rgb = torch.rand(3, He, We)
    depth = torch.ones(He, We)
    c2w = torch.eye(4)
    _, rgb_out = lift_pano_to_world_pointcloud(rgb, depth, c2w)
    rgb_expected = rgb.permute(1, 2, 0).reshape(-1, 3)
    assert torch.allclose(rgb_expected, rgb_out)


def test_lift_filters_invalid_depth():
    """Zero / negative / NaN depth pixels should be dropped."""
    He, We = 4, 4
    rgb = torch.rand(3, He, We)
    depth = torch.ones(He, We)
    depth[0, 0] = 0.0
    depth[0, 1] = float("nan")
    depth[0, 2] = -1.0
    c2w = torch.eye(4)
    pts, _ = lift_pano_to_world_pointcloud(rgb, depth, c2w)
    assert pts.shape[0] == He * We - 3


# ── render_pointcloud_to_perspective ──

def test_render_single_point_in_front():
    """One point at world (0, 0, 5), camera looking +Z from origin → projects to image center."""
    xyz = torch.tensor([[0.0, 0.0, 5.0]])
    rgb = torch.tensor([[1.0, 0.5, 0.2]])
    c2w = torch.eye(4).unsqueeze(0)              # [1, 4, 4]
    fx, fy = 100.0, 100.0
    cx, cy = 8.0, 4.0
    intr = torch.tensor([fx, fy, cx, cy])
    rendered, vis = render_pointcloud_to_perspective(
        xyz, rgb, c2w, intr, pers_h=8, pers_w=16, chunk_size=1
    )
    # Point should project to (cx, cy) = (8, 4)  → pixel (4, 8)
    assert rendered[0, 0, 4, 8].item() == pytest.approx(1.0, abs=1e-3)
    assert rendered[0, 1, 4, 8].item() == pytest.approx(0.5, abs=1e-3)
    assert rendered[0, 2, 4, 8].item() == pytest.approx(0.2, abs=1e-3)
    assert vis[0, 0, 4, 8].item() == 1.0
    # All other pixels should be empty
    visible_count = (vis[0, 0] > 0).sum().item()
    assert visible_count == 1


def test_render_zbuffer_picks_closest():
    """Two points projecting to the same pixel — closer (smaller z) wins."""
    xyz = torch.tensor([
        [0.0, 0.0, 5.0],   # closer, RED
        [0.0, 0.0, 8.0],   # farther, BLUE
    ])
    rgb = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    c2w = torch.eye(4).unsqueeze(0)
    intr = torch.tensor([100.0, 100.0, 8.0, 4.0])
    rendered, _ = render_pointcloud_to_perspective(
        xyz, rgb, c2w, intr, pers_h=8, pers_w=16, chunk_size=1
    )
    px = rendered[0, :, 4, 8]
    assert px[0].item() == pytest.approx(1.0, abs=1e-3)
    assert px[2].item() == pytest.approx(0.0, abs=1e-3)


def test_render_filters_behind_camera():
    """Points behind the camera (z < 0) should not appear."""
    xyz = torch.tensor([[0.0, 0.0, -5.0]])
    rgb = torch.tensor([[1.0, 1.0, 1.0]])
    c2w = torch.eye(4).unsqueeze(0)
    intr = torch.tensor([100.0, 100.0, 8.0, 4.0])
    _, vis = render_pointcloud_to_perspective(
        xyz, rgb, c2w, intr, pers_h=8, pers_w=16, chunk_size=1
    )
    assert vis.sum().item() == 0


def test_render_multi_frame_shape():
    """Rendering to 5 frames returns the expected output shape."""
    N = 100
    xyz = torch.randn(N, 3) * 2.0 + torch.tensor([0.0, 0.0, 5.0])
    rgb = torch.rand(N, 3)
    T = 5
    c2w = torch.eye(4).unsqueeze(0).expand(T, 4, 4).clone()
    intr = torch.tensor([100.0, 100.0, 16.0, 8.0])
    rendered, vis = render_pointcloud_to_perspective(
        xyz, rgb, c2w, intr, pers_h=16, pers_w=32, chunk_size=2
    )
    assert rendered.shape == (T, 3, 16, 32)
    assert vis.shape == (T, 1, 16, 32)


# ── lift_and_render end-to-end ──

def test_lift_and_render_e2e_shapes():
    He, We = 16, 32
    rgb = torch.rand(3, He, We)
    depth = torch.ones(He, We) * 5.0
    pano_c2w = torch.eye(4)
    T = 3
    target_c2w = torch.eye(4).unsqueeze(0).expand(T, 4, 4).clone()
    intr = torch.tensor([50.0, 50.0, 16.0, 8.0])
    rendered, vis = lift_and_render(
        rgb, depth, pano_c2w, target_c2w, intr, pers_h=16, pers_w=32, chunk_size=1,
    )
    assert rendered.shape == (T, 3, 16, 32)
    assert vis.shape == (T, 1, 16, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
