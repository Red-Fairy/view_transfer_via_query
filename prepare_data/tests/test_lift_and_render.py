"""Unit tests for lift_and_render."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import math
import tempfile
import numpy as np
import torch
import pytest

from view_transfer_via_query.prepare_data.lift_and_render import (
    equirect_pixel_to_unit_ray_ue,
    lift_pano_to_world_pointcloud,
    render_pointcloud_to_perspective,
    lift_and_render,
    load_depth_ue,
    build_cubemap_mesh_world,
    _build_cubemap_topology,
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


# ── Sky-handling fix (Phase 1) ──

def test_load_depth_ue_fills_sky_with_finite_distance():
    """Sky-sentinel pixels (raw EXR ≥ 65000) become finite (10 km default), not NaN."""
    raw = np.array([[100.0, 65504.0], [200.0, 65504.0]], dtype=np.float32)  # cm
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
        np.save(tf.name, raw)
        depth_m = load_depth_ue(tf.name)  # default infinity_depth_m=1e4 (10 km)
    os.unlink(tf.name)
    assert np.all(np.isfinite(depth_m)), "no NaNs"
    # Non-sky: cm → m
    assert depth_m[0, 0] == pytest.approx(1.0)
    assert depth_m[1, 0] == pytest.approx(2.0)
    # Sky pixels filled with 1e4 m (default)
    assert depth_m[0, 1] == pytest.approx(1e4)
    assert depth_m[1, 1] == pytest.approx(1e4)


def test_load_depth_ue_custom_infinity():
    raw = np.array([[65504.0]], dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
        np.save(tf.name, raw)
        depth_m = load_depth_ue(tf.name, infinity_depth_m=5.0e3)
    os.unlink(tf.name)
    assert depth_m[0, 0] == pytest.approx(5.0e3)


def test_lift_keeps_sky_points_after_fix():
    """After Phase-1 fix, sky pixels (now at large finite distance) are kept by
    `lift_pano_to_world_pointcloud` (vs. dropped before, when they were NaN)."""
    He, We = 8, 16
    rgb = torch.rand(3, He, We)
    # Half the rows are sky-distance (1e4 m), half are normal (5 m)
    depth = torch.full((He, We), 5.0)
    depth[:He // 2] = 1e4
    c2w = torch.eye(4)
    pts, _ = lift_pano_to_world_pointcloud(rgb, depth, c2w)
    # All pixels should be kept (1e4 < default valid_depth_max=1e6)
    assert pts.shape[0] == He * We


# ── Cubemap mesh topology ──

def test_cubemap_topology_shapes():
    topo = _build_cubemap_topology(face_res=8, device=torch.device("cpu"))
    K = topo["unique_dirs"].shape[0]
    F = topo["faces"].shape[0]
    # K should be < 6 * 8 * 8 (some shared at edges/corners)
    assert K < 6 * 8 * 8
    # Each cube face contributes 2 * (8-1)^2 triangles
    assert F == 6 * 2 * (8 - 1) ** 2
    # Unit directions
    assert torch.allclose(topo["unique_dirs"].norm(dim=-1), torch.ones(K), atol=1e-5)
    # UV in [0, 1]
    assert topo["equirect_uv"].min() >= 0.0
    assert topo["equirect_uv"].max() <= 1.0


def test_cubemap_topology_inward_winding_at_origin():
    """For a cubemap with INWARD winding, triangle normals (computed via cross
    product of edges) should point TOWARD the origin (i.e., negative dot product
    with the triangle's outward radial direction)."""
    topo = _build_cubemap_topology(face_res=8, device=torch.device("cpu"))
    verts = topo["unique_dirs"]   # unit-distance vertices
    faces = topo["faces"].long()
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    centers = (v0 + v1 + v2) / 3.0
    # Inward → normal · centroid < 0 (centroid points outward from origin)
    dot = (normals * centers).sum(dim=-1)
    n_inward = (dot < 0).sum().item()
    n_outward = (dot > 0).sum().item()
    # Allow a few degenerate (zero-area) ones near corners
    assert n_inward > 0.95 * (n_inward + n_outward), \
        f"expected mostly inward; inward={n_inward} outward={n_outward}"


# ── Cubemap mesh build ──

def test_build_cubemap_mesh_world_shapes():
    He, We = 32, 64
    rgb = torch.rand(3, He, We)
    depth = torch.full((He, We), 5.0)
    pano_c2w = torch.eye(4)
    verts, faces, colors = build_cubemap_mesh_world(
        rgb, depth, pano_c2w, face_res=8,
    )
    K = verts.shape[0]
    assert verts.shape == (K, 3)
    assert colors.shape == (K, 3)
    assert faces.dim() == 2 and faces.shape[1] == 3
    # All vertices at radial distance ≈ 5 (since depth is uniform)
    radial = verts.norm(dim=-1)
    assert torch.allclose(radial, torch.full_like(radial, 5.0), atol=1e-3)


# ── Mesh rasterization (requires nvdiffrast + CUDA) ──

@pytest.mark.skipif(not torch.cuda.is_available(), reason="nvdiffrast needs CUDA")
def test_mesh_rasterize_smoke():
    try:
        import nvdiffrast.torch  # noqa: F401
    except ImportError:
        pytest.skip("nvdiffrast not installed")
    from view_transfer_via_query.prepare_data.lift_and_render import render_mesh_to_perspective

    device = torch.device("cuda")
    He, We = 32, 64
    rgb = torch.rand(3, He, We, device=device)
    depth = torch.full((He, We), 5.0, device=device)
    pano_c2w = torch.eye(4, device=device)
    verts, faces, colors = build_cubemap_mesh_world(
        rgb, depth, pano_c2w, face_res=8,
    )

    # Target camera at origin looking +Z (OpenCV); should see the mesh shell
    T = 2
    target_c2w = torch.eye(4, device=device).unsqueeze(0).expand(T, 4, 4).contiguous()
    intrinsics = torch.tensor([50.0, 50.0, 16.0, 8.0], device=device)
    rendered, vis = render_mesh_to_perspective(
        verts, faces, colors, target_c2w, intrinsics,
        pers_h=16, pers_w=32, backface_cull=True,
    )
    assert rendered.shape == (T, 3, 16, 32)
    assert vis.shape == (T, 1, 16, 32)
    # Most pixels should be hit (camera inside the shell, looking outward)
    hit_frac = vis.mean().item()
    assert hit_frac > 0.5, f"only {hit_frac:.2%} pixels hit by mesh"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="nvdiffrast needs CUDA")
def test_lift_and_render_use_mesh_branch():
    try:
        import nvdiffrast.torch  # noqa: F401
    except ImportError:
        pytest.skip("nvdiffrast not installed")
    device = torch.device("cuda")
    He, We = 32, 64
    rgb = torch.rand(3, He, We, device=device)
    depth = torch.full((He, We), 5.0, device=device)
    pano_c2w = torch.eye(4, device=device)
    T = 2
    target_c2w = torch.eye(4, device=device).unsqueeze(0).expand(T, 4, 4).contiguous()
    intr = torch.tensor([50.0, 50.0, 16.0, 8.0], device=device)
    rendered, vis = lift_and_render(
        rgb, depth, pano_c2w, target_c2w, intr,
        pers_h=16, pers_w=32, use_mesh=True, mesh_face_res=8,
    )
    assert rendered.shape == (T, 3, 16, 32)
    assert vis.shape == (T, 1, 16, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
