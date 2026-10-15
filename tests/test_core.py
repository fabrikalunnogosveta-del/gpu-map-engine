"""Behavioral contracts for gpu-map-engine (not just 'code runs')."""

from __future__ import annotations

import torch

from mapengine import (
    GPUCanvas,
    WATER_PRESETS,
    WaterTexture,
    alpha_blend,
    animate_water,
    build_terrain_variants,
    contour_lines,
    depth_scale,
    hillshade,
    homography_grid,
    paste_rgba,
    solve_homography,
    warp_perspective,
    y_sort,
)
from mapengine.perspective import apply as tilt_apply
from mapengine.perspective import invert_homography, tilt_corners


def test_homography_identity():
    src = [(0, 0), (10, 0), (10, 10), (0, 10)]
    H = solve_homography(src, src)
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(H, eye, atol=1e-6)


def test_homography_roundtrip_points():
    src = [(0, 0), (100, 0), (100, 80), (0, 80)]
    dst = [(10, 5), (90, 8), (95, 75), (5, 70)]
    H = solve_homography(src, dst)
    Hinv = invert_homography(H)
    for (x, y) in src:
        p = torch.tensor([x, y, 1.0], dtype=torch.float64)
        q = H @ p
        q = q / q[2]
        p2 = Hinv @ q
        p2 = p2 / p2[2]
        assert abs(p2[0].item() - x) < 1e-5
        assert abs(p2[1].item() - y) < 1e-5


def test_warp_preserves_center_color():
    img = torch.zeros(3, 32, 32)
    img[:, 10:22, 10:22] = 1.0  # white square in the middle
    src, dst = tilt_corners(32, 32, tilt_deg=35.0)
    out = warp_perspective(img, src, dst)
    assert out.shape == img.shape
    # screen(16,20) -> source(~16,15) inside the square (tilt lifts the top)
    assert out[:, 20, 16].mean().item() > 0.3
    # deeper near-side sample still in the square
    assert out[:, 18, 16].mean().item() > 0.3


def test_tilt_apply_shape():
    img = torch.rand(3, 64, 48)
    out = tilt_apply(img, tilt_deg=40.0)
    assert out.shape == img.shape


def test_homography_grid_range():
    H = torch.eye(3, dtype=torch.float64)
    g = homography_grid(H, 16, 16)
    assert g.shape == (1, 16, 16, 2)
    assert g.min().item() >= -1.05 and g.max().item() <= 1.05


def test_depth_scale_monotonic():
    h = 100
    near = depth_scale(h - 1, h)
    far = depth_scale(0, h)
    mid = depth_scale(h // 2, h)
    assert far < mid < near
    assert abs(near - 1.0) < 1e-6


def test_alpha_blend_over_opaque():
    base = torch.zeros(3, 8, 8)
    layer = torch.ones(4, 4, 4)  # opaque white
    out = alpha_blend(base, layer, 2, 2)
    assert torch.allclose(out[:, 4, 4], torch.ones(3), atol=1e-5)
    assert torch.allclose(out[:, 0, 0], torch.zeros(3), atol=1e-5)


def test_alpha_blend_clips_offscreen():
    base = torch.zeros(3, 8, 8)
    layer = torch.ones(4, 4, 4)
    out = alpha_blend(base, layer, -2, -2)  # hangs off top-left
    assert out.shape == (3, 8, 8)
    assert out[:, 0, 0].sum().item() > 0  # top-left partially filled


def test_paste_rgba_and_canvas():
    c = GPUCanvas(16, 16, fill=0.0)
    layer = torch.ones(4, 4, 4)
    c.paste_rgba(layer, 4, 4)
    assert c.data[:, 6, 6].mean().item() > 0.9


def test_y_sort_orders_by_y():
    a = (0, 10, torch.zeros(4, 1, 1))
    b = (0, 2, torch.zeros(4, 1, 1))
    c = (0, 5, torch.zeros(4, 1, 1))
    ordered = y_sort([a, b, c])
    assert [item[1] for item in ordered] == [2, 5, 10]


def test_water_determinism():
    tex = WaterTexture.from_preset("deep_ocean")
    a = animate_water(tex, 32, 32, t=1.5, seed="demo")
    b = animate_water(tex, 32, 32, t=1.5, seed="demo")
    assert torch.equal(a, b)


def test_water_presets_all_build():
    for name in WATER_PRESETS:
        tex = WaterTexture.from_preset(name)
        f = tex.field(16, 16, seed="t")
        assert f.shape == (16, 16)
        assert float(f.min()) >= -1e-5 and float(f.max()) <= 1.0 + 1e-5


def test_hillshade_range():
    e = torch.zeros(32, 32)
    e[8:24, 8:24] = 10.0  # plateau
    hs = hillshade(e)
    assert hs.shape == (32, 32)
    assert float(hs.min()) >= 0.0 and float(hs.max()) <= 1.0


def test_contour_lines_hit_plateau_edges():
    e = torch.zeros(40, 40)
    e[10:30, 10:30] = 50.0
    c = contour_lines(e, spacing=25.0, width=1.5)
    assert float(c.max()) > 0.5
    # center of plateau is on a level line (50 / 25 = 2 exactly) — may be 1
    assert c.shape == (40, 40)


def test_terrain_variants_modes():
    e = torch.randn(24, 24)
    assert build_terrain_variants(e, "flat")["contour"] is None
    assert build_terrain_variants(e, "hillshade")["hillshade_mod"] is not None
    assert build_terrain_variants(e, "contour")["contour"] is not None


def test_stable_seed_not_builtin_hash():
    from mapengine.water import stable_seed

    assert stable_seed("a", "b") == stable_seed("a", "b")
    assert stable_seed("a", "b") != stable_seed("a", "c")
