"""Quickstart — run: python examples/quickstart.py"""

from __future__ import annotations

import torch

from mapengine import (
    GPUCanvas,
    WATER_PRESETS,
    WaterTexture,
    animate_water,
    build_terrain_variants,
    hillshade,
    paste_rgba,
)
from mapengine.perspective import apply as tilt
from mapengine.perspective import depth_scale, tilt_corners, warp_perspective


def main() -> None:
    # 1) Flat map -> 2.5D tilt
    flat = torch.zeros(3, 128, 192)
    flat[:, 40:88, 60:132] = 0.8  # a "land" block
    tilted = tilt(flat, tilt_deg=42.0)
    print("tilt:", tuple(tilted.shape), "mean=", round(float(tilted.mean()), 4))

    # 2) Homography round-trip corners
    src, dst = tilt_corners(192, 128, tilt_deg=42.0)
    warped = warp_perspective(flat, src, dst)
    print("warp near-center:", round(float(warped[:, 110, 96].mean()), 4))

    # 3) Canvas + alpha paste
    canvas = GPUCanvas(64, 64, fill=0.1)
    tile = torch.ones(4, 16, 16)
    canvas.paste_rgba(tile, 10, 10)
    print("canvas center:", round(float(canvas.data[:, 18, 18].mean()), 4))

    # 4) Deterministic water
    tex = WaterTexture.from_preset("deep_ocean")
    w0 = animate_water(tex, 32, 32, t=0.0, seed="demo")
    w1 = animate_water(tex, 32, 32, t=0.0, seed="demo")
    print("water determinism:", bool(torch.equal(w0, w1)))
    print("water presets:", len(WATER_PRESETS))

    # 5) Terrain
    elev = torch.zeros(64, 64)
    elev[20:44, 20:44] = 500.0
    hs = hillshade(elev)
    variants = build_terrain_variants(elev, "contour", contour_spacing=125.0)
    print("hillshade:", round(float(hs.mean()), 4), "contour hits:", int(variants["contour"].sum()))

    # 6) Depth scale (icons bigger at the bottom / near side)
    print("depth near/far:", round(depth_scale(127, 128), 3), round(depth_scale(0, 128), 3))


if __name__ == "__main__":
    main()
