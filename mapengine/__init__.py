"""gpu-map-engine — GPU-accelerated procedural 2.5D map toolkit.

Core patterns extracted from a production media pipeline:
perspective warp, tensor compositing (no PIL in render loop),
procedural water, hillshade/contour terrain.

MIT. CPU-friendly (CUDA optional). Seeded, contract-tested.
"""

from .perspective import (
    apply,
    depth_scale,
    homography_grid,
    invert_homography,
    solve_homography,
    tilt_corners,
    warp_perspective,
)
from .compositor import GPUCanvas, alpha_blend, outline_alpha, paste_rgba, y_sort
from .water import WATER_PRESETS, WaterTexture, animate_water, shade_water, stable_seed
from .terrain import build_terrain_variants, contour_lines, hillshade, shade_rgb

__version__ = "0.1.0"

__all__ = [
    "apply",
    "depth_scale",
    "homography_grid",
    "invert_homography",
    "solve_homography",
    "tilt_corners",
    "warp_perspective",
    "GPUCanvas",
    "alpha_blend",
    "outline_alpha",
    "paste_rgba",
    "y_sort",
    "WATER_PRESETS",
    "WaterTexture",
    "animate_water",
    "shade_water",
    "stable_seed",
    "build_terrain_variants",
    "contour_lines",
    "hillshade",
    "shade_rgb",
]
