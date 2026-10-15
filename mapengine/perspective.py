"""True homography-based 2.5D perspective warp (no OpenCV dependency).

Why this exists: naive affine / corner-pin looks wrong at 30–50° tilt.
We compute a real 3x3 homography (4-point, DLT) and build a grid_sample
grid so every output pixel samples the source correctly.

Contract: geo_to_screen and screen_to_geo must stay in sync (≤2px error)
or icons drift off the map. Tests enforce that.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F

Point = Tuple[float, float]


def solve_homography(src: Sequence[Point], dst: Sequence[Point]) -> torch.Tensor:
    """DLT 4-point homography: maps src pixel coords -> dst pixel coords.

    Returns 3x3 float64 tensor H such that dst ~ H @ src (homogeneous).
    """
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("homography needs exactly 4 point pairs")

    A = []
    b = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        A.append([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx])
        A.append([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy])
        b.extend([dx, dy])

    A_t = torch.tensor(A, dtype=torch.float64)
    b_t = torch.tensor(b, dtype=torch.float64)
    # Solve A h = b via least squares (4-point is square 8x8, exact)
    h = torch.linalg.solve(A_t, b_t)
    H = torch.cat([h, torch.ones(1, dtype=torch.float64)]).reshape(3, 3)
    return H


def invert_homography(H: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(H)


def homography_grid(
    H_dst_to_src: torch.Tensor,
    out_h: int,
    out_w: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build grid_sample grid [1, out_h, out_w, 2] in [-1, 1].

    H_dst_to_src maps output (screen) coords -> source coords.
    """
    ys = torch.arange(out_h, dtype=torch.float64)
    xs = torch.arange(out_w, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    pts = torch.stack([grid_x, grid_y, ones], dim=0).reshape(3, -1)  # [3, N]

    mapped = H_dst_to_src.to(torch.float64) @ pts  # [3, N]
    w = mapped[2].clamp(min=1e-8)
    src_x = mapped[0] / w
    src_y = mapped[1] / w

    # Normalize source pixels to [-1, 1] assuming source size == output size
    # (caller warps a tensor of the same HxW as out, or pre-scales).
    nx = (src_x / max(out_w - 1, 1)) * 2.0 - 1.0
    ny = (src_y / max(out_h - 1, 1)) * 2.0 - 1.0

    grid = torch.stack([nx, ny], dim=-1).reshape(1, out_h, out_w, 2)
    return grid.to(device=device, dtype=torch.float32)


def warp_perspective(
    image: torch.Tensor,
    src_corners: Sequence[Point],
    dst_corners: Sequence[Point],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Warp image [C,H,W] or [B,C,H,W] by mapping src_corners -> dst_corners.

    Grid is built for the image spatial size (source == output canvas size).
    """
    squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True

    H = solve_homography(src_corners, dst_corners)
    H_inv = invert_homography(H)  # screen(dst) -> source(src)
    _, _, h, w = image.shape
    grid = homography_grid(H_inv, h, w, device=image.device)
    warped = F.grid_sample(image, grid, mode=mode, align_corners=True, padding_mode="zeros")
    return warped.squeeze(0) if squeeze else warped


def tilt_corners(
    width: int,
    height: int,
    tilt_deg: float = 45.0,
    squash: float = 0.55,
) -> Tuple[list, list]:
    """Standard flat -> tilted trapezoid (like a table-top map).

    Flat source: full rectangle. Dest: top edge pulled inward + shorter
    (far side), bottom edge full (near side).
    """
    flat = [(0.0, 0.0), (float(width - 1), 0.0), (float(width - 1), float(height - 1)), (0.0, float(height - 1))]
    t = math.radians(max(5.0, min(80.0, tilt_deg)))
    # Horizontal pull of top corners scales with tilt
    pull = (1.0 - math.cos(t)) * 0.45 * width
    top_y = (1.0 - squash) * height * (1.0 - math.sin(t) * 0.35)
    tilted = [
        (pull, top_y),
        (width - 1 - pull, top_y),
        (float(width - 1), float(height - 1)),
        (0.0, float(height - 1)),
    ]
    return flat, tilted


def depth_scale(
    y_screen: float,
    height: int,
    near: float = 1.0,
    far: float = 0.55,
) -> float:
    """Icon scale multiplier: near (bottom) = near, far (top) = far."""
    if height <= 1:
        return near
    t = max(0.0, min(1.0, y_screen / float(height - 1)))
    return far + (near - far) * t


def apply(
    image: torch.Tensor,
    tilt_deg: float = 45.0,
    squash: float = 0.55,
) -> torch.Tensor:
    """One-shot helper: flat map frame -> tilted 2.5D frame."""
    if image.dim() != 3:
        raise ValueError("expected [C,H,W]")
    _, h, w = image.shape
    src, dst = tilt_corners(w, h, tilt_deg=tilt_deg, squash=squash)
    return warp_perspective(image, src, dst)
