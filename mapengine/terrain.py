"""Terrain shading — hillshade + contour overlays from a DEM / height array.

Production note: hillshade uses large-scale gradients only (blur first) so
pixel noise does not look like sandpaper. Contours are alpha overlays, not
destructive edits.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _blur(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    if k <= 1:
        return x
    pad = k // 2
    x4 = x.unsqueeze(0).unsqueeze(0)
    y = F.avg_pool2d(x4, kernel_size=k, stride=1, padding=pad)
    return y.squeeze(0).squeeze(0)


def hillshade(
    elevation: torch.Tensor,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
    grad_scale: float = 1.4,
    blur: int = 3,
) -> torch.Tensor:
    """Classic hillshade in [0,1], input elevation [H,W] float.

    grad_scale softens relief (2.0 aggressive, 1.4 natural).
    """
    if elevation.dim() != 2:
        raise ValueError("elevation must be [H,W]")
    z = _blur(elevation.to(torch.float32), k=blur) * float(z_factor)

    gy, gx = torch.gradient(z)
    gx = gx * float(grad_scale)
    gy = gy * float(grad_scale)

    az = torch.tensor(360.0 - azimuth_deg + 90.0, dtype=torch.float32).deg2rad()
    alt = torch.tensor(altitude_deg, dtype=torch.float32).deg2rad()

    slope = torch.atan(torch.sqrt(gx * gx + gy * gy))
    aspect = torch.atan2(-gx, gy)

    shaded = (
        torch.sin(alt) * torch.cos(slope)
        + torch.cos(alt) * torch.sin(slope) * torch.cos(az - aspect)
    )
    return shaded.clamp(0.0, 1.0)


def contour_lines(
    elevation: torch.Tensor,
    spacing: float = 20.0,
    width: float = 1.0,
    blur: int = 1,
) -> torch.Tensor:
    """Binary-ish contour mask [H,W] in [0,1] for every `spacing` elevation unit."""
    if spacing <= 0:
        raise ValueError("spacing must be > 0")
    z = _blur(elevation.to(torch.float32), k=blur)
    # Distance to nearest multiple of spacing
    nearest = torch.round(z / spacing) * spacing
    dist = (z - nearest).abs()
    # Inside a band of width/2 around the isoline
    mask = (dist < (width * 0.5)).to(torch.float32)
    return mask


def build_terrain_variants(
    elevation: torch.Tensor,
    mode: str = "hillshade",
    contour_spacing: float = 20.0,
    hillshade_strength: float = 0.55,
) -> Dict[str, Optional[torch.Tensor]]:
    """Return overlay tensors for a terrain mode. None means 'no overlay'.

    modes: flat | hillshade | contour | textured
    """
    e = elevation.to(torch.float32)
    out: Dict[str, Optional[torch.Tensor]] = {
        "hillshade_mod": None,
        "contour": None,
        "height": e,
    }
    if mode == "flat":
        return out
    if mode in ("hillshade", "textured"):
        hs = hillshade(e)
        # Modulate around mid-gray so it darkens/lightens without crushing
        out["hillshade_mod"] = hs * float(hillshade_strength)
    if mode == "contour":
        out["contour"] = contour_lines(e, spacing=contour_spacing)
    return out


def shade_rgb(elevation: torch.Tensor, low=(0.18, 0.32, 0.18), high=(0.72, 0.68, 0.55)) -> torch.Tensor:
    """Quick elevation color ramp [3,H,W]."""
    e = elevation.to(torch.float32)
    e = (e - e.min()) / max(float(e.max() - e.min()), 1e-6)
    r = e * (high[0] - low[0]) + low[0]
    g = e * (high[1] - low[1]) + low[1]
    b = e * (high[2] - low[2]) + low[2]
    return torch.stack([r, g, b], dim=0)
