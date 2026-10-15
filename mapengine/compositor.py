"""GPU tensor compositing — paste RGBA with alpha, zero PIL in the render loop.

Why: PIL paste per frame is slow and syncs the CPU. We blend on float
tensors [3,H,W] (or RGBA [4,H,W]) so a render loop never leaves the device.

Laws (from production):
  L1 — blend math on float tensors
  L2 — no PIL in the per-frame path
  L3 — caller controls z-order
  L5 — optional y-sort for depth (icons later on screen = nearer)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch


def alpha_blend(
    base: torch.Tensor,
    layer: torch.Tensor,
    x: int,
    y: int,
    alpha_scale: float = 1.0,
) -> torch.Tensor:
    """Blend RGBA layer [4,h,w] onto RGB base [3,H,W] at (x,y). In-place on clone.

    Precondition: 0 <= alpha_scale <= 1. Layer may hang off the canvas (clipped).
    """
    if base.dim() != 3 or base.shape[0] != 3:
        raise ValueError("base must be [3,H,W]")
    if layer.dim() != 3 or layer.shape[0] != 4:
        raise ValueError("layer must be [4,h,w]")

    out = base.clone()
    _, H, W = out.shape
    _, h, w = layer.shape

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return out

    lx0 = x0 - x
    ly0 = y0 - y
    lx1 = lx0 + (x1 - x0)
    ly1 = ly0 + (y1 - y0)

    src = layer[:, ly0:ly1, lx0:lx1]
    a = (src[3:4] * float(alpha_scale)).clamp(0.0, 1.0)  # [1,h,w]
    rgb = src[0:3]

    patch = out[:, y0:y1, x0:x1]
    out[:, y0:y1, x0:x1] = rgb * a + patch * (1.0 - a)
    return out


def paste_rgba(
    canvas: torch.Tensor,
    layer: torch.Tensor,
    x: int,
    y: int,
    alpha_scale: float = 1.0,
) -> torch.Tensor:
    """Alias of alpha_blend for call-site clarity in render loops."""
    return alpha_blend(canvas, layer, x, y, alpha_scale=alpha_scale)


def y_sort(boxes: Sequence[Tuple[int, int, torch.Tensor]]) -> List[Tuple[int, int, torch.Tensor]]:
    """Sort drawables by screen Y (painter's algorithm: far first).

    Each item is (x, y, rgba_tensor). Stable for equal y.
    """
    return sorted(boxes, key=lambda item: item[1])


class GPUCanvas:
    """RGB float canvas [3,H,W] with paste helpers. Device-aware (CUDA or CPU)."""

    def __init__(self, height: int, width: int, device: str | torch.device = "cpu", fill: float = 0.0):
        self.height = int(height)
        self.width = int(width)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.data = torch.full((3, self.height, self.width), float(fill), device=self.device)

    def paste_rgba(self, layer: torch.Tensor, x: int, y: int, alpha_scale: float = 1.0) -> None:
        self.data = alpha_blend(
            self.data.to("cpu") if self.data.device.type != "cpu" and layer.device.type == "cpu" else self.data,
            layer.to(self.data.device),
            x,
            y,
            alpha_scale=alpha_scale,
        )

    def to(self, device: str | torch.device) -> "GPUCanvas":
        self.device = torch.device(device) if isinstance(device, str) else device
        self.data = self.data.to(self.device)
        return self

    def numpy(self):
        return self.data.detach().cpu().permute(1, 2, 0).numpy()


def outline_alpha(icon: torch.Tensor, outline_px: int = 1, color: Tuple[float, float, float] = (0.0, 0.0, 0.0), alpha: float = 0.7) -> torch.Tensor:
    """Bake a thin outline into an RGBA icon [4,H,W]. Call once at preload — not per frame."""
    if icon.dim() != 3 or icon.shape[0] != 4:
        raise ValueError("icon must be [4,H,W]")
    alpha_ch = icon[3:4]
    k = 2 * outline_px + 1
    dilated = torch.nn.functional.max_pool2d(
        alpha_ch.unsqueeze(0), kernel_size=k, stride=1, padding=outline_px
    ).squeeze(0)
    outline_mask = (dilated - alpha_ch).clamp(0.0, 1.0)
    result = icon.clone()
    if max(color) > 1:
        r, g, b = (c / 255.0 for c in color)
    else:
        r, g, b = (float(c) for c in color)
    denom = (alpha_ch + outline_mask * alpha).clamp(min=1e-3)
    for i, v in enumerate((r, g, b)):
        result[i] = (icon[i] * alpha_ch + v * outline_mask * alpha) / denom
    result[3] = (alpha_ch + outline_mask * alpha).clamp(0.0, 1.0)
    return result
