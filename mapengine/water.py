"""Procedural water — deterministic noise fields, zero per-frame GPU allocs.

Production pattern: precompute a phase buffer once, then per frame only
offset / multiply. Hash-based seeds must be process-stable (sha256, not hash()).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch


def stable_seed(*parts: str) -> int:
    """Process-stable seed (python hash() is randomized per process)."""
    h = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


WATER_PRESETS: Dict[str, Dict] = {
    "glass_smooth": {"noise": "simplex", "scale": 0.5, "octaves": 2, "speed": 0.1},
    "silk_ripple": {"noise": "simplex", "scale": 1.2, "octaves": 3, "speed": 0.3},
    "deep_ocean": {"noise": "fractal", "scale": 0.8, "octaves": 2, "speed": 0.08},
    "storm_surge": {"noise": "fractal", "scale": 3.5, "octaves": 6, "speed": 2.0},
    "river_current": {"noise": "ridges", "scale": 4.0, "octaves": 4, "speed": 1.2},
    "lagoon_mirror": {"noise": "simplex", "scale": 0.3, "octaves": 1, "speed": 0.06},
}


def _value_noise(shape, scale: float, octaves: int, seed: int, device="cpu") -> torch.Tensor:
    """Cheap value-noise field in [0,1], multi-octave. Not Perlin — fine for water."""
    h, w = shape
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    acc = torch.zeros(h, w, device="cpu")
    amp = 1.0
    total = 0.0
    res = 2
    for _ in range(max(1, int(octaves))):
        gh, gw = max(2, int(res)), max(2, int(res * w / max(h, 1)))
        grid = torch.rand(gh, gw, generator=gen)
        up = torch.nn.functional.interpolate(
            grid.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=True
        ).squeeze()
        acc = acc + up * amp
        total += amp
        amp *= 0.5
        res *= 2
    field = acc / max(total, 1e-6)
    return field.to(device)


def _fractal(shape, scale: float, octaves: int, seed: int, device="cpu") -> torch.Tensor:
    return _value_noise(shape, scale * 4, octaves, seed, device=device)


def _ridges(shape, scale: float, octaves: int, seed: int, device="cpu") -> torch.Tensor:
    n = _fractal(shape, scale, octaves, seed, device=device)
    return (1.0 - (n * 2.0 - 1.0).abs()).clamp(0.0, 1.0)


@dataclass
class WaterTexture:
    name: str
    noise: str = "simplex"
    scale: float = 1.0
    octaves: int = 3
    speed: float = 0.3

    @classmethod
    def from_preset(cls, name: str) -> "WaterTexture":
        if name not in WATER_PRESETS:
            raise KeyError(f"unknown water preset: {name}")
        p = WATER_PRESETS[name]
        return cls(name=name, noise=p["noise"], scale=p["scale"], octaves=p["octaves"], speed=p["speed"])

    def field(self, height: int, width: int, seed: str = "water", device="cpu") -> torch.Tensor:
        s = stable_seed(self.name, seed, str(self.octaves))
        if self.noise == "fractal":
            return _fractal((height, width), self.scale, self.octaves, s, device=device)
        if self.noise == "ridges":
            return _ridges((height, width), self.scale, self.octaves, s, device=device)
        return _value_noise((height, width), self.scale, self.octaves, s, device=device)


def animate_water(
    base: WaterTexture,
    height: int,
    width: int,
    t: float,
    seed: str = "water",
    device="cpu",
) -> torch.Tensor:
    """Water height-ish field at time t (seconds). Deterministic for fixed t,seed."""
    # Two shifted fields scrolled toward each other = cheap "breathing" water.
    shift = int(round(math.sin(t * base.speed * 2.0 * math.pi) * 3))
    a = base.field(height, width, seed=seed, device=device)
    b = base.field(height, width, seed=seed + ":b", device=device)
    if shift != 0:
        a = torch.roll(a, shifts=shift, dims=1)
        b = torch.roll(b, shifts=-shift, dims=1)
    return ((a + b) * 0.5).clamp(0.0, 1.0)


def shade_water(field: torch.Tensor, tint=(0.15, 0.35, 0.55)) -> torch.Tensor:
    """Map [H,W] water field to RGB [3,H,W]."""
    r, g, b = tint
    h = field.clamp(0, 1)
    return torch.stack([h * r + 0.05, h * g + 0.08, h * b + 0.12], dim=0)
