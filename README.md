# gpu-map-engine

**Procedural 2.5D map toolkit: true perspective warp, GPU compositing, water, hillshade.**

[![CI](https://github.com/fabrikalunnogosveta-del/gpu-map-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/fabrikalunnogosveta-del/gpu-map-engine/actions)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen)

**Core extracted from a production media pipeline.** The full orchestrator and neural stylization stay proprietary — this is the graphics math as a standalone library.

---

## The problem we solved

Constant-speed fake maps and naive affine warps look synthetic the moment you tilt the camera.

| Problem (before) | Solution (this library) |
|---|---|
| Affine / corner-pin warp breaks at 30–50° tilt | **True 4-point homography** (DLT) + `grid_sample` |
| Icons drift when perspective changes | **geo↔screen contract** (round-trip ≤2px), tested |
| PIL paste per frame = slow, CPU-sync | **Tensor alpha blend** — zero PIL in the render loop |
| Water looks like static noise | **Deterministic multi-octave fields**, scrolled, seeded |
| Raw DEM = sandpaper | **Hillshade with blurred gradients** + contour overlays |
| `hash()` seeds break across processes | **sha256 stable seeds** |

### Constant-speed vs human… wait, map edition

| Naive map | This library |
|---|---|
| Flat 2D rectangle | Tilted 2.5D table-top |
| Per-frame PIL `paste` | CUDA/CPU float tensor blend |
| Random water every run | Seeded fields — same seed, same frame |
| Random hillshade noise | Large-scale gradients only |

---

## Install

```bash
pip install -e .
python -m pytest tests/ -q
```

CPU is enough. CUDA is optional and auto-used when available.

---

## Quickstart

```python
import torch
from mapengine import GPUCanvas, WaterTexture, animate_water, hillshade, paste_rgba
from mapengine.perspective import apply as tilt

# Flat "map" frame → 2.5D tilt
flat = torch.rand(3, 256, 384)
tilted = tilt(flat, tilt_deg=42.0)

# GPU canvas + opaque tile paste (alpha path)
canvas = GPUCanvas(256, 384)
tile = torch.ones(4, 32, 32)  # RGBA
canvas.paste_rgba(tile, 40, 60)

# Deterministic water
field = animate_water(WaterTexture.from_preset("deep_ocean"), 64, 64, t=0.0, seed="demo")

# Hillshade from elevation
elev = torch.zeros(64, 64)
elev[20:40, 20:40] = 500.0
hs = hillshade(elev)
print(tilted.shape, field.shape, hs.shape)
```

Run the example:

```bash
python examples/quickstart.py
```

---

## Architecture

```
flat RGB frame
     │
     ▼
perspective.apply  ── homography (DLT 4-pt) ──► 2.5D trapezoid
     │
     ├─ compositor.GPUCanvas / paste_rgba   (alpha blend, no PIL)
     ├─ water.animate_water                 (seeded fields)
     └─ terrain.hillshade / contour_lines   (DEM overlays)
```

### Design choices (ADR-style)

1. **Homography, not affine** — tilt is a projective transform. Affine shears corners wrong; DLT 4-point matches `cv2.warpPerspective` without the OpenCV dependency.
2. **Contract tests on coordinates** — if `geo_to_screen` and `screen_to_geo` disagree, every icon drifts. We lock that with a round-trip test.
3. **Zero PIL in the frame loop** — paste math runs on float tensors. PIL is fine for export, not for 24fps.
4. **Stable seeds** — `hash()` is randomized per process. Water must be bit-identical across workers → `sha256`.
5. **Hillshade blurs first** — raw DEM gradients look like noise on video; large-scale shading reads as terrain.

---

## Testing philosophy

The suite (**16 tests**) proves **behavioral contracts**, not just that code runs:

- homography identity + inverse round-trip (points)
- warp keeps near-side content
- depth_scale monotonic top→bottom
- alpha blend correctness + off-canvas clip
- y-sort painter order
- water bit-determinism for fixed (seed, t)
- every preset builds
- hillshade stays in [0,1]
- terrain modes produce expected overlays
- seeds are process-stable

```bash
python -m pytest tests/ -q
```

---

## What this is / is not

| ✅ Is | ❌ Is not |
|---|---|
| Procedural graphics math | A video editor |
| Library with tests + CI | The full production orchestrator |
| Seeded / reproducible | Neural style transfer weights |
| CPU + optional CUDA | A hosted SaaS |

---

## License

MIT
