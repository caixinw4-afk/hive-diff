"""Small image / geometry helpers shared by the pipeline, datasets and evaluation."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    """Linear [0, 1] -> sRGB-encoded [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def render_ldr(hdr: np.ndarray, ev: float) -> np.ndarray:
    """Expose linear HDR by 2**ev, clip, sRGB-encode -> uint8 (rounded)."""
    y = srgb_oetf(hdr * (2.0 ** ev))
    return (255.0 * y + 0.5).astype(np.uint8)


def log_tonemap(hdr: np.ndarray, k: float = 500.0) -> np.ndarray:
    """Per-frame log tone-mapping used for the tone-mapped targets: log1p(k x / max) / log1p(k)."""
    norm = max(float(hdr.max()), 1e-8)
    return np.clip(np.log1p(k * hdr / norm) / np.log1p(k), 0.0, 1.0)


def to_uint8(x01: np.ndarray) -> np.ndarray:
    return (255.0 * np.clip(x01, 0.0, 1.0) + 0.5).astype(np.uint8)


def tile_windows(height: int, width: int, window: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Sliding windows ``(w_start, h_start, w_end, h_end)`` covering a ``height x width`` grid.

    Windows are ``window`` wide, stride ``int(window * (1 - overlap))``, row-major; the last
    window in each direction is shifted back so it ends exactly at the border. A grid no larger
    than the window yields one window.
    """
    if height <= window and width <= window:
        return [(0, 0, min(width, window), min(height, window))]
    stride = max(1, int(window * (1.0 - overlap)))

    def starts(size):
        if size <= window:
            return [0]
        n = (size - window) // stride + 1
        if (size - window) % stride:
            n += 1
        return [min(i * stride, size - window) for i in range(n)]

    return [(ws, hs, ws + window, hs + window) for hs in starts(height) for ws in starts(width)]


def gaussian_blend_weight(h: int, w: int, device=None, dtype=None) -> torch.Tensor:
    """Gaussian weight map (sigma = min(h, w) / 3, centred at (h/2, w/2)) for blending tiles."""
    sigma = min(h, w) / 3.0
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    wgt = torch.exp(-((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2) / (2.0 * sigma ** 2))
    return wgt.to(dtype) if dtype is not None else wgt
