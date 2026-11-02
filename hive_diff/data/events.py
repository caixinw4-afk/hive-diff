"""Event-tensor construction shared by all datasets.

Event files store per-frame-interval voxels with ``bins`` fine time bins. For output frame ``k``
we stack the files of frames ``t-1 .. t+T-1`` (``T+1`` files, ``bins*(T+1)`` fine bins) and
build ``len(widths)`` coarse channels, each the sum of a window of ``w`` fine bins centred on
``c_k = bins*k + bins-1`` (the last bin before frame ``k``): fine indices
``[c_k - w//2, c_k + w - w//2)``. Settings used by the provided datasets:

* ``widths=(1, 3, 5, 7)`` on 4-bin files (EventHDR evaluation / finetuning): symmetric windows.
* ``widths=(0, 4, 8, 12)`` on 12-bin files (synthetic training): channel 0 is empty (zeros).
* ``widths=(0, 2, 4, 6)`` on 12-bin files (real-video inference / LDR finetuning).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def build_event_windows(stack: np.ndarray, num_frames: int, bins: int, widths: Sequence[int]) -> np.ndarray:
    """``stack``: (bins*(num_frames+1), H, W) -> (num_frames, len(widths), H, W) float32."""
    out = np.zeros((num_frames, len(widths)) + stack.shape[1:], np.float32)
    for k in range(num_frames):
        c = bins * k + bins - 1
        for i, w in enumerate(widths):
            if w <= 0:
                continue
            lo, hi = max(c - w // 2, 0), min(c + (w - w // 2), stack.shape[0])
            out[k, i] = stack[lo:hi].sum(0)
    return out


def normalize_per_frame_maxabs(ev: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Divide every frame's channels jointly by that frame's max |value| (EventHDR eval rule)."""
    out = ev.astype(np.float32).copy()
    for k in range(out.shape[0]):
        m = float(np.abs(out[k]).max())
        if m > 0:
            out[k] /= (m + eps)
    return out


def normalize_per_channel_quantile(ev: np.ndarray, q: float = 0.995) -> np.ndarray:
    """Per frame and channel: divide by the ``q``-quantile of |x| (q clipped to [0.9, 0.999]),
    then clip to [-1, 1] (training rule)."""
    q = float(np.clip(q, 0.90, 0.999))
    out = np.empty_like(ev, dtype=np.float32)
    for k in range(ev.shape[0]):
        for c in range(ev.shape[1]):
            x = ev[k, c]
            s = float(np.quantile(np.abs(x), q)) if x.size else 1.0
            if not np.isfinite(s) or s <= 0:
                s = 1.0
            out[k, c] = np.clip(x / s, -1.0, 1.0)
    return out


def load_voxel(path: str, bins: int) -> np.ndarray:
    """Load one event file as (bins, H, W) float32, accepting (H, W, bins), (H, W) or (1, H, W)."""
    v = np.asarray(np.load(path, mmap_mode="r")).astype(np.float32)
    if v.ndim == 2:
        v = v[None]
    elif v.ndim == 3 and v.shape[0] != bins and v.shape[-1] == bins:
        v = v.transpose(2, 0, 1)
    if v.shape[0] > bins:
        v = v[:bins]
    elif v.shape[0] < bins:
        v = np.concatenate([v, np.zeros((bins - v.shape[0],) + v.shape[1:], np.float32)])
    return v
