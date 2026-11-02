"""EventHDR real-capture test set loader.

Folder layout (one folder per clip)::

    <root>/<clip>/hdr/npy/*.npy          linear HDR frames (H, W) or (H, W, 3); uint16 or float
    <events_root>/<clip>/events/npy/*.npy  per-frame event voxels (4, H, W) float32, one file per frame

A sample is a window of ``num_frames`` consecutive frames starting at ``x`` (``1 <= x <= N - T``;
event file ``x - 1`` is needed too). It provides:

* ``reference_pil`` — frame ``x`` rendered as an LDR image at ``reference_ev`` stops (default +7),
  sRGB-encoded, bilinearly resized to ``out_size``.
* ``pixel_values`` — the tone-mapped ground-truth frames, (T, 3, S, S) in [-1, 1], fp16.
* ``event_values`` — (T, 4, S*up_scale, S*up_scale) fp16 in [-1, 1]. For output frame k the four
  channels are nested windows of 1, 3, 5 and 7 fine bins centred on the last bin of event file
  ``x + k - 1``; all four are divided by the per-frame max-abs value.
* ``exposure_value`` — the EV fed to the exposure token (``token_ev``, default 2).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..utils.image_ops import log_tonemap, render_ldr, to_uint8
from .events import build_event_windows, load_voxel, normalize_per_frame_maxabs


def _load_hdr(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return arr


def _load_event_voxel(path: str, height: int, width: int, bins: int = 4) -> np.ndarray:
    v = load_voxel(path, bins)
    if v.shape[1:] != (height, width):
        v = np.stack([np.asarray(PIL.Image.fromarray(b).resize((width, height), PIL.Image.NEAREST)) for b in v])
    return v


class EventHDRTestDataset(Dataset):
    def __init__(self, root: str, events_root: Optional[str] = None, num_frames: int = 20, bins: int = 4,
                 out_size: int = 512, up_scale: float = 1.0, reference_ev: float = 7.0, token_ev: float = 2.0,
                 tonemap_k: float = 500.0, window_bins=(1, 3, 5, 7),
                 clips: Optional[List[str]] = None):
        self.root, self.events_root = root, events_root or root
        self.T, self.bins, self.out_size, self.up_scale = num_frames, bins, out_size, up_scale
        self.reference_ev, self.token_ev = reference_ev, token_ev
        self.tonemap_k = tonemap_k
        self.window_bins = tuple(window_bins)
        self.clips: List[Dict] = []
        for name in sorted(os.listdir(root)):
            if clips is not None and name not in clips:
                continue
            hdr_dir = os.path.join(root, name, "hdr", "npy")
            ev_dir = os.path.join(self.events_root, name, "events", "npy")
            if not (os.path.isdir(hdr_dir) and os.path.isdir(ev_dir)):
                continue
            hdr = sorted(os.listdir(hdr_dir)); ev = sorted(os.listdir(ev_dir))
            n = min(len(hdr), len(ev))
            if n < self.T + 1:
                continue
            self.clips.append(dict(name=name, hdr=[os.path.join(hdr_dir, f) for f in hdr[:n]],
                                   events=[os.path.join(ev_dir, f) for f in ev[:n]], n=n,
                                   count=n - self.T))  # starts x = 1 .. n - T
        self.offsets = np.cumsum([0] + [c["count"] for c in self.clips])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def locate(self, idx: int):
        c = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        return self.clips[c], 1 + idx - int(self.offsets[c])

    def clip_names(self) -> List[str]:
        return [c["name"] for c in self.clips]

    def __getitem__(self, idx: int) -> Dict:
        clip, x = self.locate(idx)
        T, S = self.T, self.out_size
        frames = [_load_hdr(clip["hdr"][x + k]) for k in range(T)]
        H, W = frames[0].shape[:2]

        ref_u8 = render_ldr(frames[0], self.reference_ev)
        reference_pil = PIL.Image.fromarray(ref_u8).resize((S, S), PIL.Image.BILINEAR)
        reference = torch.from_numpy(ref_u8).permute(2, 0, 1).float()[None] / 127.5 - 1.0
        reference = F.interpolate(reference, size=(S, S), mode="bilinear", align_corners=False)[0].half()

        gt = np.stack([to_uint8(log_tonemap(fr, self.tonemap_k)) for fr in frames])  # (T,H,W,3)
        gt = torch.from_numpy(gt).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        pixel_values = F.interpolate(gt, size=(S, S), mode="bilinear", align_corners=False).half()

        stack = np.concatenate([_load_event_voxel(clip["events"][x - 1 + k], H, W, self.bins) for k in range(T + 1)])
        coarse = normalize_per_frame_maxabs(build_event_windows(stack, T, self.bins, self.window_bins))
        ev_size = int(round(S * self.up_scale))
        event_values = F.interpolate(torch.from_numpy(coarse), size=(ev_size, ev_size), mode="nearest").half()

        return dict(folder=clip["name"], start_idx=x, hdr_indices=list(range(x, x + T)), reference_pil=reference_pil,
                    reference=reference, pixel_values=pixel_values, event_values=event_values,
                    exposure_value=float(self.token_ev))
