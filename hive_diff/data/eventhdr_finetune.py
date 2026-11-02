"""EventHDR real-capture data for finetuning the Generator (LDR-continuation target).

Same folder layout as :class:`EventHDRTestDataset` (``<root>/<clip>/hdr/npy`` + ``events/npy``
with 4-bin voxels). Each item renders ``T`` HDR frames at a fixed exposure (+7 EV by default),
resizes them to ``out_size`` (no crop), uses frame ``t`` as the reference and the ``T`` frames
as the denoising target. Events use the centred 1/3/5/7-bin windows, optionally warped by one
fixed homography (``warp_seed``, default 42) and normalised
per channel by the 0.995 |x| quantile.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.image_ops import render_ldr
from .events import build_event_windows, load_voxel, normalize_per_channel_quantile
from .eventhdr_test import _load_hdr
from .hdm_synthetic import random_homography, warp_planes

cv2.setNumThreads(0)


class EventHDRFinetuneDataset(Dataset):
    def __init__(self, root: str, num_frames: int = 20, bins: int = 4, out_size: int = 512, exposure_ev: float = 7.0,
                 window_bins: Sequence[int] = (1, 3, 5, 7), event_quantile: float = 0.995, warp_seed: Optional[int] = 42,
                 clips: Optional[List[str]] = None):
        self.T, self.bins, self.S, self.ev = num_frames, bins, out_size, exposure_ev
        self.window_bins, self.event_quantile = tuple(window_bins), event_quantile
        self.warp = None
        if warp_seed is not None:
            rs = np.random.RandomState(warp_seed)
            self.warp = random_homography(out_size, out_size, strength=1.0, rng=rs, base=(1.0, 0.03, 0.01, 0.015, 1e-4))
        self.clips: List[Dict] = []
        for name in sorted(os.listdir(root)):
            if clips is not None and name not in clips:
                continue
            hdr_dir, ev_dir = os.path.join(root, name, "hdr", "npy"), os.path.join(root, name, "events", "npy")
            if not (os.path.isdir(hdr_dir) and os.path.isdir(ev_dir)):
                continue
            hdr, ev = sorted(os.listdir(hdr_dir)), sorted(os.listdir(ev_dir))
            n = min(len(hdr), len(ev))
            if n < self.T + 1:
                continue
            self.clips.append(dict(name=name, hdr=[os.path.join(hdr_dir, f) for f in hdr[:n]],
                                   events=[os.path.join(ev_dir, f) for f in ev[:n]], n=n, count=n - self.T))
        self.offsets = np.cumsum([0] + [c["count"] for c in self.clips])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, idx: int) -> Dict:
        c = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        clip = self.clips[c]
        t = 1 + int(idx - self.offsets[c])
        T, B, S = self.T, self.bins, self.S
        frames = []
        for k in range(T):
            hdr = _load_hdr(clip["hdr"][min(t + k, clip["n"] - 1)])
            frames.append(cv2.resize(render_ldr(hdr, self.ev), (S, S), interpolation=cv2.INTER_LINEAR))
        ldr = torch.from_numpy(np.stack(frames).astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2).half()

        stack = np.zeros(((T + 1) * B, S, S), np.float32)
        for k in range(T + 1):
            fi = t + k - 1
            if fi >= clip["n"]:
                continue
            vox = load_voxel(clip["events"][fi], B)
            stack[k * B:(k + 1) * B] = np.stack([cv2.resize(p, (S, S), interpolation=cv2.INTER_LINEAR) for p in vox])
        coarse = build_event_windows(stack, T, B, self.window_bins)
        if self.warp is not None:
            coarse = np.stack([warp_planes(f, self.warp) for f in coarse])
        events = torch.from_numpy(normalize_per_channel_quantile(coarse, self.event_quantile)).half()
        return dict(folder=clip["name"], start_idx=t, reference=ldr[0], ldr_gt=ldr, event_values=events,
                    exposure_value=float(self.ev))
