"""LDR video + 12-bin event voxels (own captures, DSEC-style exports) for finetuning and demos.

Layout::

    <frames_root>/<scene>/*.png                  LDR frames, sorted lexicographically
    <events_root>/<scene>/[voxels_npy/]*.npy     one voxel per frame interval, (12, H, W) (file i precedes frame i)

``mode="train"``: frames are resized to ``base_size`` and a shared random ``crop`` is taken; the
reference is frame ``t`` and the target the ``T`` frames ``t .. t+T-1`` (Generator finetuning,
LDR-continuation target). ``mode="infer"``: the reference is frame ``t`` resized to ``out_size``
(LANCZOS) and events are produced at ``event_scale`` times that size for tiled generation.
Event channels: windows of 0/2/4/6 fine bins centred on the last bin of each frame's own file,
per-channel quantile normalisation (0.995 train / 0.999 infer).
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset

from .events import build_event_windows, load_voxel, normalize_per_channel_quantile


def _resize_planes(planes: np.ndarray, size: Tuple[int, int], resample=PIL.Image.LANCZOS) -> np.ndarray:
    if planes.shape[1:] == (size[1], size[0]):
        return planes
    return np.stack([np.asarray(PIL.Image.fromarray(p).resize(size, resample), dtype=np.float32) for p in planes])


class LdrVideoDataset(Dataset):
    def __init__(self, frames_root: str, events_root: str, num_frames: int = 20, mode: str = "infer", bins: int = 12,
                 out_size: int = 512, base_size: int = 1024, crop: int = 512, event_scale: float = 2.0,
                 window_bins: Sequence[int] = (0, 2, 4, 6), event_quantile: Optional[float] = None,
                 exposure_value: float = 0.0, scenes: Optional[List[str]] = None):
        assert mode in ("train", "infer")
        self.T, self.mode, self.bins = num_frames, mode, bins
        self.out_size, self.base_size, self.crop, self.event_scale = out_size, base_size, crop, event_scale
        self.window_bins = tuple(window_bins)
        self.event_quantile = event_quantile if event_quantile is not None else (0.995 if mode == "train" else 0.999)
        self.exposure_value = exposure_value
        self.scenes: List[Dict] = []
        for name in sorted(os.listdir(frames_root)):
            if scenes is not None and name not in scenes:
                continue
            fdir = os.path.join(frames_root, name)
            edir = os.path.join(events_root, name, "voxels_npy")
            if not os.path.isdir(edir):
                edir = os.path.join(events_root, name)
            if not (os.path.isdir(fdir) and os.path.isdir(edir)):
                continue
            png = sorted(f for f in os.listdir(fdir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
            npy = sorted(f for f in os.listdir(edir) if f.lower().endswith(".npy"))
            n = min(len(png), len(npy))
            if n < self.T + 1:
                continue
            self.scenes.append(dict(name=name, png=[os.path.join(fdir, f) for f in png[:n]],
                                    npy=[os.path.join(edir, f) for f in npy[:n]], n=n, count=n - self.T))
        self.offsets = np.cumsum([0] + [s["count"] for s in self.scenes])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def scene_names(self) -> List[str]:
        return [s["name"] for s in self.scenes]

    def index_of(self, scene: str, start: int) -> int:
        i = self.scene_names().index(scene)
        return int(self.offsets[i]) + start - 1

    def __getitem__(self, idx: int) -> Dict:
        s = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        scene = self.scenes[s]
        t = 1 + int(idx - self.offsets[s])
        T, B = self.T, self.bins
        out: Dict = dict(folder=scene["name"], start_idx=t, exposure_value=float(self.exposure_value))

        if self.mode == "infer":
            ref = PIL.Image.open(scene["png"][t]).convert("RGB").resize((self.out_size, self.out_size), PIL.Image.LANCZOS)
            out["reference_pil"] = ref
            out["reference_path"] = scene["png"][t]
            out["reference"] = (torch.from_numpy(np.asarray(ref, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0).half()
            es = int(round(self.out_size * self.event_scale))
            size, cx, cy, cw, ch = (es, es), 0, 0, es, es
        else:
            size = (self.base_size, self.base_size)
            cw = ch = self.crop
            cx = random.randint(0, self.base_size - cw) if self.base_size > cw else 0
            cy = random.randint(0, self.base_size - ch) if self.base_size > ch else 0
            frames = []
            for k in range(T):
                img = PIL.Image.open(scene["png"][min(t + k, scene["n"] - 1)]).convert("RGB").resize(size)
                frames.append(np.asarray(img, dtype=np.float32)[cy:cy + ch, cx:cx + cw] / 127.5 - 1.0)
            out["ldr_gt"] = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).half()
            out["reference"] = out["ldr_gt"][0]

        stack = np.zeros(((T + 1) * B, ch, cw), np.float32)
        for k in range(T + 1):
            fi = t + k - 1
            if fi >= scene["n"]:
                continue
            vox = _resize_planes(load_voxel(scene["npy"][fi], B), size)
            stack[k * B:(k + 1) * B] = vox[:, cy:cy + ch, cx:cx + cw]
        coarse = build_event_windows(stack, T, B, self.window_bins)
        out["event_values"] = torch.from_numpy(normalize_per_channel_quantile(coarse, self.event_quantile)).half()
        return out
