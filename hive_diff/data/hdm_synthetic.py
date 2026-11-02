"""Synthetic training data: camera-shake HDR video (HdM-HDR-2014 footage) with simulated events.

Layout (see ``data_prep/hdm_synthetic/README.md`` for how to generate it)::

    <video_root>/<scene>/hdr_shaken_exr/*.exr    linear HDR frames
    <tone_root>/<scene>/preview_png/*.png        graded/tone-mapped target frames (same count)
    <events_root>/<scene>/frames_np/*.npy        (H, W, 12) signed event counts per frame interval

Two target modes:

* ``target="ldr"`` (stage 1, Generator): the targets are the HDR frames rendered at the same
  random exposure as the reference (so the model learns to continue an LDR video from events);
  reference = first target frame. Start index ``t >= 1``.
* ``target="tonemapped"`` (stage 2, Shader): the targets are the graded PNG frames; the previous
  clip (frames ``t-T .. t-1``) is also returned for clip-continuity training. Start ``t >= T``.

Events for frame ``k`` are nested windows of fine bins centred on the last bin of the frame's
own event file (see :mod:`hive_diff.data.events`); they may be warped by a small random homography to
simulate frame/event misalignment. Frames are never warped.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset

from ..utils.image_ops import render_ldr
from .events import build_event_windows, load_voxel, normalize_per_channel_quantile

cv2.setNumThreads(0)


def read_hdr(path: str) -> np.ndarray:
    """Read an EXR/HDR/TIFF/PNG frame as linear float32 RGB (H, W, 3)."""
    if path.lower().endswith(".exr"):
        import OpenEXR  # pip install OpenEXR (>=3.2); pip wheels of OpenCV cannot read EXR
        with OpenEXR.File(path) as f:
            ch = f.channels()
            if "RGB" in ch:
                img = np.asarray(ch["RGB"].pixels)[..., :3]
            elif "RGBA" in ch:
                img = np.asarray(ch["RGBA"].pixels)[..., :3]
            else:
                img = np.stack([np.asarray(ch[c].pixels) for c in ("R", "G", "B")], axis=-1)
        img = img.astype(np.float32)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        else:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
        if np.issubdtype(img.dtype, np.integer):
            img = img.astype(np.float32) / float(np.iinfo(img.dtype).max)
    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(img, 0.0)


def random_homography(width: int, height: int, strength: float = 0.25, rng=None,
                      base=(1.0, 0.02, 0.01, 0.01, 1e-4)) -> np.ndarray:
    """Near-identity homography about the image centre. ``base`` = max rotation (deg), scale,
    shear, translation (fraction of size) and perspective; each is drawn uniformly in
    ``+-strength*base``. ``rng`` may be ``random`` or a ``numpy.random.RandomState``."""
    rng = rng or random
    u = lambda a: float(rng.uniform(-a, a))  # noqa: E731
    rot_deg, sc, sh, tr, pe = base
    rot = np.deg2rad(u(rot_deg * strength))
    sx, sy = 1 + u(sc * strength), 1 + u(sc * strength)
    shear = u(sh * strength)
    tx, ty = u(tr * strength) * width, u(tr * strength) * height
    px, py = u(pe * strength), u(pe * strength)
    A = np.array([[sx * np.cos(rot), -sy * np.sin(rot) + shear, 0], [sx * np.sin(rot), sy * np.cos(rot), 0], [0, 0, 1]])
    P = np.array([[1, 0, tx], [0, 1, ty], [px, py, 1]], dtype=np.float64)
    cx, cy = width / 2.0, height / 2.0
    T_fwd = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    T_bwd = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    return T_fwd @ P @ A @ T_bwd


def warp_planes(planes: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = planes.shape[-2:]
    return np.stack([cv2.warpPerspective(p, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                     for p in planes])


class HdmSyntheticDataset(Dataset):
    def __init__(self, video_root: str, tone_root: str, events_root: str, num_frames: int = 20, target: str = "tonemapped",
                 fine_bins: int = 12, crop: Tuple[int, int] = (512, 512), random_crop: bool = True,
                 ev_range: Optional[Tuple[int, int]] = None, window_bins: Sequence[int] = (0, 4, 8, 12),
                 event_quantile: float = 0.995, warp_prob: float = 0.75, warp_strength: float = 0.25,
                 scenes: Optional[List[str]] = None):
        assert target in ("tonemapped", "ldr")
        self.T, self.target, self.bins = num_frames, target, fine_bins
        self.crop_w, self.crop_h, self.random_crop = crop[0], crop[1], random_crop
        self.ev_range = ev_range or ((-5, 4) if target == "tonemapped" else (-7, 4))
        self.window_bins, self.event_quantile = tuple(window_bins), event_quantile
        self.warp_prob, self.warp_strength = warp_prob, warp_strength
        self.min_start = self.T if target == "tonemapped" else 1
        self.scenes: List[Dict] = []
        for name in sorted(os.listdir(video_root)):
            if scenes is not None and name not in scenes:
                continue
            exr_dir = os.path.join(video_root, name, "hdr_shaken_exr")
            png_dir = os.path.join(tone_root, name, "preview_png")
            ev_dir = os.path.join(events_root, name, "frames_np")
            if not all(os.path.isdir(d) for d in (exr_dir, png_dir, ev_dir)):
                continue
            exr = sorted(f for f in os.listdir(exr_dir) if f.lower().endswith(".exr"))
            png = sorted(f for f in os.listdir(png_dir) if f.lower().endswith(".png"))
            npy = sorted(f for f in os.listdir(ev_dir) if f.lower().endswith(".npy"))
            n = min(len(exr), len(npy), len(png))
            if n < self.min_start + self.T:
                continue
            self.scenes.append(dict(name=name, exr=[os.path.join(exr_dir, f) for f in exr[:n]],
                                    png=[os.path.join(png_dir, f) for f in png[:n]],
                                    npy=[os.path.join(ev_dir, f) for f in npy[:n]], n=n,
                                    count=n - self.T - self.min_start + 1))
        self.offsets = np.cumsum([0] + [s["count"] for s in self.scenes])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _crop_origin(self, W0: int, H0: int) -> Tuple[int, int]:
        if not self.random_crop or W0 <= self.crop_w or H0 <= self.crop_h:
            return max((W0 - self.crop_w) // 2, 0), max((H0 - self.crop_h) // 2, 0)
        return random.randint(0, W0 - self.crop_w), random.randint(0, H0 - self.crop_h)

    def _crop(self, arr: np.ndarray, cx: int, cy: int) -> np.ndarray:
        """Crop an (H, W, C) image."""
        return arr[cy:cy + self.crop_h, cx:cx + self.crop_w]

    def _crop_planes(self, arr: np.ndarray, cx: int, cy: int) -> np.ndarray:
        """Crop a (C, H, W) stack of planes."""
        return arr[:, cy:cy + self.crop_h, cx:cx + self.crop_w]

    def _tone_frames(self, scene: Dict, indices: Sequence[int], size: Tuple[int, int], cx: int, cy: int) -> torch.Tensor:
        frames = []
        for i in indices:
            img = PIL.Image.open(scene["png"][int(np.clip(i, 0, scene["n"] - 1))]).convert("RGB")
            if img.size != size:
                img = img.resize(size)
            arr = self._crop(np.asarray(img, dtype=np.float32), cx, cy) / 127.5 - 1.0
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))
        return torch.stack(frames).half()

    def __getitem__(self, idx: int) -> Dict:
        s = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        scene = self.scenes[s]
        t = self.min_start + int(idx - self.offsets[s])
        T, B = self.T, self.bins

        hdr0 = read_hdr(scene["exr"][t])
        H0, W0 = hdr0.shape[:2]
        cx, cy = self._crop_origin(W0, H0)
        H_warp = random_homography(W0, H0, self.warp_strength) if random.random() < self.warp_prob else None
        ev = random.randint(*self.ev_range)

        out: Dict = dict(folder=scene["name"], start_idx=t, exposure_value=float(ev))
        if self.target == "ldr":
            ldr = [render_ldr(hdr0, ev)] + [render_ldr(read_hdr(scene["exr"][min(t + k, scene["n"] - 1)]), ev) for k in range(1, T)]
            ldr = np.stack([self._crop(f, cx, cy) for f in ldr]).astype(np.float32) / 127.5 - 1.0
            out["ldr_gt"] = torch.from_numpy(ldr).permute(0, 3, 1, 2).half()
            out["reference"] = out["ldr_gt"][0]
        else:
            ref = self._crop(render_ldr(hdr0, ev), cx, cy).astype(np.float32) / 127.5 - 1.0
            out["reference"] = torch.from_numpy(ref).permute(2, 0, 1).half()
            out["pixel_values"] = self._tone_frames(scene, range(t, t + T), (W0, H0), cx, cy)
            out["prev_pixel_values"] = self._tone_frames(scene, range(t - T, t), (W0, H0), cx, cy)

        stack = np.zeros(((T + 1) * B, self.crop_h, self.crop_w), np.float32)
        for k in range(T + 1):
            fi = t + k - 1
            if fi >= scene["n"]:
                continue
            vox = load_voxel(scene["npy"][fi], B)
            if vox.shape[1:] != (H0, W0):
                vox = np.stack([cv2.resize(p, (W0, H0), interpolation=cv2.INTER_LINEAR) for p in vox])
            if H_warp is not None:
                vox = warp_planes(vox, H_warp)
            stack[k * B:(k + 1) * B] = self._crop_planes(vox, cx, cy)
        coarse = build_event_windows(stack, T, B, self.window_bins)
        out["event_values"] = torch.from_numpy(normalize_per_channel_quantile(coarse, self.event_quantile)).half()
        return out
