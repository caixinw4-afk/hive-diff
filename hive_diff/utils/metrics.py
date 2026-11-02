"""Tone-mapped image metrics: PSNR, histogram-matched PSNR, SSIM and LPIPS (via pyiqa)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse < 1e-10 else -10.0 * np.log10(mse)


def psnr_hist_matched(recon: np.ndarray, gt: np.ndarray) -> float:
    """PSNR after matching the reconstruction's per-channel histogram to the ground truth."""
    from skimage.exposure import match_histograms
    matched = match_histograms(recon, gt, channel_axis=-1)
    return psnr(matched, gt)


class ToneMappedMetrics:
    """All inputs are float RGB images in [0, 1] with shape (H, W, 3)."""

    def __init__(self, device: Optional[str] = None, use_pyiqa: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ssim = self.lpips = None
        if use_pyiqa:
            import pyiqa
            self.ssim = pyiqa.create_metric("ssim", device=self.device)
            self.lpips = pyiqa.create_metric("lpips", device=self.device)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()[None].to(self.device)

    @torch.no_grad()
    def __call__(self, recon: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        out = {"PSNR_tm": psnr(recon, gt), "PSNR_tm_hist": psnr_hist_matched(recon, gt)}
        if self.ssim is not None:
            r, g = self._to_tensor(recon), self._to_tensor(gt)
            out["SSIM_tm"] = float(self.ssim(r, g).item())
            out["LPIPS_tm"] = float(self.lpips(r, g).item())
        return out
