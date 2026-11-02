"""Helpers shared by the training scripts."""
from __future__ import annotations

import torch
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing


def log_normal(shape, loc: float, scale: float, generator: torch.Generator, device) -> torch.Tensor:
    """Samples of exp(N(loc, scale^2))."""
    return (torch.randn(shape, generator=generator, device=device) * scale + loc).exp()


def clip_embed(image_encoder, feature_extractor, images: torch.Tensor) -> torch.Tensor:
    """images: (B, 3, H, W) in [-1, 1] -> (B, 1, 1024) CLIP image embeddings (SVD preprocessing)."""
    x = _resize_with_antialiasing(images.float(), (224, 224))
    x = (x + 1.0) / 2.0
    x = feature_extractor(images=x, do_normalize=True, do_center_crop=False, do_resize=False, do_rescale=False,
                          return_tensors="pt").pixel_values
    return image_encoder(x.to(images.device, image_encoder.dtype)).image_embeds.unsqueeze(1)
