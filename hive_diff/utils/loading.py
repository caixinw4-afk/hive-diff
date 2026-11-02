"""Build a :class:`HiveDiffPipeline` from the SVD base weights and HiVE-Diff checkpoints."""
from __future__ import annotations

import os
from typing import Optional

import torch
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..models import ControlledSVDUNet, EventControlNet, ExposureTokenMLP, ShaderControlNet
from ..pipelines import HiveDiffPipeline


def load_svd_components(svd_path: str, dtype: torch.dtype = torch.float16):
    variant = "fp16" if dtype == torch.float16 else None
    feature_extractor = CLIPImageProcessor.from_pretrained(svd_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(svd_path, subfolder="image_encoder", torch_dtype=dtype)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae", variant=variant, torch_dtype=dtype)
    unet = ControlledSVDUNet.from_pretrained(svd_path, subfolder="unet", variant=variant, torch_dtype=dtype)
    scheduler = EulerDiscreteScheduler.from_pretrained(svd_path, subfolder="scheduler")
    return feature_extractor, image_encoder, vae, unet, scheduler


def load_shader_dir(shader_dir: str, dtype: torch.dtype = torch.float32):
    """``shader_dir`` holds ``shader/`` (ControlNet weights) and optionally ``ev_token.pt``."""
    shader = ShaderControlNet.from_pretrained(os.path.join(shader_dir, "shader"), torch_dtype=dtype)
    token_path = os.path.join(shader_dir, "ev_token.pt")
    token = ExposureTokenMLP.load(token_path).to(dtype) if os.path.exists(token_path) else None
    return shader, token


def load_pipeline(svd_path: str, generator_path: str, shader_dir: Optional[str] = None, device: str = "cuda",
                  dtype: torch.dtype = torch.float16, controlnet_dtype: Optional[torch.dtype] = torch.float32,
                  residual_mode: str = "multi") -> HiveDiffPipeline:
    """``dtype`` is used for the SVD base (UNet / VAE / CLIP). ``controlnet_dtype`` (default fp32) is
    used for the two ControlNets and the exposure token; when it differs from ``dtype`` the pipeline
    runs them under autocast. Pass ``controlnet_dtype=None`` to use ``dtype`` everywhere (faster, less memory)."""
    feature_extractor, image_encoder, vae, unet, scheduler = load_svd_components(svd_path, dtype)
    unet.residual_mode = residual_mode
    controlnet_dtype = controlnet_dtype or dtype
    generator = EventControlNet.from_pretrained(generator_path, torch_dtype=controlnet_dtype)
    shader = token = None
    if shader_dir:
        shader, token = load_shader_dir(shader_dir, controlnet_dtype)
    pipe = HiveDiffPipeline(vae=vae, image_encoder=image_encoder, unet=unet, event_controlnet=generator,
                            shader_controlnet=shader, exposure_token=token, scheduler=scheduler,
                            feature_extractor=feature_extractor)
    pipe.set_progress_bar_config(disable=True)
    for m in (vae, image_encoder, unet, generator, shader, token):
        if m is not None:
            m.requires_grad_(False)
    return pipe.to(device)
