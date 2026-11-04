"""HIVE-Diff inference pipeline: reference LDR frame + events -> HDR (tone-mapped) video clip.

Built on the stock Stable Video Diffusion components (VAE, CLIP image encoder, UNet weights,
Euler scheduler). The custom parts are the two ControlNets, the exposure token and the
sampling loop (per-frame classifier-free guidance, optional tiled denoising for large frames,
and latent conditioning on the previous clip for long videos).
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import List, Optional, Union

import PIL.Image
import torch
from diffusers import AutoencoderKLTemporalDecoder, DiffusionPipeline, EulerDiscreteScheduler
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    retrieve_timesteps,
)
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..models import ControlledSVDUNet, EventControlNet, ExposureTokenMLP, ShaderControlNet
from ..utils.image_ops import gaussian_blend_weight, tile_windows


@dataclass
class HiveDiffOutput(BaseOutput):
    frames: Union[List[List[PIL.Image.Image]], torch.Tensor]
    latents: torch.Tensor  # (B, T, 4, h, w) un-scaled VAE latents, for chaining clips


class HiveDiffPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "image_encoder->event_controlnet->shader_controlnet->unet->vae"

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: ControlledSVDUNet,
        event_controlnet: EventControlNet,
        shader_controlnet: Optional[ShaderControlNet],
        exposure_token: Optional[ExposureTokenMLP],
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae, image_encoder=image_encoder, unet=unet, event_controlnet=event_controlnet,
            shader_controlnet=shader_controlnet, exposure_token=exposure_token,
            scheduler=scheduler, feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=self.vae_scale_factor)

    # ------------------------------------------------------------------ conditioning
    def _encode_clip_image(self, image: PIL.Image.Image, device, dtype, do_cfg: bool) -> torch.Tensor:
        x = self.video_processor.pil_to_numpy(image)
        x = self.video_processor.numpy_to_pt(x) * 2.0 - 1.0
        x = _resize_with_antialiasing(x, (224, 224))
        x = (x + 1.0) / 2.0
        x = self.feature_extractor(images=x, do_normalize=True, do_center_crop=False, do_resize=False,
                                   do_rescale=False, return_tensors="pt").pixel_values
        emb = self.image_encoder(x.to(device=device, dtype=dtype)).image_embeds.unsqueeze(1)  # (1,1,1024)
        if do_cfg:
            emb = torch.cat([torch.zeros_like(emb), emb])
        return emb

    def _exposure_embedding(self, ev: float, device, dtype, do_cfg: bool, scale: float) -> torch.Tensor:
        if self.exposure_token is None:
            return 0.0
        tok = self.exposure_token(torch.tensor([float(ev)], device=device)) * scale
        tok = tok.to(dtype).unsqueeze(1)  # (1,1,1024)
        if do_cfg:
            tok = torch.cat([torch.zeros_like(tok), tok])
        return tok

    def _encode_vae_image(self, image: torch.Tensor, do_cfg: bool) -> torch.Tensor:
        lat = self.vae.encode(image.to(self.vae.dtype)).latent_dist.mode()
        if do_cfg:
            lat = torch.cat([torch.zeros_like(lat), lat])
        return lat

    # ------------------------------------------------------------------ main call
    @torch.no_grad()
    def __call__(
        self,
        image: PIL.Image.Image,
        events: torch.Tensor,
        exposure_value: float = 0.0,
        height: int = 512,
        width: int = 512,
        num_frames: int = 20,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = 8,
        s_churn: float = 0.5,
        previous_clip: Optional[torch.Tensor] = None,
        prev_latent_scale: float = 0.3,
        generator_conditioning_scale: float = 0.0,
        shader_conditioning_scale: float = 1.0,
        exposure_token_scale: float = 1.0,
        tile_size: int = 512,
        tile_overlap: float = 0.1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        """Generate ``num_frames`` frames.

        ``image``: reference LDR frame (PIL RGB) of size (width, height).
        ``events``: (T, C_ev, height, width) tensor in [-1, 1], one slice per output frame.
        ``exposure_value``: EV at which the reference was rendered (fed to the exposure token).
        ``previous_clip``: (1, T, 4, h, w) un-scaled latents of the previous clip (the last-frame
        latent repeated T times) for clip-to-clip continuity; ``None`` for the first clip.
        ``prev_latent_scale`` weights those latents at inference (0.3 works well for long chains).
        ``fps``: written to the SVD micro-conditioning as ``fps - 1`` like the stock SVD pipeline;
        the released models were trained with ``fps_id: 7`` and sampled with the default ``fps=7``.
        """
        device = self._execution_device
        dtype = self.unet.dtype
        mixed = self.event_controlnet.dtype != dtype
        autocast = torch.autocast(device.type, dtype=dtype) if mixed else contextlib.nullcontext()
        with autocast:
            return self._generate(image, events, exposure_value, height, width, num_frames, num_inference_steps,
                                  min_guidance_scale, max_guidance_scale, fps, motion_bucket_id, noise_aug_strength,
                                  decode_chunk_size, s_churn, previous_clip, prev_latent_scale, generator_conditioning_scale,
                                  shader_conditioning_scale, exposure_token_scale, tile_size, tile_overlap, generator, latents,
                                  output_type, return_dict, device, dtype)

    def _generate(self, image, events, exposure_value, height, width, num_frames, num_inference_steps, min_guidance_scale,
                  max_guidance_scale, fps, motion_bucket_id, noise_aug_strength, decode_chunk_size, s_churn, previous_clip,
                  prev_latent_scale, generator_conditioning_scale, shader_conditioning_scale, exposure_token_scale, tile_size,
                  tile_overlap, generator, latents, output_type, return_dict, device, dtype):
        f = self.vae_scale_factor
        height, width = math.ceil(height / f) * f, math.ceil(width / f) * f
        do_cfg = max_guidance_scale > 1.0
        decode_chunk_size = decode_chunk_size or num_frames

        clip_emb = self._encode_clip_image(image, device, dtype, do_cfg)
        clip_emb = clip_emb + self._exposure_embedding(exposure_value, device, dtype, do_cfg, exposure_token_scale)

        add_time_ids = torch.tensor([[fps - 1, motion_bucket_id, noise_aug_strength]], device=device, dtype=dtype)
        add_time_ids = add_time_ids.repeat(2 if do_cfg else 1, 1)

        image_pt = self.video_processor.preprocess(image, height=height, width=width).to(device=device, dtype=dtype)
        image_pt = image_pt + noise_aug_strength * randn_tensor(image_pt.shape, generator=generator, device=device, dtype=dtype)

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device)

        h, w = height // f, width // f
        shape = (1, num_frames, self.unet.config.in_channels // 2, h, w)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(device=device, dtype=dtype)

        events = events.unsqueeze(0).to(device=device, dtype=dtype)
        if do_cfg:
            events = torch.cat([events, events])
        prev = None
        if previous_clip is not None:
            prev = previous_clip.to(device=device, dtype=dtype)
            if do_cfg:
                prev = torch.cat([prev, prev])

        guidance = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames, device=device, dtype=dtype)
        guidance = guidance.view(1, num_frames, 1, 1, 1)

        views = tile_windows(h, w, tile_size // f, tile_overlap)
        view_cache = []  # per view: reference latents + gaussian weight
        for (ws, hs, we, he) in views:
            img_lat = self._encode_vae_image(image_pt[:, :, hs * f:he * f, ws * f:we * f], do_cfg)
            img_lat = img_lat.to(dtype).unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
            weight = gaussian_blend_weight(he - hs, we - ws, device=device, dtype=dtype).view(1, 1, 1, he - hs, we - ws)
            view_cache.append((img_lat, weight))

        value = torch.zeros_like(latents)
        count = torch.zeros_like(latents)
        with self.progress_bar(total=num_inference_steps) as bar:
            for t in timesteps:
                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)
                step_i = self.scheduler.step_index
                value.zero_(); count.zero_()
                for (ws, hs, we, he), (img_lat, weight) in zip(views, view_cache):
                    lat_view = latents[:, :, :, hs:he, ws:we]
                    lmi = torch.cat([lat_view] * 2) if do_cfg else lat_view
                    lmi = self.scheduler.scale_model_input(lmi, t)
                    cond_in = torch.cat([lmi, img_lat], dim=2)
                    if prev is not None:
                        shader_in = torch.cat([lmi, prev_latent_scale * prev[:, :, :, hs:he, ws:we]], dim=2)
                    else:
                        shader_in = lmi
                    ev_view = events[:, :, :, hs * f:he * f, ws * f:we * f]

                    gen = self.event_controlnet(cond_in, t, clip_emb, add_time_ids, events=ev_view)
                    down, mid = gen.down_block_res_samples, gen.mid_block_res_sample
                    if self.shader_controlnet is not None:
                        shd = self.shader_controlnet(
                            shader_in, t, clip_emb, add_time_ids, event_embedding=gen.event_embedding,
                            generator_down_features=gen.raw_down_features, generator_mid_feature=gen.raw_mid_feature,
                        )
                        down = tuple(generator_conditioning_scale * d + shader_conditioning_scale * s
                                     for d, s in zip(down, shd.down_block_res_samples))
                        mid = generator_conditioning_scale * mid + shader_conditioning_scale * shd.mid_block_res_sample
                    down = tuple(d.to(dtype) for d in down)
                    noise_pred = self.unet(cond_in, t, clip_emb, add_time_ids, down_block_additional_residuals=down,
                                           mid_block_additional_residual=mid.to(dtype)).sample
                    if do_cfg:
                        uncond, cond = noise_pred.chunk(2)
                        noise_pred = uncond + guidance * (cond - uncond)
                    denoised = self.scheduler.step(noise_pred, t, lat_view, s_churn=s_churn, generator=generator).prev_sample
                    self.scheduler._step_index = step_i
                    value[:, :, :, hs:he, ws:we] += weight * denoised
                    count[:, :, :, hs:he, ws:we] += weight
                self.scheduler._step_index = step_i + 1
                latents = torch.where(count > 0, value / count, value)
                bar.update()

        latents_out = latents / self.vae.config.scaling_factor
        if output_type == "latent":
            frames = latents_out
        else:
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return frames, latents_out
        return HiveDiffOutput(frames=frames, latents=latents_out)

    def decode_latents(self, latents: torch.Tensor, num_frames: int, decode_chunk_size: int) -> torch.Tensor:
        latents = latents.flatten(0, 1) / self.vae.config.scaling_factor
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            chunk = latents[i:i + decode_chunk_size].to(self.vae.dtype)
            frames.append(self.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
        frames = torch.cat(frames)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        return frames.float()

    @staticmethod
    def chain_latents(latents: torch.Tensor) -> torch.Tensor:
        """Build ``previous_clip`` for the next call from this call's ``latents`` (B, T, 4, h, w)."""
        last = latents[:, -1:]
        return last.repeat(1, latents.shape[1], 1, 1, 1)
