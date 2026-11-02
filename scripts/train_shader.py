#!/usr/bin/env python
"""Stage 2: train the Shader ControlNet (+ exposure token) on top of a frozen Generator.

Trainable: ShaderControlNet (initialised from the SVD UNet down path) and ExposureTokenMLP.
Frozen: SVD VAE / CLIP / UNet and the stage-1 EventControlNet.

Loss: EDM-weighted denoising loss on the SVD latents (sigma ~ LogNormal(0.7, 1.6)), plus a
small boundary term that ties the first predicted frame to the previous clip's last frame when
the previous clip is used as conditioning (probability ramps 0.3 -> 0.7 over 7000 steps).

Example::

    accelerate launch scripts/train_shader.py --config configs/train_shader_20f.yaml
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hive_diff.data.hdm_synthetic import HdmSyntheticDataset  # noqa: E402
from hive_diff.models import EventControlNet, ExposureTokenMLP, ShaderControlNet  # noqa: E402
from hive_diff.utils.loading import load_svd_components  # noqa: E402
from hive_diff.utils.training import clip_embed, log_normal  # noqa: E402

DEFAULTS = dict(
    svd="stabilityai/stable-video-diffusion-img2vid", generator=None, shader_init=None, output_dir="outputs/shader",
    video_root=None, tone_root=None, events_root=None, num_frames=20, crop=512, fine_bins=12, window_bins=[0, 4, 8, 12],
    ev_min=-5, ev_max=4, warp_prob=0.75, batch_size=1, grad_accum=16, num_workers=8, lr=5e-5, lr_scheduler="constant",
    lr_warmup_steps=0, max_steps=7000, weight_decay=1e-2, adam_betas=[0.9, 0.999], seed=123, mixed_precision="fp16",
    gradient_checkpointing=True, cond_dropout=0.1, prev_prob_start=0.3, prev_prob_end=0.7, prev_ramp_steps=7000,
    boundary_weight=0.01, generator_scale=0.0, shader_scale=1.0, residual_mode="multi", fps_id=7,
    motion_bucket_id=127, checkpoint_every=500, keep_checkpoints=2, resume=None, log_every=10,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML with any of the keys in DEFAULTS")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif isinstance(v, list):
            p.add_argument(f"--{k}", type=float, nargs="+", default=None)
        else:
            p.add_argument(f"--{k}", type=type(v) if v is not None else str, default=None)
    a = p.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        with open(a.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update({k: v for k, v in vars(a).items() if k != "config" and v is not None})
    for k in ("generator", "video_root", "tone_root", "events_root"):
        if not cfg[k]:
            p.error(f"--{k} is required")
    return argparse.Namespace(**cfg)


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision=args.mixed_precision,
                              log_with="tensorboard", project_dir=os.path.join(args.output_dir, "logs"))
    if args.seed is not None:
        set_seed(args.seed)
    device = accelerator.device
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(vars(args), f)
        accelerator.init_trackers("hive_shader")

    feature_extractor, image_encoder, vae, unet, _ = load_svd_components(args.svd, weight_dtype)
    unet.residual_mode = args.residual_mode
    generator = EventControlNet.from_pretrained(args.generator, torch_dtype=weight_dtype)
    shader = ShaderControlNet.from_pretrained(args.shader_init) if args.shader_init else ShaderControlNet.from_unet(unet)
    shader = shader.float()
    ev_token = ExposureTokenMLP(unet.config.cross_attention_dim)
    for m in (vae, image_encoder, unet, generator):
        m.requires_grad_(False).to(device)
    if args.gradient_checkpointing:
        shader.enable_gradient_checkpointing()

    params = list(shader.parameters()) + list(ev_token.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=tuple(args.adam_betas), weight_decay=args.weight_decay, eps=1e-8)
    dataset = HdmSyntheticDataset(args.video_root, args.tone_root, args.events_root, num_frames=args.num_frames,
                                  fine_bins=args.fine_bins, crop=(args.crop, args.crop), ev_range=(args.ev_min, args.ev_max),
                                  window_bins=[int(b) for b in args.window_bins], warp_prob=args.warp_prob)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0)
    lr_sched = get_scheduler(args.lr_scheduler, optimizer, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                             num_training_steps=args.max_steps * accelerator.num_processes)
    shader, ev_token, optimizer, loader, lr_sched = accelerator.prepare(shader, ev_token, optimizer, loader, lr_sched)
    accelerator.register_for_checkpointing(lr_sched)

    global_step = 0
    if args.resume:
        path = args.resume
        if path == "latest":
            cks = sorted((d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")), key=lambda d: int(d.split("-")[1]))
            path = os.path.join(args.output_dir, cks[-1]) if cks else None
        if path:
            accelerator.load_state(path)
            global_step = int(os.path.basename(path).split("-")[1])
            accelerator.print(f"resumed from {path} at step {global_step}")

    rng = torch.Generator(device=device).manual_seed(args.seed or 0)
    T = args.num_frames
    sf = vae.config.scaling_factor
    accelerator.print(f"dataset {len(dataset)} windows over {len(dataset.scenes)} scenes; steps {args.max_steps}")
    shader.train(); ev_token.train()
    t_last = time.time()
    while global_step < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(shader, ev_token):
                ref = batch["reference"].to(device, weight_dtype)                  # (B,3,H,W)
                frames = batch["pixel_values"].to(device, weight_dtype)            # (B,T,3,H,W)
                prev_frames = batch["prev_pixel_values"].to(device, weight_dtype)
                events = batch["event_values"].to(device, weight_dtype)            # (B,T,4,H,W)
                ev = batch["exposure_value"].float().to(device)
                B = ref.shape[0]

                with torch.no_grad():
                    cond_sigma = log_normal((B,), -3.0, 0.5, rng, device).to(weight_dtype)
                    noise_aug = cond_sigma[0]
                    ref_noisy = ref + torch.randn_like(ref) * cond_sigma.view(B, 1, 1, 1)
                    cond_latents = vae.encode(ref_noisy).latent_dist.sample()      # un-scaled (B,4,h,w)

                    ramp = min(global_step / max(args.prev_ramp_steps, 1), 1.0)
                    use_prev = torch.rand((), generator=rng, device=device).item() < args.prev_prob_start + (args.prev_prob_end - args.prev_prob_start) * ramp
                    prev_last = z_prev_last = None
                    if use_prev:
                        prev_scaled = vae.encode(prev_frames.flatten(0, 1)).latent_dist.sample() * sf
                        prev_scaled = prev_scaled.reshape(B, T, *prev_scaled.shape[1:])
                        z_prev_last = prev_scaled[:, -1]
                        prev_noisy = prev_scaled / sf + torch.randn_like(prev_scaled) * (cond_sigma * 2.0).view(B, 1, 1, 1, 1)
                        prev_last = prev_noisy[:, -1:]

                    clip = clip_embed(image_encoder, feature_extractor, ref_noisy)  # (B,1,1024)
                    add_time_ids = torch.tensor([[args.fps_id, args.motion_bucket_id, float(noise_aug)]], device=device, dtype=weight_dtype).repeat(B, 1)

                    if args.cond_dropout > 0:
                        p = torch.rand(B, generator=rng, device=device)
                        prompt_drop = (p < 2 * args.cond_dropout).view(B, 1, 1)
                        image_drop = ((p >= args.cond_dropout) & (p < 3 * args.cond_dropout)).view(B, 1, 1, 1)
                        clip = clip * (~prompt_drop)
                        cond_latents = cond_latents * (~image_drop)
                        if prev_last is not None:
                            prev_last = prev_last * (~image_drop).unsqueeze(1)
                    else:
                        prompt_drop = torch.zeros(B, 1, 1, dtype=torch.bool, device=device)

                    latents = vae.encode(frames.flatten(0, 1)).latent_dist.sample() * sf
                    latents = latents.reshape(B, T, *latents.shape[1:]).float()
                    sigma = log_normal((B,), 0.7, 1.6, rng, device)
                    timesteps = 0.25 * sigma.log()
                    s = sigma.view(B, 1, 1, 1, 1)
                    noisy = latents + torch.randn_like(latents) * s
                    inp4 = (noisy / (s ** 2 + 1).sqrt()).to(weight_dtype)
                    cond_rep = cond_latents.to(weight_dtype).unsqueeze(1).repeat(1, T, 1, 1, 1)
                    inp = torch.cat([inp4, cond_rep], dim=2)

                tok = ev_token(ev).to(weight_dtype).unsqueeze(1) * (~prompt_drop)
                enc = clip.to(weight_dtype) + tok
                shader_in = torch.cat([inp4, prev_last.to(weight_dtype).repeat(1, T, 1, 1, 1)], dim=2) if prev_last is not None else inp4

                with torch.no_grad():
                    g_down, g_mid, g_raw, g_raw_mid, g_emb = generator(inp, timesteps, enc, add_time_ids, events=events, return_dict=False)
                s_down, s_mid = shader(shader_in, timesteps, enc, add_time_ids, event_embedding=g_emb, generator_down_features=g_raw,
                                       generator_mid_feature=g_raw_mid, conditioning_scale=args.shader_scale, return_dict=False)
                down = [(args.generator_scale * d + r).to(weight_dtype) for d, r in zip(g_down, s_down)]
                mid = (args.generator_scale * g_mid + s_mid).to(weight_dtype)
                pred = unet(inp, timesteps, enc, add_time_ids, down_block_additional_residuals=down, mid_block_additional_residual=mid).sample.float()

                c_out = -s / (s ** 2 + 1).sqrt()
                c_skip = 1.0 / (s ** 2 + 1)
                denoised = pred * c_out + c_skip * noisy
                w = (1 + s ** 2) / s ** 2
                loss = (w * (denoised - latents) ** 2).mean(dim=(1, 2, 3, 4)).mean()
                if z_prev_last is not None:
                    loss = loss + args.boundary_weight * F.mse_loss(denoised[:, 0], z_prev_last.float())

                accelerator.backward(loss)
                optimizer.step(); lr_sched.step(); optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0:
                    dt = time.time() - t_last; t_last = time.time()
                    accelerator.log({"loss": loss.item(), "lr": lr_sched.get_last_lr()[0]}, step=global_step)
                    accelerator.print(f"step {global_step:6d} loss {loss.item():.4f} prev={int(use_prev)} {dt / args.log_every:.2f}s/step")
                if global_step % args.checkpoint_every == 0 and accelerator.is_main_process:
                    ck = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(ck)
                    accelerator.unwrap_model(shader).save_pretrained(os.path.join(ck, "shader"))
                    torch.save(accelerator.unwrap_model(ev_token).state_dict(), os.path.join(ck, "ev_token.pt"))
                    cks = sorted((d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")), key=lambda d: int(d.split("-")[1]))
                    for old in cks[:-args.keep_checkpoints]:
                        shutil.rmtree(os.path.join(args.output_dir, old), ignore_errors=True)
                if global_step >= args.max_steps:
                    break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(shader).save_pretrained(os.path.join(args.output_dir, "final", "shader"))
        torch.save(accelerator.unwrap_model(ev_token).state_dict(), os.path.join(args.output_dir, "final", "ev_token.pt"))
    accelerator.end_training()


if __name__ == "__main__":
    main()
