#!/usr/bin/env python
"""Stage 1: train (or finetune) the event-conditioned Generator ControlNet.

Target: the LDR video rendered at the reference exposure (the model learns to continue an LDR
frame through time from events). Loss: EDM-weighted denoising loss on SVD latents
(sigma ~ LogNormal(0.7, 1.6)) + 0.1 x event-reconstruction loss from an auxiliary decoder on
the event-encoder features. Optional distillation from a frozen copy of the initial weights
(per-channel cosine distance on the event embedding and on the residuals) when finetuning on
real data.

Datasets (``--data``):
  hdm       synthetic HdM camera-shake HDR + 12-bin events (from-scratch training)
  eventhdr  EventHDR real captures, 4-bin events, +7 EV LDR target (finetuning)
  ldr       LDR PNG frames + 12-bin events, e.g. own captures (finetuning)

Example::

    accelerate launch scripts/train_generator.py --config configs/train_generator_20f.yaml
"""
from __future__ import annotations

import argparse
import copy
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
from hive_diff.data import EventHDRFinetuneDataset, HdmSyntheticDataset, LdrVideoDataset  # noqa: E402
from hive_diff.models import EventControlNet, EventReconstructor  # noqa: E402
from hive_diff.utils.loading import load_svd_components  # noqa: E402
from hive_diff.utils.training import clip_embed, log_normal  # noqa: E402

DEFAULTS = dict(
    svd="stabilityai/stable-video-diffusion-img2vid", init=None, output_dir="outputs/generator", data="hdm",
    video_root=None, tone_root=None, events_root=None, frames_root=None, num_frames=20, crop=512, base_size=1024,
    fine_bins=12, window_bins=[0, 4, 8, 12], ev_min=None, ev_max=None, warp_prob=0.75, finetune_warp_seed=42,
    batch_size=1, grad_accum=16, num_workers=8, lr=5e-5, lr_scheduler="constant", lr_warmup_steps=0, max_steps=12000,
    weight_decay=1e-2, adam_betas=[0.9, 0.999], seed=123, mixed_precision="fp16", gradient_checkpointing=True,
    cond_dropout=0.1, aux_weight=0.1, distill_residual_weight=0.0, distill_encoder_weight=0.0, distill_warmup_steps=500,
    residual_mode="multi", fps_id=7, motion_bucket_id=127, checkpoint_every=500, keep_checkpoints=2, resume=None,
    log_every=10,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
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
    return argparse.Namespace(**cfg)


def build_dataset(args):
    T = args.num_frames
    windows = [int(b) for b in args.window_bins]
    if args.data == "hdm":
        ev_range = (int(args.ev_min), int(args.ev_max)) if args.ev_min is not None else None
        return HdmSyntheticDataset(args.video_root, args.tone_root, args.events_root, num_frames=T, target="ldr",
                                   fine_bins=args.fine_bins, crop=(args.crop, args.crop), ev_range=ev_range,
                                   window_bins=windows, warp_prob=args.warp_prob)
    if args.data == "eventhdr":
        return EventHDRFinetuneDataset(args.video_root, num_frames=T, out_size=args.crop, window_bins=windows,
                                       warp_seed=args.finetune_warp_seed)
    if args.data == "ldr":
        return LdrVideoDataset(args.frames_root, args.events_root, num_frames=T, mode="train", bins=args.fine_bins,
                               base_size=args.base_size, crop=args.crop, window_bins=windows)
    raise ValueError(args.data)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean over (batch x channel) rows of 1 - cos(a_row, b_row), rows = flattened spatial maps."""
    a = F.normalize(a.flatten(0, 1).flatten(1).float(), dim=1)
    b = F.normalize(b.flatten(0, 1).flatten(1).float(), dim=1)
    return (1.0 - (a * b).sum(1)).mean()


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
        accelerator.init_trackers("hive_generator")

    feature_extractor, image_encoder, vae, unet, _ = load_svd_components(args.svd, weight_dtype)
    unet.residual_mode = args.residual_mode
    controlnet = EventControlNet.from_pretrained(args.init) if args.init else EventControlNet.from_unet(unet)
    controlnet = controlnet.float()
    aux = EventReconstructor(unet.config.block_out_channels[0], controlnet.config.event_channels)
    if args.init and os.path.exists(os.path.join(os.path.dirname(args.init.rstrip("/")), "aux_reconstructor.pt")):
        aux.load_state_dict(torch.load(os.path.join(os.path.dirname(args.init.rstrip("/")), "aux_reconstructor.pt"), map_location="cpu"))
    teacher = None
    if args.distill_residual_weight > 0 or args.distill_encoder_weight > 0:
        teacher = copy.deepcopy(controlnet).to(device, weight_dtype).eval().requires_grad_(False)
    for m in (vae, image_encoder, unet):
        m.requires_grad_(False).to(device)
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    params = list(controlnet.parameters()) + list(aux.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=tuple(args.adam_betas), weight_decay=args.weight_decay, eps=1e-8)
    dataset = build_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0)
    lr_sched = get_scheduler(args.lr_scheduler, optimizer, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                             num_training_steps=args.max_steps * accelerator.num_processes)
    controlnet, aux, optimizer, loader, lr_sched = accelerator.prepare(controlnet, aux, optimizer, loader, lr_sched)
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
    T, sf = args.num_frames, vae.config.scaling_factor
    accelerator.print(f"dataset {len(dataset)} windows; steps {args.max_steps}; distill={'on' if teacher else 'off'}")
    controlnet.train(); aux.train()
    t_last = time.time()
    while global_step < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(controlnet, aux):
                ref = batch["reference"].to(device, weight_dtype)
                target = batch["ldr_gt"].to(device, weight_dtype)            # (B,T,3,H,W)
                events = batch["event_values"].to(device, weight_dtype)      # (B,T,4,H,W)
                B = ref.shape[0]
                with torch.no_grad():
                    cond_sigma = log_normal((B,), -3.0, 0.5, rng, device).to(weight_dtype)
                    ref_noisy = ref + torch.randn_like(ref) * cond_sigma.view(B, 1, 1, 1)
                    cond_latents = vae.encode(ref_noisy).latent_dist.sample()
                    clip = clip_embed(image_encoder, feature_extractor, ref_noisy).to(weight_dtype)
                    add_time_ids = torch.tensor([[args.fps_id, args.motion_bucket_id, float(cond_sigma[0])]], device=device,
                                                dtype=weight_dtype).repeat(B, 1)
                    if args.cond_dropout > 0:
                        p = torch.rand(B, generator=rng, device=device)
                        clip = clip * (~(p < 2 * args.cond_dropout)).view(B, 1, 1)
                        cond_latents = cond_latents * (~((p >= args.cond_dropout) & (p < 3 * args.cond_dropout))).view(B, 1, 1, 1)
                    latents = vae.encode(target.flatten(0, 1)).latent_dist.sample() * sf
                    latents = latents.reshape(B, T, *latents.shape[1:]).float()
                    sigma = log_normal((B,), 0.7, 1.6, rng, device)
                    timesteps = 0.25 * sigma.log()
                    s = sigma.view(B, 1, 1, 1, 1)
                    noisy = latents + torch.randn_like(latents) * s
                    inp4 = (noisy / (s ** 2 + 1).sqrt()).to(weight_dtype)
                    inp = torch.cat([inp4, cond_latents.to(weight_dtype).unsqueeze(1).repeat(1, T, 1, 1, 1)], dim=2)

                g_down, g_mid, _, _, g_emb = controlnet(inp, timesteps, clip, add_time_ids, events=events, return_dict=False)
                down = [d.to(weight_dtype) for d in g_down]
                pred = unet(inp, timesteps, clip, add_time_ids, down_block_additional_residuals=down,
                            mid_block_additional_residual=g_mid.to(weight_dtype)).sample.float()
                c_out, c_skip = -s / (s ** 2 + 1).sqrt(), 1.0 / (s ** 2 + 1)
                denoised = pred * c_out + c_skip * noisy
                w = (1 + s ** 2) / s ** 2
                loss_edm = (w * (denoised - latents) ** 2).mean(dim=(1, 2, 3, 4)).mean()
                recon = aux(g_emb)
                loss_aux = F.mse_loss(recon.float(), events.flatten(0, 1).float())
                loss = loss_edm + args.aux_weight * loss_aux
                loss_distill = torch.zeros((), device=device)
                if teacher is not None and global_step >= args.distill_warmup_steps:
                    with torch.no_grad():
                        t_down, t_mid, _, _, t_emb = teacher(inp, timesteps, clip, add_time_ids, events=events, return_dict=False)
                    loss_distill = args.distill_encoder_weight * cosine_distance(g_emb, t_emb)
                    res = [cosine_distance(a, b) for a, b in zip(g_down, t_down)]
                    res.append(cosine_distance(g_mid, t_mid))
                    loss_distill = loss_distill + args.distill_residual_weight * torch.stack(res).mean()
                    loss = loss + loss_distill
                accelerator.backward(loss)
                optimizer.step(); lr_sched.step(); optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0:
                    dt = time.time() - t_last; t_last = time.time()
                    accelerator.log({"loss": loss.item(), "loss_edm": loss_edm.item(), "loss_aux": loss_aux.item(),
                                     "loss_distill": loss_distill.item(), "lr": lr_sched.get_last_lr()[0]}, step=global_step)
                    accelerator.print(f"step {global_step:6d} loss {loss.item():.4f} edm {loss_edm.item():.4f} aux {loss_aux.item():.4f} "
                                      f"distill {loss_distill.item():.4f} {dt / args.log_every:.2f}s/step")
                if global_step % args.checkpoint_every == 0 and accelerator.is_main_process:
                    ck = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(ck)
                    accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(ck, "controlnet"))
                    torch.save(accelerator.unwrap_model(aux).state_dict(), os.path.join(ck, "aux_reconstructor.pt"))
                    cks = sorted((d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")), key=lambda d: int(d.split("-")[1]))
                    for old in cks[:-args.keep_checkpoints]:
                        shutil.rmtree(os.path.join(args.output_dir, old), ignore_errors=True)
                if global_step >= args.max_steps:
                    break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(args.output_dir, "final", "controlnet"))
        torch.save(accelerator.unwrap_model(aux).state_dict(), os.path.join(args.output_dir, "final", "aux_reconstructor.pt"))
    accelerator.end_training()


if __name__ == "__main__":
    main()
