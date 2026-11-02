#!/usr/bin/env python
"""Evaluate HiVE-Diff on the EventHDR real-capture test set.

Example (20-frame model)::

    python scripts/eval_eventhdr.py \
        --svd stabilityai/stable-video-diffusion-img2vid \
        --generator checkpoints/hive_20f/generator --shader checkpoints/hive_20f \
        --data_root /path/eventhdr_test --num_frames 20 --item_start 355 \
        --output_dir outputs/eventhdr_20f

``--shader`` points at a directory holding ``shader/`` (ControlNet weights) and ``ev_token.pt``.
Clips are chained: the last-frame latent of one clip conditions the next one unless
``--independent_clips`` is given. Metrics (PSNR, histogram-matched PSNR, SSIM, LPIPS on the
tone-mapped frames) are written under ``<output_dir>/metrics``; frames, ground truth and the
inputs (reference frame + event slices) are saved as PNG.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import PIL.Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hive_diff.data import EventHDRTestDataset  # noqa: E402
from hive_diff.utils.loading import load_pipeline  # noqa: E402
from hive_diff.utils.metrics import ToneMappedMetrics  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--svd", default="stabilityai/stable-video-diffusion-img2vid")
    p.add_argument("--generator", required=True, help="EventControlNet directory (config.json + safetensors)")
    p.add_argument("--shader", default=None, help="directory with shader/ and ev_token.pt; omit for generator-only")
    p.add_argument("--data_root", required=True)
    p.add_argument("--events_root", default=None)
    p.add_argument("--clips", nargs="*", default=None, help="clip folder names to evaluate (default: all)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_frames", type=int, default=20)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--item_start", type=int, default=355)
    p.add_argument("--item_stride", type=int, default=0, help="0 = num_frames")
    p.add_argument("--max_items", type=int, default=0, help="stop after this many clips (0 = all)")
    p.add_argument("--independent_clips", action="store_true")
    p.add_argument("--prev_latent_scale", type=float, default=0.3)
    p.add_argument("--token_ev", type=float, default=2.0)
    p.add_argument("--reference_ev", type=float, default=7.0)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--s_churn", type=float, default=0.5)
    p.add_argument("--min_guidance", type=float, default=1.0)
    p.add_argument("--max_guidance", type=float, default=3.0)
    p.add_argument("--tile_overlap", type=float, default=0.1)
    p.add_argument("--generator_scale", type=float, default=0.0)
    p.add_argument("--shader_scale", type=float, default=1.0)
    p.add_argument("--decode_chunk", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--controlnet_dtype", choices=["fp16", "fp32"], default="fp32", help="precision of the ControlNets; fp32 runs them under autocast")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--all_frames", action="store_true", help="score every frame (default drops first/last)")
    p.add_argument("--no_pyiqa", action="store_true", help="skip SSIM/LPIPS")
    return p.parse_args()


def save_png(path, arr_u8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PIL.Image.fromarray(arr_u8).save(path)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.output_dir
    os.makedirs(os.path.join(out, "metrics"), exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    pipe = load_pipeline(args.svd, args.generator, args.shader, device=device,
                         dtype=torch.float16 if args.dtype == "fp16" else torch.float32,
                         controlnet_dtype=torch.float32 if args.controlnet_dtype == "fp32" else torch.float16)
    ds = EventHDRTestDataset(args.data_root, args.events_root, num_frames=args.num_frames, out_size=args.size,
                             reference_ev=args.reference_ev, token_ev=args.token_ev, clips=args.clips)
    print(f"dataset: {len(ds)} windows over clips {ds.clip_names()}")
    metrics = ToneMappedMetrics(device=device, use_pyiqa=not args.no_pyiqa)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    stride = args.item_stride or args.num_frames

    rows, per_clip = [], defaultdict(list)
    prev = None
    n_done = 0
    for ci, clip in enumerate(ds.clips):
        start, stop = int(ds.offsets[ci]), int(ds.offsets[ci + 1])
        for idx in range(start + args.item_start, stop, stride):
            if args.max_items and n_done >= args.max_items:
                break
            if args.independent_clips:
                prev = None
            item = ds[idx]
            events = item["event_values"].to(device)
            t0 = time.time()
            result = pipe(item["reference_pil"], events, item["exposure_value"], height=args.size, width=args.size,
                          num_frames=args.num_frames, num_inference_steps=args.steps, s_churn=args.s_churn,
                          min_guidance_scale=args.min_guidance, max_guidance_scale=args.max_guidance,
                          previous_clip=prev, prev_latent_scale=args.prev_latent_scale,
                          generator_conditioning_scale=args.generator_scale, shader_conditioning_scale=args.shader_scale,
                          tile_overlap=args.tile_overlap, decode_chunk_size=args.decode_chunk, generator=gen)
            prev = pipe.chain_latents(result.latents)
            frames = [np.asarray(im) for im in result.frames[0]]
            print(f"[{item['folder']}] start {item['start_idx']:6d}  {time.time() - t0:5.1f}s", flush=True)

            gt = ((item["pixel_values"].float() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()
            tag = f"{item['start_idx']:06d}"
            for i, fr in enumerate(frames):
                save_png(os.path.join(out, "test_images", item["folder"], f"{tag}_{i:02d}.png"), fr)
                save_png(os.path.join(out, "gt_test_images", item["folder"], f"{tag}_{i:02d}.png"), (gt[i] * 255 + 0.5).astype(np.uint8))
            save_png(os.path.join(out, "inputs", item["folder"], f"{tag}_reference.png"), np.asarray(item["reference_pil"]))
            ev = item["event_values"].float().numpy()
            for k in range(ev.shape[0]):
                strip = np.concatenate([((np.clip(ev[k, b], -1, 1) + 1) * 127.5).astype(np.uint8) for b in range(ev.shape[1])], axis=1)
                save_png(os.path.join(out, "inputs", item["folder"], f"{tag}_events_{k:02d}.png"), strip)

            sel = range(len(frames)) if args.all_frames else range(1, len(frames) - 1)
            for i in sel:
                m = metrics(frames[i].astype(np.float32) / 255.0, gt[i])
                m.update(Clip=item["folder"], Start=item["start_idx"], Frame=i)
                rows.append(m); per_clip[item["folder"]].append(m)
            n_done += 1

    keys = [k for k in rows[0] if k not in ("Clip", "Start", "Frame")] if rows else []
    with open(os.path.join(out, "metrics", "metrics_individual_frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Clip", "Start", "Frame"] + keys); w.writeheader(); w.writerows(rows)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    with open(os.path.join(out, "metrics", "metrics_per_clip.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Clip"] + keys)
        for c, rs in per_clip.items():
            w.writerow([c] + [f"{np.mean([r[k] for r in rs]):.4f}" for k in keys])
    with open(os.path.join(out, "metrics", "metrics_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Metric", "Value"])
        for k, v in summary.items():
            w.writerow([k, f"{v:.4f}"])
    print("SUMMARY", json.dumps({k: round(v, 4) for k, v in summary.items()}))


if __name__ == "__main__":
    main()
