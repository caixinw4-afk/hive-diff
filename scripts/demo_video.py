#!/usr/bin/env python
"""Run HIVE-Diff on your own LDR video + event voxels (no ground truth).

Input layout::

    <frames_root>/<scene>/*.png                 LDR frames
    <events_root>/<scene>/[voxels_npy/]*.npy    12-bin event voxel per frame interval (file i precedes frame i)

Clips of ``--num_frames`` are generated back to back (stride = num_frames) and chained through
the previous clip's last latent. With ``--rescale 2`` generation runs at 1024x1024 using 3x3
overlapping tiles (the default setting for real videos).

Example::

    python scripts/demo_video.py --generator checkpoints/hive_20f/generator --shader checkpoints/hive_20f \
        --frames_root data/my_capture/imgs --events_root data/my_capture/events --scene 001 \
        --output_dir outputs/demo_001
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import PIL.Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hive_diff.data import LdrVideoDataset  # noqa: E402
from hive_diff.utils.loading import load_pipeline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--svd", default="stabilityai/stable-video-diffusion-img2vid")
    p.add_argument("--generator", required=True)
    p.add_argument("--shader", default=None)
    p.add_argument("--frames_root", required=True)
    p.add_argument("--events_root", required=True)
    p.add_argument("--scene", default=None, help="scene folder (default: every scene)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_frames", type=int, default=20)
    p.add_argument("--bins", type=int, default=12)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--rescale", type=float, default=2.0, help="generation size = size * rescale (tiled)")
    p.add_argument("--clip_start", type=int, default=5)
    p.add_argument("--num_clips", type=int, default=0, help="0 = as many as fit")
    p.add_argument("--exposure_value", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--s_churn", type=float, default=0.5)
    p.add_argument("--min_guidance", type=float, default=2.5)
    p.add_argument("--max_guidance", type=float, default=3.5)
    p.add_argument("--prev_latent_scale", type=float, default=0.3)
    p.add_argument("--tile_overlap", type=float, default=0.1)
    p.add_argument("--decode_chunk", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--controlnet_dtype", choices=["fp16", "fp32"], default="fp32", help="precision of the ControlNets; fp32 runs them under autocast")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(args.svd, args.generator, args.shader, device=device,
                         controlnet_dtype=torch.float32 if args.controlnet_dtype == "fp32" else torch.float16)
    ds = LdrVideoDataset(args.frames_root, args.events_root, num_frames=args.num_frames, mode="infer", bins=args.bins,
                         out_size=args.size, event_scale=args.rescale, exposure_value=args.exposure_value,
                         scenes=[args.scene] if args.scene else None)
    assert args.clip_start >= 1, "--clip_start must be >= 1 (event file clip_start-1 is needed)"
    gen_size = int(round(args.size * args.rescale))
    rng = torch.Generator(device=device).manual_seed(args.seed)
    for scene in ds.scenes:
        prev = None
        n_clips = 0
        starts = range(args.clip_start, scene["n"] - args.num_frames, args.num_frames)
        for start in starts:
            if args.num_clips and n_clips >= args.num_clips:
                break
            item = ds[ds.index_of(scene["name"], start)]
            ref = PIL.Image.open(item["reference_path"]).convert("RGB").resize((gen_size, gen_size), PIL.Image.LANCZOS)
            t0 = time.time()
            out = pipe(ref, item["event_values"].to(device), item["exposure_value"], height=gen_size, width=gen_size,
                       num_frames=args.num_frames, num_inference_steps=args.steps, s_churn=args.s_churn,
                       min_guidance_scale=args.min_guidance, max_guidance_scale=args.max_guidance, previous_clip=prev,
                       prev_latent_scale=args.prev_latent_scale, tile_overlap=args.tile_overlap,
                       decode_chunk_size=args.decode_chunk, generator=rng)
            prev = pipe.chain_latents(out.latents)
            odir = os.path.join(args.output_dir, "test_images", scene["name"])
            idir = os.path.join(args.output_dir, "inputs", scene["name"])
            os.makedirs(odir, exist_ok=True); os.makedirs(idir, exist_ok=True)
            for k, fr in enumerate(out.frames[0]):
                fr.resize((args.size, args.size), PIL.Image.LANCZOS).save(os.path.join(odir, f"{start:06d}_{k:02d}.png"))
            item["reference_pil"].save(os.path.join(idir, f"{start:06d}_reference.png"))
            ev = item["event_values"].float().numpy()
            for k in range(ev.shape[0]):
                strip = np.concatenate([((np.clip(ev[k, b], -1, 1) + 1) * 127.5).astype(np.uint8) for b in range(ev.shape[1])], axis=1)
                PIL.Image.fromarray(strip).resize((args.size * ev.shape[1], args.size)).save(os.path.join(idir, f"{start:06d}_events_{k:02d}.png"))
            n_clips += 1
            print(f"[{scene['name']}] clip start {start:6d}  {time.time() - t0:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
