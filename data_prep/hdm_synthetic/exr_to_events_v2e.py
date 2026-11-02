#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate events from 8-bit frame folders with v2e.

Input layout: ``<in_root>/<scene>/<input_subdir>/*.png`` (``preview_png`` by default). Outputs:
``<out_root>/<scene>/v2e/`` (events.txt, aedat4, logs) and ``<out_root>/<scene>/previews/``.
Frames are passed to v2e in folder-of-images mode, so the true frame rate must be given with
``--fps``; SuperSloMo upsampling is enabled when ``--slomo_model`` is set.
"""

from __future__ import annotations
import argparse, shlex, shutil, subprocess, sys, os
from pathlib import Path

def find_scene_dirs(in_root: Path, subdir: str, also_try_exr: bool, exts: list[str]) -> list[tuple[Path,str]]:
    """Return list of (frames_dir, scene_name) that contain at least one image with allowed extensions."""
    hits = []
    # Normalize to lowercase extensions without dots
    exts = [e.lower().lstrip(".") for e in exts]
    # primary: images in the given subdir
    for p in sorted((in_root).rglob(f"{subdir}")):
        pp = Path(p)
        found = False
        for ext in exts:
            # case-insensitive matching by checking both lower/upper
            if list(pp.glob(f"*.{ext}")) or list(pp.glob(f"*.{ext.upper()}")):
                found = True
                break
        if found:
            hits.append((pp, pp.parent.name))

    # fallback: EXR dir if requested and nothing found above
    if not hits and also_try_exr:
        for p in sorted((in_root).rglob("hdr_shaken_exr")):
            if list(Path(p).glob("*.exr")) or list(Path(p).glob("*.EXR")):
                hits.append((Path(p), Path(p).parent.name))

    return hits

def run(cmd: list[str]):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def move_previews(out_dir: Path, preview_dir: Path):
    preview_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "*.avi","*.mp4","*preview*","dvs*.avi","dvs*.mp4",
        "video_slomo.avi","video_orig.avi","frames_*.png","preview_*"
    ]
    moved = False
    for pat in patterns:
        for f in out_dir.glob(pat):
            try:
                shutil.move(str(f), str(preview_dir / f.name))
                moved = True
            except Exception as e:
                print(f"[warn] could not move {f.name}: {e}")
    print(f"[info] previews -> {preview_dir}" if moved else "[info] no previews to move")

def run_v2e(
    img_dir: Path, out_dir: Path, fps: float,
    v2e_exec: str = "v2e",
    # DVS model (slightly lower thresholds for tone-mapped PNGs)
    pos_thres: float = 0.05, neg_thres: float = 0.05, sigma_thres: float = 0.015,
    cutoff_hz: float = 50.0, leak_rate_hz: float = 0.1, refractory_period: float = 5e-4,
    # SloMo / timing (safe defaults)
    auto_ts_res: bool = True, timestamp_resolution: float = 3e-3,
    slomo_model: str | None = None, input_slowmotion_factor: int | None = None,
    disable_slomo: bool = False, batch_size: int = 1,
    # Outputs
    write_aedat4: bool = True, write_events_txt: bool = True,
    # Preview
    enable_preview: bool = True, preview_mode: str = "time", preview_param: float = 0.01,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [*shlex.split(v2e_exec),
           "--input", str(img_dir),
           "--input_frame_rate", str(fps),
           "--output_folder", str(out_dir),
           "--pos_thres", str(pos_thres),
           "--neg_thres", str(neg_thres),
           "--sigma_thres", str(sigma_thres),
           "--cutoff_hz", str(cutoff_hz),
           "--leak_rate_hz", str(leak_rate_hz),
           "--refractory_period", str(refractory_period),
           "--batch_size", str(batch_size),
           "--no_preview",   # headless-safe
           "--overwrite"]
    # SloMo
    if disable_slomo:
        cmd += ["--disable_slomo", "1"]
    else:
        if auto_ts_res:
            cmd += ["--auto_timestamp_resolution", "1", "--timestamp_resolution", str(timestamp_resolution)]
        else:
            cmd += ["--auto_timestamp_resolution", "0", "--timestamp_resolution", str(timestamp_resolution)]
        if slomo_model:
            cmd += ["--slomo_model", slomo_model]
        if input_slowmotion_factor and input_slowmotion_factor > 1:
            cmd += ["--input_slowmotion_factor", str(input_slowmotion_factor)]
    # Event outputs
    if write_events_txt:
        cmd += ["--dvs_text", str(out_dir / "events.txt")]
    if write_aedat4:
        cmd += ["--dvs_aedat4", str(out_dir / "events.aedat4")]
    # Preview to files (no GUI)
    if enable_preview:
        if preview_mode == "time":
            cmd += ["--dvs_exposure", "duration", str(preview_param)]
        else:
            cmd += ["--dvs_exposure", "count", str(int(preview_param))]
    else:
        cmd += ["--output_suppress_video", "1", "--dvs_fps", "0"]
    run(cmd)

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in_root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--v2e_exec", type=str, default="python -m v2e")  # safer than console entry-point
    ap.add_argument("--input_subdir", type=str, default="preview_png",
                    help="subfolder name under each scene containing the frames")
    ap.add_argument("--fallback_exr", action="store_true",
                    help="if no PNGs found, also try <scene>/hdr_shaken_exr/*.exr")
    # model
    ap.add_argument("--pos_thres", type=float, default=0.05)
    ap.add_argument("--neg_thres", type=float, default=0.05)
    ap.add_argument("--sigma_thres", type=float, default=0.015)
    ap.add_argument("--cutoff_hz", type=float, default=50.0)
    ap.add_argument("--leak_rate_hz", type=float, default=0.1)
    ap.add_argument("--refractory_period", type=float, default=5e-4)
    # timing
    ap.add_argument("--auto_ts_res", action="store_true", default=True)
    ap.add_argument("--no_auto_ts_res", action="store_false", dest="auto_ts_res")
    ap.add_argument("--timestamp_resolution", type=float, default=3e-3)
    ap.add_argument("--slomo_model", type=str, default=None)
    ap.add_argument("--input_slowmotion_factor", type=int, default=None)
    ap.add_argument("--disable_slomo", action="store_true", default=False)
    ap.add_argument("--batch_size", type=int, default=12)
    # outputs
    ap.add_argument("--write_aedat4", action="store_true", default=True)
    ap.add_argument("--write_events_txt", action="store_true", default=True)
    ap.add_argument("--enable_preview", action="store_true", default=True)
    ap.add_argument("--preview_mode", choices=["time","count"], default="time")
    ap.add_argument("--preview_param", type=float, default=0.01)
    ap.add_argument("--clean_existing", action="store_true")
    ap.add_argument("--tmp_dir", type=Path, default=None)
    ap.add_argument("--input_exts", type=str, default="png,tif,tiff",
                    help="comma-separated list of image extensions to look for (case-insensitive)")

    args = ap.parse_args()

    if args.tmp_dir: os.environ["TMPDIR"] = str(args.tmp_dir)
    os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    os.environ.setdefault("MPLBACKEND","Agg")
    # PNG path does not need EXR flag, but it's harmless:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR","1")

    scenes = find_scene_dirs(
        args.in_root,
        args.input_subdir,
        args.fallback_exr,
        exts=[e.strip() for e in args.input_exts.split(",") if e.strip()]
    )
    if not scenes:
        sys.exit(f"No frames found under {args.in_root} (looked for */{args.input_subdir}/*.png)")

    print(f"Found {len(scenes)} scene(s).")
    for frames_dir, scene in scenes:
        out_dir = args.out_root / scene / "v2e"
        preview_dir = args.out_root / scene / "previews"
        if args.clean_existing:
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(preview_dir, ignore_errors=True)

        run_v2e(
            img_dir=frames_dir, out_dir=out_dir, fps=args.fps,
            v2e_exec=args.v2e_exec,
            pos_thres=args.pos_thres, neg_thres=args.neg_thres, sigma_thres=args.sigma_thres,
            cutoff_hz=args.cutoff_hz, leak_rate_hz=args.leak_rate_hz, refractory_period=args.refractory_period,
            auto_ts_res=args.auto_ts_res, timestamp_resolution=args.timestamp_resolution,
            slomo_model=args.slomo_model, input_slowmotion_factor=args.input_slowmotion_factor,
            disable_slomo=args.disable_slomo, batch_size=args.batch_size,
            write_aedat4=args.write_aedat4, write_events_txt=args.write_events_txt,
            enable_preview=args.enable_preview, preview_mode=args.preview_mode, preview_param=args.preview_param,
        )
        move_previews(out_dir, preview_dir)

if __name__ == "__main__":
    main()
