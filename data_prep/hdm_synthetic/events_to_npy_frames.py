#!/usr/bin/env python3
"""Bin v2e event streams into per-frame count tensors.

Reads ``<in_root>/<scene>/v2e/events.txt`` (t x y p) and, for every source-frame interval at
``--fps``, accumulates the events into ``--bins`` equal time bins, written as an ``(H, W, bins)``
array to ``<out_root>/<scene>/frames_np/frame_%06d.npy`` (``--mode signed``: ON minus OFF counts;
``count``: all events). ``--png`` also writes one contrast-clipped preview per bin under
``frames_png/``. Scenes are processed in parallel (``--workers``) with threaded writes
(``--io_workers``).
"""

from __future__ import annotations
import argparse, math, sys
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ------------------------ discovery & helpers ------------------------

def find_events_files(in_root: Path) -> List[Path]:
    """Find .../<scene>/v2e/events.txt under in_root."""
    return sorted(in_root.rglob("v2e/events.txt"))

def sniff_size_and_last_t(events_txt: Path) -> Tuple[int,int,float]:
    """
    Infer (H, W) and maximum timestamp tmax (seconds) from events.txt.
    Lines expected as: t x y p
    """
    max_x = max_y = -1
    tmax = 0.0
    with open(events_txt, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                t = float(parts[0]); x = int(parts[1]); y = int(parts[2])
            except Exception:
                continue
            if x > max_x: max_x = x
            if y > max_y: max_y = y
            if t > tmax: tmax = t
    if max_x < 0 or max_y < 0:
        raise RuntimeError(f"{events_txt}: no valid events to infer frame size.")
    return (max_y + 1, max_x + 1, float(tmax))

def detect_polarity_scheme(events_txt: Path, sample_lines: int = 200000) -> str:
    """
    Inspect up to sample_lines for polarity values.
    Returns:
        '01'  -> only 0 and 1 found (treat 0=OFF, 1=ON)
        'pm1' -> -1 and/or +1 present
        'unknown' -> fallback
    """
    has_pos1 = has_neg1 = has_zero = False
    n = 0
    with open(events_txt, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                pr = float(parts[3])
            except Exception:
                continue
            if pr ==  1: has_pos1 = True
            if pr == -1: has_neg1 = True
            if pr ==  0: has_zero  = True
            n += 1
            if n >= sample_lines:
                break
    if has_neg1:
        return "pm1"
    if has_pos1 and has_zero and not has_neg1:
        return "01"
    return "unknown"

def rel_scene_path(events_txt: Path, in_root: Path) -> Path:
    """.../<scene>/v2e/events.txt → '<scene>' relative to in_root."""
    return events_txt.parent.parent.relative_to(in_root)

# ------------------------ visualization ------------------------

def bin_to_png_u8(frame2d: np.ndarray, mode: str, clip_percentile: float = 99.5) -> np.ndarray:
    """
    Map a single-bin 2D array to 8-bit grayscale with percentile clipping.
    - mode='count': brighter = more events
    - mode='signed': 128=neutral; >128 ON-dominant, <128 OFF-dominant
    """
    f = frame2d.astype(np.float32, copy=False)
    if mode == "signed":
        g = np.abs(f)
        nz = g[g > 0]
        if nz.size == 0:
            return np.full_like(f, 128, dtype=np.uint8)
        a = float(np.percentile(nz, clip_percentile))
        a = max(a, 1.0)
        return np.clip(f * (127.0 / a) + 128.0, 0, 255).astype(np.uint8)
    else:
        hi = float(np.percentile(f, clip_percentile))
        hi = max(hi, 1.0)
        return np.clip(f * (255.0 / hi), 0, 255).astype(np.uint8)

# ------------------------ threaded I/O helper ------------------------

class IOWriter:
    """
    Asynchronous writer for .npy/.png to avoid blocking compute.
    Use context manager or call .drain() to ensure completion.
    """
    def __init__(self, max_workers: int = 4):
        self.pool = ThreadPoolExecutor(max_workers=max_workers) if max_workers > 0 else None
        self.futures = []

    def submit_npy(self, path: Path, array: np.ndarray, dtype: str):
        if self.pool is None:
            np.save(path, array.astype(dtype, copy=False))
            return
        # copy to isolate from later mutations
        arr = np.array(array, copy=True).astype(dtype, copy=False)
        self.futures.append(self.pool.submit(np.save, path, arr))

    def submit_png(self, path: Path, array_u8: np.ndarray):
        if self.pool is None:
            Image.fromarray(array_u8, mode="L").save(path)
            return
        img = np.array(array_u8, copy=True)
        self.futures.append(self.pool.submit(lambda p, a: Image.fromarray(a, "L").save(p), path, img))

    def drain(self):
        if self.pool is None:
            return
        for fut in self.futures:
            fut.result()
        self.futures.clear()

    def close(self):
        self.drain()
        if self.pool is not None:
            self.pool.shutdown(wait=True)

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()

# ------------------------ core conversion (single scene) ------------------------

def convert_events_to_nbin_from_events(
    events_txt: Path,
    out_npy_dir: Path,
    out_png_dir: Optional[Path],
    fps: float = 30.0,
    bins: int = 4,
    mode: str = "count",           # 'count' or 'signed'
    dtype: str = "int32",
    start_frame: int = 0,
    png_percentile: float = 99.5,
    polarity: str = "auto",        # 'auto', '01', 'pm1'
    min_frames: Optional[int] = None, # optional floor for number of frames
    io_workers: int = 4,
) -> int:
    """
    Stream events.txt and produce (H,W,bins) tensors for each original frame window.
    Also saves one PNG per bin if out_png_dir is provided.
    Returns number of frames written.
    """
    if bins <= 0:
        raise ValueError("--bins must be a positive integer")

    # Output dirs
    out_npy_dir.mkdir(parents=True, exist_ok=True)
    if out_png_dir is not None:
        out_png_dir.mkdir(parents=True, exist_ok=True)

    # Infer geometry & duration
    H, W, tmax = sniff_size_and_last_t(events_txt)
    frame_dur = 1.0 / fps
    sub_dur   = frame_dur / bins

    # Number of frames from timestamps (include last partial)
    n_frames = int(math.ceil((tmax + 1e-12) * fps))
    if min_frames is not None:
        n_frames = max(n_frames, int(min_frames))

    # Polarity scheme
    scheme = detect_polarity_scheme(events_txt) if polarity == "auto" else polarity
    print(f"  size={W}x{H}, frames={n_frames}, tmax={tmax:.6f}s, bins={bins}, polarity={scheme}")

    # Accumulator for current frame (bins,H,W) → save as (H,W,bins)
    acc = np.zeros((bins, H, W), dtype=np.int64)
    current_k = 0

    # For pretty zero-padded bin indices in filenames
    bin_digits = max(2, len(str(max(0, bins - 1))))

    # async I/O writer
    with IOWriter(max_workers=io_workers) as writer:

        def save_frame_outputs(k: int):
            """Save current N-bin frame k, then clear accumulator."""
            out_array = np.moveaxis(acc, 0, -1)  # (H,W,bins)
            npy_path = out_npy_dir / f"frame_{start_frame + k:06d}.npy"
            writer.submit_npy(npy_path, out_array, dtype)

            if out_png_dir is not None:
                # compute PNGs from current acc (before clearing)
                for b in range(bins):
                    img = bin_to_png_u8(acc[b], mode=mode, clip_percentile=png_percentile)
                    png_path = out_png_dir / f"frame_{start_frame + k:06d}_bin{b:0{bin_digits}d}.png"
                    writer.submit_png(png_path, img)
            acc.fill(0)

        # Stream events
        with open(events_txt, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 4:
                    continue
                try:
                    t = float(parts[0]); x = int(parts[1]); y = int(parts[2]); p_raw = float(parts[3])
                except Exception:
                    continue

                # Map to original frame index
                k = int(t // frame_dur)
                if k < 0:
                    continue
                if k >= n_frames:
                    # If timestamps slightly exceed tmax rounding, clamp to last frame
                    k = n_frames - 1

                # Flush completed frames
                while current_k < k:
                    save_frame_outputs(current_k)
                    current_k += 1

                # Sub-bin inside frame k
                t_in_frame = t - (k * frame_dur)
                b = int(t_in_frame // sub_dur)
                if b >= bins:
                    b = bins - 1  # guard at exact boundary

                # Accumulate
                if 0 <= x < W and 0 <= y < H:
                    if mode == "signed":
                        if scheme == "01":
                            p = +1 if p_raw > 0 else -1
                        elif scheme == "pm1":
                            p = +1 if p_raw > 0 else (-1 if p_raw < 0 else 0)
                        else:
                            p = +1 if p_raw > 0 else (-1 if p_raw < 0 else 0)
                        acc[b, y, x] += p
                    else:
                        acc[b, y, x] += 1

        # Flush remaining frames
        while current_k < n_frames:
            save_frame_outputs(current_k)
            current_k += 1

        # Ensure all pending writes are complete
        writer.drain()

    return n_frames

# ------------------------ worker wrapper (for processes) ------------------------

def _process_one_scene(ev: Path, in_root: Path, out_root: Path, args_dict: dict) -> tuple[str, int, Optional[str]]:
    """
    Called in a worker process. Returns (scene_rel_str, frames_written, error_message_or_None).
    """
    try:
        scene_rel = rel_scene_path(ev, in_root)
        out_npy = out_root / scene_rel / "frames_np"
        out_png = (out_root / scene_rel / "frames_png") if args_dict["png"] else None

        if args_dict["clean_existing"]:
            for d in [out_npy, out_png]:
                if d and d.exists():
                    # fast recursive delete
                    for p in sorted(d.rglob("*"), reverse=True):
                        try:
                            p.unlink()
                        except IsADirectoryError:
                            p.rmdir()
                    d.rmdir()

        print(f"[convert] {scene_rel} -> {out_npy}" + (f" & {out_png}" if out_png else ""))
        n = convert_events_to_nbin_from_events(
            events_txt=ev,
            out_npy_dir=out_npy,
            out_png_dir=out_png,
            fps=args_dict["fps"],
            bins=args_dict["bins"],
            mode=args_dict["mode"],
            dtype=args_dict["dtype"],
            start_frame=args_dict["start_frame"],
            png_percentile=args_dict["png_percentile"],
            polarity=args_dict["polarity"],
            min_frames=args_dict["min_frames"],
            io_workers=args_dict["io_workers"],
        )
        return (str(scene_rel), n, None)
    except Exception as e:
        return (ev.as_posix(), 0, f"{e}")

# ------------------------ entry point ------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert **/v2e/events.txt to per-frame N-bin .npy and per-bin PNGs (no tone-mapped frames needed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--in_root",  type=Path, required=True, help="Root containing <scene>/v2e/events.txt")
    ap.add_argument("--out_root", type=Path, required=True, help="Root to write <scene>/(frames_np, frames_png)")
    ap.add_argument("--fps", type=float, default=30.0, help="Original FPS for time binning")
    ap.add_argument("--bins", type=int, default=4, help="Number of temporal bins per frame")
    ap.add_argument("--mode", choices=["count","signed"], default="count", help="Per-pixel accumulation mode")
    ap.add_argument("--dtype", type=str, default="int32", help="Output dtype of .npy arrays")
    ap.add_argument("--start_frame", type=int, default=0, help="Starting index for output filenames")
    ap.add_argument("--png", action="store_true", help="Save PNG previews (one file per bin)")
    ap.add_argument("--png_percentile", type=float, default=99.5, help="Percentile for contrast clipping")
    ap.add_argument("--polarity", choices=["auto","01","pm1"], default="auto",
                    help="Polarity interpretation: auto, 0/1, or ±1")
    ap.add_argument("--clean_existing", action="store_true",
                    help="Remove existing frames_np/frames_png before writing")
    ap.add_argument("--min_frames", type=int, default=None,
                    help="Optional floor for #frames if you want to enforce a minimum")

    # NEW: parallelism knobs
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of processes to use across scenes (≥1).")
    ap.add_argument("--io_workers", type=int, default=4,
                    help="Threads for async .npy/.png writing per process (≥0).")

    args = ap.parse_args()

    ev_files = find_events_files(args.in_root)
    if not ev_files:
        print(f"No events.txt found under {args.in_root} (expected **/v2e/events.txt).", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ev_files)} scene(s).")
    total_frames = 0

    # Prepare a serial-friendly dict of args for workers
    args_dict = dict(
        fps=args.fps,
        bins=args.bins,
        mode=args.mode,
        dtype=args.dtype,
        start_frame=args.start_frame,
        png=args.png,
        png_percentile=args.png_percentile,
        polarity=args.polarity,
        clean_existing=args.clean_existing,
        min_frames=args.min_frames,
        io_workers=max(0, args.io_workers),
    )

    if args.workers <= 1 or len(ev_files) == 1:
        # Serial path (one process)
        for ev in ev_files:
            scene_rel, n, err = _process_one_scene(ev, args.in_root, args.out_root, args_dict)
            if err:
                print(f"  ERROR converting {scene_rel}: {err}", file=sys.stderr)
            else:
                print(f"  wrote {n} frame(s)  (PNGs: {n*args.bins if args.png else 0})")
                total_frames += n
    else:
        # Parallel across scenes
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one_scene, ev, args.in_root, args.out_root, args_dict): ev for ev in ev_files}
            for fut in as_completed(futures):
                scene_rel, n, err = fut.result()
                if err:
                    print(f"  ERROR converting {scene_rel}: {err}", file=sys.stderr)
                else:
                    print(f"  wrote {n} frame(s)  (PNGs: {n*args.bins if args.png else 0})")
                    total_frames += n

    print(f"Done. Total frames written: {total_frames}")

if __name__ == "__main__":
    # On some platforms, Pillow benefits from libjpeg-turbo via env var;
    # you can also tweak OMP_NUM_THREADS to keep BLAS from oversubscribing.
    # Example:
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
