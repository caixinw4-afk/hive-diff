#!/usr/bin/env python3
# ---- set BEFORE importing numpy/opencv ----
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Optional, List, Tuple
import argparse, time, math
from pathlib import Path
import numpy as np
import cv2
import imageio.v3 as iio

try:
    cv2.setNumThreads(0)
except Exception:
    pass

# ---------------- I/O helpers ----------------

def read_hdr_rgb_float(path: str) -> np.ndarray:
    """Read EXR/TIFF/PNG/etc. as float32 RGB, clamp NaN/Inf/negatives."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float32) / float(np.iinfo(img.dtype).max)
    else:
        img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.maximum(img, 0.0)
    return img

def write_exr(path: Path, rgb_float: np.ndarray):
    """Save RGB float32 image to .exr (linear)."""
    assert path.suffix.lower() == ".exr", f"Expected .exr, got {path}"
    im = cv2.cvtColor(rgb_float.astype(np.float32, copy=False), cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path.as_posix(), im)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")

def center_crop(img: np.ndarray, w: int, h: int) -> np.ndarray:
    H, W = img.shape[:2]
    x0 = max((W - w)//2, 0); y0 = max((H - h)//2, 0)
    return img[y0:y0+h, x0:x0+w, :]

def crop_xy(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return img[y:y+h, x:x+w, :]

def list_scenes(root: Path) -> List[Path]:
    return [d for d in sorted(root.iterdir()) if d.is_dir()]

def list_frames_with_ext(scene_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    exts = tuple(e.lower() for e in exts)
    return [p for p in sorted(scene_dir.iterdir())
            if p.is_file() and p.suffix.lower() in exts]

# ------------- Constant-speed FOV motion path -------------

def _clip_angle(delta: float, max_abs: float) -> float:
    return float(np.clip(delta, -max_abs, +max_abs))

def _rand_small_angle(rng: np.random.RandomState, max_turn_rad: float) -> float:
    return float(rng.uniform(-max_turn_rad, +max_turn_rad))

def gen_const_speed_path(
    num_frames: int,
    W_src: int, H_src: int,
    crop_w: int, crop_h: int,
    speed_px: float = 2.0,
    max_turn_deg: float = 2.0,
    wall_avoid_gain: float = 0.035,
    edge_margin_px: int = 24,
    seed: Optional[int] = None,
    initial_heading_deg: Optional[float] = None,
) -> np.ndarray:
    """Generate a path with constant per-frame displacement (speed_px)."""
    rng = np.random.RandomState(seed)
    max_x = max(W_src - crop_w, 0)
    max_y = max(H_src - crop_h, 0)
    x = max_x / 2.0
    y = max_y / 2.0
    if initial_heading_deg is None:
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
    else:
        theta = math.radians(initial_heading_deg)

    max_turn_rad = math.radians(max(1e-6, max_turn_deg))
    path = np.zeros((num_frames, 2), dtype=np.int32)

    def steer_from_walls(x_cur: float, y_cur: float, theta_cur: float) -> float:
        d_left  = x_cur
        d_right = max_x - x_cur
        d_top   = y_cur
        d_bot   = max_y - y_cur

        def weight(d: float) -> float:
            if d >= edge_margin_px:
                return 0.0
            t = 1.0 - (d / max(edge_margin_px, 1e-6))
            return float(t * t)

        wL = weight(d_left); wR = weight(d_right)
        wT = weight(d_top);  wB = weight(d_bot)
        rep_x = (+wL - wR)
        rep_y = (+wT - wB)
        if rep_x == 0.0 and rep_y == 0.0:
            return 0.0
        desired = math.atan2(rep_y, rep_x)
        diff = math.atan2(math.sin(desired - theta_cur), math.cos(desired - theta_cur))
        return float(wall_avoid_gain * diff)

    for i in range(num_frames):
        wall_delta   = steer_from_walls(x, y, theta)
        random_delta = _rand_small_angle(rng, max_turn_rad * 0.5)
        dtheta = _clip_angle(wall_delta + random_delta, max_turn_rad)
        theta = theta + dtheta

        dx = speed_px * math.cos(theta)
        dy = speed_px * math.sin(theta)
        x_new = x + dx
        y_new = y + dy

        bounced = False
        if x_new < 0.0 or x_new > max_x:
            theta = np.pi - theta; bounced = True
        if y_new < 0.0 or y_new > max_y:
            theta = -theta; bounced = True
        if bounced:
            dx = speed_px * math.cos(theta)
            dy = speed_px * math.sin(theta)
            x_new = np.clip(x + dx, 0.0, float(max_x))
            y_new = np.clip(y + dy, 0.0, float(max_y))

        x, y = x_new, y_new
        path[i, 0] = int(round(x))
        path[i, 1] = int(round(y))
    return path

# ------------- Dual-scene processing (EXR + TIF/PNG) -------------

def process_scene_dual(
    scene_exr_dir: Path, scene_lq_dir: Path,
    out_scene_exr: Path, out_scene_lq: Path,
    speed_px: float, max_turn_deg: float,
    wall_avoid_gain: float, edge_margin_px: int,
    seed: Optional[int],
    preview_pct: float,
    initial_heading_deg: Optional[float],
    center_w: int, center_h: int,
    crop_w: int, crop_h: int,
) -> str:
    t0 = time.perf_counter()

    # outputs
    exr_frames_dir = out_scene_exr / "hdr_shaken_exr"
    exr_png_dir    = out_scene_exr / "preview_png"
    lq_png_dir     = out_scene_lq / "preview_png"
    exr_frames_dir.mkdir(parents=True, exist_ok=True)
    exr_png_dir.mkdir(parents=True, exist_ok=True)
    lq_png_dir.mkdir(parents=True, exist_ok=True)

    # inputs
    exr_list = list_frames_with_ext(scene_exr_dir, (".exr",))
    # ↓ now accepts TIFF **and** PNG
    lq_list  = list_frames_with_ext(scene_lq_dir, (".tif", ".tiff", ".png"))
    if not exr_list:
        return f"[SKIP] {scene_exr_dir.name}: no EXR frames"
    if not lq_list:
        return f"[SKIP] {scene_lq_dir.name}: no TIFF/PNG frames"
    if len(exr_list) != len(lq_list):
        return f"[ERR ] {scene_exr_dir.name}: frame count mismatch exr={len(exr_list)} vs lq={len(lq_list)}"

    # center-crop geometry based on EXR (assumed identical sizes)
    first_exr = read_hdr_rgb_float(str(exr_list[0]))
    base = center_crop(first_exr, center_w, center_h)
    Hc, Wc = base.shape[:2]
    if crop_w > Wc or crop_h > Hc:
        return f"[ERR ] {scene_exr_dir.name}: crop {crop_w}x{crop_h} exceeds centered size {Wc}x{Hc}"

    # one shared path for BOTH modalities
    path = gen_const_speed_path(
        num_frames=len(exr_list),
        W_src=Wc, H_src=Hc,
        crop_w=crop_w, crop_h=crop_h,
        speed_px=speed_px,
        max_turn_deg=max_turn_deg,
        wall_avoid_gain=wall_avoid_gain,
        edge_margin_px=edge_margin_px,
        seed=seed,
        initial_heading_deg=initial_heading_deg,
    )

    # iterate frames
    for i, (pexr, plq) in enumerate(zip(exr_list, lq_list)):
        stem = f"frame_{i:010d}"

        # EXR side: crop + save EXR and preview PNG
        exr_img = read_hdr_rgb_float(str(pexr))
        exr_img = center_crop(exr_img, center_w, center_h)
        x0, y0 = int(path[i, 0]), int(path[i, 1])
        exr_crop = crop_xy(exr_img, x0, y0, crop_w, crop_h)
        write_exr(exr_frames_dir / f"{stem}.exr", exr_crop)

        prc_exr = np.percentile(exr_crop, preview_pct)
        prc_exr = float(prc_exr) if prc_exr > 1e-12 else 1.0
        vis_exr = np.clip(exr_crop / prc_exr, 0.0, 1.0)
        iio.imwrite((exr_png_dir / f"{stem}.png").as_posix(),
                    (vis_exr * 255.0 + 0.5).astype(np.uint8))

        # LQ side (TIFF/PNG): crop + preview PNG
        lq_img = read_hdr_rgb_float(str(plq))
        lq_img = center_crop(lq_img, center_w, center_h)
        lq_crop = crop_xy(lq_img, x0, y0, crop_w, crop_h)

        prc_lq = np.percentile(lq_crop, preview_pct)
        prc_lq = float(prc_lq) if prc_lq > 1e-12 else 1.0
        vis_lq = np.clip(lq_crop / prc_lq, 0.0, 1.0)
        iio.imwrite((lq_png_dir / f"{stem}.png").as_posix(),
                    (vis_lq * 255.0 + 0.5).astype(np.uint8))

    dt = time.perf_counter() - t0
    return (f"[OK ] {scene_exr_dir.name} → EXR:{exr_frames_dir} (+PNGs) | "
            f"LQ:{lq_png_dir} (PNGs only) | {len(exr_list)} frames | {dt:.2f}s")

# ------------- Scene worker (picklable) -------------

def _scene_worker(task: Tuple[str, str, str, str, float, float, float, int, Optional[int], float, Optional[float], int, int, int, int]):
    (scene_exr_dir_str, scene_lq_dir_str,
     out_scene_exr_str, out_scene_lq_str,
     speed_px, max_turn_deg, wall_avoid_gain, edge_margin_px,
     seed, preview_pct, initial_heading_deg,
     center_w, center_h, crop_w, crop_h) = task
    try:
        return process_scene_dual(
            Path(scene_exr_dir_str), Path(scene_lq_dir_str),
            Path(out_scene_exr_str), Path(out_scene_lq_str),
            speed_px, max_turn_deg, wall_avoid_gain, edge_margin_px,
            seed, preview_pct, initial_heading_deg,
            center_w, center_h, crop_w, crop_h
        )
    except Exception as e:
        return f"[ERR ] {Path(scene_exr_dir_str).name}: {e}"

# -------------------- CLI --------------------

def main():
    prog_t0 = time.perf_counter()
    ap = argparse.ArgumentParser(
        "Apply identical moving FOV to paired EXR & LQ (TIFF/PNG) roots (same subfolders & frame counts). "
        "Outputs: EXR crops (.exr) + PNG previews; LQ previews only."
    )
    ap.add_argument("--in_root_exr", type=str, required=True, help="Root with EXR video subfolders.")
    ap.add_argument("--in_root_tif", type=str, required=True,
                    help="Root with TIFF/PNG video subfolders (.tif/.tiff/.png).")
    ap.add_argument("--out_root_exr", type=str, required=True, help="Output root for EXR side.")
    ap.add_argument("--out_root_tif", type=str, required=True, help="Output root for TIFF/PNG side.")

    # Motion / FOV
    ap.add_argument("--speed_px", type=float, default=2.0, help="Pixels per frame (fixed step length).")
    ap.add_argument("--max_turn_deg", type=float, default=2.0, help="Max heading change per frame in degrees.")
    ap.add_argument("--wall_avoid_gain", type=float, default=0.035, help="Steering gain away from walls.")
    ap.add_argument("--edge_margin_px", type=int, default=24, help="Start steering when within this many pixels of an edge.")

    # Randomness controls
    ap.add_argument("--seed", type=int, default=None, help="Base seed (None = non-deterministic).")
    ap.add_argument("--seed_per_scene", action="store_true",
                    help="Different seed per scene from base seed XOR scene name.")
    ap.add_argument("--initial_heading_deg", type=float, default=None,
                    help="Fix initial heading in degrees (None = randomized).")

    # Preview
    ap.add_argument("--preview_pct", type=float, default=99.5, help="Percentile for preview normalization.")

    # FOV sizing
    ap.add_argument("--center_w", type=int, default=1840, help="Centered pre-crop width before FOV motion.")
    ap.add_argument("--center_h", type=int, default=1000, help="Centered pre-crop height before FOV motion.")
    ap.add_argument("--crop_w",   type=int, default=1740, help="Width of moving FOV window.")
    ap.add_argument("--crop_h",   type=int, default=900,  help="Height of moving FOV window.")

    # Parallel
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()-1), help="Processes for parallel scenes.")
    ap.add_argument("--recursive", action="store_true", help="If set, also look for scenes one level deeper under EXR root.")
    args = ap.parse_args()

    in_root_exr = Path(args.in_root_exr)
    in_root_lq  = Path(args.in_root_tif)   # name kept for backward-compat
    out_root_exr = Path(args.out_root_exr); out_root_exr.mkdir(parents=True, exist_ok=True)
    out_root_lq  = Path(args.out_root_tif); out_root_lq.mkdir(parents=True, exist_ok=True)

    # discover scenes by EXR root (authoritative)
    if args.recursive:
        exr_candidates = [d for d in in_root_exr.rglob("*") if d.is_dir()]
    else:
        exr_candidates = list_scenes(in_root_exr)

    # map EXR scene → matching LQ (TIFF/PNG) scene by same relative path name
    scenes = []
    for exr_scene in exr_candidates:
        if not list_frames_with_ext(exr_scene, (".exr",)):
            continue
        rel = exr_scene.relative_to(in_root_exr)
        lq_scene = in_root_lq / rel
        if lq_scene.is_dir():
            scenes.append((exr_scene, lq_scene))
        else:
            print(f"[WARN] Missing TIFF/PNG scene for {rel}; skipping")

    if not scenes:
        raise SystemExit(f"No paired EXR & TIFF/PNG scene subfolders found under {in_root_exr} and {in_root_lq}")

    print(f"[INFO] Found {len(scenes)} paired scene(s)")

    # build tasks
    tasks = []
    for exr_scene, lq_scene in scenes:
        rel = exr_scene.relative_to(in_root_exr)
        out_scene_exr = out_root_exr / rel
        out_scene_lq  = out_root_lq / rel

        if args.seed_per_scene:
            base = 0 if args.seed is None else (args.seed & 0xffffffff)
            h = (hash(str(rel)) & 0xffffffff)
            scene_seed = int((base ^ h) & 0xffffffff)
        else:
            scene_seed = args.seed

        tasks.append((
            str(exr_scene), str(lq_scene),
            str(out_scene_exr), str(out_scene_lq),
            float(args.speed_px), float(args.max_turn_deg), float(args.wall_avoid_gain), int(args.edge_margin_px),
            scene_seed, float(args.preview_pct),
            None if args.initial_heading_deg is None else float(args.initial_heading_deg),
            int(args.center_w), int(args.center_h), int(args.crop_w), int(args.crop_h),
        ))

    # run scenes in parallel
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = [ex.submit(_scene_worker, t) for t in tasks]
        for fut in as_completed(futs):
            print(fut.result())

    prog_dt = time.perf_counter() - prog_t0
    print(f"[TIME] Total wall time: {prog_dt:.2f}s")

if __name__ == "__main__":
    main()
