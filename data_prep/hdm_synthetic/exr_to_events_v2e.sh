#!/usr/bin/env bash

# Activate your env if needed:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate vid2e
export PYTHONNOUSERSITE=1 
export OPENCV_IO_ENABLE_OPENEXR=1      # allow OpenCV to read EXR
export QT_QPA_PLATFORM=offscreen       # headless Qt
export MPLBACKEND=Agg                  # headless matplotlib (if v2e plots)

IN_ROOT="/path/to/data/hdm_moving_mat_png_temp"
OUT_ROOT="/path/to/data/hdm_moving_mat_v2e"
FPS="30"  # <-- set true FPS here
SLOMO="/path/to/v2e/SuperSloMo39.ckpt"
TMPDIR_CUSTOM="/path/to/tmp"  # large disk


# Preview settings


mkdir -p "$TMPDIR_CUSTOM"

# --------- OPTION A: keep SloMo, reduce memory (recommended first) ----------
python exr_to_events_v2e.py \
  --in_root "$IN_ROOT" \
  --out_root "$OUT_ROOT" \
  --fps "$FPS" \
  --v2e_exec "python -m v2e" \
  --input_subdir preview_png \
  --input_exts png,tif,tiff,exr \
  --slomo_model "$SLOMO" \
  --batch_size 12 \
  --timestamp_resolution 0.003 \
  --no_auto_ts_res \
  --pos_thres 0.05 --neg_thres 0.05 --sigma_thres 0.015 \
  --enable_preview --preview_mode time --preview_param 0.01 \
  --clean_existing \
  --tmp_dir "$TMPDIR_CUSTOM"