#!/usr/bin/env bash
export OPENCV_IO_ENABLE_OPENEXR=1
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg

IN_ROOT="/path/to/hdm_graded/hdm_moving_camera_graded"
OUT_ROOT="/path/to/hdm_graded/hdm_moving_camera_v2e"
SLOMO="/path/to/v2e/SuperSloMo39.ckpt"
TMP_DIR="/path/to/tmp"
mkdir -p "$TMP_DIR"

python exr_to_events_v2e.py \
  --in_root "$IN_ROOT" \
  --out_root "$OUT_ROOT" \
  --fps 30 \
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
  --tmp_dir "$TMP_DIR"
