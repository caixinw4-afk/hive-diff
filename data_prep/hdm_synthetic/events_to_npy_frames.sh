#!/usr/bin/env bash
python events_to_npy_frames.py \
  --in_root  /path/to/hdm_graded/hdm_moving_camera_v2e \
  --out_root /path/to/hdm_graded/hdm_moving_camera_v2e_graded_npy \
  --fps 30 \
  --mode signed \
  --png \
  --bins 12 \
  --clean_existing \
  --workers 12 --io_workers 12
