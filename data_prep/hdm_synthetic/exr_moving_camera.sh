#!/usr/bin/env bash
python exr_moving_camera.py \
  --in_root_exr  /path/to/HdM-HDR-2014_Original-HDR-Camera-Footage \
  --in_root_tif  /path/to/HdM-HDR-2014_tonemapped \
  --out_root_exr /path/to/hdm_graded/hdm_moving_camera \
  --out_root_tif /path/to/hdm_graded/hdm_moving_camera_graded \
  --workers 12 \
  --speed_px 3.5 \
  --max_turn_deg 2 \
  --crop_w 1200 --crop_h 650
