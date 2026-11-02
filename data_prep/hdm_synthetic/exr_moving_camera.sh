python exr_moving_camera.py \
  --in_root_tif  /path/to/data/HdM-tone-mat \
  --in_root_exr /path/to/data/HdM-HDR-2014_Original-HDR-Camera-Footage \
  --out_root_tif /path/to/data/hdm_moving_mat_png \
  --out_root_exr /path/to/data/hdm_moving_mat_exr \
  --workers 12 \
  --speed_px 3.5 \
  --max_turn_deg 2 \
  --crop_w 1200 --crop_h 650