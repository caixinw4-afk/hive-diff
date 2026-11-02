python events_to_npy_frames.py \
  --in_root  /path/to/data/hdm_moving_mat_v2e_home \
  --out_root /path/to/data/hdm_moving_mat_v2e_npy \
  --fps 30 \
  --mode signed \
  --png \
  --bins 12 \
  --clean_existing \
  --workers 12 --io_workers 12