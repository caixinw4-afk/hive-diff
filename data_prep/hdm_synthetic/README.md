# HdM synthetic training data (camera-shake HDR video + simulated events)

Source footage: HdM-HDR-2014 (Stuttgart Media University). Download it from the HdM page and
generate the set with the scripts here. The split used in the paper is in
`split_manifest.json` (4 held-out test scenes: bistro_01, carousel_fireworks_07, showgirl_02,
smith_hammering; validation: poker_fullshot).

Pipeline (one scene = one HdM clip):

1. `exr_moving_camera.py` — reads the original HDR EXR frames and a tone-mapped/graded
   8-bit copy of the same clip, cuts a moving field-of-view window (random-walk camera path,
   `--speed_px 3.5 --max_turn_deg 2 --crop_w 1200 --crop_h 650`) and writes
   `<scene>/hdr_shaken_exr/*.exr` (linear HDR target) and `<scene>/preview_png/*.png`.
2. `exr_to_events_v2e.py` — runs v2e on the 8-bit frames (30 fps, SuperSloMo upsampling,
   `--pos_thres 0.05 --neg_thres 0.05 --sigma_thres 0.015 --timestamp_resolution 0.003`)
   and writes `<scene>/v2e/events.txt`.
3. `events_to_npy_frames.py` — bins events per source frame into `(H, W, bins)` int32 count
   tensors (`--mode signed --bins 12`, or `--bins 4` for the 4-bin variant) at
   `<scene>/frames_np/frame_%06d.npy`.
4. `gen_tonemap.m` (MATLAB) — the graded 8-bit reference used as `tone_root`
   (percentile clip 0.5/99.5, MATLAB `tonemap` with 4x4 tiles, saturation 3).
   TODO: provide a Python equivalent.

Reading EXR in Python: the training loaders use the `OpenEXR` package (pip wheels of OpenCV cannot
decode EXR). The generator scripts here still use `cv2` with `OPENCV_IO_ENABLE_OPENEXR=1`, which
works with a conda-forge OpenCV build.

Status: scripts copied verbatim from the research tree; CLI defaults in the `.sh` files are
the ones used for the released models. Paths inside the `.sh` files still point at the
original cluster and must be edited.
