# HdM synthetic training data (camera-shake HDR video + simulated events)

Source footage: HdM-HDR-2014 (Stuttgart Media University), one folder of linear EXR frames per
clip. The footage itself cannot be redistributed; download it from the HdM page and run the steps
below. The split used in the paper is `split_manifest.json` (21 training scenes, validation
`poker_fullshot`, test `bistro_01`, `carousel_fireworks_07`, `showgirl_02`, `smith_hammering`).

Each `.sh` wrapper holds the settings used for the released models; set the root paths before
running. Steps 1 and 2 need OpenCV with EXR support (`OPENCV_IO_ENABLE_OPENEXR=1`, e.g. the
conda-forge build); step 3 needs [v2e](https://github.com/SensorsINI/v2e) and its SuperSloMo
checkpoint.

1. `gen_tonemap.m` (MATLAB) — tone-maps the original EXR frames to 8-bit PNG (global 0.5/99.5
   percentile clip, MATLAB `tonemap` with 4x4 tiles, saturation 3):
   `HdM-HDR-2014_tonemapped/<scene>/*.png`.
2. `exr_moving_camera.py` — cuts the same moving field-of-view window (random-walk camera path,
   `--speed_px 3.5 --max_turn_deg 2 --crop_w 1200 --crop_h 650`) out of the EXR frames and the
   tone-mapped PNGs of each clip and writes
   `hdm_moving_camera/<scene>/hdr_shaken_exr/*.exr` (linear HDR, training `video_root`) and
   `hdm_moving_camera_graded/<scene>/preview_png/*.png` (graded 8-bit target, `tone_root`).
   `hdm_moving_camera/<scene>/preview_png/` is a percentile-normalised preview of the EXR frames
   and is not used for training.
3. `exr_to_events_v2e.py` — runs v2e on `hdm_moving_camera_graded/<scene>/preview_png` (30 fps,
   SuperSloMo upsampling, `--pos_thres 0.05 --neg_thres 0.05 --sigma_thres 0.015
   --timestamp_resolution 0.003`) and writes `hdm_moving_camera_v2e/<scene>/v2e/events.txt`.
4. `events_to_npy_frames.py` — bins the events of every frame interval into an `(H, W, bins)`
   int32 count tensor (`--mode signed --bins 12`; `--bins 4` for the 4-bin variant):
   `hdm_moving_camera_v2e_graded_npy/<scene>/frames_np/frame_%06d.npy`.
5. Move the validation and test scenes of each root into `<root>_validate` / `<root>_test`
   as listed in `split_manifest.json`; the training configs point at the training roots only.
