# Data formats

## Event voxels

All loaders expect one `.npy` file per frame interval holding signed event counts binned in time:
shape `(bins, H, W)` or `(H, W, bins)`, any float or integer dtype. File `i` holds the events
between frame `i-1` and frame `i` (so the file *precedes* the frame with the same index), and the
last time bin of a file is the one closest to the frame.

The network consumes 4 channels per output frame. For frame `k` of a clip starting at frame `t`,
the loaders stack the files `t-1 .. t+T-1` and sum nested windows of fine bins centred on the last
bin of file `t+k-1` (`hive_diff/data/events.py`):

| data | bins / file | window widths (fine bins) | normalisation |
|---|---|---|---|
| synthetic HdM training | 12 | 0, 4, 8, 12 | per channel, 0.995 quantile of \|x\| |
| EventHDR evaluation | 4 | 1, 3, 5, 7 | per frame, max \|x\| |
| EventHDR finetuning | 4 | 1, 3, 5, 7 | per channel, 0.995 quantile of \|x\| |
| LDR video (own captures, demo) | 12 | 0, 2, 4, 6 | per channel, 0.999 quantile of \|x\| |

Values end up in `[-1, 1]`, fp16, at the pixel resolution of the generated video (the event
encoder downsamples by 8 to the latent grid).

## Synthetic training set (HdM camera shake)

```
hdm_moving_camera/<scene>/hdr_shaken_exr/frame_%010d.exr        linear HDR, 650x1200   (video_root)
hdm_moving_camera_graded/<scene>/preview_png/frame_%010d.png    graded 8-bit target    (tone_root)
hdm_moving_camera_v2e_graded_npy/<scene>/frames_np/frame_%06d.npy   (650, 1200, 12) int32 events (events_root)
```

Splits: 21 training scenes, 1 validation scene (`poker_fullshot`), 4 test scenes (`bistro_01`,
`carousel_fireworks_07`, `showgirl_02`, `smith_hammering`); see
`data_prep/hdm_synthetic/split_manifest.json`. The loaders take every scene found under a root,
so the validation and test scenes are kept in `<root>_validate` and `<root>_test` and the training
configs point at the training roots. Download the HdM-HDR-2014 footage and generate the set with
the scripts in `data_prep/hdm_synthetic/`.

## EventHDR test set

```
eventhdr_test/<clip>/hdr/npy/image%09d.npy       (240, 320) uint16 linear HDR
eventhdr_test/<clip>/events/npy/image%09d.npy    (4, 240, 320) float32 event voxel
```

The evaluation protocol renders the reference frame at +7 EV (gain 128, sRGB), tone-maps the
ground truth with `log1p(500 x / max) / log1p(500)`, resizes both to 512x512 and scores clips
starting at dataset index 355 (frame 356) with stride = clip length (`scripts/eval_eventhdr.py`).
The paper numbers score every frame of each clip (`--all_frames`).

## Own captures (`LdrVideoDataset`)

```
<frames_root>/<scene>/*.png
<events_root>/<scene>/voxels_npy/*.npy     (12, H, W) per frame interval
```

Frames are resized to 512x512 for the reference and events to 1024x1024 when generating with
`--rescale 2` (3x3 overlapping tiles). The exposure token is fed 0 for such inputs.
