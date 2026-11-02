# HiVE-Diff: Event-Guided HDR Video Diffusion

HiVE-Diff turns a single LDR reference frame plus the event stream of an event camera into a
tone-mapped HDR video clip, recovering detail in regions where the frame is over- or
under-exposed. It is built on the frozen Stable Video Diffusion (SVD) image-to-video model and two
ControlNets:

```
reference LDR frame ──► CLIP token (+ exposure token) ──┐
                     ─► VAE latent ────────────────────┐ │
events (4 windows/frame) ► Generator ControlNet (stage 1) │ │   residuals
                                 │ raw features         ▼ ▼
previous clip latent ─► Shader ControlNet (stage 2) ──► frozen SVD UNet ──► VAE decoder ──► frames
```

* **Generator** (stage 1): an event-conditioned ControlNet that learns, on synthetic data, to
  continue the reference frame through time from events.
* **Shader** (stage 2): a second ControlNet fed with the Generator's features, the previous clip's
  latent and an exposure token; it produces tone-mapped HDR frames and keeps consecutive clips
  consistent, so arbitrarily long videos can be generated clip by clip.

## Installation

```bash
conda create -n hive python=3.11 && conda activate hive
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[eval]"
```

Download the SVD base weights once (requires accepting Stability AI's license on Hugging Face):

```bash
huggingface-cli download stabilityai/stable-video-diffusion-img2vid --include "vae/*" "unet/*" "image_encoder/*" "feature_extractor/*" "scheduler/*" "model_index.json"
```

## Model zoo

| name | frames / clip | Generator | Shader | notes |
|---|---|---|---|---|
| `hive_5f` | 5 | `generator` | `shader/` + `ev_token.pt` | synthetic training only |
| `hive_5f_eventhdr` | 5 | finetuned on EventHDR | same Shader | real-capture results |
| `hive_20f` | 20 | `generator` | `shader/` + `ev_token.pt` | long clips, slower |
| `hive_20f_eventhdr` | 20 | finetuned on EventHDR | same Shader | default setting |

The weights are not uploaded yet. [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) lists every checkpoint
with its training recipe; download links will be added there.

## Quick start

Generate from your own LDR frames + event voxels (12 time bins per frame interval, summed into
4 nested windows per frame, see [docs/DATA.md](docs/DATA.md) for the format):

```bash
python scripts/demo_video.py \
  --generator checkpoints/hive_20f_eventhdr/generator --shader checkpoints/hive_20f \
  --frames_root data/my_capture/imgs --events_root data/my_capture/events --scene 001 \
  --output_dir outputs/demo_001
```

Evaluate on the EventHDR test set (PSNR / histogram-matched PSNR / SSIM / LPIPS on tone-mapped
frames; `--all_frames` scores every frame of a clip, which is the setting used in the paper):

```bash
python scripts/eval_eventhdr.py \
  --generator checkpoints/hive_20f_eventhdr/generator --shader checkpoints/hive_20f \
  --data_root data/eventhdr_test --num_frames 20 --item_start 355 --all_frames \
  --output_dir outputs/eventhdr_20f
```

Python API:

```python
from hive_diff.utils.loading import load_pipeline
pipe = load_pipeline("stabilityai/stable-video-diffusion-img2vid", "checkpoints/hive_20f/generator", "checkpoints/hive_20f")
out = pipe(reference_pil, events, exposure_value=0.0, num_frames=20)   # events: (20, 4, 512, 512) in [-1, 1]
frames, latents = out.frames[0], out.latents                          # chain: pipe.chain_latents(latents)
```

## Training

Data preparation for the synthetic set is described in
[data_prep/hdm_synthetic/README.md](data_prep/hdm_synthetic/README.md).

```bash
# stage 1: Generator on synthetic data
accelerate launch scripts/train_generator.py --config configs/train_generator_20f.yaml
# stage 1b (optional): finetune the Generator on real captures, with distillation from its initial weights
accelerate launch scripts/train_generator.py --config configs/finetune_generator_eventhdr_20f.yaml
# stage 2: Shader + exposure token on top of the frozen Generator
accelerate launch scripts/train_shader.py --config configs/train_shader_20f.yaml
```

Every option in the YAML files can be overridden on the command line (`--lr 1e-5`). Training the
20-frame models needs an 80 GB GPU (batch 1, gradient checkpointing); 5-frame models fit in 48 GB.
The validation and test scenes of the synthetic set live in separate roots, so the training
configs only see the 21 training scenes.

## Tests

```bash
pytest tests/                          # CPU-only unit tests on small models
HIVE_GENERATOR_CKPT=checkpoints/hive_20f/generator HIVE_SHADER_CKPT=checkpoints/hive_20f/shader pytest tests/
```

With the two environment variables set, the tests also check that the released checkpoints match
the model definitions key by key.

## Repository layout

```
hive_diff/models/      EventControlNet, ShaderControlNet, ControlledSVDUNet, ExposureTokenMLP, EventReconstructor
hive_diff/pipelines/   HiveDiffPipeline (sampling loop: per-frame CFG, tiling, clip chaining)
hive_diff/data/        datasets: synthetic HdM, EventHDR test/finetune, LDR video; event-window construction
hive_diff/utils/       model loading, image ops, metrics
scripts/               train_generator.py, train_shader.py, eval_eventhdr.py, demo_video.py
configs/               training recipes of the released models
data_prep/             synthetic data generation (camera-shake HDR video + v2e events)
docs/                  data formats, model zoo
```

## Acknowledgements

HiVE-Diff builds on [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
and [diffusers](https://github.com/huggingface/diffusers). The synthetic training data uses
[HdM-HDR-2014](https://www.hdm-stuttgart.de/vmlab/hdm-hdr-2014) footage and
[v2e](https://github.com/SensorsINI/v2e); real-capture evaluation uses the EventHDR dataset.
The SVD weights and the HdM footage come with their own (non-commercial) licenses.

## License

The code is released under Apache-2.0, see `LICENSE`.
