# Model zoo

All checkpoints are `diffusers`-style directories (`config.json` + `diffusion_pytorch_model.safetensors`)
and load with `EventControlNet.from_pretrained` / `ShaderControlNet.from_pretrained`. A "shader
directory" holds `shader/` and `ev_token.pt` (the exposure-token MLP state dict).

Download links will be added when the weights are uploaded.

| release name | frames | contents | training recipe |
|---|---|---|---|
| `hive_5f/generator` | 5 | EventControlNet | `configs/train_generator_20f.yaml` with `num_frames: 5`, ~9.6k steps |
| `hive_5f/` (shader dir) | 5 | ShaderControlNet + `ev_token.pt` | `configs/train_shader_20f.yaml` with `num_frames: 5`, step 25500 |
| `hive_5f_eventhdr/generator` | 5 | Generator finetuned on EventHDR, no distillation, step 200 | `configs/finetune_generator_eventhdr_20f.yaml`, `num_frames: 5`, distill weights 0 |
| `hive_5f_eventhdr_distill/generator` | 5 | same with distillation (0.1 residual / 0.3 encoder, warmup 500), step 700 | |
| `hive_5f_captured/generator` | 5 | Generator finetuned on in-house captures (`LdrVideoDataset`), steps 400 / 900 | `--data ldr`, lr 5e-6 |
| `hive_20f/generator` | 20 | EventControlNet | `configs/train_generator_20f.yaml`, step 11500 |
| `hive_20f/` (shader dir) | 20 | ShaderControlNet + `ev_token.pt` | `configs/train_shader_20f.yaml`, step 7000 |
| `hive_20f_eventhdr/generator` | 20 | Generator finetuned on EventHDR, step 400 (no distill) / 600 (distill) | `configs/finetune_generator_eventhdr_20f.yaml` |

Notes

* The Shader is trained once per clip length on the synthetic data and is reused unchanged with
  every finetuned Generator of the same clip length.
* Base model: `stabilityai/stable-video-diffusion-img2vid` (14-frame SVD), fp16 variant.
