"""Structural tests with tiny configurations (CPU only): shapes, head counts, residual
multiplicities, zero-initialised heads, and (optionally) state-dict compatibility with a
checkpoint given via HIVE_GENERATOR_CKPT / HIVE_SHADER_CKPT (directories)."""
import json
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hive_diff.models import ControlledSVDUNet, EventControlNet, ExposureTokenMLP, ShaderControlNet  # noqa: E402

SMALL = dict(block_out_channels=(32, 64, 64, 64), num_attention_heads=(1, 2, 2, 2), cross_attention_dim=16,
             addition_time_embed_dim=8, projection_class_embeddings_input_dim=24, layers_per_block=2,
             transformer_layers_per_block=1, num_frames=2, in_channels=8, out_channels=4)


def test_forward_shapes():
    torch.manual_seed(0)
    unet = ControlledSVDUNet(**SMALL, sample_size=8)
    gen = EventControlNet.from_unet(unet, event_channels=4)
    shd = ShaderControlNet.from_unet(unet)
    assert len(gen.controlnet_down_blocks) == 12 and len(shd.shader_down_blocks) == 12
    assert unet.skip_multiplicities() == [4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1]
    B, F, h = 1, 2, 8
    x = torch.randn(B, F, 8, h, h)
    ev = torch.randn(B, F, 4, 8 * h, 8 * h).clamp(-1, 1)
    ehs, atid, t = torch.randn(B, 1, 16), torch.tensor([[6., 127., 0.02]]), torch.tensor([1.5])
    with torch.no_grad():
        g = gen(x, t, ehs, atid, events=ev)
        s = shd(x, t, ehs, atid, event_embedding=g.event_embedding, generator_down_features=g.raw_down_features,
                generator_mid_feature=g.raw_mid_feature)
        out = unet(x, t, ehs, atid, down_block_additional_residuals=s.down_block_res_samples,
                   mid_block_additional_residual=s.mid_block_res_sample).sample
    shapes = [tuple(r.shape) for r in g.raw_down_features]
    assert shapes[0] == (B * F, 32, h, h) and shapes[3] == (B * F, 32, h // 2, h // 2) and shapes[-1] == (B * F, 64, h // 8, h // 8)
    assert tuple(g.event_embedding.shape) == (B * F, 32, h, h) and tuple(out.shape) == (B, F, 4, h, h)
    assert all(float(r.abs().max()) == 0 for r in s.down_block_res_samples) and float(s.mid_block_res_sample.abs().max()) == 0
    assert tuple(ExposureTokenMLP(16)(torch.tensor([2.0])).shape) == (1, 16)


def _safetensors_shapes(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}


def test_checkpoint_keys():
    gen_dir, shd_dir = os.environ.get("HIVE_GENERATOR_CKPT"), os.environ.get("HIVE_SHADER_CKPT")
    if not (gen_dir and shd_dir):
        print("HIVE_GENERATOR_CKPT / HIVE_SHADER_CKPT not set; skipping")
        return
    for cls, d in ((EventControlNet, gen_dir), (ShaderControlNet, shd_dir)):
        cfg = json.load(open(os.path.join(d, "config.json")))
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        with torch.device("meta"):
            model = cls(**cfg)
        ours = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        theirs = _safetensors_shapes(os.path.join(d, "diffusion_pytorch_model.safetensors"))
        missing = set(ours) - set(theirs)
        mismatch = {k for k in set(ours) & set(theirs) if ours[k] != theirs[k]}
        assert not missing and not mismatch, (cls.__name__, sorted(missing)[:3], sorted(mismatch)[:3])


if __name__ == "__main__":
    test_forward_shapes(); test_checkpoint_keys(); print("model tests OK")
