"""Second-stage "Shader" ControlNet.

Same backbone as :class:`EventControlNet` (SVD UNet down path + mid block), but instead of
raw events it receives the first-stage network's *features*:

* the event embedding map is added to its ``conv_in`` output,
* every raw down-path feature of the first stage is added to the matching Shader feature
  (skip tensors and the running hidden state), and the raw mid feature to the mid output.

Its own input is the noisy latent (4 channels) optionally concatenated with the latent of the
previous clip's last frame (4 more channels) for clip-to-clip continuity. Only the 13
zero-initialised heads are new parameters; everything else starts from the UNet weights.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.controlnets.controlnet import ControlNetOutput
from diffusers.models.modeling_utils import ModelMixin

from .event_controlnet import SVDDownPathMixin


class ShaderControlNet(ModelMixin, ConfigMixin, SVDDownPathMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int, ...]] = 2,
        cross_attention_dim: Union[int, Tuple[int, ...]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        num_attention_heads: Union[int, Tuple[int, ...]] = (5, 10, 20, 20),
        num_frames: int = 14,
        conditioning_channels: int = 32,
        conditioning_embedding_out_channels: Tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.shader_down_blocks, self.shader_mid_block = self._build_backbone(
            in_channels, down_block_types, block_out_channels, addition_time_embed_dim,
            projection_class_embeddings_input_dim, layers_per_block, cross_attention_dim,
            transformer_layers_per_block, num_attention_heads,
        )

    @classmethod
    def from_unet(cls, unet, load_weights: bool = True):
        kwargs = cls._unet_backbone_kwargs(unet)
        model = cls(**kwargs)
        if load_weights:
            model._copy_backbone_from_unet(unet)
        return model

    def _fit_input_channels(self, x: torch.Tensor) -> torch.Tensor:
        c_in = self.conv_in.in_channels
        if x.shape[1] < c_in:
            x = F.pad(x, (0, 0, 0, 0, 0, c_in - x.shape[1]))
        elif x.shape[1] > c_in:
            x = x[:, :c_in]
        return x

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        event_embedding: Optional[torch.Tensor] = None,
        generator_down_features: Optional[Sequence[torch.Tensor]] = None,
        generator_mid_feature: Optional[torch.Tensor] = None,
        generator_feature_scale: float = 1.0,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        """``sample``: (B, F, 4 or 8, h, w). Features come from :class:`EventControlNet`."""
        batch_size, num_frames = sample.shape[:2]
        emb = self._embed_time(sample, timestep, added_time_ids, num_frames)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        x = self.conv_in(self._fit_input_channels(sample.flatten(0, 1)))
        if event_embedding is not None:
            if event_embedding.dim() == 5:
                event_embedding = event_embedding.flatten(0, 1)
            x = x + event_embedding
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=x.dtype, device=x.device)

        gen = list(generator_down_features) if generator_down_features is not None else None
        if gen is not None:
            x = x + (generator_feature_scale * gen[0]).to(x.dtype)
        skips = [x]
        cursor = 1
        for block in self.down_blocks:
            if getattr(block, "has_cross_attention", False):
                x, res = block(hidden_states=x, temb=emb, encoder_hidden_states=encoder_hidden_states,
                               image_only_indicator=image_only_indicator)
            else:
                x, res = block(hidden_states=x, temb=emb, image_only_indicator=image_only_indicator)
            if gen is not None:
                fused = []
                for r in res:
                    g = (generator_feature_scale * gen[cursor]).to(r.dtype)
                    fused.append(r + g)
                    cursor += 1
                res = tuple(fused)
                x = x + g  # the last fused feature of this block also feeds the next block
            skips.extend(res)
        mid = self.mid_block(hidden_states=x, temb=emb, encoder_hidden_states=encoder_hidden_states,
                             image_only_indicator=image_only_indicator)
        if generator_mid_feature is not None:
            mid = mid + (generator_feature_scale * generator_mid_feature).to(mid.dtype)

        down = tuple(head(s) * conditioning_scale for head, s in zip(self.shader_down_blocks, skips))
        mid_out = self.shader_mid_block(mid) * conditioning_scale
        if not return_dict:
            return down, mid_out
        return ControlNetOutput(down_block_res_samples=down, mid_block_res_sample=mid_out)
