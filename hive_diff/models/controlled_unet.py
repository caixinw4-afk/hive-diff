"""SVD UNet that accepts ControlNet residuals.

The constructor and weights are the stock ``UNetSpatioTemporalConditionModel``; only ``forward``
is extended with ``down_block_additional_residuals`` / ``mid_block_additional_residual``.

``residual_mode`` controls how the down residuals reach the skip connections:

* ``"multi"`` (default) — residual *i* is added ``(n_blocks - b_i)`` times, where ``b_i`` is the
  index of the down block that produced skip *i* (``conv_in`` counts as block 0); with four
  blocks this is x4 / x3 / x2 / x1 by resolution level; the provided checkpoints expect it.
* ``"single"`` — each residual is added once to its skip tensor (classic ControlNet).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from diffusers.models.unets.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionModel,
    UNetSpatioTemporalConditionOutput,
)


class ControlledSVDUNet(UNetSpatioTemporalConditionModel):
    residual_mode: str = "multi"

    def skip_multiplicities(self) -> list:
        """Number of times each down residual is added under ``residual_mode="multi"``."""
        n = len(self.down_blocks)
        mult = [n]  # conv_in output, produced before block 0
        for b, block in enumerate(self.down_blocks):
            count = len(block.resnets) + (1 if getattr(block, "downsamplers", None) else 0)
            mult.extend([n - b] * count)
        return mult

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        down_block_additional_residuals: Optional[Sequence[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = any(s % default_overall_up_factor != 0 for s in sample.shape[-2:])
        upsample_size = None

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            dtype = torch.float64 if isinstance(timestep, float) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif timesteps.dim() == 0:
            timesteps = timesteps[None].to(sample.device)
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        emb = self.time_embedding(self.time_proj(timesteps).to(sample.dtype))
        time_embeds = self.add_time_proj(added_time_ids.flatten()).reshape(batch_size, -1).to(emb.dtype)
        emb = emb + self.add_embedding(time_embeds)

        sample = sample.flatten(0, 1)
        emb = emb.repeat_interleave(num_frames, dim=0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
        sample = self.conv_in(sample)
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        skips = [sample]
        for block in self.down_blocks:
            if getattr(block, "has_cross_attention", False):
                sample, res = block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states,
                                    image_only_indicator=image_only_indicator)
            else:
                sample, res = block(hidden_states=sample, temb=emb, image_only_indicator=image_only_indicator)
            skips.extend(res)

        if down_block_additional_residuals is not None:
            if len(down_block_additional_residuals) != len(skips):
                raise ValueError(f"expected {len(skips)} down residuals, got {len(down_block_additional_residuals)}")
            if self.residual_mode == "multi":
                mults = self.skip_multiplicities()
            elif self.residual_mode == "single":
                mults = [1] * len(skips)
            else:
                raise ValueError(f"unknown residual_mode {self.residual_mode!r}")
            skips = [s + m * r for s, r, m in zip(skips, down_block_additional_residuals, mults)]

        sample = self.mid_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states,
                                image_only_indicator=image_only_indicator)
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        skips = tuple(skips)
        for i, block in enumerate(self.up_blocks):
            is_final = i == len(self.up_blocks) - 1
            res = skips[-len(block.resnets):]
            skips = skips[: -len(block.resnets)]
            if not is_final and forward_upsample_size:
                upsample_size = skips[-1].shape[2:]
            if getattr(block, "has_cross_attention", False):
                sample = block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res,
                               encoder_hidden_states=encoder_hidden_states, upsample_size=upsample_size,
                               image_only_indicator=image_only_indicator)
            else:
                sample = block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res,
                               upsample_size=upsample_size, image_only_indicator=image_only_indicator)

        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
        if not return_dict:
            return (sample,)
        return UNetSpatioTemporalConditionOutput(sample=sample)
