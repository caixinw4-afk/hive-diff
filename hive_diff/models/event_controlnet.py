"""Event-conditioned ControlNet for Stable Video Diffusion (the "Generator" stage).

Built from the diffusers SVD blocks: the network is a copy of the SVD UNet's down path + mid block (initialised from the UNet weights), with

* an event encoder that maps a per-frame event tensor ``(B, F, C_ev, 8h, 8w)`` to a
  ``(B*F, C0, h, w)`` feature map that is added to the ``conv_in`` output, and
* zero-initialised 1x1 "head" convolutions producing one residual per UNet skip tensor
  plus one for the mid block.

Besides the head outputs, the forward pass also exposes the raw (pre-head) feature pyramid and
the event embedding, because the second-stage :class:`ShaderControlNet` consumes them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_3d_blocks import UNetMidBlockSpatioTemporal, get_down_block
from diffusers.utils import BaseOutput


def zero_conv(channels: int) -> nn.Conv2d:
    conv = nn.Conv2d(channels, channels, kernel_size=1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


def _as_tuple(value, n: int) -> tuple:
    return tuple(value) if isinstance(value, (list, tuple)) else (value,) * n


class EventConditioningEncoder(nn.Module):
    """Small conv stack: pixel-resolution events -> latent-resolution feature map (stride 8).

    Layout: ``conv_in`` (3x3) then six 3x3 convs where every second one has stride 2, then a
    zero-initialised ``conv_out``. SiLU follows every conv except ``conv_out``.
    """

    def __init__(self, out_channels: int, event_channels: int = 4,
                 hidden_channels: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.conv_in = nn.Conv2d(event_channels, hidden_channels[0], kernel_size=3, padding=1)
        blocks = []
        for c_in, c_out in zip(hidden_channels[:-1], hidden_channels[1:]):
            blocks.append(nn.Conv2d(c_in, c_in, kernel_size=3, padding=1))
            blocks.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2))
        self.blocks = nn.ModuleList(blocks)
        self.conv_out = nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        if events.dim() == 5:  # (B, F, C, H, W) -> (B*F, C, H, W)
            events = events.flatten(0, 1)
        x = F.silu(self.conv_in(events))
        for block in self.blocks:
            x = F.silu(block(x))
        return self.conv_out(x)


@dataclass
class EventControlNetOutput(BaseOutput):
    down_block_res_samples: Tuple[torch.Tensor, ...]
    mid_block_res_sample: torch.Tensor
    raw_down_features: Optional[Tuple[torch.Tensor, ...]] = None
    raw_mid_feature: Optional[torch.Tensor] = None
    event_embedding: Optional[torch.Tensor] = None


class SVDDownPathMixin:
    """Shared construction/forward helpers for the two ControlNets (down path of the SVD UNet)."""

    def _build_backbone(self, in_channels, down_block_types, block_out_channels, addition_time_embed_dim,
                        projection_class_embeddings_input_dim, layers_per_block, cross_attention_dim,
                        transformer_layers_per_block, num_attention_heads):
        n = len(down_block_types)
        layers_per_block = _as_tuple(layers_per_block, n)
        cross_attention_dim = _as_tuple(cross_attention_dim, n)
        transformer_layers_per_block = _as_tuple(transformer_layers_per_block, n)
        num_attention_heads = _as_tuple(num_attention_heads, n)
        c0 = block_out_channels[0]
        time_embed_dim = c0 * 4

        self.conv_in = nn.Conv2d(in_channels, c0, kernel_size=3, padding=1)
        self.time_proj = Timesteps(c0, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(c0, time_embed_dim)
        self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList()
        heads = [zero_conv(c0)]  # one head for the conv_in output
        c_prev = c0
        for i, block_type in enumerate(down_block_types):
            c_out = block_out_channels[i]
            is_final = i == n - 1
            self.down_blocks.append(
                get_down_block(
                    block_type,
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=c_prev,
                    out_channels=c_out,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final,
                    resnet_eps=1e-5,
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                    resnet_act_fn="silu",
                )
            )
            heads.extend(zero_conv(c_out) for _ in range(layers_per_block[i]))
            if not is_final:
                heads.append(zero_conv(c_out))
            c_prev = c_out
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
        )
        return nn.ModuleList(heads), zero_conv(block_out_channels[-1])

    def _embed_time(self, sample, timestep, added_time_ids, num_frames):
        batch_size = sample.shape[0]
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            dtype = torch.float64 if isinstance(timestep, float) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif timesteps.dim() == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(batch_size)
        emb = self.time_embedding(self.time_proj(timesteps).to(sample.dtype))
        aug = self.add_time_proj(added_time_ids.flatten()).reshape(batch_size, -1).to(emb.dtype)
        emb = emb + self.add_embedding(aug)
        return emb.repeat_interleave(num_frames, dim=0)

    def _copy_backbone_from_unet(self, unet):
        for name in ("conv_in", "time_proj", "time_embedding", "add_time_proj", "add_embedding",
                     "down_blocks", "mid_block"):
            getattr(self, name).load_state_dict(getattr(unet, name).state_dict())

    @staticmethod
    def _unet_backbone_kwargs(unet) -> dict:
        cfg = unet.config
        return dict(
            sample_size=cfg.sample_size,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            down_block_types=cfg.down_block_types,
            block_out_channels=cfg.block_out_channels,
            addition_time_embed_dim=cfg.addition_time_embed_dim,
            projection_class_embeddings_input_dim=cfg.projection_class_embeddings_input_dim,
            layers_per_block=cfg.layers_per_block,
            cross_attention_dim=cfg.cross_attention_dim,
            transformer_layers_per_block=cfg.transformer_layers_per_block,
            num_attention_heads=cfg.num_attention_heads,
            num_frames=cfg.num_frames,
        )

    def _set_gradient_checkpointing(self, *args, **kwargs):
        # diffusers < 0.33 calls (module, value=...) per module; newer versions call
        # (enable=..., gradient_checkpointing_func=...) once on the model.
        if args and isinstance(args[0], nn.Module):
            module = args[0]
            value = kwargs.get("value", args[1] if len(args) > 1 else False)
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
            return
        enable = kwargs.get("enable", True)
        func = kwargs.get("gradient_checkpointing_func")
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable
                if func is not None:
                    module._gradient_checkpointing_func = func


class EventControlNet(ModelMixin, ConfigMixin, SVDDownPathMixin):
    """Stage-1 ControlNet: events -> residuals for the frozen SVD UNet.

    ``in_channels`` is 8: the 4 noisy latent channels concatenated with the 4 reference-image
    latent channels, exactly like the SVD UNet input. ``event_channels`` is the number of event
    bins per frame (default 4). ``conditioning_channels`` and ``up_block_types`` are accepted so that
    config files written with the full set of SVD UNet keys load, and are otherwise unused.
    """

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
        up_block_types: Optional[Tuple[str, ...]] = None,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int, ...]] = 2,
        cross_attention_dim: Union[int, Tuple[int, ...]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        num_attention_heads: Union[int, Tuple[int, ...]] = (5, 10, 20, 20),
        num_frames: int = 14,
        event_channels: int = 4,
        conditioning_channels: int = 32,
        conditioning_embedding_out_channels: Tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.controlnet_down_blocks, self.controlnet_mid_block = self._build_backbone(
            in_channels, down_block_types, block_out_channels, addition_time_embed_dim,
            projection_class_embeddings_input_dim, layers_per_block, cross_attention_dim,
            transformer_layers_per_block, num_attention_heads,
        )
        self.controlnet_cond_embedding = EventConditioningEncoder(
            block_out_channels[0], event_channels, tuple(conditioning_embedding_out_channels)
        )

    @classmethod
    def from_unet(cls, unet, event_channels: int = 4,
                  conditioning_embedding_out_channels=(32, 64, 128, 256), load_weights: bool = True):
        model = cls(event_channels=event_channels,
                    conditioning_embedding_out_channels=tuple(conditioning_embedding_out_channels),
                    **cls._unet_backbone_kwargs(unet))
        if load_weights:
            model._copy_backbone_from_unet(unet)
        return model

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        events: Optional[torch.Tensor] = None,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
    ) -> Union[EventControlNetOutput, tuple]:
        """``sample``: (B, F, 8, h, w) latents; ``events``: (B, F, C_ev, 8h, 8w) in [-1, 1]."""
        batch_size, num_frames = sample.shape[:2]
        emb = self._embed_time(sample, timestep, added_time_ids, num_frames)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        x = self.conv_in(sample.flatten(0, 1))
        event_embedding = None
        if events is not None:
            event_embedding = self.controlnet_cond_embedding(events)
            x = x + event_embedding
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=x.dtype, device=x.device)

        raw = [x]
        for block in self.down_blocks:
            if getattr(block, "has_cross_attention", False):
                x, res = block(hidden_states=x, temb=emb, encoder_hidden_states=encoder_hidden_states,
                               image_only_indicator=image_only_indicator)
            else:
                x, res = block(hidden_states=x, temb=emb, image_only_indicator=image_only_indicator)
            raw.extend(res)
        raw_mid = self.mid_block(hidden_states=x, temb=emb, encoder_hidden_states=encoder_hidden_states,
                                 image_only_indicator=image_only_indicator)

        down = tuple(head(r) * conditioning_scale for head, r in zip(self.controlnet_down_blocks, raw))
        mid = self.controlnet_mid_block(raw_mid) * conditioning_scale
        if not return_dict:
            return down, mid, tuple(raw), raw_mid, event_embedding
        return EventControlNetOutput(down, mid, tuple(raw), raw_mid, event_embedding)
