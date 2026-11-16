"""Pipeline-facing wrapper around the tiled KVAE.

The SR pipeline calls ``encode`` for a ``(latent, split_list)`` tuple and
``decode`` for an object with ``.sample``. ``SplitVAE`` returns a bare tensor
from both, so this module adapts that contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from kandinsky_sr.core.components.split_vae.split_kvae import SplitVAE, build_split_kvae
from kandinsky_sr.core.components.video_kvae.cached_model import DecoderOutput


class SplitSRVAE(nn.Module):
    """``SplitVAE`` with the encode/decode contract the SR pipeline calls."""

    def __init__(self, vae: SplitVAE) -> None:
        super().__init__()
        self.vae = vae
        self.config = vae.config
        self.decode_segment_frames = 0
        self.decode_spatial_splits = 1
        self.decode_spatial_overlap = 4

    @staticmethod
    def normalize_data(data: torch.Tensor) -> torch.Tensor:
        return SplitVAE.normalize_data(data)

    @staticmethod
    def denormalize_data(data: torch.Tensor) -> torch.Tensor:
        return SplitVAE.denormalize_data(data)

    def encode(self, x: torch.Tensor, seg_len: int = 16) -> tuple[torch.Tensor, None]:
        del seg_len
        return self.vae.encode(x), None

    def decode(self, z: torch.Tensor, split_list: list[int] | None = None) -> DecoderOutput:
        del split_list
        with torch._dynamo.utils.disable_cache_limit():
            sample = self.vae.decode(z)
        return DecoderOutput(sample=sample)


def build_split_sr_vae(vae_conf: Any) -> SplitSRVAE:
    """Build the tiled KVAE from the training config's ``vae`` section."""
    conf = SimpleNamespace(
        checkpoint_path=str(vae_conf.checkpoint_path),
        dtype=getattr(vae_conf, "dtype", "bf16"),
        decoder_backend="torch",
    )
    return SplitSRVAE(build_split_kvae(conf))
