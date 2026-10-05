"""Load-time KVAE decode precompile scope.

What & why: ``kandy-sr`` compiles the decode ahead of the run; compiling every
trained base pins decoder weights per graph (magi) and can exhaust the GPU,
so the default is the single base the clip tiles into. How: fake components
whose ``vae.decode`` records the latent shapes it was called with; no CUDA.
Corner cases: landscape / portrait / square sources, explicit ``bases`` list,
``bases=None`` meaning every trained base.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kandinsky_sr import constants
from kandinsky_sr.pipeline.warmup import clip_base_resolution, compile_vae_decode


class RecordingVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.shapes: list[tuple[int, ...]] = []

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(z.shape))
        return z


def _components() -> SimpleNamespace:
    return SimpleNamespace(
        vae=RecordingVAE(),
        dit=SimpleNamespace(in_visual_dim=64),
        sr_params=SimpleNamespace(visual_size=[512], scale_factor={512: [1.0, 1.0, 1.0]}),
    )


@pytest.mark.parametrize(
    ("source_hw", "expected"),
    [((512, 768), (512, 768)), ((768, 512), (768, 512)), ((500, 500), (512, 512)), ((576, 864), (512, 768))],
)
def test_clip_base_follows_the_aspect_ratio(source_hw: tuple[int, int], expected: tuple[int, int]) -> None:
    assert clip_base_resolution(_components(), source_hw) == expected


def test_compile_only_the_requested_base() -> None:
    components = _components()
    factor = constants.VAE_SPATIAL_FACTOR
    t_latent = (constants.MAX_NUM_FRAMES - 1) // constants.VAE_TEMPORAL_FACTOR + 1

    compile_vae_decode(components, "cpu", bases=[(512, 768)])

    assert components.vae.shapes == [(1, 64, t_latent, 512 // factor, 768 // factor)]


def test_compile_every_trained_base_by_default() -> None:
    components = _components()

    compile_vae_decode(components, "cpu")

    assert len(components.vae.shapes) == len(constants.RESOLUTIONS[512])
