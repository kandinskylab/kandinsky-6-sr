"""Regressions for CPU loading and non-overlapping SR residency; no GPU needed."""
# ruff: noqa: PLR2004 -- literal expected shapes/strip counts in regression tests

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from kandinsky_sr.core.algo.tiling_utils import blend_spatial_decode, spatial_decode_spans
from kandinsky_sr.pipeline import stages, warmup
from kandinsky_sr.pipeline.sr_pipeline import RunConfig


class StrictOffload:
    strategy = "block"

    def __init__(self):
        self.active = ()
        self.events = []

    @contextmanager
    def use(self, *names):
        assert not self.active, f"Nested residency: {self.active} -> {names}"
        self.active = names
        self.events.append(names)
        try:
            yield
        finally:
            self.active = ()


@pytest.mark.parametrize("pixels", [False, True])
@pytest.mark.parametrize("fail_decode", [False, True])
def test_native_stages_never_overlap_dit_and_vae(monkeypatch, pixels, fail_decode):
    offload = StrictOffload()
    latent = torch.ones(1, 2, 2, 1)
    state = SimpleNamespace(batch_size=1, duration=1, height=2, width=2)

    def encode(*args):
        assert offload.active == ("vae",)
        return latent

    def prepare(**kwargs):
        assert offload.active == ("dit",)
        assert kwargs["lq_latents"] is latent
        assert kwargs["n_samples"] == 1
        return state

    def text(**kwargs):
        assert offload.active == ("dit",)
        return object()

    def denoise(**kwargs):
        assert offload.active == ("dit",)
        return latent

    def decode(**kwargs):
        assert offload.active == ("vae",)
        assert kwargs["spatial_splits"] == 2
        if fail_decode:
            raise RuntimeError("decode failed")
        return latent

    monkeypatch.setattr(stages, "_encode_lq_videos", encode)
    monkeypatch.setattr(stages, "prepare_latents", prepare)
    monkeypatch.setattr(stages, "text_encode", text)
    monkeypatch.setattr(stages, "denoise", denoise)
    monkeypatch.setattr(stages, "vae_decode", decode)
    inputs = {"lq_videos": [torch.zeros(1)]} if pixels else {"lq_latents": latent, "n_samples": 1}
    kwargs = dict(
        dit=object(), vae=object(), scale_factor=(1, 1, 1), offload=offload, decode_spatial_splits=2, **inputs
    )
    if fail_decode:
        with pytest.raises(RuntimeError, match="decode failed"):
            stages.run_stages(**kwargs)
    else:
        assert stages.run_stages(**kwargs) is latent
    assert offload.active == ()
    assert offload.events == ([("vae",)] if pixels else []) + [("dit",), ("vae",)]


def test_latent_upscaler_finishes_before_native_or_custom_sampler(monkeypatch):
    offload = StrictOffload()
    config = RunConfig(device="cpu", resolution_scale=2, num_steps=3)
    components = SimpleNamespace(
        dit=object(), vae=object(), cached_text_embeds=None, sr_params=SimpleNamespace(scheduler_scale=5)
    )
    monkeypatch.setattr(stages, "_component_params", lambda _: components.sr_params)
    monkeypatch.setattr(stages, "_generate_kwargs", lambda *args, **kwargs: kwargs)

    def prepare(chunk):
        assert offload.active == ("latent_upscaler",)
        return chunk

    def native(**kwargs):
        assert not offload.active
        assert kwargs["offload"] is offload
        assert kwargs["decode_spatial_splits"] == 2
        return torch.zeros(1, 3, 1, 2, 2)

    def custom(**kwargs):
        assert offload.active == ("dit", "vae")
        return torch.zeros(1, 3, 1, 2, 2)

    config.decode_spatial_splits = 2
    monkeypatch.setattr(stages, "run_stages", native)
    for sampler in [None, custom]:
        components.sampler = sampler
        result = stages._run_tile_batches(
            [torch.zeros(1, 2, 2, 1)], components, config, (1, 1, 1), stage=None, offload=offload, prepare_chunk=prepare
        )
        assert len(result) == 1
        assert not offload.active


@pytest.mark.parametrize("width,splits,overlap", [(48, 2, 4), (47, 2, 4), (48, 3, 0), (48, 3, 4)])
def test_decode_strips_have_no_gaps_or_dark_seams(width, splits, overlap):
    spans = spatial_decode_spans(width, splits, overlap)
    parts = [torch.full((1, 3, 1, 8, (end - start) * 16), 128, dtype=torch.uint8) for start, end in spans]
    decoded = blend_spatial_decode(parts, spans, width, overlap)
    assert decoded.shape[-1] == width * 16
    assert torch.all(decoded == 128)


def test_compiled_vae_warmup_only_decodes_strip_widths(monkeypatch):
    widths = []
    vae = SimpleNamespace(
        parameters=lambda: iter(()),
        decode_spatial_splits=2,
        decode_spatial_overlap=4,
        decode=lambda z: widths.append(z.shape[-1]),
    )
    components = SimpleNamespace(
        dit=SimpleNamespace(in_visual_dim=1), vae=vae, sr_params=SimpleNamespace(visual_size=[512])
    )
    monkeypatch.setattr(warmup.constants, "VAE_SPATIAL_FACTOR", 16)
    warmup.compile_vae_decode(components, "cpu", bases=[(512, 768)])
    assert widths == [28]
