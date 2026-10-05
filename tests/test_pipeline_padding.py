"""The pipeline pads any-size sources for the latent-upscaler path and crops the result back.

What & why: callers (the CLI, ComfyUI, the HF demo) hand the pipeline
whatever the source video is; the LU path's constraints (VAE stride, one
tile) are the pipeline's business. The contract: (1) a pixel source is
padded before the whole-clip encode and the stitched SR is cropped to
``source * scale``, (2) a raw latent source is padded in latent units the
same way, (3) the pixel path (no matching LU) is left alone, (4) the
progress reporter counts the tiles of the padded source, (5) the warmup's
throwaway encode sees the padded clip too.

How: ``Kandinsky6SRPipeline`` with fake modules; the encode and the tiled
runs are replaced by recorders that return correctly shaped tensors. No
CUDA. Corner cases: 240-high source at x2 (tile 256x384) and the
non-stride 426 width; an already aligned 256x384 source; a batch of two
non-aligned clips (one shared padding, per-sample crop and tile count).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from kandinsky_sr.pipeline import sr_pipeline as sr_module
from kandinsky_sr.pipeline import stages, warmup
from kandinsky_sr.pipeline.sr_pipeline import Kandinsky6SRPipeline, RunConfig

STRIDE = 16
FRAMES = 2
LATENT_CHANNELS = 64


def make_pipeline(*, latent_upscaler: object | None = object()) -> Kandinsky6SRPipeline:
    return Kandinsky6SRPipeline(
        dit=SimpleNamespace(piflow_params=None, in_visual_dim=LATENT_CHANNELS),
        vae=object(),
        latent_upscaler=latent_upscaler,
        device=torch.device("cpu"),
        sr_params=SimpleNamespace(scale_factor={512: [1.0, 1.0, 1.0]}, visual_size=[512]),
        num_steps=5,
        resolution_scale=2,
        spatial_factor=STRIDE,
    )


def same_storage(seen: torch.Tensor | None, source: torch.Tensor) -> bool:
    """The pipeline normalises inputs into views, so "untouched" means same memory and shape, not same object."""
    return seen is not None and seen.data_ptr() == source.data_ptr() and seen.shape == source.shape


def sr_of(source: torch.Tensor, scale: int) -> torch.Tensor:
    """What a tiled run returns for a ``[T, C, H, W]`` source: ``[C, T, H*scale, W*scale]`` uint8."""
    return torch.zeros(3, source.shape[0], source.shape[-2] * scale, source.shape[-1] * scale, dtype=torch.uint8)


class Recorder:
    def __init__(self) -> None:
        self.encoded: torch.Tensor | None = None
        self.latents: torch.Tensor | None = None
        self.started: tuple[int, int] | None = None

    def encode(self, video: torch.Tensor, _vae: object, _device: object) -> torch.Tensor:
        self.encoded = video
        return torch.zeros(video.shape[0], LATENT_CHANNELS, video.shape[-2] // STRIDE, video.shape[-1] // STRIDE)

    def tiled(self, latents: torch.Tensor, _components: Any, config: RunConfig, **_kwargs: Any) -> torch.Tensor:
        self.latents = latents
        return sr_of(latents, STRIDE * config.resolution_scale)

    def start(self, total_tiles: int, steps_per_tile: int) -> None:
        self.started = (total_tiles, steps_per_tile)

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(sr_module, "encode_lq_video_to_lr_latent", rec.encode)
    monkeypatch.setattr(sr_module, "latent_upscaler_for_scale", lambda bank, _scale: bank)
    monkeypatch.setattr(stages, "run_tiled_sr", rec.tiled)
    return rec


def test_pixel_source_is_padded_for_the_encode_and_the_sr_is_cropped_back(recorder: Recorder) -> None:
    """240x426 at x2: encode sees 256x432 (tile 256x384, stride 16); the SR comes back as 480x852."""
    video = torch.randint(0, 256, (FRAMES, 3, 240, 426), dtype=torch.uint8)

    result = make_pipeline()(video=video, progress=recorder)

    assert recorder.encoded is not None
    assert tuple(recorder.encoded.shape) == (FRAMES, 3, 256, 432)
    assert torch.equal(recorder.encoded[..., :240, :426], video)
    assert tuple(result.frames.shape) == (1, 3, FRAMES, 480, 852)
    expected_tiles = sr_module._tile_geometry(256, 432, 512, 2, 0.2, STRIDE)[2].total_tiles
    assert recorder.started == (expected_tiles, 4)


def test_pixel_batch_is_padded_once_and_every_sample_is_cropped_back(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two 240x426 clips at x2 share one padding: 256x432 batched encode, 480x852 results, tiles counted per sample."""
    batch = 2
    seen: dict[str, Any] = {}

    def encode_batch(videos: list[torch.Tensor], _vae: object, _device: object) -> torch.Tensor:
        seen["encoded"] = videos
        return torch.zeros(len(videos), videos[0].shape[0], LATENT_CHANNELS, 256 // STRIDE, 432 // STRIDE)

    def batched_tiles(samples: list[torch.Tensor], _components: Any, config: RunConfig, **kwargs: Any) -> torch.Tensor:
        seen["latent_input"] = kwargs["latent_input"]
        return torch.stack([sr_of(sample, STRIDE * config.resolution_scale) for sample in samples])

    monkeypatch.setattr(sr_module, "encode_lq_videos_to_lr_latents", encode_batch)
    monkeypatch.setattr(sr_module, "_run_batched_tiles", batched_tiles)
    videos = torch.randint(0, 256, (batch, FRAMES, 3, 240, 426), dtype=torch.uint8)

    result = make_pipeline()(video=videos, progress=recorder)

    assert [tuple(video.shape) for video in seen["encoded"]] == [(FRAMES, 3, 256, 432)] * batch
    assert all(
        torch.equal(padded[..., :240, :426], video) for padded, video in zip(seen["encoded"], videos, strict=True)
    )
    assert seen["latent_input"] is True
    assert tuple(result.frames.shape) == (batch, 3, FRAMES, 480, 852)
    expected_tiles = sr_module._tile_geometry(256, 432, 512, 2, 0.2, STRIDE)[2].total_tiles
    assert recorder.started == (expected_tiles * batch, 4)


def test_aligned_pixel_source_is_passed_through(recorder: Recorder) -> None:
    video = torch.randint(0, 256, (FRAMES, 3, 256, 384), dtype=torch.uint8)

    result = make_pipeline()(video=video)

    assert same_storage(recorder.encoded, video)
    assert tuple(result.frames.shape) == (1, 3, FRAMES, 512, 768)


def test_latent_source_is_padded_in_latent_units(recorder: Recorder) -> None:
    """A 15x27 latent (240x432 px) at x2 is padded to 16x27 and the SR cropped to 480x864."""
    latents = torch.randn(FRAMES, LATENT_CHANNELS, 15, 27)

    result = make_pipeline()(latents=latents)

    assert recorder.latents is not None
    assert tuple(recorder.latents.shape) == (FRAMES, LATENT_CHANNELS, 16, 27)
    assert torch.equal(recorder.latents[..., :15, :], latents)
    assert tuple(result.frames.shape) == (1, 3, FRAMES, 480, 864)


def test_pixel_path_without_latent_upscaler_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, torch.Tensor] = {}

    def from_pixels(video: torch.Tensor, _components: Any, config: RunConfig, **_kwargs: Any) -> torch.Tensor:
        seen["video"] = video
        return sr_of(video, config.resolution_scale)

    monkeypatch.setattr(stages, "run_tiled_sr_from_pixels", from_pixels)
    video = torch.randint(0, 256, (FRAMES, 3, 240, 426), dtype=torch.uint8)

    result = make_pipeline(latent_upscaler=None)(video=video)

    assert same_storage(seen["video"], video)
    assert tuple(result.frames.shape) == (1, 3, FRAMES, 480, 852)


def test_warmup_encodes_the_padded_source(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def encode(video: torch.Tensor, _vae: object, _device: object) -> None:
        seen["encoded"] = video

    def upscale_tile(tile: torch.Tensor, *_args: object) -> torch.Tensor:
        seen["tile"] = tile
        return tile

    monkeypatch.setattr(warmup, "encode_lq_video_to_lr_latent", encode)
    monkeypatch.setattr(warmup, "upscale_lr_latent_tile", upscale_tile)
    monkeypatch.setattr(warmup, "generate_sample_sr", lambda **_kwargs: None)
    monkeypatch.setattr(warmup.torch.cuda, "empty_cache", lambda: None)
    components = SimpleNamespace(
        dit=object(),
        vae=SimpleNamespace(config=SimpleNamespace(latent_channels=64)),
        sr_params=SimpleNamespace(
            visual_size=[512],
            scale_factor={512: [1.0, 1.0, 1.0]},
            scheduler_scale=5.0,
            lq_noise_scale=0.7,
            lq_noise_type="ddpm",
            lq_channel_noise_scale=0.0,
            cap_noise_timestep=False,
        ),
        cached_text_embeds=None,
        spatial_factor=STRIDE,
    )
    run_config = RunConfig(device="cpu", num_steps=5, overlap=0.2, resolution_scale=2)
    video = torch.randint(0, 256, (FRAMES, 3, 240, 426), dtype=torch.uint8)

    warmup.warmup_target_pass(components, run_config, (240, 426), FRAMES, SimpleNamespace(in_channels=64), video)

    assert tuple(seen["encoded"].shape) == (FRAMES, 3, 256, 432)
    assert tuple(seen["tile"].shape[-2:]) == (256 // STRIDE, 384 // STRIDE)
