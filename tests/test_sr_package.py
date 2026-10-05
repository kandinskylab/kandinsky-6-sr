"""Tests for the packaged super-resolution route."""

import inspect
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kandinsky_sr import cli as sr_cli
from kandinsky_sr.core.components.model import compiled_kvae
from kandinsky_sr.core.components.video_kvae.cached_layers import CachedCausalResnetBlock3D
from kandinsky_sr.pipeline import components as components_module
from kandinsky_sr.pipeline import sr_pipeline as sr_module
from kandinsky_sr.pipeline import stages
from kandinsky_sr.pipeline.sr_pipeline import Kandinsky6SRPipeline, resolve_scale_request

BATCH_SIZE = 2


def test_sr_rejects_hunyuan_vae_configuration():
    with pytest.raises(ValueError, match="only 'video-kvae'"):
        components_module.build_vae_for_backend(SimpleNamespace(), "hunyuan", "torch")


def test_fractional_scale_uses_pixel_pre_upscale_and_x2_tiling():
    assert resolve_scale_request(2.25) == (2, 1.125)


SR_YAML = """
sr:
  enabled: false
  checkpoint_path: /models/dit
  vae_path: /models/kvae/kvae_20_v2
  latent_upscaler_config: /models/lu/bank.yaml
  resolution_scale: 2.25
  num_steps: 5
"""


def test_native_sr_pipeline_uses_seed_for_runtime_randomness():
    signature = inspect.signature(Kandinsky6SRPipeline.__call__)

    assert "seed" in signature.parameters
    assert "generator" not in signature.parameters
    assert not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    assert {
        "video",
        "latents",
        "resolution_scale",
        "num_steps",
        "seed",
        "overlap",
        "tiles_batch_size",
        "kvae_bridge",
        "cached_text_embeds",
        "save_path",
        "fps",
        "audio",
        "source_video",
        "audio_sample_rate",
        "show_progress",
        "progress",
    }.issubset(signature.parameters)


def _write_sr_yaml(tmp_path: Path) -> Path:
    config_path = tmp_path / "sr.yaml"
    config_path.write_text(SR_YAML)
    return config_path


def test_sr_cli_reads_config_settings_without_gating_on_enabled(tmp_path):
    config_path = _write_sr_yaml(tmp_path)
    options = sr_cli.CommonOptions(config_path=config_path, output_dir=tmp_path)

    config = sr_cli._prepare_common(options)

    assert config.enabled is False
    assert options.checkpoint_path is None
    assert options.vae_path is None
    assert options.latent_upscaler_config is None
    assert sr_cli._resolve_resolution_scale(None, config) == config.resolution_scale


def test_sr_cli_explicit_options_override_config(tmp_path):
    config_path = _write_sr_yaml(tmp_path)
    override_num_steps = 7
    options = sr_cli.CommonOptions(
        config_path=config_path,
        checkpoint_path="/override/checkpoint",
        num_steps=override_num_steps,
        output_dir=tmp_path,
    )

    config = sr_cli._prepare_common(options)

    assert config.checkpoint_path == "/override/checkpoint"
    assert config.num_steps == override_num_steps
    assert config.vae_path == "/models/kvae/kvae_20_v2"


def test_dynamic_kvae_regions_are_the_default_compile_boundary(monkeypatch):
    monkeypatch.setenv("KVAE_COMPILE_DYNAMIC", "1")
    monkeypatch.delenv("KVAE_COMPILE_DYNAMIC_MODE", raising=False)
    block = CachedCausalResnetBlock3D(
        in_channels=32,
        out_channels=32,
        dropout=0.0,
        temb_channels=0,
    )
    compiled_owners = []

    def fake_compile(fn, **kwargs):
        compiled_owners.append((fn.__self__, kwargs))
        return fn

    monkeypatch.setattr(compiled_kvae.torch, "compile", fake_compile)

    expected_compiled_kernels = 3  # one resblock + two norms
    assert compiled_kvae.compile_kvae_leaves(block) == expected_compiled_kernels
    assert block in [owner for owner, _ in compiled_owners]
    assert block.conv1 not in [owner for owner, _ in compiled_owners]
    assert block.conv2 not in [owner for owner, _ in compiled_owners]
    assert block.conv1.conv not in [owner for owner, _ in compiled_owners]
    assert block.conv2.conv not in [owner for owner, _ in compiled_owners]
    assert block.norm1 in [owner for owner, _ in compiled_owners]
    assert block.norm2 in [owner for owner, _ in compiled_owners]
    assert all(kwargs["dynamic"] is True for _, kwargs in compiled_owners)


def test_dynamic_inner_conv_mode_is_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("KVAE_COMPILE_DYNAMIC", "1")
    monkeypatch.setenv("KVAE_COMPILE_DYNAMIC_MODE", "inner_conv")
    block = CachedCausalResnetBlock3D(
        in_channels=32,
        out_channels=32,
        dropout=0.0,
        temb_channels=0,
    )
    compiled_owners = []

    def fake_compile(fn, **kwargs):
        compiled_owners.append((fn.__self__, kwargs))
        return fn

    monkeypatch.setattr(compiled_kvae.torch, "compile", fake_compile)

    expected_compiled_kernels = 4  # two convolutions + two norms
    assert compiled_kvae.compile_kvae_leaves(block) == expected_compiled_kernels
    assert block not in [owner for owner, _ in compiled_owners]
    assert block.conv1.conv in [owner for owner, _ in compiled_owners]
    assert block.conv2.conv in [owner for owner, _ in compiled_owners]


def test_sr_stage_orchestrator_calls_reusable_stages_in_order(monkeypatch):
    latent_state = stages.SRLatentState(
        lq_latent=torch.zeros(1, 1, 1, 1),
        image=torch.zeros(1, 1, 1, 1),
        batch_size=1,
        duration=1,
        height=1,
        width=1,
    )
    text_state = stages.SRTextState(
        text_embeds={},
        text_cu_seqlens=torch.zeros(2, dtype=torch.int32),
        null_text_embeds={},
        null_text_cu_seqlens=torch.zeros(2, dtype=torch.int32),
    )
    calls: list[str] = []
    stage_names: list[str] = []

    monkeypatch.setattr(stages, "prepare_latents", lambda **_kwargs: calls.append("prepare") or latent_state)
    monkeypatch.setattr(stages, "text_encode", lambda **_kwargs: calls.append("text") or text_state)
    monkeypatch.setattr(stages, "denoise", lambda **_kwargs: calls.append("denoise") or torch.zeros(1, 1, 1, 1))
    monkeypatch.setattr(stages, "vae_decode", lambda **_kwargs: calls.append("decode") or torch.zeros(1, 3, 1, 1, 1))

    result = stages.run_stages(
        dit=SimpleNamespace(),
        vae=SimpleNamespace(),
        scale_factor=(1.0, 1.0, 1.0),
        lq_latents=torch.zeros(1, 1, 1, 1),
        n_samples=1,
        device="cpu",
        stage=lambda name: stage_names.append(name) or nullcontext(),
    )

    assert calls == ["prepare", "text", "denoise", "decode"]
    assert stage_names == ["prepare_latents", "text_encode", "denoise", "vae_decode"]
    assert result.shape == (1, 3, 1, 1, 1)


def test_sr_tile_batches_keep_dit_and_vae_resident(monkeypatch):
    class RecordingOffload:
        def __init__(self):
            self.events = []

        @contextmanager
        def use(self, *names, **_kwargs):
            self.events.append(("enter", names))
            try:
                yield
            finally:
                self.events.append(("exit", names))

    offload = RecordingOffload()
    calls = []
    seeds = []

    def run_stages(**kwargs):
        calls.append(kwargs["lq_videos"])
        seeds.append(kwargs["seed"])
        return torch.zeros(len(kwargs["lq_videos"]), 1, 1, 1, 1)

    monkeypatch.setattr(stages, "run_stages", run_stages)
    components = SimpleNamespace(
        dit=SimpleNamespace(),
        vae=SimpleNamespace(),
        sr_params=SimpleNamespace(),
        sampler=None,
    )
    run_config = SimpleNamespace(tiles_batch_size=2, num_steps=5, seed=42, device="cpu")

    outputs = stages._run_tile_batches(
        [
            torch.zeros(1, 1, 1, 1),
            torch.ones(1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1),
            torch.ones(1, 1, 1, 1),
        ],
        components,
        run_config,
        (1.0,),
        stage=None,
        offload=offload,
        sample_batch_size=2,
    )

    tile_count = 2  # four tile-major inputs = two tiles x BATCH_SIZE samples, one chunk per tile
    assert len(outputs) == BATCH_SIZE * tile_count
    assert len(calls) == tile_count
    assert seeds == [42, 43]
    assert offload.events == [("enter", ("dit", "vae")), ("exit", ("dit", "vae"))]


def test_kvae_bridge_reencodes_source_latents_before_lu(monkeypatch):
    # One x2 tile (256x384) of the 512 base: the latent path pads smaller sources.
    source_video = torch.zeros(2, 3, 256, 384, dtype=torch.uint8)
    source_latents = torch.zeros(2, 16, 16, 24)
    sr_latents = torch.zeros(2, 64, 16, 24)
    captured: dict[str, torch.Tensor] = {}

    def encode(video, _vae, _device):
        captured["source_video"] = video
        return sr_latents

    def tiled(latents, _components, _config, **_kwargs):
        captured["sr_latents"] = latents
        return torch.zeros(3, 2, 512, 768, dtype=torch.uint8)

    monkeypatch.setattr(
        sr_module,
        "_decode_source_latent_video",
        lambda _latents, _source_vae, _device: source_video,
    )
    monkeypatch.setattr(sr_module, "encode_lq_video_to_lr_latent", encode)
    monkeypatch.setattr(sr_module, "latent_upscaler_for_scale", lambda _bank, _scale: object())
    monkeypatch.setattr(stages, "run_tiled_sr", tiled)

    pipeline = Kandinsky6SRPipeline(
        dit=object(),
        vae=object(),
        latent_upscaler=object(),
        device=torch.device("cpu"),
        sr_params=SimpleNamespace(scale_factor={512: [1.0, 1.0, 1.0]}, visual_size=[512]),
        num_steps=5,
        resolution_scale=2,
        overlap=0.25,
        tiles_batch_size=1,
        source_vae=object(),
        kvae_bridge=True,
    )

    result = pipeline(latents=source_latents)

    assert captured["source_video"] is source_video
    assert captured["sr_latents"] is sr_latents
    assert result.frames.shape == (1, 3, 2, 512, 768)


def test_native_sr_batches_samples_in_one_tile_batch(monkeypatch):
    grid = SimpleNamespace(total_tiles=2, tile_h=2, tile_w=2, tops=(0,), lefts=(0, 2))
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        sr_module,
        "_tile_geometry",
        lambda *_args, **_kwargs: ((2, 4), (2, 2), grid),
    )

    def extract(sample, _grid):
        marker = int(sample.flatten()[0])
        return [torch.full((sample.shape[0], sample.shape[1], 2, 2), marker + tile) for tile in range(2)]

    def run_tile_batches(tile_inputs, _components, run_config, *_args, **kwargs):
        seen["tile_inputs"] = [int(tile.flatten()[0]) for tile in tile_inputs]
        seen["tiles_batch_size"] = run_config.tiles_batch_size
        return [tile.permute(3, 0, 1, 2).float() for tile in tile_inputs]

    monkeypatch.setattr(sr_module, "extract_all_tiles", extract)
    monkeypatch.setattr(
        sr_module, "_upsample_tiles_to_base", lambda tiles, _h, _w: [tile.permute(0, 2, 3, 1) for tile in tiles]
    )
    monkeypatch.setattr(stages, "_run_tile_batches", run_tile_batches)
    monkeypatch.setattr(sr_module, "stitch_tiles_hanning", lambda tiles, *_args, **_kwargs: tiles[0])

    pipeline = Kandinsky6SRPipeline(
        dit=SimpleNamespace(in_visual_dim=1),
        vae=object(),
        latent_upscaler=None,
        device="cpu",
        sr_params=SimpleNamespace(scale_factor={512: [1.0, 1.0, 1.0]}, visual_size=[512]),
        tiles_batch_size=1,
    )
    result = pipeline(
        video=torch.stack(
            [
                torch.full((5, 1, 2, 4), 10, dtype=torch.uint8),
                torch.full((5, 1, 2, 4), 20, dtype=torch.uint8),
            ]
        ),
        resolution_scale=2,
    )

    assert seen["tile_inputs"] == [10, 20, 11, 21]
    assert seen["tiles_batch_size"] == BATCH_SIZE
    assert result.frames.shape == (BATCH_SIZE, 1, 5, 2, 2)
    assert int(result.frames[0, 0, 0, 0, 0]) == 10  # noqa: PLR2004
    assert int(result.frames[1, 0, 0, 0, 0]) == 20  # noqa: PLR2004


def test_native_sr_batches_latent_samples_in_one_tile_batch(monkeypatch):
    pixel_grid = SimpleNamespace(total_tiles=2, tile_h=4, tile_w=4, tops=(0,), lefts=(0, 4))
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        sr_module,
        "_tile_geometry",
        lambda *_args, **_kwargs: ((8, 16), (4, 4), pixel_grid),
    )

    def extract(sample, _grid):
        marker = int(sample.flatten()[0])
        return [torch.full((sample.shape[0], sample.shape[1], 2, 2), marker + tile) for tile in range(2)]

    def upscale(tile, *_args, **_kwargs):
        return tile.permute(0, 2, 3, 1)

    def run_tile_batches(tile_inputs, _components, run_config, *_args, **kwargs):
        seen["tile_inputs"] = [int(tile.flatten()[0]) for tile in tile_inputs]
        seen["tiles_batch_size"] = run_config.tiles_batch_size
        prepared = kwargs["prepare_chunk"](tile_inputs)
        return [tile.permute(3, 0, 1, 2).float() for tile in prepared]

    monkeypatch.setattr(sr_module, "extract_all_tiles", extract)
    monkeypatch.setattr(sr_module, "upscale_lr_latent_tile", upscale)
    monkeypatch.setattr(sr_module, "latent_upscaler_for_scale", lambda _bank, _scale: object())
    monkeypatch.setattr(stages, "_run_tile_batches", run_tile_batches)

    def stitch(tiles, grid, *_args, **_kwargs):
        seen["stitch_grid"] = grid
        return tiles[0]

    monkeypatch.setattr(sr_module, "stitch_tiles_hanning", stitch)

    pipeline = Kandinsky6SRPipeline(
        dit=SimpleNamespace(in_visual_dim=1),
        vae=object(),
        latent_upscaler=object(),
        device="cpu",
        sr_params=SimpleNamespace(scale_factor={512: [1.0, 1.0, 1.0]}, visual_size=[512]),
        spatial_factor=2,
        tiles_batch_size=1,
    )
    result = pipeline(
        latents=torch.stack(
            [
                torch.full((5, 1, 2, 4), 10.0),
                torch.full((5, 1, 2, 4), 20.0),
            ]
        ),
        resolution_scale=2,
    )

    assert seen["tile_inputs"] == [10, 20, 11, 21]
    assert seen["tiles_batch_size"] == BATCH_SIZE
    assert seen["stitch_grid"] is pixel_grid
    assert result.frames.shape == (BATCH_SIZE, 1, 5, 2, 2)
