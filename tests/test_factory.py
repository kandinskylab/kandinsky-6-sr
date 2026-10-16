"""Factory contract: config gating and offload injection.

What & why: ``load_sr_pipeline`` is the embedding entry point. It must stay
inert when SR is disabled, and it must register the loaded modules on the
caller-provided offload handle under the names the stages use — that is the
whole dependency-inversion contract with the K6 pipeline. How: the component
loader and the pipeline class are monkeypatched, no weights or CUDA involved.
Corner cases: disabled + force, missing checkpoint path, no latent upscaler.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import kandinsky_sr.pipeline.components as components_module
import kandinsky_sr.pipeline.warmup as warmup_module
from kandinsky_sr.pipeline import factory
from kandinsky_sr.pipeline.config import SRConfig

MIN_OVERLAP = 0.3


class RecordingOffload:
    """Minimal ``OffloadHandle`` that records registrations."""

    strategy = "module"

    def __init__(self) -> None:
        self.registered: dict[str, object] = {}

    def register(self, name: str, module: object) -> None:
        self.registered[name] = module

    @contextmanager
    def use(self, *names: str, prefetch=None):
        yield


def _patch_loaders(monkeypatch: pytest.MonkeyPatch, *, latent_upscaler: object | None) -> dict[str, object]:
    seen: dict[str, object] = {}
    components = SimpleNamespace(
        dit=object(),
        vae=SimpleNamespace(decode_segment_frames=0),
        latent_upscaler=latent_upscaler,
        sr_params=SimpleNamespace(visual_size=[512], scale_factor={512: [1.0, 1.0, 1.0]}),
        cached_text_embeds=None,
    )

    def fake_load_components(**kwargs):
        seen["components_kwargs"] = kwargs
        return components

    def fake_pipeline(**kwargs):
        seen["pipeline_kwargs"] = kwargs
        return "pipeline"

    def fake_compile(loaded, device):
        seen.setdefault("compiled", []).append((loaded, device))

    # The factory binds both names at import time, so patch them where they are used.
    monkeypatch.setattr(factory, "load_sr_components", fake_load_components)
    monkeypatch.setattr(factory, "Kandinsky6SRPipeline", fake_pipeline)
    monkeypatch.setattr(warmup_module, "compile_vae_decode", fake_compile)
    monkeypatch.setattr(
        warmup_module,
        "warmup",
        lambda dit, scale_factor, device: seen.setdefault("warmed", (dit, scale_factor, device)),
    )
    seen["components"] = components
    return seen


def test_disabled_config_returns_none_unless_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_loaders(monkeypatch, latent_upscaler=object())
    config = SRConfig(checkpoint_path="/dit", latent_upscaler_config="/lu.yaml")

    assert factory.load_sr_pipeline(config, "cpu") is None
    assert factory.load_sr_pipeline(config, "cpu", force=True) == "pipeline"
    assert seen["components_kwargs"]["checkpoint_path"] == "/dit"


def test_missing_checkpoint_raises() -> None:
    with pytest.raises(ValueError, match="checkpoint_path is empty"):
        factory.load_sr_pipeline(SRConfig(enabled=True, latent_upscaler_config="/lu.yaml"), "cpu")


def test_modules_are_registered_on_injected_offload(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_loaders(monkeypatch, latent_upscaler=object())
    offload = RecordingOffload()
    config = SRConfig(enabled=True, checkpoint_path="/dit", latent_upscaler_config="/lu.yaml")

    factory.load_sr_pipeline(config, "cpu", offload=offload)

    components = seen["components"]
    assert offload.registered == {
        "dit": components.dit,
        "vae": components.vae,
        "latent_upscaler": components.latent_upscaler,
    }
    assert seen["pipeline_kwargs"]["offload"] is offload


def test_without_latent_upscaler_only_dit_and_vae_are_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_loaders(monkeypatch, latent_upscaler=None)
    offload = RecordingOffload()
    config = SRConfig(enabled=True, checkpoint_path="/dit", latent_upscaler_config="none")

    factory.load_sr_pipeline(config, "cpu", offload=offload)

    assert set(offload.registered) == {"dit", "vae"}


def test_factory_only_builds_no_warmup_or_precompile(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_loaders(monkeypatch, latent_upscaler=None)
    config = SRConfig(enabled=True, checkpoint_path="/dit", latent_upscaler_config="none")

    factory.load_sr_pipeline(config, "cuda:0")

    assert "compiled" not in seen
    assert "warmed" not in seen


def test_overlap_reaches_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_loaders(monkeypatch, latent_upscaler=None)
    config = SRConfig(enabled=True, checkpoint_path="/dit", latent_upscaler_config="none", overlap=MIN_OVERLAP)

    factory.load_sr_pipeline(config, "cuda:0")

    assert seen["pipeline_kwargs"]["overlap"] == MIN_OVERLAP


def test_bare_cuda_device_resolves_to_the_current_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """``kandy generate`` passes ``"cuda"``; torch.cuda.set_device needs an index."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    assert components_module.resolve_cuda_device("cuda") == torch.device("cuda", 1)
    assert components_module.resolve_cuda_device("cuda:3") == torch.device("cuda", 3)
    assert components_module.resolve_cuda_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("strategy", ["module", "block"])
def test_offload_loads_all_components_and_cold_lu_on_cpu(monkeypatch, strategy):
    seen = _patch_loaders(monkeypatch, latent_upscaler=object())
    selected_devices = []
    monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)
    handle = RecordingOffload()
    handle.strategy = strategy
    config = SRConfig(enabled=True, checkpoint_path="/bundle", lu_load_scales=("2x",),
                      decode_segment_frames=4, decode_spatial_splits=2)
    factory.load_sr_pipeline(config, "cuda:3", offload=handle)
    assert seen["components_kwargs"]["device"] == "cpu"
    assert seen["components_kwargs"]["lu_load_scales"] == ("2x",)
    assert selected_devices == [torch.device("cuda:3")]
    assert seen["pipeline_kwargs"]["device"] == "cuda:3"
    assert seen["pipeline_kwargs"]["decode_spatial_splits"] == config.decode_spatial_splits
    assert seen["components"].vae.decode_segment_frames == config.decode_segment_frames


def test_resident_load_keeps_requested_device(monkeypatch):
    seen = _patch_loaders(monkeypatch, latent_upscaler=None)
    factory.load_sr_pipeline(SRConfig(enabled=True, checkpoint_path="/bundle"), "cuda:2")
    assert seen["components_kwargs"]["device"] == "cuda:2"


def test_block_offload_rejects_magi_before_loading(monkeypatch):
    seen = _patch_loaders(monkeypatch, latent_upscaler=None)
    handle = RecordingOffload()
    handle.strategy = "block"
    with pytest.raises(ValueError, match="Magi graphs"):
        factory.load_sr_pipeline(SRConfig(enabled=True, checkpoint_path="/bundle", vae_backend="magi"),
                                 "cuda:0", offload=handle)
    assert "components_kwargs" not in seen
