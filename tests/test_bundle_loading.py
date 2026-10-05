"""The native loaders fed with a Diffusers bundle's components.

What & why: once a reference is resolved to a bundle component, each loader has
to read it as it reads a native checkpoint — the DiT its training config and
weights, the latent-upscaler bank one upscaler at a time out of the shared
weights file, and the factory / CLI must accept a config that names only the
bundle.

How: the ``make_bundle`` fixture; the upscaler architecture is replaced by a
single-parameter module so the real ``load_single_latent_upscaler`` runs on
CPU, and ``load_sr_components`` is stubbed where only argument passing matters.

Corner cases: x2 and x4 weights must not be swapped; lazily loaded scales keep
their prefix; behavioural overrides stored in the bundle apply under explicit
ones; a native checkpoint without an upscaler setting is still refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from kandinsky_sr import cli as sr_cli
from kandinsky_sr.core.algo import latent_upscaler as lu_module
from kandinsky_sr.core.algo.checkpoint import CheckpointConfig
from kandinsky_sr.pipeline import components, factory
from kandinsky_sr.pipeline.components import (
    LazyLatentUpscalerBank,
    load_latent_upscaler_bank,
    load_training_config,
    resolve_dit_state_dict,
    resolve_model_references,
)
from kandinsky_sr.pipeline.config import SRConfig

from .conftest import IN_VISUAL_DIM, KVAE_SCALING_FACTOR, PIFLOW_GRID_POINTS, BundleFactory

UPSCALER_VALUES = {"2x": 2.0, "4x": 4.0}


class ToyUpscaler(torch.nn.Module):
    """Has the one parameter the toy bundle stores per upscaler."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))


@pytest.fixture
def toy_upscaler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lu_module, "build_upsampler", lambda _config: ToyUpscaler())


def test_training_config_is_read_from_the_transformer_component(make_bundle: BundleFactory) -> None:
    conf, vae_name = load_training_config(str(make_bundle() / "transformer"), "torch")

    assert vae_name == "video-kvae"
    assert conf.dit.params.out_visual_dim == IN_VISUAL_DIM
    assert conf.trainer.piflow.dx_num_grid_points == PIFLOW_GRID_POINTS


def test_dit_weights_are_read_from_the_transformer_component(make_bundle: BundleFactory) -> None:
    state = resolve_dit_state_dict(str(make_bundle() / "transformer"), CheckpointConfig(), "cpu")

    assert list(state) == ["out_layer.weight"]


def test_each_upscaler_gets_its_own_weights(make_bundle: BundleFactory, toy_upscaler: None) -> None:
    bank = load_latent_upscaler_bank(str(make_bundle() / "latent_upscaler"), None, "cpu", KVAE_SCALING_FACTOR)

    assert bank.scales == (2, 4)
    for scale, value in UPSCALER_VALUES.items():
        upscaler = bank[scale]
        assert torch.equal(upscaler.weight.float(), torch.full((2,), value))
        assert upscaler.target_scale == scale
        assert upscaler.scaling_factor == KVAE_SCALING_FACTOR


def test_lazily_loaded_upscaler_reads_its_own_weights(make_bundle: BundleFactory, toy_upscaler: None) -> None:
    bank = load_latent_upscaler_bank(str(make_bundle() / "latent_upscaler"), ("2x",), "cpu", KVAE_SCALING_FACTOR)

    assert isinstance(bank, LazyLatentUpscalerBank)
    assert sorted(bank.lazy_specs) == ["4x"]
    assert torch.equal(bank.for_scale(4).weight.float(), torch.full((2,), UPSCALER_VALUES["4x"]))


def test_overrides_stored_in_the_bundle_apply_under_explicit_ones(make_bundle: BundleFactory) -> None:
    transformer = make_bundle() / "transformer"
    config = json.loads((transformer / "config.json").read_text())
    config["attribute_overrides"] = {"visual_cond": False, "instruct_type": "latent"}
    (transformer / "config.json").write_text(json.dumps(config))
    conf, _ = load_training_config(str(transformer), "torch")

    assert components.merged_dit_overrides(conf, {"instruct_type": "noise"}) == {
        "visual_cond": False,
        "instruct_type": "noise",
    }
    assert components.merged_dit_overrides(conf, None) == {"visual_cond": False, "instruct_type": "latent"}


def test_native_training_config_carries_no_overrides(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("vae: {name: video-kvae}\ndit: {params: {}}\n")
    conf, _ = load_training_config(str(tmp_path), "torch")

    assert components.merged_dit_overrides(conf, {"visual_cond": True}) == {"visual_cond": True}
    assert components.merged_dit_overrides(conf, None) == {}


def test_native_checkpoint_without_an_upscaler_setting_is_refused(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"x")

    with pytest.raises(ValueError, match="latent_upscaler_config"):
        resolve_model_references(str(tmp_path), "/kvae/prefix", None)


def test_factory_accepts_a_config_that_names_only_the_bundle(
    make_bundle: BundleFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_components(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise StopIteration

    monkeypatch.setattr(factory, "load_sr_components", fake_components)
    config = SRConfig(enabled=True, checkpoint_path=str(make_bundle()))

    with pytest.raises(StopIteration):
        factory.load_sr_pipeline(config, "cpu")

    assert seen["checkpoint_path"] == config.checkpoint_path
    assert seen["vae_path"] is None
    assert seen["latent_upscaler_config"] is None


def test_cli_accepts_a_config_that_names_only_the_bundle(make_bundle: BundleFactory, tmp_path: Path) -> None:
    options = sr_cli.CommonOptions(checkpoint_path=str(make_bundle()), output_dir=tmp_path / "out")

    sr = sr_cli._prepare_common(options)

    assert sr.vae_path is None
    assert sr.latent_upscaler_config is None
