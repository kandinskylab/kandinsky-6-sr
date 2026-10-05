"""A ``Kandinsky6SRPipeline`` Diffusers bundle read back into the native loaders' inputs.

What & why: the models are published only as Diffusers bundles, so the native
pipeline must rebuild what it used to read from the native checkpoints — the
training config next to the DiT, the KVAE sidecar pair and the latent-upscaler
bank — out of the bundle's component folders.

How: a tiny bundle written by the ``make_bundle`` fixture (real config layout,
toy weights); plain filesystem, no downloads, no model builds.

Corner cases: a distilled (π-Flow) vs a flow-matching bundle, the ``n_grid``-wide
output head folded back to the DiT's own width, JSON's string resolution keys,
both spellings of the KVAE architecture (the flat fields of the published
bundles and the nested sections the converter writes), a repeated sidecar
request, the two upscalers sharing one weights file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from kandinsky_sr.core.algo.safetensors_io import load_safetensors_state_dict
from kandinsky_sr.pipeline.diffusers_bundle import (
    COMPONENT_WEIGHTS,
    is_bundle_dir,
    is_component_dir,
    kvae_sidecar_from_component,
    lu_bank_conf_from_component,
    training_config_from_transformer,
)

from .conftest import (
    FLOW_MATCHING_SHIFT,
    IN_VISUAL_DIM,
    KVAE_DECODER_PARAMS,
    KVAE_ENCODER_PARAMS,
    KVAE_SCALING_FACTOR,
    LU_MODEL,
    PIFLOW_GRID_POINTS,
    PIFLOW_SHIFT,
    BundleFactory,
    VaeLayout,
)


def test_bundle_and_component_directories_are_told_apart(make_bundle: BundleFactory, tmp_path: Path) -> None:
    root = make_bundle()

    assert is_bundle_dir(root)
    assert not is_component_dir(root)
    assert is_component_dir(root / "vae")
    assert not is_bundle_dir(root / "vae")
    assert not is_component_dir(root / "scheduler")
    assert not is_bundle_dir(tmp_path / "missing")


def test_distilled_bundle_rebuilds_the_piflow_training_config(make_bundle: BundleFactory) -> None:
    conf = training_config_from_transformer(make_bundle("piflow") / "transformer")

    assert conf.vae.name == "video-kvae"
    assert conf.dit.params.out_visual_dim == IN_VISUAL_DIM
    assert OmegaConf.to_container(conf.trainer.piflow) == {
        "nfe": 2,
        "dx_num_grid_points": PIFLOW_GRID_POINTS,
        "shift": PIFLOW_SHIFT,
        "num_policy_substeps": 128,
        "final_step_size_scale": 0.5,
        "eps": 1e-06,
    }
    assert OmegaConf.to_container(conf.trainer.params) == {
        "lq_noise_scale": 0.7,
        "lq_noise_type": "ddpm",
        "lq_channel_noise_scale": 0.0,
        "cap_noise_timestep": False,
        "scheduler_scale": PIFLOW_SHIFT,
    }


def test_flow_matching_bundle_has_no_piflow_section(make_bundle: BundleFactory) -> None:
    conf = training_config_from_transformer(make_bundle("flow_matching") / "transformer")

    assert "piflow" not in conf.trainer
    assert conf.dit.params.out_visual_dim == IN_VISUAL_DIM
    assert conf.trainer.params.scheduler_scale == FLOW_MATCHING_SHIFT


def test_training_config_keeps_the_native_layout(make_bundle: BundleFactory) -> None:
    """Resolution keys come back as ints and the bundle-only fields leave ``dit.params``."""
    conf = training_config_from_transformer(make_bundle() / "transformer")

    assert OmegaConf.to_container(conf.common) == {
        "visual_size": [512],
        "fps": 24,
        "scale_factor": {512: [1.0, 2.0, 2.0]},
    }
    assert sorted(conf.dit.params.attention_params) == [512, 1024]
    assert "sr_params" not in conf.dit.params
    assert "attribute_overrides" not in conf.dit.params
    assert OmegaConf.to_container(conf.dit.attribute_overrides) == {}


def test_piflow_head_that_is_not_a_multiple_of_the_grid_is_refused(make_bundle: BundleFactory) -> None:
    transformer = make_bundle() / "transformer"
    config = json.loads((transformer / "config.json").read_text())
    config["out_visual_dim"] = IN_VISUAL_DIM * PIFLOW_GRID_POINTS + 1
    (transformer / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="out_visual_dim"):
        training_config_from_transformer(transformer)


def test_transformer_without_its_scheduler_is_refused(make_bundle: BundleFactory) -> None:
    """Without the sampler settings a distilled DiT would silently run as a flow-matching one."""
    root = make_bundle("piflow")
    (root / "scheduler" / "scheduler_config.json").unlink()

    with pytest.raises(FileNotFoundError, match="scheduler_config.json"):
        training_config_from_transformer(root / "transformer")


def test_unknown_scheduler_is_refused(make_bundle: BundleFactory) -> None:
    root = make_bundle("flow_matching")
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps({"_class_name": "DDIMScheduler"}))

    with pytest.raises(ValueError, match="DDIMScheduler"):
        training_config_from_transformer(root / "transformer")


@pytest.mark.parametrize("vae_layout", ["flat", "nested"])
def test_vae_component_becomes_a_kvae_sidecar_pair(make_bundle: BundleFactory, vae_layout: VaeLayout) -> None:
    """Either spelling of the architecture yields the encoder / decoder parameters the builders take."""
    vae = make_bundle(vae_layout=vae_layout) / "vae"

    prefix = Path(kvae_sidecar_from_component(vae))

    assert prefix.parent == vae
    sidecar = yaml.safe_load(prefix.with_suffix(".yaml").read_text())
    assert sidecar["scaling_factor"] == KVAE_SCALING_FACTOR
    assert sidecar["model"] == {"encoder_params": KVAE_ENCODER_PARAMS, "decoder_params": KVAE_DECODER_PARAMS}
    weights = prefix.with_suffix(".safetensors")
    assert weights.is_symlink()
    assert weights.resolve() == (vae / COMPONENT_WEIGHTS).resolve()


def test_sidecar_request_is_repeatable(make_bundle: BundleFactory) -> None:
    vae = make_bundle() / "vae"

    assert kvae_sidecar_from_component(vae) == kvae_sidecar_from_component(vae)


def test_non_vae_component_is_refused_as_a_kvae(make_bundle: BundleFactory) -> None:
    with pytest.raises(ValueError, match="not a KVAE component"):
        kvae_sidecar_from_component(make_bundle() / "transformer")


def test_kvae_config_missing_an_architecture_field_is_refused(make_bundle: BundleFactory) -> None:
    vae = make_bundle() / "vae"
    config = json.loads((vae / "config.json").read_text())
    del config["decoder_ch_mult"]
    (vae / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="decoder_ch_mult"):
        kvae_sidecar_from_component(vae)


def test_latent_upscaler_component_becomes_a_bank_conf(make_bundle: BundleFactory) -> None:
    component = make_bundle() / "latent_upscaler"

    conf = lu_bank_conf_from_component(component)

    assert conf.latent_upscaler.enabled is True
    entries = {entry.target_scale: entry for entry in conf.latent_upscaler.models}
    assert sorted(entries) == ["2x", "4x"]
    for scale, entry in entries.items():
        assert entry.checkpoint == str(component / COMPONENT_WEIGHTS)
        assert entry.state_prefix == {"2x": "_models.0.", "4x": "_models.1."}[scale]
        assert OmegaConf.to_container(entry.model) == LU_MODEL


def test_non_bank_component_is_refused_as_a_latent_upscaler(make_bundle: BundleFactory) -> None:
    with pytest.raises(ValueError, match="models"):
        lu_bank_conf_from_component(make_bundle() / "vae")


def test_one_upscaler_is_read_out_of_the_shared_weights_file(make_bundle: BundleFactory) -> None:
    weights = str(make_bundle() / "latent_upscaler" / COMPONENT_WEIGHTS)

    state = load_safetensors_state_dict(weights, key_prefix="_models.1.")

    assert list(state) == ["weight"]
    assert torch.equal(state["weight"], torch.full((2,), 4.0))


def test_unknown_weights_prefix_is_an_error(make_bundle: BundleFactory) -> None:
    weights = str(make_bundle() / "latent_upscaler" / COMPONENT_WEIGHTS)

    with pytest.raises(KeyError, match="_models.8x."):
        load_safetensors_state_dict(weights, key_prefix="_models.8x.")
