"""Model-path references: local directories resolve like Hub snapshots.

What & why: ``sr.vae_path`` may name the KVAE directory instead of the sidecar
prefix, and ``sr.latent_upscaler_config`` the LU directory instead of the bank
YAML — the single pair / bank YAML inside is discovered. How: tmp directories
with dummy files; no downloads. Corner cases: explicit prefix passes through
untouched, zero or two candidates raise, the KVAE sidecar YAML is not mistaken
for a bank YAML and vice versa.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from kandinsky_sr.pipeline.hub import resolve_kvae_reference, resolve_lu_bank_reference


def _touch(*paths: Path) -> None:
    for path in paths:
        path.write_bytes(b"x")


def test_kvae_directory_resolves_to_the_sidecar_prefix(tmp_path: Path) -> None:
    _touch(tmp_path / "kvae_20_v2.yaml", tmp_path / "kvae_20_v2.safetensors", tmp_path / "README.md")

    assert resolve_kvae_reference(str(tmp_path)) == str(tmp_path / "kvae_20_v2")


def test_kvae_prefix_passes_through(tmp_path: Path) -> None:
    prefix = str(tmp_path / "kvae_20_v2")

    assert resolve_kvae_reference(prefix) == prefix


@pytest.mark.parametrize("pairs", [0, 2])
def test_kvae_directory_requires_exactly_one_pair(tmp_path: Path, pairs: int) -> None:
    for index in range(pairs):
        _touch(tmp_path / f"kvae_{index}.yaml", tmp_path / f"kvae_{index}.ckpt")

    with pytest.raises(ValueError, match="exactly one KVAE sidecar pair"):
        resolve_kvae_reference(str(tmp_path))


def test_lu_directory_resolves_to_the_bank_yaml(tmp_path: Path) -> None:
    _touch(
        tmp_path / "latent_upscaler_multi_scale_kvae.yaml",
        tmp_path / "latent_upscaler_2x.safetensors",
        tmp_path / "latent_upscaler_4x.safetensors",
    )

    assert resolve_lu_bank_reference(str(tmp_path)) == str(tmp_path / "latent_upscaler_multi_scale_kvae.yaml")


def test_lu_directory_ignores_sidecar_yamls(tmp_path: Path) -> None:
    _touch(tmp_path / "bank.yaml", tmp_path / "kvae.yaml", tmp_path / "kvae.safetensors")

    assert resolve_lu_bank_reference(str(tmp_path)) == str(tmp_path / "bank.yaml")


def test_lu_yaml_path_passes_through(tmp_path: Path) -> None:
    bank = tmp_path / "bank.yaml"
    _touch(bank)

    assert resolve_lu_bank_reference(str(bank)) == str(bank)


HF_KVAE_CONFIG_JSON = {
    "data": {"input_norm": "m11"},
    "model": {
        "encoder_params": {"in_channels": 3, "ch": 128, "z_channels": 64},
        "decoder_params": {"out_ch": 3, "ch": 256, "z_channels": 64},
    },
}


def test_public_kvae_checkout_becomes_a_sidecar_pair(tmp_path: Path) -> None:
    """config.json + model.safetensors (kandinskylab/KVAE-3D-2.0-*) resolve like a sidecar."""
    (tmp_path / "config.json").write_text(json.dumps(HF_KVAE_CONFIG_JSON))
    _touch(tmp_path / "model.safetensors", tmp_path / "README.md")

    prefix = resolve_kvae_reference(str(tmp_path))

    assert prefix == str(tmp_path / "kvae")
    assert (tmp_path / "kvae.safetensors").resolve() == (tmp_path / "model.safetensors").resolve()
    sidecar = yaml.safe_load((tmp_path / "kvae.yaml").read_text())
    assert sidecar["model"]["encoder_params"]["ch"] == HF_KVAE_CONFIG_JSON["model"]["encoder_params"]["ch"]
    assert sidecar["model"]["decoder_params"]["ch"] == HF_KVAE_CONFIG_JSON["model"]["decoder_params"]["ch"]
    assert sidecar["scaling_factor"] == pytest.approx(0.910344004631042)
    # idempotent: a second resolve reuses the generated pair
    assert resolve_kvae_reference(str(tmp_path)) == prefix


def test_public_kvae_scaling_factor_from_config_wins(tmp_path: Path) -> None:
    config = {**HF_KVAE_CONFIG_JSON, "scaling_factor": 0.5}
    (tmp_path / "config.json").write_text(json.dumps(config))
    _touch(tmp_path / "model.safetensors")

    resolve_kvae_reference(str(tmp_path))

    assert yaml.safe_load((tmp_path / "kvae.yaml").read_text())["scaling_factor"] == pytest.approx(0.5)


def test_sidecar_pair_takes_precedence_over_a_public_checkout(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(HF_KVAE_CONFIG_JSON))
    _touch(tmp_path / "model.safetensors", tmp_path / "kvae_20_v2.yaml", tmp_path / "kvae_20_v2.safetensors")

    assert resolve_kvae_reference(str(tmp_path)) == str(tmp_path / "kvae_20_v2")
