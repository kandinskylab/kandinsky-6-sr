"""SR config YAML loading.

What & why: ``load_sr_config_yaml`` is the single entry point for both the
CLI and the embedding factory, so it must read only the ``sr:`` section,
resolve relative model paths against the YAML's directory, and leave Hugging
Face repo ids untouched. How: pure unit tests over tmp YAML files.
Corner cases: missing ``sr:`` section, ``"none"`` sentinel, absolute paths,
full k6 pipeline YAML with unrelated top-level sections.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kandinsky_sr.pipeline.config import SRConfig, load_sr_config_yaml

DEFAULT_RESOLUTION_SCALE = 2.25


def test_relative_paths_resolve_against_yaml_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "sr.yaml"
    config_path.write_text("sr:\n  checkpoint_path: dit\n  latent_upscaler_config: bank.yaml\n  vae_path: /abs/kvae\n")

    sr = load_sr_config_yaml(config_path)

    assert sr.checkpoint_path == str(tmp_path / "dit")
    assert sr.latent_upscaler_config == str(tmp_path / "bank.yaml")
    assert sr.vae_path == "/abs/kvae"


def test_hf_repo_ids_and_none_pass_through(tmp_path: Path) -> None:
    config_path = tmp_path / "sr.yaml"
    config_path.write_text(
        "sr:\n"
        "  checkpoint_path: kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s\n"
        "  vae_path: kandinskylab/KVAE-3D-2.0-t4s16-SR\n"
        "  latent_upscaler_config: none\n"
    )

    sr = load_sr_config_yaml(config_path)

    assert sr.checkpoint_path == "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s"
    assert sr.vae_path == "kandinskylab/KVAE-3D-2.0-t4s16-SR"
    assert sr.latent_upscaler_config == "none"


def test_bundle_references_pass_through(tmp_path: Path) -> None:
    """A bundle repo id and a component of it are Hub references, not paths relative to the YAML."""
    bundle = "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"
    config_path = tmp_path / "sr.yaml"
    config_path.write_text(
        "sr:\n"
        f"  checkpoint_path: {bundle}\n"
        f"  vae_path: {bundle}/vae\n"
        f"  latent_upscaler_config: {bundle}/latent_upscaler\n"
    )

    sr = load_sr_config_yaml(config_path)

    assert sr.checkpoint_path == bundle
    assert sr.vae_path == f"{bundle}/vae"
    assert sr.latent_upscaler_config == f"{bundle}/latent_upscaler"


def test_full_pipeline_yaml_reads_only_sr_section(tmp_path: Path) -> None:
    config_path = tmp_path / "k6.yaml"
    config_path.write_text("model:\n  name: k6\noffload:\n  strategy: module\nsr:\n  enabled: true\n  device: cuda:1\n")

    sr = load_sr_config_yaml(config_path)

    assert sr.enabled is True
    assert sr.device == "cuda:1"


def test_missing_sr_section_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "other.yaml"
    config_path.write_text("model:\n  name: k6\n")

    with pytest.raises(ValueError, match="no 'sr:' section"):
        load_sr_config_yaml(config_path)


def test_defaults() -> None:
    sr = SRConfig()

    assert sr.device == "cuda:0"
    assert sr.resolution_scale == DEFAULT_RESOLUTION_SCALE
    assert sr.enabled is False


@pytest.mark.parametrize("scale", [2, 2.25, 4])
def test_resolution_scale_accepts_the_three_supported_values(scale: float) -> None:
    assert SRConfig(resolution_scale=scale).resolution_scale == scale


def test_resolution_scale_rejects_other_values() -> None:
    with pytest.raises(ValidationError):
        SRConfig(resolution_scale=3)
