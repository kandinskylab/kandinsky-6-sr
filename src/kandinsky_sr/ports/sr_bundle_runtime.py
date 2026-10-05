"""Implementation of SR-to-Diffusers bundle conversion.

This module is intentionally kept outside the generated port. It uses the
native SR builders to read resolved local or Hugging Face training checkpoints,
while the resulting bundle contains Diffusers-compatible component wrappers
and no absolute source paths. The generated ``convert_sr_checkpoint.py`` is a
small, reviewable CLI front-end to this implementation.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from diffusers import __version__ as diffusers_version
from omegaconf import DictConfig, OmegaConf
from pydantic import TypeAdapter
from safetensors.torch import save_file

from kandinsky_sr.core.algo.checkpoint import CheckpointConfig
from kandinsky_sr.core.algo.latent_upscaler import resolve_lu_state_dict
from kandinsky_sr.core.algo.piflow_sampler import build_piflow_sampler_params
from kandinsky_sr.core.components.latent_upscaler.config import ModelConfig
from kandinsky_sr.core.components.video_kvae.cached_model import CachedCausalVAE
from kandinsky_sr.pipeline.components import (
    _extract_sr_params,
    _resolve_config,
    resolve_dit_state_dict,
    resolve_lu_bank_checkpoints,
    resolve_vae_source,
)
from kandinsky_sr.pipeline.diffusers_bundle import lu_state_prefix
from kandinsky_sr.pipeline.hub import (
    resolve_checkpoint_reference,
    resolve_kvae_reference,
    resolve_lu_bank_reference,
)

_SAFE_TENSOR_FILENAME = "diffusion_pytorch_model.safetensors"
_PIPELINE_FILENAME = "pipeline_kandinsky6_sr.py"
_OUTPUT_FILENAME = "pipeline_output.py"
_SCHEDULER_FILENAME = "scheduling_piflow.py"
_ATTENTION_FILENAME = "attention_processor.py"
_TI2VA_FILENAME = "modeling_kandinsky6.py"
_SHARED_FILENAME = "modeling_kandinsky6_sr.py"
_COMPONENT_FILES = {
    "transformer": ("transformer_kandinsky6_sr.py", "transformer_kandinsky6_sr.py"),
    "vae": ("autoencoder_kandinsky6_sr.py", "autoencoder_kandinsky6_sr.py"),
    "latent_upscaler": ("latent_upscaler.py", "latent_upscaler.py"),
}
_LEGACY_COMPONENT_FILES = {
    "transformer": "sr_dit.py",
    "vae": "sr_vae.py",
    "latent_upscaler": "sr_latent_upscaler.py",
}
_MODEL_CONFIG_ADAPTER = TypeAdapter(ModelConfig)


def _container(value: Any) -> Any:
    """Convert OmegaConf/Pydantic values to ordinary Python containers."""
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _json_safe(value: Any) -> Any:  # noqa: PLR0911
    """Make metadata JSON-safe without preserving machine-local paths."""
    value = _container(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        path = Path(value)
        return path.name if path.is_absolute() else value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (AttributeError, TypeError, ValueError):
            pass
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def _require_file(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {value}")
    return value


def _copy_file(path: str | Path, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_require_file(path, label), destination)


def _save_component(directory: Path, config: Mapping[str, Any], state_dict: Mapping[str, Any]) -> None:
    """Write one standard Diffusers ``ModelMixin`` component directory."""
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "config.json", config)
    tensors: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"component state contains non-tensor value at {key!r}: {type(value).__name__}")
        tensors[str(key)] = value.contiguous() if not value.is_contiguous() else value
    save_file(tensors, str(directory / _SAFE_TENSOR_FILENAME))


def _get_sr_checkpoint(config: Any, override: str | Path | None) -> Path:
    value = override or getattr(config.sr, "checkpoint_path", None)
    if not value:
        raise ValueError("SR checkpoint is missing; set sr.checkpoint_path or pass --checkpoint-path")
    return Path(resolve_checkpoint_reference(str(value)))


def _extract_piflow_params(training_config: Any) -> dict[str, Any] | None:
    piflow = getattr(getattr(training_config, "trainer", None), "piflow", None)
    if piflow is None or not bool(getattr(piflow, "enabled", True)):
        return None
    n_grid = int(getattr(piflow, "dx_num_grid_points", 0))
    return build_piflow_sampler_params(piflow, n_grid=n_grid)


def _kvae_configs(vae_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load KVAE architecture and return encoder, decoder, and full config."""
    config_path = _require_file(f"{vae_path}.yaml", "KVAE config")
    config = OmegaConf.load(config_path)
    encoder = config.encoder_params if "encoder_params" in config else config.model.encoder_params
    decoder = config.decoder_params if "decoder_params" in config else config.model.decoder_params
    return (
        _json_safe(OmegaConf.to_container(encoder, resolve=True)),
        _json_safe(OmegaConf.to_container(decoder, resolve=True)),
        _json_safe(OmegaConf.to_container(config, resolve=True)),
    )


def _convert_kvae(vae_path: Path, output_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Convert a causal KVAE checkpoint into a Diffusers autoencoder."""
    encoder_config, decoder_config, full_config = _kvae_configs(vae_path)
    native_vae = CachedCausalVAE(encoder_conf=encoder_config, decoder_conf=decoder_config)
    safetensors_path = Path(f"{vae_path}.safetensors")
    checkpoint = safetensors_path if safetensors_path.is_file() else Path(f"{vae_path}.ckpt")
    native_vae.init_from_ckpt(str(_require_file(checkpoint, "KVAE weights")))
    state_dict = native_vae.state_dict()
    scaling_factor = float(full_config.get("scaling_factor", 1.0))
    component_config = {
        "vae_type": "video-kvae",
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
        "scaling_factor": scaling_factor,
        "spatial_factor": 16,
        "temporal_factor": int(encoder_config.get("temporal_compress_times", 4)),
    }
    _save_component(output_dir / "vae", component_config, state_dict)
    return ["autoencoder_kandinsky6_sr", "Kandinsky6SRVAE"], component_config


def _convert_vae(
    sr_config: Any,
    output_dir: Path,
    vae_override: str | Path | None,
) -> tuple[list[str], dict[str, Any]]:
    """Convert the KVAE selected by the SR training config."""
    configured_path = vae_override or getattr(sr_config, "vae_path", None)
    vae_path = resolve_kvae_reference(resolve_vae_source(str(configured_path) if configured_path else None))
    prefix = Path(vae_path)
    if prefix.suffix == ".ckpt":
        prefix = prefix.with_suffix("")
    return _convert_kvae(prefix, output_dir)


def _convert_dit(
    sr_checkpoint: Path,
    training_config: Any,
    sr_config: Any,
    sr_params: Mapping[str, Any],
    output_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Save native SR DiT weights under the Diffusers transformer layout."""
    state_dict = resolve_dit_state_dict(str(sr_checkpoint), CheckpointConfig(), "cpu")
    model_config = dict(OmegaConf.to_container(training_config.dit.params, resolve=True))
    piflow_params = _extract_piflow_params(training_config)
    if piflow_params is not None:
        model_config["out_visual_dim"] = int(model_config["out_visual_dim"]) * int(piflow_params["n_grid"])
    attribute_overrides = _json_safe(getattr(sr_config, "dit_overrides", None) or {})
    component_config = dict(model_config)
    component_config["attribute_overrides"] = attribute_overrides
    component_config["sr_params"] = _json_safe(sr_params)
    _save_component(
        output_dir / "transformer",
        component_config,
        state_dict,
    )
    return ["transformer_kandinsky6_sr", "Kandinsky6SRTransformer3DModel"], component_config


def _convert_latent_upscaler(
    sr_config: Any,
    vae_component_config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[list[str] | None, dict[str, Any] | None]:
    """Convert all configured x2/x4 latent-upscaler entries into one bank."""
    config_path = getattr(sr_config, "latent_upscaler_config", None)
    if not config_path:
        raise ValueError(
            "SR bundle conversion requires sr.latent_upscaler_config; set it to a local LU bank YAML or 'none'"
        )
    if str(config_path).lower() == "none":
        return None, None
    config_path = resolve_lu_bank_reference(str(config_path))
    config_file = _require_file(config_path, "latent upscaler config")
    bank = OmegaConf.load(config_file)
    resolve_lu_bank_checkpoints(bank, config_file.absolute().parent)
    models = getattr(getattr(bank, "latent_upscaler", None), "models", None)
    if models is None:
        raise ValueError(f"latent upscaler config has no latent_upscaler.models list: {config_path}")

    serialized_models: list[dict[str, Any]] = []
    serialized_state: dict[str, torch.Tensor] = {}
    seen_scales: set[str] = set()
    scaling_factor = float(vae_component_config.get("scaling_factor", 1.0))
    for item in models:
        target_scale = str(getattr(item, "target_scale", "4x"))
        if not target_scale.endswith("x"):
            target_scale = f"{target_scale}x"
        if target_scale in seen_scales:
            raise ValueError(f"duplicate latent upscaler target scale: {target_scale}")
        seen_scales.add(target_scale)
        model_config = _json_safe(OmegaConf.to_container(item.model, resolve=True))
        _MODEL_CONFIG_ADAPTER.validate_python(model_config)
        checkpoint = _require_file(item.checkpoint, f"latent upscaler {target_scale} checkpoint")
        weights = resolve_lu_state_dict(str(checkpoint), use_ema=bool(getattr(item, "use_ema", True)))
        serialized_models.append({"target_scale": target_scale, "model": model_config})
        for key, value in weights.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"latent upscaler {target_scale} has non-tensor value at {key!r}")
            serialized_state[f"{lu_state_prefix(target_scale)}{key}"] = value

    component_config = {"models": serialized_models, "scaling_factor": scaling_factor}
    _save_component(output_dir / "latent_upscaler", component_config, serialized_state)
    return ["latent_upscaler", "Kandinsky6SRLatentUpscalerBank"], component_config


def _copy_generated_code(
    output_dir: Path,
    generated_dir: Path,
    *,
    use_patched_diffusers: bool = False,
) -> None:
    """Copy the generated SR pipeline and output types into the bundle."""
    # ``source_vae`` is an optional runtime-only input for latent SR. Remove a
    # stale directory when reusing an output path from older converter runs.
    source_vae = output_dir / "source_vae"
    if source_vae.exists():
        shutil.rmtree(source_vae)
    (output_dir / "scheduler" / "scheduler_config.json").unlink(missing_ok=True)
    (output_dir / "kandinsky6_attention.py").unlink(missing_ok=True)
    (output_dir / _ATTENTION_FILENAME).unlink(missing_ok=True)
    (output_dir / _SHARED_FILENAME).unlink(missing_ok=True)
    (output_dir / _SCHEDULER_FILENAME).unlink(missing_ok=True)
    for component_name in _COMPONENT_FILES:
        (output_dir / component_name / "kandinsky6_attention.py").unlink(missing_ok=True)
        (output_dir / component_name / _ATTENTION_FILENAME).unlink(missing_ok=True)
        (output_dir / component_name / _SHARED_FILENAME).unlink(missing_ok=True)
        (output_dir / component_name / _LEGACY_COMPONENT_FILES[component_name]).unlink(missing_ok=True)
    legacy_dit_dir = output_dir / "dit"
    if legacy_dit_dir.is_dir():
        shutil.rmtree(legacy_dit_dir)
    if use_patched_diffusers:
        (output_dir / _PIPELINE_FILENAME).unlink(missing_ok=True)
        (output_dir / _OUTPUT_FILENAME).unlink(missing_ok=True)
        (output_dir / _SCHEDULER_FILENAME).unlink(missing_ok=True)
        for component_name, (filename, source_name) in _COMPONENT_FILES.items():
            (output_dir / component_name / filename).unlink(missing_ok=True)
            (output_dir / component_name / source_name).unlink(missing_ok=True)
    else:
        _copy_file(generated_dir / _PIPELINE_FILENAME, output_dir / _PIPELINE_FILENAME, "generated SR pipeline")
        _copy_file(generated_dir / _OUTPUT_FILENAME, output_dir / _OUTPUT_FILENAME, "generated pipeline output")
        _copy_file(generated_dir / _SCHEDULER_FILENAME, output_dir / _SCHEDULER_FILENAME, "generated PiFlow scheduler")
        _copy_file(generated_dir / _TI2VA_FILENAME, output_dir / _TI2VA_FILENAME, "generated TI2VA transformer")
        _copy_file(
            generated_dir / _TI2VA_FILENAME,
            output_dir / "transformer" / _TI2VA_FILENAME,
            "generated TI2VA transformer component",
        )
        # Diffusers resolves custom component classes relative to the component
        # directory.  Keep a copy beside each custom component so
        # ``from_pretrained`` works from a completely local bundle.
        for component_name, (filename, source_name) in _COMPONENT_FILES.items():
            component_source = generated_dir / source_name
            _copy_file(component_source, output_dir / filename, f"SR {component_name} wrapper")
            component_dir = output_dir / component_name
            if component_dir.is_dir():
                _copy_file(
                    component_source,
                    component_dir / filename,
                    f"SR Diffusers {component_name} component wrapper",
                )


def _patched_component_ref(component: list[str], enabled: bool) -> list[str]:
    if not enabled:
        return component
    return ["diffusers", component[1]]


def _piflow_scheduler_config(piflow_params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "_class_name": "PiflowScheduler",
        "_diffusers_version": str(diffusers_version),
        "num_train_timesteps": 1000,
        "shift": float(piflow_params["shift"]),
        "n_grid": int(piflow_params["n_grid"]),
        "nfe": int(piflow_params["nfe"]),
        "eps": float(piflow_params["eps"]),
        "final_step_size_scale": float(piflow_params["final_step_size_scale"]),
        "num_policy_substeps": int(piflow_params["num_policy_substeps"]),
    }


def _model_index(  # noqa: PLR0913
    *,
    diffusers_vae: list[str],
    transformer: list[str],
    latent_upscaler: list[str] | None,
    dit_checkpoint: Path,
    training_config: Any,
    scheduler: list[str],
    use_patched_diffusers: bool = False,
) -> dict[str, Any]:
    transformer = _patched_component_ref(transformer, use_patched_diffusers)
    diffusers_vae = _patched_component_ref(diffusers_vae, use_patched_diffusers)
    latent_upscaler = (
        _patched_component_ref(latent_upscaler, use_patched_diffusers) if latent_upscaler is not None else None
    )
    index: dict[str, Any] = {
        "_class_name": (
            "Kandinsky6SRPipeline"
            if use_patched_diffusers
            else ["pipeline_kandinsky6_sr", "Kandinsky6SRPipeline"]
        ),
        "_diffusers_version": str(diffusers_version),
        "transformer": transformer,
        "vae": diffusers_vae,
        "_kandinsky6_sr": {
            "format_version": 1,
            "source_checkpoint": dit_checkpoint.name,
            "vae_name": str(getattr(getattr(training_config, "vae", None), "name", "")),
            "latent_upscaler": "latent_upscaler" if latent_upscaler is not None else None,
            "portable_components": True,
            "uses_patched_diffusers": use_patched_diffusers,
        },
    }
    index["scheduler"] = scheduler
    if latent_upscaler is not None:
        index["latent_upscaler"] = latent_upscaler
    return index


def _write_readme(output_dir: Path, *, use_patched_diffusers: bool = False) -> None:
    _write_json(
        output_dir / "sr_config.json",
        {
            "description": "Kandinsky 6 SR Diffusers bundle",
            "default_resolution_scale": 2.25,
            "component_loading": "Diffusers ModelMixin wrappers",
        },
    )
    loading = (
        "The pipeline, output, DiT, VAE, and latent-upscaler classes resolve from an installed patched "
        "Diffusers package.\n\n"
        "Load it with `Kandinsky6SRPipeline.from_pretrained(path, "
        "torch_dtype=torch.bfloat16)`.\n\n"
        if use_patched_diffusers
        else "Load it with `Kandinsky6SRPipeline.from_pretrained(path, "
        "trust_remote_code=True, torch_dtype=torch.bfloat16)`.\n\n"
    )
    (output_dir / "README.md").write_text(
        "# Kandinsky 6 SR Diffusers export\n\n"
        "This bundle contains the SR DiT, KVAE, and configured latent-upscaler bank.\n\n"
        + loading
        + "The production route is `resolution_scale=2.25`: a 1.125x pixel "
        "pre-upscale followed by the x2 latent-upscaler path.\n\n"
        "The exported wrappers use eager model execution. Apply Torch/Magi "
        "compilation after loading if the runtime supports it.\n",
        encoding="utf-8",
    )


def convert_sr_checkpoint(  # noqa: PLR0913
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    vae_path: str | Path | None = None,
    latent_upscaler_config: str | Path | None = None,
    output_dir: str | Path = "outputs/kandinsky6_sr_diffusers",
    generated_dir: str | Path | None = None,
    use_patched_diffusers: bool = False,
) -> Path:
    """Convert an SR checkpoint from local paths or Hugging Face repo ids."""
    config = OmegaConf.load(config_path)
    sr_config = config.sr
    sr_checkpoint = _get_sr_checkpoint(config, checkpoint_path)
    training_config = _resolve_config(str(sr_checkpoint))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = Path(generated_dir).resolve() if generated_dir is not None else output

    # The SR config's LU override is the only input mutation: all other model
    # architecture values come from the training YAML next to the SR DiT.
    if latent_upscaler_config is not None:
        OmegaConf.set_struct(config, value=False)
        sr_config.latent_upscaler_config = str(latent_upscaler_config)

    piflow_params = _extract_piflow_params(training_config)
    sr_params = _extract_sr_params(training_config).model_dump(mode="json")
    transformer_class, _ = _convert_dit(sr_checkpoint, training_config, sr_config, sr_params, output)
    vae_class, vae_config = _convert_vae(sr_config, output, vae_path)
    latent_class, latent_config = _convert_latent_upscaler(sr_config, vae_config, output)
    _copy_generated_code(
        output,
        generated,
        use_patched_diffusers=use_patched_diffusers,
    )
    if piflow_params is not None:
        _write_json(output / "scheduler" / "scheduler_config.json", _piflow_scheduler_config(piflow_params))
        scheduler = (
            ["diffusers", "PiflowScheduler"] if use_patched_diffusers else ["scheduling_piflow", "PiflowScheduler"]
        )
    else:
        from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: PLC0415

        FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=float(sr_params["scheduler_scale"]),
        ).save_pretrained(output / "scheduler")
        scheduler = ["diffusers", "FlowMatchEulerDiscreteScheduler"]
    index = _model_index(
        diffusers_vae=vae_class,
        transformer=transformer_class,
        latent_upscaler=latent_class,
        dit_checkpoint=sr_checkpoint,
        training_config=training_config,
        scheduler=scheduler,
        use_patched_diffusers=use_patched_diffusers,
    )
    _write_json(output / "model_index.json", index)
    _write_readme(output, use_patched_diffusers=use_patched_diffusers)
    return output


__all__ = ["convert_sr_checkpoint"]
