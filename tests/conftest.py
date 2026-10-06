"""Shared pytest setup: the ComfyUI node pack on ``sys.path`` and a tiny Diffusers-bundle factory."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from safetensors.torch import save_file

COMFYUI_PACK_DIR = Path(__file__).resolve().parent.parent / "comfyui"

if str(COMFYUI_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(COMFYUI_PACK_DIR))

SchedulerKind = Literal["piflow", "flow_matching"]
VaeLayout = Literal["flat", "nested"]
BundleFactory = Callable[..., Path]

WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
IN_VISUAL_DIM = 64
PIFLOW_GRID_POINTS = 10
KVAE_SCALING_FACTOR = 0.910344004631042
PIFLOW_SHIFT = 3.5
FLOW_MATCHING_SHIFT = 5.0
# The KVAE architecture as the native sidecar YAML spells it...
KVAE_ENCODER_PARAMS = {
    "in_channels": 3,
    "ch": 128,
    "ch_mult": [0.125, 1, 2, 4, 8],
    "num_res_blocks": 2,
    "resolution": 0,
    "z_channels": IN_VISUAL_DIM,
    "double_z": True,
    "temporal_compress_times": 4,
    "fix_pxs": True,
    "norm_type": "rms_norm",
    "downsample_version": 2,
    "temporal_compress_start_level": 1,
    "padding_mode": "zeros",
}
KVAE_DECODER_PARAMS = {
    "out_ch": 3,
    "ch": 256,
    "ch_mult": [0.0625, 1, 2, 4, 8],
    "num_res_blocks": 2,
    "resolution": 0,
    "z_channels": IN_VISUAL_DIM,
    "temporal_compress_times": 4,
    "norm_type": "rms_norm",
    "temporal_compress_start_level": 1,
    "padding_mode": "zeros",
}
KVAE_COMPONENT_HEADER = {
    "vae_type": "video-kvae",
    "scaling_factor": KVAE_SCALING_FACTOR,
    "spatial_factor": 16,
    "temporal_factor": 4,
}
# ...and as the published bundles spell it: one flat set of fields, the decoder's width prefixed.
KVAE_FLAT_CONFIG = {
    **KVAE_COMPONENT_HEADER,
    "in_channels": 3,
    "out_channels": 3,
    "z_channels": IN_VISUAL_DIM,
    "ch": 128,
    "ch_mult": [0.125, 1, 2, 4, 8],
    "decoder_ch": 256,
    "decoder_ch_mult": [0.0625, 1, 2, 4, 8],
    "num_res_blocks": 2,
    "resolution": 0,
    "padding_mode": "zeros",
    "temporal_compress_times": 4,
    "temporal_compress_start_level": 1,
    "norm_type": "rms_norm",
    "double_z": True,
    "downsample_version": 2,
    "fix_pxs": True,
}
KVAE_NESTED_CONFIG = {
    **KVAE_COMPONENT_HEADER,
    "encoder_config": KVAE_ENCODER_PARAMS,
    "decoder_config": KVAE_DECODER_PARAMS,
}
LU_MODEL = {"architecture": "multi_scale", "in_channels": IN_VISUAL_DIM, "hidden_channels": 8}

SR_PARAMS = {
    "cap_noise_timestep": False,
    "fps": 24,
    "lq_channel_noise_scale": 0.0,
    "lq_noise_scale": 0.7,
    "lq_noise_type": "ddpm",
    "scale_factor": {"512": [1.0, 2.0, 2.0]},
    "visual_size": [512],
}
PIFLOW_SCHEDULER = {
    "_class_name": "PiflowScheduler",
    "_diffusers_version": "0.41.0.dev0",
    "eps": 1e-06,
    "final_step_size_scale": 0.5,
    "n_grid": PIFLOW_GRID_POINTS,
    "nfe": 2,
    "num_policy_substeps": 128,
    "num_train_timesteps": 1000,
    "shift": PIFLOW_SHIFT,
}
FLOW_MATCHING_SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.39.0",
    "num_train_timesteps": 1000,
    "shift": FLOW_MATCHING_SHIFT,
}


def transformer_config(scheduler: SchedulerKind) -> dict[str, Any]:
    """``transformer/config.json`` as the converter writes it (JSON turns the resolution keys into strings)."""
    distilled = scheduler == "piflow"
    return {
        "attention_params": {"512": {"type": "nabla", "P": 0.8}, "1024": {"type": "flash", "window": 3}},
        "attribute_overrides": {},
        "axes_dims": [16, 24, 24],
        "in_visual_dim": IN_VISUAL_DIM,
        "instruct_type": "hybrid_anchor",
        "model_dim": 1792,
        "out_visual_dim": IN_VISUAL_DIM * PIFLOW_GRID_POINTS if distilled else IN_VISUAL_DIM,
        "sr_params": {**SR_PARAMS, "scheduler_scale": PIFLOW_SHIFT if distilled else FLOW_MATCHING_SHIFT},
        "use_text": False,
        "visual_cond": True,
    }


def write_component(directory: Path, config: dict[str, Any], tensors: dict[str, torch.Tensor]) -> None:
    """Write one Diffusers component: ``config.json`` next to the safetensors weights."""
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(directory / WEIGHTS_NAME))


@pytest.fixture
def make_bundle(tmp_path: Path) -> BundleFactory:
    """Return a factory writing a minimal ``Kandinsky6SRPipeline`` bundle and returning its root."""

    def build(scheduler: SchedulerKind = "piflow", name: str = "bundle", vae_layout: VaeLayout = "flat") -> Path:
        root = tmp_path / name
        write_component(root / "transformer", transformer_config(scheduler), {"out_layer.weight": torch.ones(2)})
        vae_config = KVAE_FLAT_CONFIG if vae_layout == "flat" else KVAE_NESTED_CONFIG
        write_component(root / "vae", vae_config, {"encoder.conv_in.weight": torch.zeros(1)})
        write_component(
            root / "latent_upscaler",
            {
                "models": [{"target_scale": "4x", "model": LU_MODEL}, {"target_scale": "2x", "model": LU_MODEL}],
                "scaling_factor": KVAE_SCALING_FACTOR,
            },
            {"_models.0.weight": torch.full((2,), 2.0), "_models.1.weight": torch.full((2,), 4.0)},
        )
        scheduler_config = PIFLOW_SCHEDULER if scheduler == "piflow" else FLOW_MATCHING_SCHEDULER
        (root / "scheduler").mkdir()
        (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler_config))
        (root / "model_index.json").write_text(json.dumps({"_class_name": "Kandinsky6SRPipeline"}))
        return root

    return build
