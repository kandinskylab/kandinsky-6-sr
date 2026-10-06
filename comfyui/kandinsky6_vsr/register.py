"""Register the Kandinsky 6 SR DiT as a ComfyUI model type and checkpoint detector."""

from functools import wraps

import comfy.model_detection
import comfy.supported_models

from .checkpoint_keys import native_dit_state
from .sr_contract import DIT_CONFIG, PIFLOW_SAMPLING
from .supported_model import SR_IMAGE_MODEL, Kandinsky6SR

_DETECTOR_MARKER = "_kandinsky6_sr_detector"
_SR_MARKER_KEYS = (
    "pooled_bias",
    "visual_transformer_blocks.0.visual_modulation.out_layer.weight",
)


def is_sr_state_dict(state_dict, key_prefix):
    return all(f"{key_prefix}{name}" in state_dict for name in _SR_MARKER_KEYS)


def detect_sr_dit(state_dict, key_prefix):
    """Return the ComfyUI ``unet_config`` of a released Kandinsky 6 SR DiT."""
    state_dict = native_dit_state(state_dict)

    def shape(name):
        return tuple(state_dict[f"{key_prefix}{name}"].shape)

    blocks = 0
    while f"{key_prefix}visual_transformer_blocks.{blocks}.visual_modulation.out_layer.weight" in state_dict:
        blocks += 1
    model_dim, visual_input_dim = shape("visual_embeddings.in_layer.weight")
    found = {
        "model_dim": model_dim,
        "visual_embed_dim": visual_input_dim,
        "time_dim": shape("pooled_bias")[0],
        "ff_dim": shape("visual_transformer_blocks.0.feed_forward.in_layer.weight")[0],
        "head_dim": shape("visual_transformer_blocks.0.self_attention.query_norm.weight")[0],
        "num_visual_blocks": blocks,
    }
    expected = {
        "model_dim": int(DIT_CONFIG["model_dim"]),
        "visual_embed_dim": 2 * int(DIT_CONFIG["in_visual_dim"]) + 1,
        "time_dim": int(DIT_CONFIG["time_dim"]),
        "ff_dim": int(DIT_CONFIG["ff_dim"]),
        "head_dim": sum(int(value) for value in DIT_CONFIG["axes_dims"]),
        "num_visual_blocks": int(DIT_CONFIG["num_visual_blocks"]),
    }
    mismatches = [
        f"{name}: checkpoint={found[name]!r}, release={value!r}"
        for name, value in expected.items()
        if found[name] != value
    ]
    output_dim = shape("out_layer.out_layer.weight")[0]
    out_visual_dim = int(DIT_CONFIG["out_visual_dim"])
    n_grid = output_dim // out_visual_dim
    if output_dim % out_visual_dim or n_grid not in (1, int(PIFLOW_SAMPLING["n_grid"])):
        mismatches.append(f"output head: {output_dim} channels")
    if f"{key_prefix}text_embeddings.in_layer.weight" in state_dict:
        mismatches.append("use_text: the released SR DiT is text-free")
    if mismatches:
        raise ValueError(
            f"Kandinsky 6 SR checkpoint does not match the released architecture ({'; '.join(mismatches)})."
        )
    return {"image_model": SR_IMAGE_MODEL, "n_grid": n_grid}


def _register_model(model_class):
    # A reloaded node pack creates a new class; replace the old entry in place.
    for index, model in enumerate(comfy.supported_models.models):
        if model.__module__ == model_class.__module__ and model.__name__ == model_class.__name__:
            comfy.supported_models.models[index] = model_class
            break
    else:
        comfy.supported_models.models.append(model_class)


def register():
    _register_model(Kandinsky6SR)

    current = comfy.model_detection.detect_unet_config
    wrapped = current
    while wrapped is not None:
        if getattr(wrapped, _DETECTOR_MARKER, False):
            return
        wrapped = getattr(wrapped, "__wrapped__", None)

    @wraps(current)
    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        if is_sr_state_dict(state_dict, key_prefix):
            return detect_sr_dit(state_dict, key_prefix)
        return current(state_dict, key_prefix, *args, **kwargs)

    setattr(detect_unet_config, _DETECTOR_MARKER, True)
    comfy.model_detection.detect_unet_config = detect_unet_config
