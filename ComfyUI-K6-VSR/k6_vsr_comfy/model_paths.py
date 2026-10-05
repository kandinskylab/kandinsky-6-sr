"""Model reference resolution for the ComfyUI nodes.

A model reference is a Hugging Face reference (a repo id, or a component of a
Diffusers bundle repo such as ``namespace/name/vae``; resolved by
``kandinsky_sr`` at load time), an absolute local path, or a name under
ComfyUI's ``models/kandinsky_vsr`` folder. Anything else is handed to
``kandinsky_sr`` verbatim, so every path form the ``kandy-sr`` CLI accepts
works here too.
"""

from __future__ import annotations

from pathlib import Path

from kandinsky_sr.pipeline.hub import is_hub_reference

try:
    import folder_paths  # ComfyUI's model directory registry; absent outside of ComfyUI
except ImportError:
    folder_paths = None

MODEL_FOLDER = "kandinsky_vsr"


def comfy_models_directory() -> Path | None:
    """Return ``<ComfyUI>/models/kandinsky_vsr``, or ``None`` outside of ComfyUI."""
    if folder_paths is None:
        return None
    return Path(folder_paths.models_dir) / MODEL_FOLDER


def resolve_model_reference(value: str) -> str:
    """Resolve a node's model reference to what ``kandinsky_sr`` expects.

    Raises:
        ValueError: If the reference is empty.
    """
    reference = value.strip()
    if not reference:
        raise ValueError("model reference is empty: pass a Hugging Face repo id or a local path")
    if reference.lower() == "none" or is_hub_reference(reference) or Path(reference).is_absolute():
        return reference
    models_dir = comfy_models_directory()
    if models_dir is not None and (models_dir / reference).exists():
        return str(models_dir / reference)
    return reference


def resolve_optional_model_reference(value: str) -> str | None:
    """Resolve a reference that may be left empty; empty means "take it from the checkpoint's bundle"."""
    return resolve_model_reference(value) if value.strip() else None
