"""Free the GPU memory of a pipeline that ComfyUI still references.

ComfyUI caches a node's output by its inputs and keeps the previous output
alive until the current run finishes. When any loader input changes, the new
pipeline would therefore be built next to the old one; the old weights are
dropped here first.
"""

from __future__ import annotations

import gc
from typing import Any

import torch

MODULE_ATTRIBUTES = ("dit", "vae", "latent_upscaler")


def free_module_weights(module: torch.nn.Module) -> None:
    """Replace every parameter and buffer of ``module`` with an empty CPU tensor."""
    for tensor in (*module.parameters(), *module.buffers()):
        tensor.data = torch.empty(0)


def release_pipeline(pipeline: Any) -> None:
    """Drop the weights of a pipeline's DiT, VAE and latent upscaler and return the memory to the allocator."""
    for name in MODULE_ATTRIBUTES:
        module = getattr(pipeline, name, None)
        if isinstance(module, torch.nn.Module):
            free_module_weights(module)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
