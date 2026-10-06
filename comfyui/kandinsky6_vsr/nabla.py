"""Native NABLA sparse attention for the Kandinsky6 SR ComfyUI port.

Only PyTorch flex-attention and the content-dependent mask builders are
compiled. The diffusion model itself remains a regular ComfyUI-managed model,
so weight patching, offloading and first-load behaviour stay intact.
"""

from __future__ import annotations

import logging
import threading

import torch
from torch.nn.attention.flex_attention import flex_attention

from .runtime.core.components.model.utils import (
    FRACTAL_BLOCK_SIZE,
    local_merge,
    local_patching,
    nablaT_v2_doc,
)

LOGGER = logging.getLogger(__name__)

_flex = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs", dynamic=True)
_warmup_lock = threading.Lock()
_warmed_devices: set[str] = set()


def validate_shape(shape: tuple[int, int, int]) -> None:
    """Validate the token grid expected by the released 8x8 NABLA layout."""
    _t, height, width = shape
    block = FRACTAL_BLOCK_SIZE
    if height % block or width % block:
        raise ValueError(
            f"NABLA requires token height/width divisible by {block}, got {height}x{width}."
        )


def fractal_flatten_batch(x: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Convert ``[B,T,H,W,...]`` raster tensors to the canonical 8x8 order."""
    validate_shape(shape)
    block = FRACTAL_BLOCK_SIZE
    return torch.stack(
        [local_patching(sample, shape, (1, block, block), dim=0).flatten(0, 1) for sample in x],
        dim=0,
    )


def fractal_flatten_grid(x: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Convert one ``[T,H,W,...]`` raster grid to canonical flattened order."""
    validate_shape(shape)
    block = FRACTAL_BLOCK_SIZE
    return local_patching(x, shape, (1, block, block), dim=0).flatten(0, 1)


def fractal_unflatten_batch(x: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Restore canonical flattened tokens to ``[B,T,H,W,...]`` raster order."""
    validate_shape(shape)
    block = FRACTAL_BLOCK_SIZE
    return torch.stack(
        [local_merge(sample.reshape(-1, block**2, *sample.shape[1:]), shape, (1, block, block), dim=0) for sample in x],
        dim=0,
    )


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    shape: tuple[int, int, int],
    config: dict,
) -> torch.Tensor:
    """Run the canonical packed-document NABLA flex-attention path."""
    validate_shape(shape)
    batch, sequence, heads, head_dim = query.shape
    q = query.reshape(batch * sequence, heads, head_dim).unsqueeze(0).transpose(1, 2).contiguous()
    k = key.reshape(batch * sequence, heads, head_dim).unsqueeze(0).transpose(1, 2).contiguous()
    v = value.reshape(batch * sequence, heads, head_dim).unsqueeze(0).transpose(1, 2).contiguous()

    duration, height, width = shape
    visual_seqlens = torch.arange(
        0,
        (batch + 1) * duration,
        duration,
        dtype=torch.int32,
        device=query.device,
    )
    block_mask = nablaT_v2_doc(
        q,
        k,
        visual_seqlens,
        batch * duration,
        height // FRACTAL_BLOCK_SIZE,
        width // FRACTAL_BLOCK_SIZE,
        wT=int(config["wT"]),
        wH=int(config["wH"]),
        wW=int(config["wW"]),
        thr=float(config["P"]),
        add_sta=bool(config["add_sta"]),
        method=str(config.get("method", "topcdf")),
    )
    out = _flex(
        q,
        k,
        v,
        block_mask=block_mask,
        kernel_options={"BLOCK_M": 64, "BLOCK_N": 64},
    )
    return out.transpose(1, 2).squeeze(0).reshape(batch, sequence, heads * head_dim).contiguous()


@torch.no_grad()
def warmup(device: torch.device | str, config: dict, *, heads: int, head_dim: int) -> None:
    """Compile the dynamic NABLA kernels once on a tiny representative grid."""
    device = torch.device(device)
    if device.type != "cuda":
        return
    key = str(device)
    with _warmup_lock:
        if key in _warmed_devices:
            return
        LOGGER.info("Kandinsky6 SR: warming NABLA flex-attention kernels on %s", device)
        # 121 input frames become 31 KVAE latents. Torch 2.8 specializes the
        # mask builder for the duration, so warm the release/default duration
        # rather than paying that compile inside the first real DiT call.
        shape = (31, 32, 32)
        q = torch.zeros((1, 31 * 1024, heads, head_dim), device=device, dtype=torch.bfloat16)
        out = attention(q, q, q, shape, config)
        torch.cuda.synchronize(device)
        del q, out
        torch.cuda.empty_cache()
        _warmed_devices.add(key)
        LOGGER.info("Kandinsky6 SR: NABLA kernels are ready")
