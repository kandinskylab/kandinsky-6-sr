"""SR NABLA at the CuteDSL block of 128.

Queries and keys are averaged over each 8×16 tile (128 tokens, two neighboring
8×8 fractal blocks in one frame). Softmax, top-cdf and the FlexAttention block
all use that grid, so the reduced matmul is four times smaller than a 64-token
pool. ``nabla_topcdf_sum`` keeps the slower path (softmax at 64, then a 2×2
sum) as the reference for the attention-error test.

``install_nabla_cutedsl`` points ``MultiheadSelfAttention.attention_flex`` at
this path. The factory does that when ``sr.nabla`` is ``cutedsl``.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
from loguru import logger
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention

POOL_BLOCK = 64
ATTN_BLOCK = 128
FRACTAL = 8

# Latent height, width of the three trained visual-512 bases (patch 1×1×1).
SR_LATENT_BASES: tuple[tuple[int, int], ...] = ((32, 32), (32, 48), (48, 32))


def sr_sequence_fits_cutedsl(frames: int, height: int, width: int) -> bool:
    """True when a fractal sequence can be blocked at 128 by pairing width tiles."""
    if frames < 1 or height % FRACTAL or width % FRACTAL:
        return False
    blocks_w = width // FRACTAL
    tokens = frames * height * width
    return blocks_w % 2 == 0 and tokens % ATTN_BLOCK == 0


def topcdf_mask(scores: Tensor, threshold: float) -> Tensor:
    """Keep the largest key-blocks whose softmax mass reaches ``threshold``.

    ``scores`` is ``(..., blocks, blocks)`` and each row sums to one.
    """
    order = scores.argsort(dim=-1)
    ranked = scores.gather(-1, order)
    keep = ranked.cumsum(-1) >= (1.0 - threshold)
    return keep.scatter(-1, order, keep)


def sum_pool_2x2(scores: Tensor) -> Tensor:
    """Sum a square score matrix over consecutive 2×2 blocks and renormalize.

    ``(..., n, n)`` with even ``n`` becomes ``(..., n/2, n/2)``. Dividing by 2
    restores rows that sum to one: each coarse row adds two fine rows.
    """
    *lead, n, m = scores.shape
    if n != m or n % 2:
        raise ValueError(f"score matrix must be square with an even side, got {tuple(scores.shape)}")
    half = n // 2
    paired = scores.reshape(*lead, half, 2, half, 2)
    return paired.sum(dim=(-3, -1)) / 2


def sta_mask(grid: tuple[int, int, int], window: tuple[int, int, int], device: torch.device) -> Tensor:
    """Sliding window on the ``(frame, row, column)`` block grid, shape ``(blocks, blocks)``."""
    frames, blocks_h, blocks_w = grid
    w_t, w_h, w_w = window
    blocks = frames * blocks_h * blocks_w
    t = torch.arange(frames, device=device)
    h = torch.arange(blocks_h, device=device)
    w = torch.arange(blocks_w, device=device)
    tt = (t[:, None] - t[None, :]).abs() <= w_t // 2
    hh = (h[:, None] - h[None, :]).abs() <= w_h // 2
    ww = (w[:, None] - w[None, :]).abs() <= w_w // 2
    cube = tt[:, None, None, :, None, None] & hh[None, :, None, None, :, None] & ww[None, None, :, None, None, :]
    return cube.reshape(blocks, blocks)


def coarsen_mask_2x2(mask: Tensor) -> Tensor:
    """OR-pool a square block mask by pairing consecutive query and key blocks.

    ``(..., n, n)`` with even ``n`` becomes ``(..., n/2, n/2)``. A coarse cell
    is kept when any of the four fine cells is kept.
    """
    *lead, n, m = mask.shape
    if n != m or n % 2:
        raise ValueError(f"block mask must be square with an even side, got {tuple(mask.shape)}")
    half = n // 2
    paired = mask.reshape(*lead, half, 2, half, 2)
    return paired.any(dim=-1).any(dim=-2)


def document_block_mask(seqlens: Tensor, frames: int, blocks_per_frame: int, device: torch.device) -> Tensor:
    """Block-diagonal mask on the 128-token grid, shape ``(blocks, blocks)``.

    ``seqlens`` is cumulative frame counts, the same tensor ``nablaT_v2_doc``
    reads. Each frame holds ``blocks_per_frame`` key blocks, so a tile boundary
    lands on a block boundary for the trained bases.
    """
    if blocks_per_frame < 1:
        raise ValueError(f"blocks_per_frame must be positive, got {blocks_per_frame}")
    lengths = torch.diff(seqlens.to(device=device, dtype=torch.long))
    if lengths.numel() == 0 or int(lengths.sum()) != frames or bool((lengths <= 0).any()):
        raise ValueError(f"visual_seqlens {seqlens.detach().cpu().tolist()} do not cover {frames} frames")
    eye = torch.eye(int(lengths.numel()), dtype=torch.bool, device=device)
    repeats = lengths * blocks_per_frame
    return eye.repeat_interleave(repeats, 0).repeat_interleave(repeats, 1)


def block_scores(
    query: Tensor,
    key: Tensor,
    pool_block: int = POOL_BLOCK,
    document: Tensor | None = None,
) -> Tensor:
    """Softmax of mean-pooled queries and keys. One row per ``pool_block`` tokens.

    ``document`` is a bool block mask. Its log is added before the softmax, so
    top-cdf spends its budget inside the tile, matching ``nablaT_v2_doc``.
    """
    tokens = query.shape[-2]
    if key.shape[-2] != tokens or tokens % pool_block:
        raise ValueError(f"expected a shared multiple of {pool_block} tokens, got q={tokens} k={key.shape[-2]}")

    def _pool(x: Tensor) -> Tensor:
        batch, heads, _, dim = x.shape
        return x.reshape(batch, heads, tokens // pool_block, pool_block, dim).mean(-2)

    pooled_key = _pool(key)
    logits = (_pool(query) @ pooled_key.transpose(-2, -1)) * pooled_key.shape[-1] ** -0.5
    if document is not None:
        logits = logits + document.to(dtype=logits.dtype).log()
    return torch.softmax(logits, dim=-1)


def _checked_grid(query: Tensor, key: Tensor, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    frames, height, width = shape
    if not sr_sequence_fits_cutedsl(frames, height, width):
        raise ValueError(f"sequence {shape} is not divisible into 128-token blocks")
    tokens = frames * height * width
    if query.shape[-2] != tokens or key.shape[-2] != tokens:
        raise ValueError(f"expected {tokens} tokens, got q={query.shape[-2]} k={key.shape[-2]}")
    return frames, height // FRACTAL, width // FRACTAL


def nabla_topcdf_or(
    query: Tensor,
    key: Tensor,
    shape: tuple[int, int, int],
    threshold: float = 0.8,
    sta: tuple[int, int, int] | None = (11, 7, 7),
) -> Tensor:
    """Top-cdf at 64, then OR of each 2×2 into a 128-token mask.

    Reference aggregation. The sort runs on the fine grid.
    """
    frames, blocks_h, blocks_w = _checked_grid(query, key, shape)
    mask = topcdf_mask(block_scores(query, key), threshold)
    if sta is not None:
        mask = mask | sta_mask((frames, blocks_h, blocks_w), sta, query.device)
    return coarsen_mask_2x2(mask)


def nabla_topcdf_sum(
    query: Tensor,
    key: Tensor,
    shape: tuple[int, int, int],
    threshold: float = 0.8,
    sta: tuple[int, int, int] | None = (11, 7, 7),
) -> Tensor:
    """Softmax at 64, sum each 2×2, then top-cdf. Reference for the 8×16 pool."""
    frames, blocks_h, blocks_w = _checked_grid(query, key, shape)
    mask = topcdf_mask(sum_pool_2x2(block_scores(query, key)), threshold)
    if sta is not None:
        window = sta_mask((frames, blocks_h, blocks_w), sta, query.device)
        mask = mask | coarsen_mask_2x2(window)
    return mask


def nabla_topcdf(  # noqa: PLR0913 — document lengths ride with the mask knobs
    query: Tensor,
    key: Tensor,
    shape: tuple[int, int, int],
    threshold: float = 0.8,
    sta: tuple[int, int, int] | None = (11, 7, 7),
    seqlens: Tensor | None = None,
) -> Tensor:
    """Mean-pool each 8×16 tile, then top-cdf on that 128-token grid.

    ``query`` and ``key`` are ``(batch, heads, tokens, dim)`` in fractal order
    ``(frame, 8×8-row, 8×8-column)``. ``shape`` is ``(frames, height, width)``
    in latent pixels. The returned mask is ``(batch, heads, tokens/128, tokens/128)``.

    ``seqlens`` are cumulative frame counts of packed tiles. One document
    (``[0, frames]``) leaves the mask unchanged. Several documents zero every
    cross-tile block, including STA cells that would otherwise cross the cut.
    """
    frames, blocks_h, blocks_w = _checked_grid(query, key, shape)
    _, height, width = shape
    document = None
    if seqlens is not None:
        document = document_block_mask(seqlens, frames, (height * width) // ATTN_BLOCK, query.device)
    mask = topcdf_mask(block_scores(query, key, ATTN_BLOCK, document), threshold)
    if sta is not None:
        window = sta_mask((frames, blocks_h, blocks_w), sta, query.device)
        mask = mask | coarsen_mask_2x2(window)
    if document is not None:
        mask = mask & document
    return mask


def block_mask_from_dense(mask: Tensor, block_size: int = ATTN_BLOCK) -> BlockMask:
    """Pack a dense ``(batch, heads, n, n)`` bool mask into a FlexAttention BlockMask."""
    counts = mask.sum(dim=-1).to(torch.int32)
    columns = mask.shape[-1]
    index = torch.arange(columns, device=mask.device).view(1, 1, 1, columns).expand_as(mask)
    filled = index.masked_fill(~mask, columns)
    indices = filled.gather(-1, filled.argsort(-1)).to(torch.int32)
    return BlockMask.from_kv_blocks(counts, indices, BLOCK_SIZE=block_size, mask_mod=None)


_CUTEDSL_KERNEL = {"BACKEND": "FLASH", "BLOCK_M": ATTN_BLOCK, "BLOCK_N": ATTN_BLOCK}


def flex_cutedsl(query: Tensor, key: Tensor, value: Tensor, block_mask: BlockMask) -> Tensor:
    """FlexAttention on the CuteDSL Flash kernel. Both block sizes are 128."""
    return flex_attention(query, key, value, block_mask=block_mask, kernel_options=_CUTEDSL_KERNEL)


@torch.compiler.disable  # the S/128 score matmul is not splittable under dynamic compile
def sr_nabla_block_mask(  # noqa: PLR0913 — document lengths ride with the mask knobs
    query: Tensor,
    key: Tensor,
    shape: tuple[int, int, int],
    threshold: float = 0.8,
    sta: tuple[int, int, int] | None = (11, 7, 7),
    seqlens: Tensor | None = None,
) -> BlockMask:
    """Block mask the SR DiT can hand to :func:`flex_cutedsl`."""
    mask = nabla_topcdf(query, key, shape, threshold, sta, seqlens)
    return block_mask_from_dense(mask, ATTN_BLOCK)


def _sta_window(sparse_params: dict[str, Any]) -> tuple[int, int, int] | None:
    if not sparse_params.get("add_sta", True):
        return None
    return (int(sparse_params["wT"]), int(sparse_params["wH"]), int(sparse_params["wW"]))


def attention_flex_cutedsl(  # noqa: PLR0913
    self: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sparse_params: dict[str, Any] | None = None,
    *,
    return_sparsity: bool = False,
) -> Tensor | tuple[Tensor, float]:
    """Drop-in ``attention_flex``: top-cdf at block 128, FlexAttention CuteDSL."""
    if sparse_params is None:
        raise ValueError("CuteDSL NABLA requires sparse_params")
    if sparse_params["attention_type"] != "nabla":
        raise ValueError(f"CuteDSL NABLA supports attention_type 'nabla', got {sparse_params['attention_type']!r}")
    if self.cached_k is not None:
        raise NotImplementedError("KV-cache on the flex/NABLA path is out of scope")

    query = query.unsqueeze(0).transpose(1, 2).contiguous()
    key = key.unsqueeze(0).transpose(1, 2).contiguous()
    value = value.unsqueeze(0).transpose(1, 2).contiguous()
    frames, height, width = sparse_params["visual_shape"]
    block_mask = sr_nabla_block_mask(
        query,
        key,
        (int(frames), int(height), int(width)),
        float(sparse_params["P"]),
        _sta_window(sparse_params),
        sparse_params["visual_seqlens"],
    )
    from kandinsky_sr.core.components.model.nn import flex  # noqa: PLC0415

    out = flex(query, key, value, block_mask=block_mask, kernel_options=_CUTEDSL_KERNEL)
    out = out.transpose(1, 2).squeeze(0).contiguous().flatten(-2, -1)
    if return_sparsity:
        sparsity = 100.0 * (1 - (1 - block_mask.sparsity() / 100) * (sparse_params["visual_seqlens"].shape[0] - 1))
        return out, sparsity
    return out


def fa4_installed() -> bool:
    """True when FlexAttention ``BACKEND=FLASH`` can import ``flash_attn.cute``."""
    try:
        return importlib.util.find_spec("flash_attn.cute") is not None
    except (ImportError, ValueError):
        return False


def resolve_nabla_backend(requested: str) -> str:
    """Return ``cutedsl`` only when flash-attn-4 is installed.

    Otherwise the block-64 NABLA stays in place. Dense flash attention then
    uses flash-attn 2, when that package is installed.
    """
    if requested != "cutedsl":
        return requested
    if fa4_installed():
        return "cutedsl"
    logger.warning(
        "SR NABLA CuteDSL needs flash-attn-4 (module flash_attn.cute). "
        "It is not installed, so attention stays on the block-64 NABLA. "
        "Dense flash paths use flash-attn 2 when the flash extra is installed."
    )
    return "flex"


def install_nabla_cutedsl() -> None:
    """Point SR self-attention at :func:`attention_flex_cutedsl`. Idempotent."""
    if install_nabla_cutedsl.done:  # type: ignore[attr-defined]
        return
    from kandinsky_sr.core.components.model.nn import MultiheadSelfAttention  # noqa: PLC0415

    MultiheadSelfAttention.attention_flex = attention_flex_cutedsl  # type: ignore[method-assign]
    install_nabla_cutedsl.done = True  # type: ignore[attr-defined]
    logger.info("SR NABLA: topcdf block 128 via FlexAttention BACKEND=FLASH")


install_nabla_cutedsl.done = False  # type: ignore[attr-defined]
