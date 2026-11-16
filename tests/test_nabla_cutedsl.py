"""Geometry and mask shape of the 128-block SR NABLA."""

from __future__ import annotations

import torch

from kandinsky_sr.core.components.model import nabla_cutedsl
from kandinsky_sr.core.components.model.nabla_cutedsl import (
    ATTN_BLOCK,
    SR_LATENT_BASES,
    coarsen_mask_2x2,
    install_nabla_cutedsl,
    nabla_topcdf,
    nabla_topcdf_sum,
    sr_nabla_block_mask,
    sr_sequence_fits_cutedsl,
)

_ODD_LATENT_FRAMES = (31, 61)
_ATTN_REL_L2 = 0.05
_MASK_DISAGREE = 0.01


def test_trained_bases_divide_by_128() -> None:
    for height, width in SR_LATENT_BASES:
        for frames in (1, *_ODD_LATENT_FRAMES):
            assert sr_sequence_fits_cutedsl(frames, height, width)


def test_temporal_pair_does_not_fit_odd_clip() -> None:
    for height, width in SR_LATENT_BASES:
        for frames in _ODD_LATENT_FRAMES:
            assert frames % 2 == 1
            assert (frames * height * width) % ATTN_BLOCK == 0


def test_coarsen_is_or_of_two_by_two() -> None:
    fine = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    coarse = coarsen_mask_2x2(fine)
    assert coarse.tolist() == [[True, False], [False, True]]


def test_topcdf_mask_is_half_the_fine_grid() -> None:
    frames, height, width, dim = 1, 16, 16, 8
    tokens = frames * height * width
    query = torch.randn(1, 2, tokens, dim)
    key = torch.randn(1, 2, tokens, dim)
    shape = (frames, height, width)
    mask = nabla_topcdf(query, key, shape, threshold=0.8, sta=None)
    assert mask.shape == (1, 2, tokens // ATTN_BLOCK, tokens // ATTN_BLOCK)
    assert mask.dtype == torch.bool
    block_mask = sr_nabla_block_mask(query, key, shape, sta=None)
    assert block_mask.BLOCK_SIZE == (ATTN_BLOCK, ATTN_BLOCK)
    assert block_mask.shape[-1] == tokens


def _blocked_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Dense attention with a 128-token block mask expanded back to tokens."""
    block = query.shape[-2] // mask.shape[-1]
    token_mask = mask.repeat_interleave(block, -2).repeat_interleave(block, -1)
    scores = (query @ key.transpose(-2, -1)) * query.shape[-1] ** -0.5
    weights = torch.softmax(scores.masked_fill(~token_mask, float("-inf")), dim=-1)
    return weights @ value


def _local_block_qk(tokens: int, dim: int, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each 64-token block matches itself and its two neighbors. ``scale`` peaks the softmax."""
    prototypes = torch.nn.functional.normalize(torch.randn(tokens // 64, dim), dim=-1)
    mixed = prototypes + 0.5 * torch.roll(prototypes, 1, 0) + 0.5 * torch.roll(prototypes, -1, 0)
    query = torch.nn.functional.normalize(mixed, dim=-1).repeat_interleave(64, 0)
    key = prototypes.repeat_interleave(64, 0)
    value = torch.randn(tokens, dim)
    return query[None, None] * scale, key[None, None] * scale, value[None, None]


def test_pool128_stays_close_to_sum() -> None:
    """Attention error of the 8×16 mean-pool against softmax-at-64 then 2×2 sum."""
    torch.manual_seed(0)
    frames, height, width, dim = 4, 32, 32, 32
    tokens = frames * height * width
    query, key, value = _local_block_qk(tokens, dim, scale=8.0)
    shape = (frames, height, width)
    pooled = nabla_topcdf(query, key, shape, sta=None)
    summed = nabla_topcdf_sum(query, key, shape, sta=None)
    left = _blocked_attention(query, key, value, pooled)
    right = _blocked_attention(query, key, value, summed)
    rel = (left - right).norm() / right.norm()
    assert rel < _ATTN_REL_L2, rel.item()
    assert (pooled != summed).float().mean() < _MASK_DISAGREE


def test_cutedsl_falls_back_to_flex_without_fa4(monkeypatch) -> None:
    monkeypatch.setattr(nabla_cutedsl, "fa4_installed", lambda: False)
    assert nabla_cutedsl.resolve_nabla_backend("cutedsl") == "flex"
    assert nabla_cutedsl.resolve_nabla_backend("flex") == "flex"
    monkeypatch.setattr(nabla_cutedsl, "fa4_installed", lambda: True)
    assert nabla_cutedsl.resolve_nabla_backend("cutedsl") == "cutedsl"


def test_install_points_sr_attention_at_cutedsl() -> None:
    from kandinsky_sr.core.components.model.nn import MultiheadSelfAttention  # noqa: PLC0415

    original = MultiheadSelfAttention.attention_flex
    install_nabla_cutedsl.done = False  # type: ignore[attr-defined]
    try:
        install_nabla_cutedsl()
        assert MultiheadSelfAttention.attention_flex is nabla_cutedsl.attention_flex_cutedsl
        install_nabla_cutedsl()
        assert MultiheadSelfAttention.attention_flex is nabla_cutedsl.attention_flex_cutedsl
    finally:
        MultiheadSelfAttention.attention_flex = original
        install_nabla_cutedsl.done = False  # type: ignore[attr-defined]
