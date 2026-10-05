"""Spatial padding of the latent-upscaler path's source.

What & why: the LU path encodes the whole clip with the KVAE (whose encoder
halves the spatial dims four times) and cuts latent tiles of ``base // scale``
pixels, so the source must be a multiple of the VAE stride and at least one
tile on each axis. Real clips are neither (568x320, 426x240, 240-high at x2)
and used to crash inside the encoder or hand the DiT a clamped tile. The
contract: (1) ``source_padding`` reports how much bottom/right padding brings
a source onto both constraints, iterating when the padded aspect picks a
different base, (2) padding mirrors the source (reflect) and only falls back
to edge replication when the source is smaller than the strip, (3) latents
are padded in latent units, (4) ``crop`` removes exactly ``padding * scale``
from the stitched SR, (5) aligned sources are returned untouched.

How: pure CPU unit tests on small tensors; no model.

Corner cases: one axis aligned and the other not; a source smaller than the
tile on both axes; a tiny source (replicate fallback); scale 1 vs 4 on the
crop; a ``min_hw_for`` that changes with the aspect.
"""

from __future__ import annotations

import pytest
import torch

from kandinsky_sr.pipeline.source_padding import SourcePadding, padded_source_hw, source_padding

STRIDE = 16
X2_TILE = (256, 384)


def x2_tile(_h: int, _w: int) -> tuple[int, int]:
    return X2_TILE


@pytest.mark.parametrize(
    ("source_hw", "expected_hw"),
    [
        ((320, 568), (320, 576)),  # stride only
        ((240, 432), (256, 432)),  # tile only
        ((240, 426), (256, 432)),  # both at once
        ((200, 300), (256, 384)),  # smaller than the tile on both axes
        ((320, 576), (320, 576)),  # already fine
    ],
)
def test_padded_source_hw_meets_stride_and_tile(source_hw: tuple[int, int], expected_hw: tuple[int, int]) -> None:
    assert padded_source_hw(*source_hw, STRIDE, x2_tile) == expected_hw
    padding = source_padding(*source_hw, STRIDE, x2_tile)
    assert padding.padded_hw(source_hw) == expected_hw


def test_padded_source_hw_without_tile_constraint_is_stride_only() -> None:
    assert padded_source_hw(240, 426, STRIDE, None) == (240, 432)


def test_padded_source_hw_iterates_when_the_padded_aspect_changes_the_tile() -> None:
    """A tile function that depends on the aspect is re-evaluated on the padded size until stable."""

    def tile_for(h: int, w: int) -> tuple[int, int]:
        return (128, 192) if w / h > 1.4 else (192, 192)  # noqa: PLR2004

    # 100x150 (1.5) -> tile 128x192 -> 128x192 (1.5) stable.
    assert padded_source_hw(100, 150, STRIDE, tile_for) == (128, 192)
    # 100x140 (1.4) -> tile 192x192 -> 192x192 (1.0) -> still 192x192 stable.
    assert padded_source_hw(100, 140, STRIDE, tile_for) == (192, 192)


def test_aligned_source_is_returned_untouched() -> None:
    video = torch.randint(0, 256, (2, 3, 256, 384), dtype=torch.uint8)
    padding = source_padding(256, 384, STRIDE, x2_tile)
    assert not padding.any
    assert padding.apply_to_video(video) is video
    assert padding.crop(video.permute(1, 0, 2, 3), 2) is not None


def test_video_padding_mirrors_the_source() -> None:
    torch.manual_seed(0)
    video = torch.randint(0, 256, (2, 3, 24, 30), dtype=torch.uint8)
    padding = source_padding(24, 30, STRIDE, None)

    padded = padding.apply_to_video(video)

    assert padding == SourcePadding(bottom=8, right=2)
    assert padded.shape == (2, 3, 32, 32)
    assert padded.dtype == torch.uint8
    assert torch.equal(padded[..., :24, :30], video)
    assert torch.equal(padded[..., :24, 30], video[..., 28])  # reflect: mirrors around the edge pixel
    assert torch.equal(padded[..., :24, 31], video[..., 27])
    assert torch.equal(padded[..., 24, :30], video[..., 22, :])


def test_tiny_source_falls_back_to_edge_replication() -> None:
    """A source smaller than the strip cannot be reflected; its edge is repeated instead of failing."""
    video = torch.randint(0, 256, (1, 3, 4, 4), dtype=torch.uint8)
    padding = source_padding(4, 4, STRIDE, x2_tile)

    padded = padding.apply_to_video(video)

    assert padded.shape == (1, 3, 256, 384)
    assert torch.equal(padded[..., :4, :4], video)
    assert torch.equal(padded[..., 200, 200], video[..., 3, 3])


def test_latent_padding_uses_latent_units() -> None:
    latent = torch.randn(3, 64, 15, 27)  # 240x432 pixels at stride 16
    padding = source_padding(240, 432, STRIDE, x2_tile)

    padded = padding.apply_to_latent(latent, STRIDE)

    assert padding == SourcePadding(bottom=16, right=0)
    assert padded.shape == (3, 64, 16, 27)
    assert torch.equal(padded[..., :15, :], latent)
    assert torch.equal(padded[..., 15, :], latent[..., 13, :])  # reflect in latent rows


@pytest.mark.parametrize("scale", [1, 4])
def test_crop_restores_source_times_scale(scale: int) -> None:
    padding = SourcePadding(bottom=6, right=8)
    frames = torch.randint(0, 256, (3, 2, 256 * scale, 576 * scale), dtype=torch.uint8)

    cropped = padding.crop(frames, scale)

    assert cropped.shape == (3, 2, 250 * scale, 568 * scale)
    assert torch.equal(cropped, frames[..., : 250 * scale, : 568 * scale])
