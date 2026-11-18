"""Contract of the split SR VAE wrapper. Weight loading stays on the GPU check."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kandinsky_sr.core.components.split_vae.adapter import SplitSRVAE
from kandinsky_sr.core.components.split_vae.split_kvae import (
    _check_fitted_temporal,
    get_n_tile_log,
    read_kvae_state,
    spatial_after_downsamples,
)

_PIXEL_MIDPOINT = 128.0


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=0.5)
        self.weight = nn.Parameter(torch.zeros(1))

    def encode(self, x: torch.Tensor, exact: bool = True) -> torch.Tensor:
        del exact
        return x[:, :1]

    def decode(self, z: torch.Tensor, seg_len: int = 16) -> torch.Tensor:
        del seg_len
        return z + 1


def test_split_sr_vae_matches_pipeline_calls() -> None:
    vae = SplitSRVAE(_Inner())
    latent, splits = vae.encode(torch.zeros(1, 4))
    assert splits is None
    assert tuple(latent.shape) == (1, 1)
    decoded = vae.decode(torch.zeros(1, 2))
    assert tuple(decoded.sample.shape) == (1, 2)
    assert torch.equal(decoded.sample, torch.ones(1, 2))
    assert float(vae.normalize_data(torch.tensor(_PIXEL_MIDPOINT))) == 0.0
    assert float(vae.denormalize_data(torch.tensor(0.0))) == _PIXEL_MIDPOINT


def test_split_sr_vae_encodes_a_batch_one_video_at_a_time() -> None:
    seen: list[int] = []

    class _One(_Inner):
        def encode(self, x: torch.Tensor, exact: bool = True) -> torch.Tensor:
            del exact
            seen.append(int(x.shape[0]))
            if x.shape[0] != 1:
                raise AssertionError(f"expected one video, got {x.shape[0]}")
            return x[:, :1] + x.sum()

    vae = SplitSRVAE(_One())
    videos = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    latent, splits = vae.encode(videos)
    alone = torch.cat([vae.vae.encode(videos[index : index + 1]) for index in range(2)], dim=0)
    assert splits is None
    assert seen[:2] == [1, 1]
    assert torch.equal(latent, alone)


def test_fitted_temporal_assets_ship_with_the_package() -> None:
    root = Path(__file__).resolve().parents[1]
    assets = root / "src" / "kandinsky_sr" / "core" / "components" / "split_vae" / "assets"
    assert (assets / "t4hw16ch256.pt").is_file()
    assert (assets / "t4hw16ch512.pt").is_file()
    assert (assets / "t4hw16ch256.json").is_file()
    assert (assets / "t4hw16ch512.json").is_file()


def test_large_frame_estimate_skips_the_missing_split_level() -> None:
    four_tiles = 2
    assert get_n_tile_log((121, 1440, 2560), 128, True) > four_tiles
    assert get_n_tile_log((121, 1080, 1920), 128, True) <= four_tiles


def test_outer_halo_lands_on_the_latent_grid() -> None:
    downs = 4
    spatial = 2**downs
    length = 2560
    halo = 32
    mid = (length // 2) // spatial * spatial
    extra = spatial_after_downsamples(mid + halo, downs) - spatial_after_downsamples(mid, downs)
    assert extra == halo // spatial
    kept = spatial_after_downsamples(mid, downs) + spatial_after_downsamples(length - mid, downs)
    assert kept == spatial_after_downsamples(length, downs)


def test_training_ckpt_keys_are_flattened(tmp_path: Path) -> None:
    path = tmp_path / "kvae.ckpt"
    weight = torch.zeros(1)
    torch.save(
        {
            "state_dict": {
                "loss.foo": weight,
                "decoder.up.0.upsample.weight": weight,
                "decoder.conv_b.conv.weight": weight,
                "encoder.down.1.downsample.temporal_conv.0.conv.weight": weight,
                "regularizer.sample.temporal_conv.conv.bias": weight,
            }
        },
        path,
    )
    state = read_kvae_state(str(path))
    assert "loss.foo" not in state
    assert "decoder.up.0.upsample.weight" not in state
    assert "decoder.conv_b.weight" in state
    assert "encoder.down.1.downsample.temporal_conv.0.conv.weight" in state
    assert "regularizer.sample.temporal_conv.bias" in state


def test_fitted_temporal_checksum_is_required(tmp_path: Path) -> None:
    fitted = tmp_path / "t4hw16ch256.pt"
    fitted.write_bytes(b"")
    source = {"encoder.down.1.downsample.temporal_conv.0.conv.weight": torch.zeros(1)}
    prefix = "encoder.down.1.downsample.temporal_conv."
    with pytest.raises(FileNotFoundError):
        _check_fitted_temporal(fitted, source, prefix)
    (tmp_path / "t4hw16ch256.json").write_text(
        '{"source_weights_sha256": "' + ("0" * 64) + '"}\n'
    )
    with pytest.raises(ValueError, match="do not match"):
        _check_fitted_temporal(fitted, source, prefix)
