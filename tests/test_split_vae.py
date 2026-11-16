"""Contract of the split SR VAE wrapper. Weight loading stays on the GPU check."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from kandinsky_sr.core.components.split_vae.adapter import SplitSRVAE

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


def test_fitted_temporal_assets_ship_with_the_package() -> None:
    root = Path(__file__).resolve().parents[1]
    assets = root / "src" / "kandinsky_sr" / "core" / "components" / "split_vae" / "assets"
    assert (assets / "t4hw16ch256.pt").is_file()
    assert (assets / "t4hw16ch512.pt").is_file()
