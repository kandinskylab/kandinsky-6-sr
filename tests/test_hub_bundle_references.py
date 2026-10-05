"""Model references that point at a Diffusers bundle: local directories and Hub repos.

What & why: ``checkpoint_path`` / ``vae_path`` / ``latent_upscaler_config`` may
name a ``Kandinsky6SRPipeline`` bundle — its root, one component folder, or the
Hub forms ``namespace/name`` and ``namespace/name/<component>`` — and must
resolve to what the native loaders take. A bundle given as the checkpoint also
supplies the VAE and the upscalers when those are left unset.

How: the ``make_bundle`` fixture as the local bundle; ``snapshot_download`` is
replaced by a recorder serving that bundle (or a native-style directory), so
nothing is downloaded and the requested file patterns can be asserted.

Corner cases: a native repo (no ``model_index.json``) keeps the old full
snapshot; a component named on a non-bundle repo; a component of the wrong
kind; a local sidecar prefix that merely looks like ``namespace/name/vae``;
``"none"`` for the upscaler is never replaced by the bundle's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kandinsky_sr.pipeline import hub
from kandinsky_sr.pipeline.components import resolve_model_references
from kandinsky_sr.pipeline.hub import (
    HubReference,
    is_hub_reference,
    parse_hub_reference,
    resolve_checkpoint_reference,
    resolve_kvae_reference,
    resolve_lu_bank_reference,
)

from .conftest import BundleFactory

BUNDLE_REPO = "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"
NATIVE_REPO = "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s"


class FakeHub:
    """Stands in for ``snapshot_download``: serves local directories and records what was asked for."""

    def __init__(self, repos: dict[str, Path]) -> None:
        self.repos = repos
        self.requests: list[tuple[str, tuple[str, ...] | None]] = []

    def __call__(self, repo_id: str, allow_patterns: list[str] | None = None, **_kwargs: Any) -> str:
        self.requests.append((repo_id, tuple(allow_patterns) if allow_patterns is not None else None))
        return str(self.repos[repo_id])


@pytest.fixture
def fake_hub(make_bundle: BundleFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    native = tmp_path / "native"
    native.mkdir()
    (native / "model.safetensors").write_bytes(b"x")
    (native / "config.yaml").write_text("dit: {}\n")
    fake = FakeHub({BUNDLE_REPO: make_bundle(), NATIVE_REPO: native})
    monkeypatch.setattr(hub, "snapshot_download", fake)
    monkeypatch.chdir(tmp_path)  # repo ids must not collide with paths under the working directory
    return fake


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (BUNDLE_REPO, HubReference(BUNDLE_REPO, None)),
        (f"{BUNDLE_REPO}/vae", HubReference(BUNDLE_REPO, "vae")),
        (f"{BUNDLE_REPO}/latent_upscaler", HubReference(BUNDLE_REPO, "latent_upscaler")),
        (f"{BUNDLE_REPO}/transformer", HubReference(BUNDLE_REPO, "transformer")),
        (f"{BUNDLE_REPO}/scheduler", None),
        ("/abs/path/to/vae", None),
        ("weights/kvae/kvae_20_v2", None),
        ("just-a-name", None),
    ],
)
def test_hub_references_are_recognised(
    value: str, expected: HubReference | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert parse_hub_reference(value) == expected
    assert is_hub_reference(value) == (expected is not None)


def test_an_existing_local_path_is_never_a_hub_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "bundle" / "vae").mkdir(parents=True)

    assert parse_hub_reference("models/bundle") is None
    assert parse_hub_reference("models/bundle/vae") is None


def test_a_local_sidecar_prefix_is_not_mistaken_for_a_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights" / "kvae").mkdir(parents=True)
    (tmp_path / "weights" / "kvae" / "vae.yaml").write_text("scaling_factor: 1.0\n")

    assert parse_hub_reference("weights/kvae/vae") is None


def test_local_bundle_root_resolves_every_model(make_bundle: BundleFactory) -> None:
    root = make_bundle()

    assert resolve_checkpoint_reference(str(root)) == str(root / "transformer")
    assert resolve_kvae_reference(str(root)) == str(root / "vae" / "kvae")
    assert resolve_lu_bank_reference(str(root)) == str(root / "latent_upscaler")


def test_local_component_directories_resolve_directly(make_bundle: BundleFactory) -> None:
    root = make_bundle()

    assert resolve_checkpoint_reference(str(root / "transformer")) == str(root / "transformer")
    assert resolve_kvae_reference(str(root / "vae")) == str(root / "vae" / "kvae")
    assert resolve_lu_bank_reference(str(root / "latent_upscaler")) == str(root / "latent_upscaler")


def test_bundle_without_the_requested_component_is_refused(make_bundle: BundleFactory) -> None:
    root = make_bundle()
    (root / "latent_upscaler" / "config.json").unlink()

    with pytest.raises(ValueError, match="no latent_upscaler component"):
        resolve_lu_bank_reference(str(root))


def test_bundle_repo_downloads_only_what_the_dit_needs(fake_hub: FakeHub) -> None:
    resolved = resolve_checkpoint_reference(BUNDLE_REPO)

    assert resolved == str(fake_hub.repos[BUNDLE_REPO] / "transformer")
    assert fake_hub.requests == [
        (BUNDLE_REPO, ("model_index.json",)),
        (BUNDLE_REPO, ("model_index.json", "transformer/*", "scheduler/*")),
    ]


@pytest.mark.parametrize("reference", [BUNDLE_REPO, f"{BUNDLE_REPO}/vae"])
def test_vae_is_taken_from_a_bundle_repo(fake_hub: FakeHub, reference: str) -> None:
    resolved = resolve_kvae_reference(reference)

    assert resolved == str(fake_hub.repos[BUNDLE_REPO] / "vae" / "kvae")
    assert fake_hub.requests[-1] == (BUNDLE_REPO, ("model_index.json", "vae/*"))


@pytest.mark.parametrize("reference", [BUNDLE_REPO, f"{BUNDLE_REPO}/latent_upscaler"])
def test_upscalers_are_taken_from_a_bundle_repo(fake_hub: FakeHub, reference: str) -> None:
    resolved = resolve_lu_bank_reference(reference)

    assert resolved == str(fake_hub.repos[BUNDLE_REPO] / "latent_upscaler")
    assert fake_hub.requests[-1] == (BUNDLE_REPO, ("model_index.json", "latent_upscaler/*"))


def test_native_repo_still_downloads_the_whole_snapshot(fake_hub: FakeHub) -> None:
    resolved = resolve_checkpoint_reference(NATIVE_REPO)

    assert resolved == str(fake_hub.repos[NATIVE_REPO])
    assert fake_hub.requests[-1] == (NATIVE_REPO, None)


def test_component_of_a_repo_that_is_not_a_bundle_is_refused(fake_hub: FakeHub) -> None:
    with pytest.raises(ValueError, match="not a Diffusers bundle"):
        resolve_kvae_reference(f"{NATIVE_REPO}/vae")


def test_component_of_the_wrong_kind_is_refused(fake_hub: FakeHub) -> None:
    with pytest.raises(ValueError, match="transformer"):
        resolve_checkpoint_reference(f"{BUNDLE_REPO}/vae")


def test_bundle_checkpoint_supplies_the_unset_vae_and_upscalers(fake_hub: FakeHub) -> None:
    root = fake_hub.repos[BUNDLE_REPO]

    assert resolve_model_references(BUNDLE_REPO, None, None) == (
        str(root / "transformer"),
        str(root / "vae" / "kvae"),
        str(root / "latent_upscaler"),
    )


def test_local_bundle_checkpoint_supplies_the_unset_vae_and_upscalers(make_bundle: BundleFactory) -> None:
    root = make_bundle()

    assert resolve_model_references(str(root), None, None) == (
        str(root / "transformer"),
        str(root / "vae" / "kvae"),
        str(root / "latent_upscaler"),
    )


def test_disabled_upscaler_stays_disabled_with_a_bundle(make_bundle: BundleFactory) -> None:
    root = make_bundle()

    assert resolve_model_references(str(root), None, "none")[2] == "none"


def test_explicit_vae_and_upscalers_win_over_the_bundle(make_bundle: BundleFactory) -> None:
    first, second = make_bundle(name="first"), make_bundle(name="second")

    _, vae, upscalers = resolve_model_references(str(first), str(second / "vae"), str(second))

    assert vae == str(second / "vae" / "kvae")
    assert upscalers == str(second / "latent_upscaler")


def test_native_checkpoint_supplies_no_vae(fake_hub: FakeHub) -> None:
    assert resolve_model_references(NATIVE_REPO, None, "none")[1:] == (None, "none")
