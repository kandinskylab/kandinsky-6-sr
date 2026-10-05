"""KVAE torch.compile defaults and the lazy flash-attn import.

What & why: the KVAE decode compiles static segment shapes by default (the
faster mode, measured x2: 45 s static vs 75-85 s dynamic on torch 2.10, and the
only one torch <= 2.9 can compile); ``KVAE_COMPILE_DYNAMIC=1`` opts into
dynamic shapes.
The DiT module must import without flash-attn, because the nabla text-free
models never call a flash path. How: pure unit tests over version strings and
environment variables; the flash import is exercised by importing the module.
Corner cases: env unset, ``0``, ``1`` and a stray value (only ``1`` enables).
"""

from __future__ import annotations

import importlib.util

import pytest

from kandinsky_sr.core.components.model import compiled_kvae, nn


def test_static_shapes_are_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KVAE_COMPILE_DYNAMIC", raising=False)
    assert compiled_kvae.kvae_compile_kwargs()["dynamic"] is False


@pytest.mark.parametrize(("env_value", "expected"), [("0", False), ("1", True), ("2", False)])
def test_env_opts_into_dynamic_shapes(monkeypatch: pytest.MonkeyPatch, env_value: str, expected: bool) -> None:
    monkeypatch.setenv("KVAE_COMPILE_DYNAMIC", env_value)
    assert compiled_kvae.kvae_compile_kwargs()["dynamic"] is expected


def test_flash_attn_is_only_needed_on_a_flash_path() -> None:
    """Importing the DiT blocks must not require flash-attn (module already imported above)."""
    if importlib.util.find_spec("flash_attn") is not None:
        pytest.skip("flash-attn installed; the lazy path cannot be observed")
    with pytest.raises(ImportError, match="kandinsky-6-sr\\[flash\\]"):
        nn.flash_attn_funcs()
