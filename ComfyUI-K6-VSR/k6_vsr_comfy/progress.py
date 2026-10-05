"""Denoising progress of an SR run shown in the ComfyUI interface."""

from __future__ import annotations

from typing import Any

try:
    from comfy import utils as comfy_utils  # ComfyUI's progress API; absent outside of ComfyUI
except ImportError:
    comfy_utils = None


class ComfyProgress:
    """``ProgressReporter`` backed by ``comfy.utils.ProgressBar`` (node + top-bar progress).

    Outside of ComfyUI (tests, notebooks) the reporter is a no-op.
    """

    def __init__(self) -> None:
        self.bar: Any | None = None

    def start(self, total_tiles: int, steps_per_tile: int) -> None:
        if comfy_utils is None:
            return
        self.bar = comfy_utils.ProgressBar(total_tiles * steps_per_tile)

    def update(self, n: int = 1) -> None:
        if self.bar is not None:
            self.bar.update(n)

    def close(self) -> None:
        self.bar = None
