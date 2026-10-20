"""Standalone LU + DiT tiled super-resolution inference."""

from __future__ import annotations

from .factory import load_sr_pipeline

__version__ = "1.1.0"

__all__ = ["load_sr_pipeline"]
