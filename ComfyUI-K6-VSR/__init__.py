"""ComfyUI-K6-VSR: Kandinsky 6 video super-resolution nodes.

The heavy lifting lives in the ``kandinsky-6-sr`` package (import name
``kandinsky_sr``); this pack only adapts it to ComfyUI's node contract. A
missing package is reported as one clear log line instead of a traceback deep
inside the model code.
"""

from __future__ import annotations

import logging

try:
    import kandinsky_sr  # noqa: F401 - presence check only
except ImportError as exc:
    MESSAGE = (
        "ComfyUI-K6-VSR needs the 'kandinsky-6-sr' package in ComfyUI's Python environment: "
        'pip install "kandinsky-6-sr @ git+https://github.com/kandinskylab/kandinsky-6-sr"'
    )
    logging.getLogger(__name__).error("[K6-VSR] %s", MESSAGE)
    raise ImportError(MESSAGE) from exc

from .k6_vsr_comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
