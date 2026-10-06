"""Kandinsky 6 video super-resolution nodes for ComfyUI."""

from pathlib import Path

import folder_paths
from server import PromptServer

from .kandinsky6_vsr.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .kandinsky6_vsr.register import register
from .model_downloads import register_routes

register()

if getattr(PromptServer, "instance", None) is not None:
    register_routes(
        PromptServer.instance.routes,
        "kandinsky6-sr",
        Path(__file__).parent / "example_workflows",
        Path(folder_paths.models_dir),
        {
            kind: list(folder_paths.folder_names_and_paths.get(kind, ([], set()))[0])
            for kind in ("diffusion_models", "text_encoders", "vae")
        },
    )

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
