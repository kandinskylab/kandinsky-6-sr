"""ComfyUI nodes: load the Kandinsky 6 VSR pipeline, then upscale an ``IMAGE`` batch with it."""

from __future__ import annotations

from typing import Any, Literal

import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from kandinsky_sr.constants import MAX_NUM_FRAMES, TARGET_FPS, TARGET_RESOLUTIONS
from kandinsky_sr.pipeline.components import scale_factor_for
from kandinsky_sr.pipeline.config import SRConfig, VaeBackend
from kandinsky_sr.pipeline.factory import load_sr_pipeline
from kandinsky_sr.pipeline.output_resize import resize_to_target, resolve_target_hw
from kandinsky_sr.pipeline.warmup import log_sampling_mode, warmup

from .frames import comfy_images_to_tchw_uint8, prepare_source_clip, sr_frames_to_comfy_images, trim_comfy_audio
from .model_paths import resolve_model_reference
from .progress import ComfyProgress
from .release import release_pipeline

PIPELINE_TYPE = "K6_VSR_PIPELINE"
CATEGORY = "video/upscaling/Kandinsky 6 VSR"

# The released Diffusers bundle: SR DiT + KVAE + latent upscalers in one repo.
DEFAULT_CHECKPOINT = "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"

RESOLUTION_SCALES = ("2", "2.25", "4")
TARGET_RESOLUTION_OPTIONS = ("none", *sorted(TARGET_RESOLUTIONS))
MAX_SEED = 2**64 - 1

# The pipeline built by the last loader run; its weights are freed before the next build.
LAST_PIPELINE: Any | None = None


def string_input(default: str, tooltip: str) -> tuple[str, dict[str, Any]]:
    """ComfyUI ``STRING`` widget spec."""
    return ("STRING", {"default": default, "tooltip": tooltip})


def combo_input(options: tuple[str, ...], default: str, tooltip: str) -> tuple[list[str], dict[str, Any]]:
    """ComfyUI combo (dropdown) widget spec."""
    return (list(options), {"default": default, "tooltip": tooltip})


class NumberWidget(BaseModel):
    """ComfyUI ``INT`` / ``FLOAT`` widget definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["INT", "FLOAT"]
    default: int | float
    minimum: int | float
    maximum: int | float
    tooltip: str
    step: float | None = None

    def spec(self) -> tuple[str, dict[str, Any]]:
        """Return the ``(type, options)`` pair ComfyUI expects in ``INPUT_TYPES``."""
        options: dict[str, Any] = {
            "default": self.default,
            "min": self.minimum,
            "max": self.maximum,
            "tooltip": self.tooltip,
        }
        if self.step is not None:
            options["step"] = self.step
        return (self.kind, options)


class LoadSettings(BaseModel):
    """Widget values of :class:`K6VSRLoadModel`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_path: str
    vae_backend: VaeBackend
    device: str

    def to_sr_config(self) -> SRConfig:
        """Build the :class:`SRConfig` the factory consumes, resolving model references."""
        return SRConfig(
            checkpoint_path=resolve_model_reference(self.checkpoint_path),
            vae_backend=self.vae_backend,
            device=self.device,
        )


class UpscaleSettings(BaseModel):
    """Widget values of :class:`K6VSRUpscale`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolution_scale: Literal["2", "2.25", "4"]
    seed: int = Field(ge=0, le=MAX_SEED)
    num_steps: int = Field(ge=2)
    overlap: float = Field(ge=0.0, lt=1.0)
    tiles_batch_size: int = Field(gt=0)
    frame_rate: float = Field(gt=0.0)
    target_resolution: str

    @property
    def scale(self) -> float:
        """The requested total upscale factor as a number."""
        return float(self.resolution_scale)


class K6VSRLoadModel:
    """Load the SR DiT, the KVAE and the latent-upscaler bank onto one GPU."""

    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = (PIPELINE_TYPE,)
    RETURN_NAMES = ("pipeline",)
    DESCRIPTION = (
        "Loads the Kandinsky 6 video super-resolution pipeline (SR DiT + KVAE + latent upscaler) from a "
        "Diffusers bundle. Model references are Hugging Face repo ids, absolute paths, or names under "
        "models/kandinsky_vsr. The VAE and latent upscalers are resolved automatically from that bundle. "
        "The pipeline stays loaded between runs while these inputs are unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI node contract
        return {
            "required": {
                "checkpoint_path": string_input(
                    DEFAULT_CHECKPOINT,
                    "Diffusers bundle with the SR DiT, VAE and latent upscalers (HF repo id or local dir).",
                ),
                "vae_backend": combo_input(
                    ("torch", "magi"),
                    "torch",
                    "KVAE compile backend; 'magi' needs the kandinsky-6-sr[magi] extra.",
                ),
                "device": string_input("cuda:0", "CUDA device for every SR component."),
            }
        }

    def load(self, **inputs: Any) -> tuple[Any]:
        """Build the pipeline and warm the flex-attention kernels once.

        Any changed input rebuilds all three models: the package loads them as
        one unit (the VAE build reads the DiT's training config). The previous
        pipeline is released first so both never sit on the GPU together.
        """
        global LAST_PIPELINE  # noqa: PLW0603 - the loader owns the GPU residency of its last output
        settings = LoadSettings(**inputs)
        config = settings.to_sr_config()
        if LAST_PIPELINE is not None:
            logger.info("Releasing the previously loaded Kandinsky 6 VSR pipeline")
            release_pipeline(LAST_PIPELINE)
            LAST_PIPELINE = None
        logger.info("Loading Kandinsky 6 VSR pipeline on {}: {}", config.device, config.model_dump(exclude_none=True))
        pipeline = load_sr_pipeline(config, config.device, force=True)
        LAST_PIPELINE = pipeline
        log_sampling_mode(pipeline, config.num_steps)
        warmup(pipeline.dit, scale_factor_for(pipeline), config.device)
        return (pipeline,)


class K6VSRUpscale:
    """Super-resolve an ``IMAGE`` batch (video frames) with a loaded pipeline."""

    CATEGORY = CATEGORY
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO")
    RETURN_NAMES = ("images", "frame_rate", "audio")
    DESCRIPTION = (
        f"Tiled x2 / x2.25 / x4 video super-resolution. Frames are resampled to {TARGET_FPS} fps and "
        f"clipped to the model's {MAX_NUM_FRAMES}-frame (5 s) budget; the returned frame_rate is the "
        "rate to encode the result at, and the optional audio is passed through cut to the result's length."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - ComfyUI node contract
        return {
            "required": {
                "pipeline": (PIPELINE_TYPE, {"tooltip": "Output of the Kandinsky 6 VSR Load Model node."}),
                "images": ("IMAGE", {"tooltip": "Video frames as one IMAGE batch (e.g. from Get Video Components)."}),
                "resolution_scale": combo_input(
                    RESOLUTION_SCALES, "2", "Total upscale factor; 2.25 = x1.125 pixel pre-upscale + x2 tiling."
                ),
                "seed": NumberWidget(kind="INT", default=42, minimum=0, maximum=MAX_SEED, tooltip="Noise seed.").spec(),
                "num_steps": NumberWidget(
                    kind="INT",
                    default=5,
                    minimum=2,
                    maximum=64,
                    tooltip=(
                        "Denoising grid points per tile (N points = N - 1 steps) for the flow-matching checkpoint "
                        "(Kandinsky-6.0-VSR-5s). Distilled checkpoints run their trained step count and ignore it."
                    ),
                ).spec(),
                "overlap": NumberWidget(
                    kind="FLOAT",
                    default=0.20,
                    minimum=0.0,
                    maximum=0.95,
                    step=0.01,
                    tooltip="Minimum tile overlap as a fraction of the tile size.",
                ).spec(),
                "tiles_batch_size": NumberWidget(
                    kind="INT",
                    default=1,
                    minimum=1,
                    maximum=16,
                    tooltip="Tiles per DiT call; more = faster, more VRAM.",
                ).spec(),
                "frame_rate": NumberWidget(
                    kind="FLOAT",
                    default=24.0,
                    minimum=1.0,
                    maximum=240.0,
                    step=0.01,
                    tooltip="Frame rate of the input frames (the fps output of Get Video Components).",
                ).spec(),
                "target_resolution": combo_input(
                    TARGET_RESOLUTION_OPTIONS,
                    "none",
                    "Optionally downscale the result to a delivery tier (aspect kept).",
                ),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {"tooltip": "Source audio (Get Video Components); returned trimmed to the SR clip."},
                ),
            },
        }

    def upscale(
        self, pipeline: Any, images: torch.Tensor, audio: dict[str, Any] | None = None, **inputs: Any
    ) -> tuple[torch.Tensor, float, dict[str, Any] | None]:
        """Run the tiled SR; return the frames as a ComfyUI ``IMAGE`` batch, their frame rate and the trimmed audio."""
        settings = UpscaleSettings(**inputs)
        video, fps = prepare_source_clip(comfy_images_to_tchw_uint8(images), settings.frame_rate)
        frames_count, _, height, width = video.shape
        logger.info("SR x{}: {} frames {}x{} @ {} fps", settings.scale, frames_count, width, height, fps)
        # no_grad (not inference_mode) keeps the compiled KVAE / flex kernels
        # under the same autograd mode the load-time warmup used, avoiding a recompile.
        with torch.no_grad():
            result = pipeline(
                video=video,
                resolution_scale=settings.scale,
                seed=settings.seed,
                num_steps=settings.num_steps,
                overlap=settings.overlap,
                tiles_batch_size=settings.tiles_batch_size,
                show_progress=True,
                progress=ComfyProgress(),
            )
        frames = result.frames[0]
        target_hw = resolve_target_hw(settings.target_resolution, tuple(frames.shape[-2:]))
        if target_hw is not None:
            frames = resize_to_target(frames, target_hw)
        logger.info("SR result: {} frames {}x{}", frames.shape[1], frames.shape[3], frames.shape[2])
        return sr_frames_to_comfy_images(frames), float(fps), trim_comfy_audio(audio, frames.shape[1], fps)


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "K6VSRLoadModel": K6VSRLoadModel,
    "K6VSRUpscale": K6VSRUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "K6VSRLoadModel": "Kandinsky 6 VSR Load Model",
    "K6VSRUpscale": "Kandinsky 6 VSR Upscale",
}
