"""Frame layout adapters between ComfyUI ``IMAGE`` batches and the SR pipeline.

ComfyUI passes video frames as one ``IMAGE`` batch: ``[T, H, W, C]`` float32
in ``[0, 1]`` on the CPU. The SR pipeline consumes ``[T, C, H, W]`` uint8 and
returns ``[C, T, H, W]`` uint8 (``SRPipelineOutput.frames[0]``).
"""

from __future__ import annotations

from typing import Any

import torch

from kandinsky_sr.pipeline.video_io import clip_to_aligned_frames, resample_to_target_fps

IMAGE_BATCH_NDIM = 4
RGB_CHANNELS = 3


def comfy_images_to_tchw_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert a ComfyUI ``[T, H, W, C]`` float batch to ``[T, 3, H, W]`` uint8.

    An alpha channel (RGBA sources) is dropped; the SR model is RGB-only.

    Raises:
        ValueError: If ``images`` is not a 4-D batch with at least 3 channels.
    """
    if images.ndim != IMAGE_BATCH_NDIM or images.shape[-1] < RGB_CHANNELS:
        raise ValueError(f"expected a ComfyUI IMAGE batch [T, H, W, C>=3], got {tuple(images.shape)}")
    rgb = images[..., :RGB_CHANNELS].detach().cpu().float()
    return (rgb * 255.0).round().clamp(0, 255).to(torch.uint8).permute(0, 3, 1, 2).contiguous()


def sr_frames_to_comfy_images(frames: torch.Tensor) -> torch.Tensor:
    """Convert the pipeline's ``[C, T, H, W]`` uint8 result to a ComfyUI ``[T, H, W, C]`` float batch."""
    return frames.detach().cpu().permute(1, 2, 3, 0).float().div(255.0).contiguous()


def prepare_source_clip(video: torch.Tensor, src_fps: float) -> tuple[torch.Tensor, int]:
    """Bring a ``[T, C, H, W]`` source to the model contract: 24 fps, at most 121 frames, ``1 + 8k`` long.

    Args:
        video: Source frames in ``[T, C, H, W]`` uint8 layout.
        src_fps: Frame rate the source plays at (sources above 24 fps are
            stride-downsampled; slower ones are kept at their native rate).

    Returns:
        ``(clip, fps)`` where ``fps`` is the rate the SR result should play at.
    """
    resampled, fps = resample_to_target_fps(video, src_fps)
    return clip_to_aligned_frames(resampled), fps


def trim_comfy_audio(audio: dict[str, Any] | None, num_frames: int, fps: float) -> dict[str, Any] | None:
    """Cut a ComfyUI ``AUDIO`` (``{"waveform": [B, C, N], "sample_rate": int}``) to ``num_frames / fps`` seconds.

    The SR result keeps at most the model's frame budget of the source; the
    audio that travels next to it must not outlive the picture.
    """
    if audio is None:
        return None
    max_samples = round(num_frames / fps * int(audio["sample_rate"]))
    return {**audio, "waveform": audio["waveform"][..., :max_samples]}
