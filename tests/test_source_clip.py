"""Source-clip contract: frame budget truncation is announced, audio never outlives the picture.

What & why: both hosts (CLI, ComfyUI) feed at most 121 frames to the model,
but a longer source used to be cut silently and its whole audio track was
muxed, so the output played sound past the last frame. A waveform handed over
as decode-sized chunks must also survive whole: ``mux_video_audio`` documents
a list as an accepted input, and keeping only its first element turned a 5 s
track into 23 ms without a word. How: unit tests on ``clip_to_aligned_frames``
with a loguru sink and on ``trim_audio_to_video`` / ``_audio_np``; no files,
no CUDA. Corner cases: a clip inside the budget logs nothing, audio shorter
than the video is left alone, a non-24 fps result trims by its own rate, a
chunked waveform is joined rather than truncated, and a single-element list
(one waveform for one batch item) is unwrapped as before.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from loguru import logger

from kandinsky_sr.constants import MAX_NUM_FRAMES
from kandinsky_sr.core.algo.mux import _audio_np, trim_audio_to_video
from kandinsky_sr.pipeline.video_io import clip_to_aligned_frames


@pytest.fixture
def warnings() -> list[str]:
    """Collect loguru WARNING messages emitted during a test."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink_id)


def test_truncation_to_the_frame_budget_is_logged(warnings: list[str]) -> None:
    clip = clip_to_aligned_frames(torch.zeros(300, 3, 4, 4, dtype=torch.uint8))
    assert clip.shape[0] == MAX_NUM_FRAMES
    assert any("only the first 121" in message for message in warnings)


def test_clip_within_the_budget_logs_nothing(warnings: list[str]) -> None:
    clip = clip_to_aligned_frames(torch.zeros(72, 3, 4, 4, dtype=torch.uint8))
    assert clip.shape[0] == 65  # noqa: PLR2004 - floor-aligned to 1 + 8k
    assert warnings == []


@pytest.mark.parametrize(
    ("num_frames", "fps", "samples", "expected"),
    [
        (121, 24, 44100 * 10, round(121 / 24 * 44100)),  # 10 s of audio for a 5 s clip
        (121, 24, 44100 * 2, 44100 * 2),  # audio shorter than the video stays whole
        (100, 25, 44100 * 10, 44100 * 4),  # a 25 fps result trims by its own rate
    ],
)
def test_audio_is_trimmed_to_the_video_duration(num_frames: int, fps: int, samples: int, expected: int) -> None:
    audio = np.zeros(samples, dtype=np.float32)
    assert trim_audio_to_video(audio, num_frames, fps, 44100).shape == (expected,)


def test_a_chunked_waveform_is_joined_not_truncated() -> None:
    """Decode-sized chunks are one track: dropping all but the first silently lost 99% of it."""
    chunks = [np.full(1024, index, dtype=np.float32) / 256 for index in range(200)]

    joined = _audio_np(chunks)

    assert joined.shape == (200 * 1024,)
    assert joined[0] == pytest.approx(0.0)
    assert joined[-1] == pytest.approx(199 / 256)


def test_a_single_waveform_in_a_list_is_unwrapped() -> None:
    """One waveform for one batch item — the historical shape — still passes through."""
    assert _audio_np([np.zeros(512, dtype=np.float32)]).shape == (512,)


def test_a_bare_waveform_is_passed_through() -> None:
    """The plain-ndarray form is untouched by the list handling."""
    assert _audio_np(np.zeros(512, dtype=np.float32)).shape == (512,)
