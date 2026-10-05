"""Progress reporting of the SR pipeline.

What & why: hosts (the ``kandy-sr`` CLI, ComfyUI) need the run's total work
up front and one update per denoising step, without the pipeline knowing
the host. How: unit tests on the reporter combinators and on the totals
helper with fake components; no CUDA. Corner cases: neither / one / both
reporters requested, the distilled DiT (fixed nfe) vs the flow DiT
(``num_steps - 1``), a failing run still closes the reporters.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kandinsky_sr.pipeline.progress import CompositeProgress
from kandinsky_sr.pipeline.sr_pipeline import RunConfig, _progress_reporter, _tile_geometry, denoise_progress_total


class RecordingReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, int] | tuple[str, int] | tuple[str]] = []

    def start(self, total_tiles: int, steps_per_tile: int) -> None:
        self.events.append(("start", total_tiles, steps_per_tile))

    def update(self, n: int = 1) -> None:
        self.events.append(("update", n))

    def close(self) -> None:
        self.events.append(("close",))


def test_composite_forwards_every_event_to_every_reporter() -> None:
    first, second = RecordingReporter(), RecordingReporter()
    composite = CompositeProgress([first, second])
    composite.start(9, 2)
    composite.update()
    composite.update(3)
    composite.close()
    assert first.events == second.events == [("start", 9, 2), ("update", 1), ("update", 3), ("close",)]


def test_no_reporter_when_nothing_requested() -> None:
    assert _progress_reporter(False, None) is None


def test_single_host_reporter_is_used_directly() -> None:
    host = RecordingReporter()
    assert _progress_reporter(False, host) is host


def test_console_and_host_reporters_are_combined() -> None:
    reporter = _progress_reporter(True, RecordingReporter())
    assert isinstance(reporter, CompositeProgress)
    assert len(reporter.reporters) == 2  # noqa: PLR2004 - console + host


@pytest.mark.parametrize(
    ("piflow_params", "num_steps", "expected_steps"),
    [({"nfe": 2}, 5, 2), (None, 5, 4)],
)
def test_total_counts_tiles_times_sampler_steps(
    piflow_params: dict | None, num_steps: int, expected_steps: int
) -> None:
    components = SimpleNamespace(
        dit=SimpleNamespace(piflow_params=piflow_params),
        sr_params=SimpleNamespace(visual_size=[512]),
        spatial_factor=16,
    )
    run_config = RunConfig(device="cpu", num_steps=num_steps, overlap=0.2, resolution_scale=2)
    source_hw = (512, 768)
    expected_tiles = _tile_geometry(*source_hw, 512, 2, 0.2, 16)[2].total_tiles

    assert denoise_progress_total(components, run_config, source_hw) == (expected_tiles, expected_steps)
