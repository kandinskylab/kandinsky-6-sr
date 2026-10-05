"""ComfyUI node pack: tensor conversions, source-clip preparation, model path resolution.

What & why: the nodes are thin adapters between ComfyUI's ``IMAGE`` batches
(``[T, H, W, C]`` float in [0, 1]) and the SR pipeline's ``[T, C, H, W]``
uint8 / ``[C, T, H, W]`` uint8 layouts; a layout or range slip there corrupts
every frame silently, so the adapters are pinned here. Model references must
resolve the same way the CLI does (HF repo ids untouched) plus the ComfyUI
convention of a ``models/kandinsky_vsr`` folder. How: pure unit tests, no CUDA,
no weights; ComfyUI's ``folder_paths`` / ``comfy.utils`` are replaced by fakes on the modules.
Corner cases: RGBA input (alpha dropped), rounding at the [0, 1] boundaries,
sources longer than the 5 s contract (clipped to ``1 + 8k`` frames), sources
above 24 fps (downsampled), a ``none`` latent-upscaler spec, model names
that exist under the ComfyUI models folder vs. paths given verbatim.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch

from k6_vsr_comfy import frames, model_paths, nodes
from k6_vsr_comfy import progress as progress_module
from k6_vsr_comfy.progress import ComfyProgress
from k6_vsr_comfy.release import release_pipeline

from .conftest import COMFYUI_PACK_DIR

MODEL_FPS = 24
MODEL_FRAME_BUDGET = 121
# 150 frames @48 fps -> 75 @24 fps -> floor-aligned to 1 + 8k.
EXPECTED_ALIGNED_FRAMES = 73


def test_comfy_images_to_tchw_uint8_roundtrips_layout_and_range() -> None:
    images = torch.tensor([[[[0.0, 0.5, 1.0]]]])  # [T=1, H=1, W=1, C=3]
    video = frames.comfy_images_to_tchw_uint8(images)
    assert video.shape == (1, 3, 1, 1)
    assert video.dtype == torch.uint8
    assert video.flatten().tolist() == [0, 128, 255]


def test_comfy_images_drop_alpha_channel() -> None:
    images = torch.rand(2, 4, 6, 4)
    assert frames.comfy_images_to_tchw_uint8(images).shape == (2, 3, 4, 6)


def test_comfy_images_reject_non_batch_input() -> None:
    with pytest.raises(ValueError, match="IMAGE batch"):
        frames.comfy_images_to_tchw_uint8(torch.rand(4, 6, 3))


def test_sr_frames_to_comfy_images_layout_and_range() -> None:
    sr = torch.zeros(3, 2, 4, 6, dtype=torch.uint8)  # [C, T, H, W]
    sr[0, 1, 0, 0] = 255
    images = frames.sr_frames_to_comfy_images(sr)
    assert images.shape == (2, 4, 6, 3)
    assert images.dtype == torch.float32
    assert images[1, 0, 0, 0].item() == pytest.approx(1.0)
    assert images.max().item() == pytest.approx(1.0)


def test_prepare_source_clip_downsamples_and_aligns_frames() -> None:
    video = torch.zeros(150, 3, 8, 8, dtype=torch.uint8)
    clip, fps = frames.prepare_source_clip(video, src_fps=48.0)
    assert fps == MODEL_FPS
    assert clip.shape[0] == EXPECTED_ALIGNED_FRAMES


def test_prepare_source_clip_caps_at_the_model_frame_budget() -> None:
    video = torch.zeros(400, 3, 8, 8, dtype=torch.uint8)
    clip, fps = frames.prepare_source_clip(video, src_fps=24.0)
    assert (clip.shape[0], fps) == (MODEL_FRAME_BUDGET, MODEL_FPS)


@pytest.fixture
def fake_models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand in for ComfyUI's ``folder_paths`` with ``models_dir`` under ``tmp_path``."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(model_paths, "folder_paths", types.SimpleNamespace(models_dir=str(models_dir)))
    return models_dir / model_paths.MODEL_FOLDER


def test_hf_repo_ids_pass_through(fake_models_dir: Path) -> None:
    ref = "kandinskylab/Kandinsky-6.0-VSR-5s"
    assert model_paths.resolve_model_reference(ref) == ref


def test_bundle_component_references_pass_through(fake_models_dir: Path) -> None:
    ref = "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers/latent_upscaler"
    assert model_paths.resolve_model_reference(ref) == ref


def test_none_spec_passes_through(fake_models_dir: Path) -> None:
    assert model_paths.resolve_model_reference("none") == "none"


def test_relative_name_resolves_under_comfy_models_folder(fake_models_dir: Path) -> None:
    (fake_models_dir / "my_dit").mkdir(parents=True)
    assert model_paths.resolve_model_reference("my_dit") == str(fake_models_dir / "my_dit")


def test_absolute_and_unknown_paths_are_kept_verbatim(fake_models_dir: Path) -> None:
    assert model_paths.resolve_model_reference("/abs/dit") == "/abs/dit"
    assert model_paths.resolve_model_reference("./local/dit") == "./local/dit"


def test_without_comfyui_the_reference_is_kept_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_paths, "folder_paths", None)
    assert model_paths.resolve_model_reference("dit_dir") == "dit_dir"


def test_empty_reference_is_rejected(fake_models_dir: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        model_paths.resolve_model_reference("  ")


def test_optional_reference_left_empty_means_unset(fake_models_dir: Path) -> None:
    assert model_paths.resolve_optional_model_reference("  ") is None
    assert model_paths.resolve_optional_model_reference("/abs/kvae") == "/abs/kvae"


def test_loader_defaults_take_every_model_from_the_distilled_bundle(fake_models_dir: Path) -> None:
    """The node's default widgets name only the bundle; the VAE and upscalers are left to it."""
    required = nodes.K6VSRLoadModel.INPUT_TYPES()["required"]
    defaults = {name: spec[1]["default"] for name, spec in required.items()}

    sr_config = nodes.LoadSettings(**defaults).to_sr_config()

    assert sr_config.checkpoint_path == "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"
    assert sr_config.vae_path is None
    assert sr_config.latent_upscaler_config is None


def test_example_workflow_uses_the_loader_defaults() -> None:
    workflow = json.loads((COMFYUI_PACK_DIR / "example_workflows" / "k6_vsr_video_upscale.json").read_text())
    loader = next(node for node in workflow["nodes"] if node["type"] == "K6VSRLoadModel")
    required = nodes.K6VSRLoadModel.INPUT_TYPES()["required"]

    assert loader["widgets_values"] == [spec[1]["default"] for spec in required.values()]


def test_release_pipeline_frees_every_module_parameter() -> None:
    """ComfyUI keeps the stale loader output alive until the run ends, so the
    node itself must drop the old weights before building a new pipeline;
    otherwise two full pipelines sit on the GPU at once (the OOM seen when
    switching the KVAE)."""
    vae = torch.nn.Linear(4, 4)
    vae.register_buffer("scale", torch.ones(3))
    pipeline = types.SimpleNamespace(dit=torch.nn.Linear(8, 8), vae=vae, latent_upscaler=None)

    release_pipeline(pipeline)

    assert all(p.numel() == 0 for p in pipeline.dit.parameters())
    assert all(p.numel() == 0 for p in pipeline.vae.parameters())
    assert all(b.numel() == 0 for b in pipeline.vae.buffers())


class FakeComfyBar:
    instances: list[FakeComfyBar] = []

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        FakeComfyBar.instances.append(self)

    def update(self, value: int) -> None:
        self.done += value


def test_comfy_progress_drives_the_comfyui_progress_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    """The UI bar counts tiles x steps and advances once per denoising step."""
    FakeComfyBar.instances.clear()
    monkeypatch.setattr(progress_module, "comfy_utils", types.SimpleNamespace(ProgressBar=FakeComfyBar))

    progress = ComfyProgress()
    progress.start(total_tiles=9, steps_per_tile=2)
    progress.update()
    progress.update(2)
    progress.close()

    (bar,) = FakeComfyBar.instances
    assert (bar.total, bar.done) == (18, 3)


def test_comfy_progress_is_a_no_op_outside_comfyui(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(progress_module, "comfy_utils", None)
    progress = ComfyProgress()
    progress.start(total_tiles=1, steps_per_tile=1)
    progress.update()
    progress.close()
    assert progress.bar is None


def test_trim_comfy_audio_cuts_to_the_clip_duration() -> None:
    audio = {"waveform": torch.zeros(1, 2, 44100 * 10), "sample_rate": 44100}
    trimmed = frames.trim_comfy_audio(audio, num_frames=121, fps=24)
    assert trimmed is not None
    assert trimmed["waveform"].shape == (1, 2, round(121 / 24 * 44100))
    assert trimmed["sample_rate"] == 44100  # noqa: PLR2004 - unchanged metadata


def test_trim_comfy_audio_passes_none_through() -> None:
    assert frames.trim_comfy_audio(None, num_frames=121, fps=24) is None
