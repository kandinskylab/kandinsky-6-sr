"""Contracts of the native ComfyUI SR package, not the removed pipeline adapter.

Metadata checks run in the ordinary Python test environment. Set COMFYUI_PATH
to a compatible ComfyUI checkout to also run the real node contracts on CPU;
no GPU, model weights, fake Comfy modules or core patches are needed.
"""

# ruff: noqa: PLR2004 - exact released widget values, shapes and frame budgets

from __future__ import annotations

import importlib
import json
import os
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from .conftest import COMFYUI_PACK_DIR

WORKFLOW = COMFYUI_PACK_DIR / "example_workflows" / "Kandinsky 6.0 Video Super Resolution.json"
BUNDLE = "Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"


@pytest.fixture
def workflow():
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def native():
    comfy_path = os.environ.get("COMFYUI_PATH")
    if not comfy_path:
        pytest.skip("Set COMFYUI_PATH to run native ComfyUI contracts on CPU.")
    assert (Path(comfy_path) / "comfy").is_dir(), "COMFYUI_PATH must point to a ComfyUI checkout"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(comfy_path)
        cli_args = importlib.import_module("comfy.cli_args")
        patch.setattr(cli_args.args, "cpu", True)
        yield SimpleNamespace(
            nodes=importlib.import_module("kandinsky6_vsr.nodes"),
            register=importlib.import_module("kandinsky6_vsr.register"),
            supported=importlib.import_module("kandinsky6_vsr.supported_model"),
            contract=importlib.import_module("kandinsky6_vsr.sr_contract"),
            detection=importlib.import_module("comfy.model_detection"),
            models=importlib.import_module("comfy.supported_models"),
            motion=importlib.import_module(
                "kandinsky6_vsr.runtime.core.components.latent_upscaler.model.motion_correspondence"
            ),
            distributed=importlib.import_module(
                "kandinsky6_vsr.runtime.core.components.video_kvae.ctx.utils_distributed"
            ),
        )


def test_package_metadata_uses_the_public_sr_identity():
    metadata = tomllib.loads((COMFYUI_PACK_DIR / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "kandinsky6-sr"
    assert metadata["tool"]["comfy"]["PublisherId"] == "kandinskylab"
    assert metadata["tool"]["comfy"]["DisplayName"] == "Kandinsky6 SR"
    assert metadata["project"]["urls"]["Documentation"].endswith("kandinsky-6-sr/tree/main/comfyui")
    assert "kandinsky-6-sr" not in metadata["project"]["dependencies"]


def test_workflow_uses_native_loaders_and_distilled_diffusers_bundle(workflow):
    widgets = {node["type"]: node.get("widgets_values") for node in workflow["nodes"]}
    assert widgets["UNETLoader"] == [f"{BUNDLE}/transformer/diffusion_pytorch_model.safetensors", "default"]
    assert widgets["Kandinsky6SRVAELoader"] == [f"{BUNDLE}/vae/diffusion_pytorch_model.safetensors"]
    assert widgets["Kandinsky6LatentUpscalerLoader"] == [
        f"{BUNDLE}/latent_upscaler/diffusion_pytorch_model.safetensors",
        "2x",
    ]
    assert widgets["Kandinsky6VSRUpscale"] == [42, "fixed", 5, 1, "2x", 0.2, 24.0, "none", False]


def test_workflow_has_preview_and_correct_registry_identity(workflow):
    assert WORKFLOW.with_suffix(".jpg").is_file()
    assert workflow["extra"]["name"] == "Kandinsky 6.0 Video Super Resolution"
    native_types = {"Kandinsky6SRVAELoader", "Kandinsky6LatentUpscalerLoader", "Kandinsky6VSRUpscale"}
    nodes = [node for node in workflow["nodes"] if node["type"] in native_types]
    assert len(nodes) == len(native_types)
    assert all(node["properties"]["cnr_id"] == "kandinsky6-sr" for node in nodes)


def test_development_tests_are_outside_the_comfy_package():
    assert not (COMFYUI_PACK_DIR / "tests").exists()
    assert "tests/" in (COMFYUI_PACK_DIR / ".comfyignore").read_text().splitlines()


def test_comfy_images_roundtrip_layout_range_and_alpha(native):
    images = torch.tensor([[[[0.0, 0.5, 1.0, 0.25]]]])
    video = native.nodes.comfy_images_to_tchw_uint8(images)
    assert video.shape == (1, 3, 1, 1)
    assert video.dtype == torch.uint8
    assert video.flatten().tolist() == [0, 128, 255]
    restored = native.nodes.sr_frames_to_comfy_images(video.permute(1, 0, 2, 3))
    assert restored.shape == (1, 1, 1, 3)
    assert restored.dtype == torch.float32
    torch.testing.assert_close(restored.flatten(), torch.tensor([0.0, 128 / 255, 1.0]))


def test_comfy_image_conversion_clamps_out_of_range_values(native):
    video = native.nodes.comfy_images_to_tchw_uint8(torch.tensor([[[[-1.0, 0.0, 2.0]]]]))
    assert video.flatten().tolist() == [0, 0, 255]


@pytest.mark.parametrize(("frames", "fps", "expected_frames"), [(150, 48.0, 73), (400, 24.0, 121)])
def test_source_clip_resampling_and_frame_budget(native, frames, fps, expected_frames):
    video, output_fps = native.nodes.resample_to_target_fps(torch.zeros(frames, 3, 8, 8, dtype=torch.uint8), fps)
    clip = native.nodes.clip_to_aligned_frames(video)
    assert output_fps == 24
    assert clip.shape[0] == expected_frames


def test_source_clip_rejects_empty_video(native):
    with pytest.raises(ValueError, match="no readable frames"):
        native.nodes.clip_to_aligned_frames(torch.zeros(0, 3, 8, 8, dtype=torch.uint8))


def test_audio_is_trimmed_without_mutating_input(native):
    audio = {"waveform": torch.zeros(1, 2, 44100 * 10), "sample_rate": 44100}
    trimmed = native.nodes.trim_comfy_audio(audio, num_frames=121, fps=24)
    assert trimmed["waveform"].shape == (1, 2, round(121 / 24 * 44100))
    assert trimmed["sample_rate"] == 44100
    assert audio["waveform"].shape[-1] == 44100 * 10
    assert native.nodes.trim_comfy_audio(None, num_frames=121, fps=24) is None


@pytest.mark.parametrize("folder", ["kandinsky6_sr", "diffusion_models"])
def test_model_files_resolve_through_comfy_paths(native, monkeypatch, tmp_path, folder):
    checkpoint = tmp_path / "diffusion_pytorch_model.safetensors"
    checkpoint.touch()
    monkeypatch.setattr(
        native.nodes.folder_paths, "get_full_path", lambda name, _: str(checkpoint) if name == folder else None
    )
    assert native.nodes._resolve_model_file("bundle/vae/diffusion_pytorch_model.safetensors", "KVAE") == checkpoint


@pytest.mark.parametrize("name", ["", "missing.safetensors", "kandinskylab/model"])
def test_missing_models_fail_without_delegating_to_hf_pipeline(native, monkeypatch, name):
    monkeypatch.setattr(native.nodes.folder_paths, "get_full_path", lambda *_: None)
    with pytest.raises(FileNotFoundError, match="Download the Diffusers|not found in ComfyUI"):
        native.nodes._resolve_model_file(name, "KVAE")


def test_native_socket_contract_and_nabla_default(native):
    inputs = native.nodes.Kandinsky6VSRUpscale.INPUT_TYPES()
    assert inputs["required"]["sr_model"][0] == "MODEL"
    assert inputs["required"]["sr_vae"] == ("K6_SR_VAE",)
    assert inputs["optional"]["latent_upscaler"][0] == "K6_LATENT_UPSCALER"
    assert inputs["optional"]["audio"] == ("AUDIO",)
    assert inputs["optional"]["use_nabla"][1]["default"] is False
    assert native.nodes.Kandinsky6VSRUpscale.RETURN_TYPES == ("IMAGE", "FLOAT", "AUDIO")
    assert native.nodes.CATEGORY == "Kandinsky6 SR"


def test_each_stage_uses_comfy_model_management(native):
    run = object.__new__(native.nodes._SRRun)
    run.memory = {"dit": 123, "vae": 456, "latent_upscaler": 789}
    patcher = object()
    for stage, memory in run.memory.items():
        with mock.patch.object(native.nodes.mm, "load_models_gpu") as load_models:
            run._load(patcher, stage)
        load_models.assert_called_once_with([patcher], memory_required=memory)


@pytest.mark.parametrize("use_nabla", [False, True])
def test_nabla_choice_is_scoped_to_a_patcher_clone(native, use_nabla):
    model = torch.nn.Module()
    model.diffusion_model = SimpleNamespace(n_grid=1)
    patcher_type = native.nodes.comfy.model_patcher.ModelPatcher
    original = patcher_type(model, torch.device("cpu"), torch.device("cpu"))
    original.model_options["transformer_options"]["foreign_option"] = "preserved"
    clone = original.clone()
    with (
        mock.patch.object(original, "clone", return_value=clone),
        mock.patch.object(native.nodes, "comfy_images_to_tchw_uint8", side_effect=InterruptedError("test stop")),
        pytest.raises(InterruptedError, match="test stop"),
    ):
        native.nodes.Kandinsky6VSRUpscale().upscale(original, None, None, 42, 5, 1, use_nabla=use_nabla)
    assert clone.model_options["transformer_options"]["k6_vsr_disable_nabla"] is not use_nabla
    assert clone.model_options["transformer_options"]["foreign_option"] == "preserved"
    assert "k6_vsr_disable_nabla" not in original.model_options["transformer_options"]


def test_disabled_nabla_does_not_warm_or_compile_kernels(native):
    patcher = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=SimpleNamespace(n_grid=1), get_dtype_inference=lambda: torch.float32),
        model_options={"transformer_options": {"k6_vsr_disable_nabla": True}},
        load_device=torch.device("cpu"),
    )
    run = native.nodes._SRRun(patcher, None, None, {}, None)
    with mock.patch.object(native.nodes, "warmup_nabla") as warmup:
        assert run.tiles([], None, seed=42, num_steps=5, tiles_batch_size=1) == []
    warmup.assert_not_called()


def test_registration_is_idempotent_and_preserves_foreign_models(native):
    foreign = mock.Mock(return_value={"foreign": True})
    with (
        mock.patch.object(native.models, "models", []),
        mock.patch.object(native.detection, "detect_unet_config", foreign),
    ):
        native.register.register()
        installed = native.detection.detect_unet_config
        native.register.register()
        assert native.detection.detect_unet_config is installed
        assert native.models.models == [native.supported.Kandinsky6SR]
        assert installed({}, "") == {"foreign": True}


@pytest.mark.parametrize("n_grid", [1, 10])
def test_dit_detection_distinguishes_regular_and_distilled_models(native, n_grid):
    config = native.contract.DIT_CONFIG
    model_dim, time_dim = config["model_dim"], config["time_dim"]
    state = {
        "pooled_bias": torch.empty(time_dim, device="meta"),
        "visual_embeddings.in_layer.weight": torch.empty(model_dim, 2 * config["in_visual_dim"] + 1, device="meta"),
        "out_layer.out_layer.weight": torch.empty(config["out_visual_dim"] * n_grid, model_dim, device="meta"),
    }
    for index in range(config["num_visual_blocks"]):
        prefix = f"visual_transformer_blocks.{index}."
        state[prefix + "visual_modulation.out_layer.weight"] = torch.empty(6 * model_dim, time_dim, device="meta")
        state[prefix + "feed_forward.in_layer.weight"] = torch.empty(config["ff_dim"], model_dim, device="meta")
        state[prefix + "self_attention.query_norm.weight"] = torch.empty(sum(config["axes_dims"]), device="meta")
    assert native.register.detect_sr_dit(state, "") == {"image_model": "kandinsky6_sr", "n_grid": n_grid}


def test_vae_loader_builds_eager_codec_without_compiling(native, monkeypatch, tmp_path):
    checkpoint = tmp_path / "vae.safetensors"
    checkpoint.touch()
    architecture = SimpleNamespace(
        encoder_params={"z_channels": 64}, decoder_params={"z_channels": 64}, scaling_factor=0.910344004631042
    )
    module = torch.nn.Linear(1, 1)
    module.init_from_ckpt = mock.Mock()
    patcher = object()
    monkeypatch.setattr(native.nodes, "_resolve_model_file", lambda *_: checkpoint)
    monkeypatch.setattr(native.nodes, "_diffusers_config", lambda _: {"spatial_factor": 16, "temporal_factor": 4})
    monkeypatch.setattr(native.nodes, "kvae_architecture", lambda _: architecture)
    monkeypatch.setattr(native.nodes, "_patcher", lambda *_: patcher)
    monkeypatch.setattr(native.nodes, "_weight_dtype", lambda _: torch.float32)
    with (
        mock.patch.object(native.nodes, "CachedCausalVAE", return_value=module) as constructor,
        mock.patch.object(torch, "compile", side_effect=AssertionError("KVAE must stay eager")) as compile_model,
    ):
        (loaded,) = native.nodes.Kandinsky6SRVAELoader().load_vae("vae.safetensors")
    assert constructor.call_args.kwargs["encoder_conf"]["z_channels"] == 64
    module.init_from_ckpt.assert_called_once_with(str(checkpoint))
    assert loaded.module is module
    assert loaded.patcher is patcher
    assert loaded.scaling_factor == architecture.scaling_factor
    assert not module.training
    assert all(not parameter.requires_grad for parameter in module.parameters())
    compile_model.assert_not_called()


@pytest.mark.parametrize("frames", [1, 2, 5, 9, 31])
def test_eager_vae_keeps_default_temporal_decode_segments(native, monkeypatch, frames):
    chunks = []

    class IdentityDecoder(torch.nn.Module):
        def forward(self, chunk, cache):
            chunks.append(chunk.shape[2])
            return chunk

    codec = object.__new__(native.nodes.CachedCausalVAE)
    torch.nn.Module.__init__(codec)
    codec.conf = {"enc": {"temporal_compress_times": 4}}
    codec.decoder = IdentityDecoder()
    monkeypatch.setattr(codec, "make_empty_cache", lambda _: {})
    monkeypatch.setenv("KVAE_COMPILE", "1")
    monkeypatch.setenv("KVAE_DECODE_SEG", "32")
    latent = torch.arange(frames * 4, dtype=torch.float32).reshape(1, 1, frames, 2, 2)
    with mock.patch.object(torch, "compile", side_effect=AssertionError("KVAE must stay eager")):
        decoded = codec.decode(latent).sample
    expected = [4] * ((frames - 1) // 4)
    if (frames - 1) % 4:
        expected.append((frames - 1) % 4)
    if expected:
        expected[0] += 1
    else:
        expected = [1]
    assert chunks == expected
    torch.testing.assert_close(decoded, latent, rtol=0, atol=0)


@pytest.mark.parametrize("available", [False, True])
def test_optional_natten_import_requires_fused_kernels(native, monkeypatch, available):
    backend = SimpleNamespace(HAS_LIBNATTEN=available)
    monkeypatch.setitem(sys.modules, "natten", backend)
    if available:
        assert native.motion.load_natten() is backend
    else:
        with pytest.raises(RuntimeError, match="fused libnatten kernels are unavailable"):
            native.motion.load_natten()


def test_missing_natten_has_an_actionable_error(native, monkeypatch):
    monkeypatch.setitem(sys.modules, "natten", None)
    with pytest.raises(RuntimeError, match="natten package is not installed"):
        native.motion.load_natten()


def test_single_device_context_parallel_remains_a_noop(native, monkeypatch):
    monkeypatch.setattr(native.distributed, "get_context_parallel_world_size", lambda: 1)
    image = torch.randn(1, 3, 9, 8, 8)
    assert native.distributed._conv_gather(image, dim=2, kernel_size=3) is image
    assert native.distributed._conv_split(image, dim=2, kernel_size=3) is image
