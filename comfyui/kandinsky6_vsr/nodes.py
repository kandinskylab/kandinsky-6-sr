"""ComfyUI nodes of Kandinsky 6 video super-resolution.

The SR DiT is a ComfyUI model (``Load Diffusion Model``); the KVAE and the
latent upscaler are the vendored canonical codecs under ComfyUI model
management. ``Kandinsky6VSRUpscale`` reproduces the canonical tiled SR:
encode the LQ video, latent-upscale each tile, denoise it from the degraded
latent and decode it, then blend the tiles with a Hann window.
"""

import contextlib
import os
from pathlib import Path

import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
import folder_paths
import torch
import torch.nn.functional as F  # noqa: N812 - PyTorch convention
import yaml

from . import sampling
from .nabla import warmup as warmup_nabla
from .runtime.core.algo.latent_upscaler import (
    latent_upscaler_for_scale,
    load_single_latent_upscaler,
    run_latent_upscaler,
)
from .runtime.core.algo.tiling_utils import TileGrid, extract_all_tiles, stitch_tiles_hanning
from .runtime.core.components.model.compiled_kvae import CompiledCachedCausalVAE
from .runtime.core.components.model.vae_io import cast_to_module_dtype, decode_latent_to_uint8
from .runtime.pipeline.diffusers_bundle import (
    COMPONENT_WEIGHTS,
    kvae_architecture,
    lu_bank_conf_from_component,
    read_component_config,
)
from .runtime.pipeline.output_resize import resize_to_target, resolve_target_hw
from .runtime.pipeline.source_padding import SourcePadding, source_padding
from .runtime.pipeline.tile_grid import compute_tile_grid_even
from .runtime.pipeline.upscale_utils import pre_upscale_video, resolve_scale_request
from .runtime.pipeline.video_io import clip_to_aligned_frames, resample_to_target_fps
from .sr_contract import ATTENTION_CONFIG, DIT_CONFIG, RESOLUTIONS, RUN_DEFAULTS, SR_COMMON, TARGET_RESOLUTIONS, VAE

SR_MODEL_DIR = os.path.join(folder_paths.models_dir, "kandinsky6", "sr")
SR_MODEL_FOLDER = "kandinsky6_sr"
folder_paths.add_model_folder_path(SR_MODEL_FOLDER, SR_MODEL_DIR)

CATEGORY = "Kandinsky6 SR"
SPATIAL_FACTOR = int(VAE["spatial_factor"])
TEMPORAL_FACTOR = int(VAE["temporal_factor"])
VISUAL_SIZE = int(SR_COMMON["visual_size"])
_SCALES = {"2.25x": 2.25, "2x": 2.0, "4x": 4.0}
_KVAE_SUFFIXES = frozenset({".safetensors", ".ckpt"})
_LU_SUFFIXES = frozenset({".safetensors", ".pt"})
_LU_BANK_SUFFIXES = frozenset({".yaml", ".yml"})

# Activation room requested from ``load_models_gpu`` for each SR stage, so
# ComfyUI keeps enough free memory for the forward pass when it decides how
# much of the weights to keep on the device. Coarse upper bounds.
_DIT_BYTES_PER_TOKEN = 96 * 1024
_LU_BYTES_PER_LATENT_VOXEL = 48 * 1024
_VAE_BYTES_PER_PIXEL = 4 * 1024
_VAE_SEGMENT_FRAMES = 17


# --- Model files ------------------------------------------------------------


def _model_files():
    names = set()
    for folder in (SR_MODEL_FOLDER, "diffusion_models"):
        try:
            names.update(folder_paths.get_filename_list(folder))
        except (KeyError, OSError):
            continue
    return sorted(names)


def _full_model_path(name):
    for folder in (SR_MODEL_FOLDER, "diffusion_models"):
        path = folder_paths.get_full_path(folder, name)
        if path is not None:
            return Path(path)
    return None


def _diffusers_config(path):
    """Inspect JSON beside a bundle component without converting or writing sidecars."""
    if path.name != COMPONENT_WEIGHTS or not (path.parent / "config.json").is_file():
        return None
    try:
        return read_component_config(path.parent)
    except (AttributeError, TypeError) as error:
        raise ValueError(f"Expected a JSON object in {path.parent / 'config.json'}") from error


def _resolve_model_file(name, kind):
    if not name:
        raise FileNotFoundError(
            f"No Kandinsky6 SR {kind} was selected. Download the Diffusers SR bundle under "
            f"ComfyUI/models/diffusion_models/ (or configure an extra '{SR_MODEL_FOLDER}' model path)."
        )
    path = _full_model_path(name)
    if path is None:
        raise FileNotFoundError(f"Kandinsky6 SR {kind} {name!r} was not found in ComfyUI model paths.")
    # Keep file symlinks: the KVAE weights and their .yaml are a same-stem pair.
    return path.absolute()


def _lu_bank_entries(bank):
    try:
        data = yaml.safe_load(bank.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return []
    upscaler = data.get("latent_upscaler") if isinstance(data, dict) else None
    models = upscaler.get("models") if isinstance(upscaler, dict) else None
    return [entry for entry in models if isinstance(entry, dict)] if isinstance(models, list) else []


def _lu_entries_for(checkpoint):
    """Released bank entries (next to *checkpoint*) that name it."""
    try:
        banks = sorted(path for path in checkpoint.parent.iterdir() if path.suffix.lower() in _LU_BANK_SUFFIXES)
    except OSError:
        return []
    return [
        entry
        for bank in banks
        for entry in _lu_bank_entries(bank)
        if Path(str(entry.get("checkpoint", ""))).name == checkpoint.name
    ]


def _list_kvae_checkpoints():
    names = []
    for name in _model_files():
        path = _full_model_path(name)
        if Path(name).suffix.lower() not in _KVAE_SUFFIXES or path is None:
            continue
        try:
            config = _diffusers_config(path)
        except (OSError, ValueError):
            continue
        if (config and config.get("vae_type") == VAE["name"]) or path.with_suffix(".yaml").is_file():
            names.append(name)
    return sorted(names)


def _list_latent_upscaler_checkpoints():
    names = []
    for name in _model_files():
        path = _full_model_path(name)
        if Path(name).suffix.lower() not in _LU_SUFFIXES or path is None:
            continue
        try:
            config = _diffusers_config(path)
        except (OSError, ValueError):
            continue
        if config is not None:
            models = config.get("models")
            if isinstance(models, list) and any(
                item.get("target_scale") in ("2x", "4x") for item in models if isinstance(item, dict)
            ):
                names.append(name)
            continue
        if (path.parent / "config.yaml").is_file() or path.with_suffix(".yaml").is_file():
            continue  # a DiT directory or a KVAE sidecar pair
        if any(str(entry.get("target_scale", "4x")) in ("2x", "4x") for entry in _lu_entries_for(path)):
            names.append(name)
    return sorted(names)


def _patcher(module, load_device, offload_device):
    patcher_type = getattr(comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher)
    size = sum(t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers()))
    return patcher_type(module, load_device=load_device, offload_device=offload_device, size=size)


def _weight_dtype(device):
    # The released codecs run in bfloat16 wherever the device supports it.
    return torch.bfloat16 if mm.should_use_bf16(device) else torch.float32


def _autocast(device):
    """The canonical bf16 autocast region of LU upscale and KVAE decode."""
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class K6SRVAE:
    """KVAE of the Kandinsky6 SR stage under ComfyUI model management."""

    def __init__(self, module, patcher):
        self.module, self.patcher = module, patcher

    @property
    def scaling_factor(self):
        return float(self.module.config.scaling_factor)


class K6LatentUpscaler:
    """Latent upscaler of the Kandinsky6 SR stage under ComfyUI model management."""

    def __init__(self, module, patcher):
        self.module, self.patcher = module, patcher


class Kandinsky6SRVAELoader:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 - ComfyUI node API
        return {
            "required": {
                "kvae_checkpoint": (
                    _list_kvae_checkpoints(),
                    {"tooltip": "SR Diffusers bundle: vae/diffusion_pytorch_model.safetensors with config.json."},
                ),
            }
        }

    RETURN_TYPES = ("K6_SR_VAE",)
    RETURN_NAMES = ("sr_vae",)
    FUNCTION = "load_vae"
    CATEGORY = CATEGORY

    def load_vae(self, kvae_checkpoint):
        path = _resolve_model_file(kvae_checkpoint, "KVAE checkpoint")
        from omegaconf import OmegaConf  # noqa: PLC0415 - imported with the codec

        component = _diffusers_config(path)
        if component is not None:
            if (component.get("spatial_factor"), component.get("temporal_factor")) != (SPATIAL_FACTOR, TEMPORAL_FACTOR):
                raise ValueError("KVAE component must use the released spatial/temporal factors (16, 4).")
            architecture = kvae_architecture(path.parent)
            if any(
                params.get("z_channels") != int(DIT_CONFIG["in_visual_dim"])
                for params in (architecture.encoder_params, architecture.decoder_params)
            ):
                raise ValueError("KVAE component must use the released 64 latent channels.")
            config = OmegaConf.create(
                {
                    "scaling_factor": architecture.scaling_factor,
                    "encoder_params": architecture.encoder_params,
                    "decoder_params": architecture.decoder_params,
                }
            )
        else:
            # Compatibility with old native installations; new releases use JSON only.
            config_path = path.with_suffix(".yaml")
            if not config_path.is_file():
                raise FileNotFoundError(f"KVAE config.json or legacy YAML sidecar not found beside {path}")
            config = OmegaConf.load(str(config_path))
        encoder = config.encoder_params if "encoder_params" in config else config.model.encoder_params
        decoder = config.decoder_params if "decoder_params" in config else config.model.decoder_params
        vae = CompiledCachedCausalVAE(encoder_conf=encoder, decoder_conf=decoder)
        vae.init_from_ckpt(str(path))
        load_device = mm.vae_device()
        vae = vae.eval().requires_grad_(False).to(dtype=_weight_dtype(load_device))
        vae.config = config
        return (K6SRVAE(vae, _patcher(vae, load_device, mm.vae_offload_device())),)


class Kandinsky6LatentUpscalerLoader:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 - ComfyUI node API
        return {
            "required": {
                "latent_upscaler_checkpoint": (
                    _list_latent_upscaler_checkpoints(),
                    {
                        "tooltip": "SR Diffusers bundle: latent_upscaler/diffusion_pytorch_model.safetensors "
                        "with config.json."
                    },
                ),
                "target_scale": (
                    ["2x", "4x"],
                    {
                        "default": "2x",
                        "tooltip": "Entry to load from the Diffusers bank: 2x for 2x/2.25x VSR, 4x for 4x VSR.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("K6_LATENT_UPSCALER",)
    RETURN_NAMES = ("latent_upscaler",)
    FUNCTION = "load_latent_upscaler"
    CATEGORY = CATEGORY

    def load_latent_upscaler(self, latent_upscaler_checkpoint, target_scale="2x"):
        from omegaconf import OmegaConf  # noqa: PLC0415 - imported with the codec

        path = _resolve_model_file(latent_upscaler_checkpoint, "latent-upscaler checkpoint")
        component = _diffusers_config(path)
        if component is not None:
            bank = lu_bank_conf_from_component(path.parent)
            entries = [entry for entry in bank.latent_upscaler.models if entry.target_scale == target_scale]
        else:
            entries = [entry for entry in _lu_entries_for(path) if str(entry.get("target_scale", "4x")) in ("2x", "4x")]
        if len(entries) != 1:
            raise FileNotFoundError(
                f"Expected exactly one latent-upscaler bank entry for {path.name} (scale={target_scale}), "
                f"found {len(entries)}. Keep the component's config.json beside its weights."
            )
        entry = OmegaConf.create(
            {
                "checkpoint": str(path),
                "use_ema": bool(entries[0].get("use_ema", True)),
                "target_scale": str(entries[0].get("target_scale", "4x")),
                "model": entries[0]["model"],
                "state_prefix": entries[0].get("state_prefix"),
            }
        )
        # The KVAE scaling factor is applied by the upscale node.
        upscaler = load_single_latent_upscaler(entry, device="cpu", vae_scaling_factor=1.0)
        load_device = mm.get_torch_device()
        upscaler = upscaler.to(dtype=_weight_dtype(load_device))
        return (K6LatentUpscaler(upscaler, _patcher(upscaler, load_device, mm.unet_offload_device())),)


# --- Canonical tiling, reproduced on ComfyUI-managed models -----------------


def _closest_base(height, width):
    ratio = width / height if height else 1.0
    return min(RESOLUTIONS[VISUAL_SIZE], key=lambda hw: abs(hw[1] / hw[0] - ratio))


def _tile_geometry(height, width, scale, overlap):
    """Base resolution, tile size and the even tile grid (canonical ``_tile_geometry``)."""
    base_h, base_w = _closest_base(height, width)
    if base_h % scale or base_w % scale:
        raise ValueError(f"scale x{scale} does not divide the base resolution {base_h}x{base_w}")
    tile_hw = (base_h // scale, base_w // scale)
    grid = compute_tile_grid_even(height, width, tile_hw, overlap, SPATIAL_FACTOR)
    return (base_h, base_w), tile_hw, grid


def _latent_grid(pixel_grid):
    values = (pixel_grid.tile_h, pixel_grid.tile_w, *pixel_grid.tops, *pixel_grid.lefts)
    if any(value % SPATIAL_FACTOR for value in values):
        raise ValueError(f"Pixel tile grid {pixel_grid} is not aligned to the VAE stride {SPATIAL_FACTOR}.")
    return TileGrid(
        pixel_grid.tile_h // SPATIAL_FACTOR,
        pixel_grid.tile_w // SPATIAL_FACTOR,
        tuple(top // SPATIAL_FACTOR for top in pixel_grid.tops),
        tuple(left // SPATIAL_FACTOR for left in pixel_grid.lefts),
    )


def _memory_required(frames, scale_hw, tiles_batch_size):
    latent_frames = (frames - 1) // TEMPORAL_FACTOR + 1
    tile_tokens = latent_frames * (VISUAL_SIZE // SPATIAL_FACTOR) ** 2
    height, width = scale_hw
    return {
        "dit": tiles_batch_size * tile_tokens * _DIT_BYTES_PER_TOKEN,
        "latent_upscaler": tiles_batch_size * tile_tokens * _LU_BYTES_PER_LATENT_VOXEL,
        "vae": _VAE_SEGMENT_FRAMES * max(height * width, VISUAL_SIZE**2) * _VAE_BYTES_PER_PIXEL,
    }


class _SRRun:
    """One tiled SR run: ComfyUI loads each model before the stage that needs it."""

    def __init__(self, sr_model, sr_vae, upscaler, memory, progress):
        self.sr_model, self.sr_vae, self.upscaler = sr_model, sr_vae, upscaler
        self.memory, self.progress = memory, progress
        self.device = sr_model.load_device

    def _load(self, patcher, stage):
        mm.load_models_gpu([patcher], memory_required=self.memory[stage])

    def encode_video(self, video):
        """Canonical ``encode_lq_video_to_lr_latent``: ``[T, 3, H, W]`` uint8 -> raw ``[T', C, h, w]``."""
        self._load(self.sr_vae.patcher, "vae")
        vae = self.sr_vae.module
        pixels = video.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        pixels = cast_to_module_dtype(vae, vae.normalize_data(pixels.float()))
        latent = vae.encode(pixels)[0]
        return latent.squeeze(0).permute(1, 0, 2, 3).float()

    def encode_tiles(self, tiles):
        """Canonical ``_encode_lq_videos``: ``[T, H, W, 3]`` pixel tiles -> scaled ``(N, h, w, C)``."""
        self._load(self.sr_vae.patcher, "vae")
        vae = self.sr_vae.module
        latents = []
        for tile in tiles:
            pixels = tile.permute(3, 0, 1, 2).unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
            latent = vae.encode(cast_to_module_dtype(vae, vae.normalize_data(pixels)))[0]
            latents.append(latent.squeeze(0).permute(1, 2, 3, 0).float() * self.sr_vae.scaling_factor)
        return torch.cat(latents, dim=0)

    def upscale_tiles(self, tiles):
        """Canonical ``upscale_lr_latent_tile``: raw ``[T, C, h, w]`` tiles -> scaled ``(N, H, W, C)``."""
        self._load(self.upscaler.patcher, "latent_upscaler")
        module = self.upscaler.module
        out = []
        for tile in tiles:
            z = cast_to_module_dtype(module, tile.permute(1, 0, 2, 3).unsqueeze(0).to(self.device, torch.float32))
            with _autocast(self.device):
                upscaled = run_latent_upscaler(module, z * self.sr_vae.scaling_factor)
            out.append(upscaled.squeeze(0).permute(1, 2, 3, 0).float())
        return torch.cat(out, dim=0)

    def decode(self, latent, batch):
        """Canonical ``vae_decode``: scaled ``(N, h, w, C)`` -> ``batch`` ``[3, T, H, W]`` uint8 tiles."""
        self._load(self.sr_vae.patcher, "vae")
        vae = self.sr_vae.module
        with _autocast(self.device):
            latents = latent.reshape(batch, -1, *latent.shape[1:]) / self.sr_vae.scaling_factor
            latents = latents.permute(0, 4, 1, 2, 3)
            decoded = [decode_latent_to_uint8(vae, latents[index : index + 1]) for index in range(batch)]
        return torch.cat(decoded, dim=0)

    def tiles(self, tile_inputs, prepare, *, seed, num_steps, tiles_batch_size):
        """Canonical ``_run_tile_batches``: the seed advances once per tile."""
        dtype = self.sr_model.model.get_dtype_inference()
        diffusion_model = self.sr_model.model.diffusion_model
        if not self.sr_model.model_options["transformer_options"].get("k6_vsr_disable_nabla", False):
            warmup_nabla(
                self.device,
                ATTENTION_CONFIG,
                heads=int(diffusion_model.num_heads),
                head_dim=int(diffusion_model.head_dim),
            )
        settings = sampling.sampling_settings(self.sr_model)
        outputs = []
        for start in range(0, len(tile_inputs), tiles_batch_size):
            mm.throw_exception_if_processing_interrupted()
            chunk = tile_inputs[start : start + tiles_batch_size]
            lq_latent = prepare(chunk).to(device=self.device, dtype=dtype)
            initial = sampling.degraded_start(
                lq_latent,
                seed=seed + start,
                noise_scale=float(settings["lq_noise_scale"]),
                noise_type=str(settings["lq_noise_type"]),
            )
            latent = sampling.sample(self.sr_model, initial, len(chunk), num_steps, self.progress, self.memory["dit"])
            outputs.extend(sample.float().cpu() for sample in self.decode(latent, len(chunk)))
        return outputs


def comfy_images_to_tchw_uint8(images):
    """ComfyUI ``IMAGE`` ``[T, H, W, C]`` float -> ``[T, 3, H, W]`` uint8 (alpha dropped)."""
    rgb = images[..., :3].detach().cpu().float()
    return (rgb * 255.0).round().clamp(0, 255).to(torch.uint8).permute(0, 3, 1, 2).contiguous()


def sr_frames_to_comfy_images(frames):
    """``[C, T, H, W]`` uint8 -> ComfyUI ``IMAGE`` ``[T, H, W, C]`` float."""
    return frames.detach().cpu().permute(1, 2, 3, 0).float().div(255.0).contiguous()


def trim_comfy_audio(audio, num_frames, fps):
    """Cut a ComfyUI ``AUDIO`` to the duration of ``num_frames`` at ``fps``."""
    if audio is None:
        return None
    max_samples = round(num_frames / fps * int(audio["sample_rate"]))
    return {**audio, "waveform": audio["waveform"][..., :max_samples]}


class Kandinsky6VSRUpscale:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 - ComfyUI node API
        return {
            "required": {
                "sr_model": ("MODEL", {"tooltip": "SR DiT from Load Diffusion Model."}),
                "sr_vae": ("K6_SR_VAE",),
                "images": ("IMAGE",),
                "seed": (
                    "INT",
                    {
                        "default": int(RUN_DEFAULTS["seed"]),
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": int(RUN_DEFAULTS["num_steps"]),
                        "min": 2,
                        "max": 100,
                        "tooltip": (
                            "Grid points of the flow-matching checkpoint; the distilled one runs its trained steps."
                        ),
                    },
                ),
                "tiles_batch_size": ("INT", {"default": int(RUN_DEFAULTS["tiles_batch_size"]), "min": 1, "max": 16}),
                "scale": (list(_SCALES), {"default": "2.25x"}),
                "overlap": (
                    "FLOAT",
                    {"default": float(RUN_DEFAULTS["overlap"]), "min": 0.0, "max": 0.95, "step": 0.01},
                ),
                "frame_rate": ("FLOAT", {"default": float(SR_COMMON["fps"]), "min": 1.0, "max": 240.0}),
                "target_resolution": (["none", *sorted(TARGET_RESOLUTIONS)], {"default": "none"}),
            },
            "optional": {
                "latent_upscaler": (
                    "K6_LATENT_UPSCALER",
                    {"tooltip": "Weights matching the scale (x2 for 2x and 2.25x). Omit for the pixel route."},
                ),
                "audio": ("AUDIO",),
                "use_nabla": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable sparse NABLA attention (compiles kernels on first use). "
                            "Off uses ComfyUI's selected attention backend without NABLA warmup."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO")
    RETURN_NAMES = ("images", "frame_rate", "audio")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Upscale video by x2, x2.25 or x4. Frames are aligned to the model's 121-frame budget at up to "
        "24 fps; the audio is trimmed to the output duration."
    )

    def upscale(  # noqa: PLR0913, PLR0917 - ComfyUI sockets and widgets
        self,
        sr_model,
        sr_vae,
        images,
        seed,
        steps,
        tiles_batch_size,
        scale="2.25x",
        overlap=float(RUN_DEFAULTS["overlap"]),
        frame_rate=float(SR_COMMON["fps"]),
        target_resolution="none",
        latent_upscaler=None,
        audio=None,
        use_nabla=False,
    ):
        if scale not in _SCALES:
            raise ValueError(f"Unknown SR scale {scale!r}.")
        if not hasattr(sr_model.model.diffusion_model, "n_grid"):
            raise ValueError("sr_model is not a Kandinsky6 SR diffusion model.")

        sr_model = sr_model.clone()
        sr_model.model_options["transformer_options"]["k6_vsr_disable_nabla"] = not use_nabla

        video, fps = resample_to_target_fps(comfy_images_to_tchw_uint8(images), frame_rate)
        video = clip_to_aligned_frames(video)
        tiling_scale, pre_upscale = resolve_scale_request(_SCALES[scale])
        if pre_upscale != 1.0:
            video = pre_upscale_video(video, pre_upscale, SPATIAL_FACTOR)
        upscaler = None
        if latent_upscaler is not None:
            if latent_upscaler_for_scale(latent_upscaler.module, tiling_scale) is None:
                raise ValueError(f"Select an x{tiling_scale} latent upscaler for {scale} SR.")
            upscaler = latent_upscaler

        frames, _, height, width = video.shape
        memory = _memory_required(frames, (height, width), tiles_batch_size)
        bar = comfy.utils.ProgressBar(1)
        run = _SRRun(sr_model, sr_vae, upscaler, memory, bar.update)

        if upscaler is not None:
            padding = source_padding(
                height, width, SPATIAL_FACTOR, lambda h, w: _tile_geometry(h, w, tiling_scale, overlap)[1]
            )
            video = padding.apply_to_video(video)
            height, width = video.shape[-2:]
            lr_latent = run.encode_video(video)
            _, _, pixel_grid = _tile_geometry(height, width, tiling_scale, overlap)
            tile_inputs = extract_all_tiles(lr_latent, _latent_grid(pixel_grid))
            prepare = run.upscale_tiles
        else:
            padding = SourcePadding()
            (base_h, base_w), _, pixel_grid = _tile_geometry(height, width, tiling_scale, overlap)
            tile_inputs = [
                F.interpolate(tile.float(), size=(base_h, base_w), mode="bilinear", align_corners=False).permute(
                    0, 2, 3, 1
                )
                for tile in extract_all_tiles(video, pixel_grid)
            ]
            prepare = run.encode_tiles

        bar.total = len(tile_inputs) * sampling.steps_per_tile(sr_model, steps)
        outputs = run.tiles(tile_inputs, prepare, seed=seed, num_steps=steps, tiles_batch_size=tiles_batch_size)
        frames_out = stitch_tiles_hanning(outputs, pixel_grid, height, width, scale=tiling_scale)
        frames_out = padding.crop(frames_out.clamp(0, 255).to(torch.uint8), tiling_scale)

        target_hw = resolve_target_hw(target_resolution, tuple(frames_out.shape[-2:]))
        if target_hw is not None:
            frames_out = resize_to_target(frames_out, target_hw)
        mm.throw_exception_if_processing_interrupted()
        return (
            sr_frames_to_comfy_images(frames_out),
            float(fps),
            trim_comfy_audio(audio, frames_out.shape[1], fps),
        )


NODE_CLASS_MAPPINGS = {
    "Kandinsky6SRVAELoader": Kandinsky6SRVAELoader,
    "Kandinsky6LatentUpscalerLoader": Kandinsky6LatentUpscalerLoader,
    "Kandinsky6VSRUpscale": Kandinsky6VSRUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Kandinsky6SRVAELoader": "Kandinsky6 SR VAE Loader",
    "Kandinsky6LatentUpscalerLoader": "Kandinsky6 SR Latent Upscaler Loader",
    "Kandinsky6VSRUpscale": "Kandinsky6 SR Upscale",
}
