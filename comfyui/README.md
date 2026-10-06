# Kandinsky6 SR

Standalone video super resolution with **2x, 2.25x and 4x** scaling.
Includes a video-upscale workflow with audio preservation and NABLA attention.
ComfyUI manages the models; the Python SR pipeline and Diffusers library are
not required. The Kandinsky 6 generation extension is optional.

Requires ComfyUI 0.8.0+, Python 3.10+ and PyTorch 2.8+ with a suitable CUDA build.
This extension does not replace ComfyUI's PyTorch installation.

## Install

Once the Registry version is available:

1. Open **ComfyUI Manager** and its custom-node list.
2. Search for `kandinsky6-sr` (publisher `kandinskylab`) and click **Install**.
3. Restart ComfyUI, then follow **Models** and **Run** below.

Alternatively, with [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)
and ComfyUI Manager installed:

```bash
comfy --workspace /path/to/ComfyUI node install kandinsky6-sr
```

Replace `/path/to/ComfyUI` with your ComfyUI folder and restart after installation.

For manual installation, clone `kandinskylab/kandinsky-6-sr` outside `custom_nodes/`,
copy the contents of `comfyui/` into `ComfyUI/custom_nodes/kandinsky6-sr/`, then run with
**ComfyUI's Python** and restart:

```bash
python -m pip install -r ComfyUI/custom_nodes/kandinsky6-sr/requirements.txt
```

## Models

Open the bundled workflow and click **Download models** in its setup note,
or choose **Kandinsky 6 → Kandinsky6 SR — Download models** from the top menu.

The button downloads the VSR DiT, KVAE and both latent-upscaler entries,
including all required JSON configs. Everything goes into
`ComfyUI/models/diffusion_models/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers/`.
Existing files and configured extra model paths are reused. Nothing is downloaded
during extension install or startup. Allow enough disk space for the HF weights.

For gated/private models, obtain access and run `hf auth login` on the ComfyUI
server first. The
[HF bundle](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers)
can also be downloaded manually, preserving its folders and JSON files.

## Run

For faster inference, try SageAttention: launch ComfyUI with `--use-sage-attention` instead of `--use-flash-attention` (requires `sageattention` in ComfyUI's Python environment and a supported NVIDIA GPU).

Open **Kandinsky 6.0 Video Super Resolution** in ComfyUI's workflow templates,
select an input video and the three model files, then run:

- **Load Diffusion Model** → `transformer/diffusion_pytorch_model.safetensors`
- **Kandinsky6 SR VAE Loader** → `vae/diffusion_pytorch_model.safetensors`
- **Kandinsky6 SR Latent Upscaler Loader** → `latent_upscaler/diffusion_pytorch_model.safetensors`

Choose `2x` in the upscaler loader for **2x/2.25x**, or `4x` for **4x**; set the
corresponding scale in **Kandinsky6 SR Upscale**. Start with the other defaults.
Frames are resampled toward 24 fps and clipped to at most 121 frames; audio is
trimmed to match. Use the VSR node's image, frame-rate and audio outputs together,
as the template does.

`use_nabla` defaults to **off**: VSR uses ComfyUI's selected attention backend
(including `--use-flash-attention` or `--use-sage-attention`) without NABLA warmup.
Turn it on to use sparse NABLA attention; its first run compiles kernels and
subsequent runs reuse them. Compare runtime and output quality for your video.
DiT and KVAE are not compiled in either mode.

Validation, packaging and manual publishing details:
[developer notes](https://github.com/kandinskylab/kandinsky-6-sr/blob/main/comfyui/docs/comfyui-development.md).
