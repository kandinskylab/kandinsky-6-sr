# ComfyUI developer notes

User setup: [README](../README.md). This folder is the standalone `kandinsky6-sr`
package; it embeds its runtime and does not depend on the Python SR pipeline.

## Packaging and publishing

Run from `comfyui/`:

```bash
comfy --skip-prompt --no-enable-telemetry node pack
```

This builds `node.zip` without publishing. The package must be Git-tracked;
`.comfyignore` excludes tests, developer notes and generated files.
To publish an explicitly approved release, set a new semantic version in
`pyproject.toml` and run `comfy node publish` from this folder. Use the Comfy
Registry publishing key for publisher `kandinskylab`, not a GitHub PAT.
No merge or push automatically publishes this package.

## Models and runtime

The template's **Download models** button downloads the complete SR Diffusers
bundle, including JSON configs, after confirmation. Installation and startup
do not download weights. Exact URLs and destinations are embedded in the
workflow metadata. Existing files are reused.

ComfyUI owns model loading and eviction through `ModelPatcher` and
`load_models_gpu`. DiT and KVAE stay eager; NABLA is off by default and its
attention kernels compile only when it is enabled. Sage Attention is recommended.
Python-pipeline KVAE compile/segment environment overrides are not used here;
the eager codec keeps the default temporal segments. Context-parallel debug
messages use standard Python logging instead of `CP_DEBUG`.
Use the upscaler loader's `2x` entry for 2x/2.25x and `4x` for 4x. Disconnecting
the latent upscaler enables the optional pixel route, not the template default.

The previous Python-pipeline adapter has been replaced by native loaders.
Use the new template instead of the old `K6VSRLoadModel` workflow.

## Validation

Before releasing, test the shipped template on GPU, including audio, repeated
runs and supported scales. Development tests live in the repository's root
`tests/`, outside the node package. Run `python -m pytest tests` from the
repository root; set `COMFYUI_PATH` to a ComfyUI checkout to include native
node contracts on CPU. Without it, only those native checks are skipped.
