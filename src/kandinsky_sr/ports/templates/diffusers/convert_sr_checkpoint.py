"""Convert a Kandinsky 6 SR checkpoint to a Diffusers bundle.

This file writes portable component weights and a ``model_index.json`` for
``Kandinsky6SRPipeline``.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path


def convert_sr_checkpoint(  # noqa: PLR0913
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    vae_path: str | Path | None = None,
    latent_upscaler_config: str | Path | None = None,
    output_dir: str | Path = "outputs/kandinsky6_sr_diffusers",
    use_patched_diffusers: bool = False,
) -> Path:
    """Convert a local or Hugging Face SR checkpoint using the installed SR package."""
    runtime = import_module("kandinsky_sr.ports.sr_bundle_runtime")
    return runtime.convert_sr_checkpoint(
        config_path,
        checkpoint_path=checkpoint_path,
        vae_path=vae_path,
        latent_upscaler_config=latent_upscaler_config,
        output_dir=output_dir,
        generated_dir=Path(__file__).resolve().parent,
        use_patched_diffusers=use_patched_diffusers,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the SR bundle converter CLI."""
    parser = argparse.ArgumentParser(
        description="Convert a local or Hugging Face K6 SR checkpoint to a Diffusers bundle"
    )
    parser.add_argument("--config", required=True, help="K6 YAML configuration containing the sr section")
    parser.add_argument("--checkpoint-path", "--checkpoint_path", help="override sr.checkpoint_path")
    parser.add_argument("--vae-path", "--vae_path", help="override sr.vae_path")
    parser.add_argument(
        "--latent-upscaler-config",
        "--latent_upscaler_config",
        help="override sr.latent_upscaler_config; use 'none' to disable the bank",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="outputs/kandinsky6_sr_diffusers",
        help="output Diffusers bundle directory",
    )
    parser.add_argument(
        "--use-patched-diffusers",
        action="store_true",
        help="resolve the SR pipeline class from the installed patched Diffusers package",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = convert_sr_checkpoint(
        args.config,
        checkpoint_path=args.checkpoint_path,
        vae_path=args.vae_path,
        latent_upscaler_config=args.latent_upscaler_config,
        output_dir=args.output_dir,
        use_patched_diffusers=args.use_patched_diffusers,
    )
    print(f"SR Diffusers bundle written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
