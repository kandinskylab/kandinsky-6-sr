# torch wheel index for this repo's own environment (validated: torch 2.10.0 / CUDA 12.8).
# Pick another CUDA build for your driver: just torch_index=https://download.pytorch.org/whl/cu130 setup
torch_index := "https://download.pytorch.org/whl/cu128"

# Every recipe below adds the torch index as an extra index for this repo's env only;
# consumers of the package (k6_video) choose their own torch build.
export UV_INDEX := torch_index
export UV_INDEX_STRATEGY := "unsafe-best-match"

lock:
    @uv lock

setup:
    @uv sync --group dev

# Also build the MagiCompiler KVAE backend (`vae_backend: magi`).
setup-magi:
    @uv sync --group dev --extra magi

# Sync into the ACTIVE conda environment instead of .venv (no venv is created).
# Extras pass through: just setup-conda --extra magi
setup-conda *ARGS:
    @if [ -z "$CONDA_PREFIX" ]; then echo "Error: no conda environment is active. Activate one first (e.g. 'conda activate myenv')." && exit 1; fi
    @echo "=== Syncing kandinsky_sr into conda env $CONDA_PREFIX ==="
    VIRTUAL_ENV="$CONDA_PREFIX" uv sync --group dev --active {{ARGS}}

lint:
    @uv run ruff check .
    @uv run ruff format --check .

fmt:
    @uv run ruff format .
    @uv run ruff check --fix .

typecheck:
    @uv run mypy src

test *ARGS:
    @uv run pytest {{ARGS}}

# Remove generated Python, test, lint, and package-build artifacts.
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    find . -type d \( -name '__pycache__' -o -name '.ruff_cache' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name 'build' -o -name 'dist' -o -name '*.egg-info' \) -prune -exec rm -rf {} +
    find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.coverage' -o -name '.coverage.*' \) -delete

# Run SR on a video. Model paths come from --config / $KANDY_SR_CONFIG or CLI options.
# just generate-sr from-video --config sr_config.yaml --input lq.mp4 --output-dir outputs
generate-sr *ARGS:
    @uv run kandy-sr {{ARGS}}
