"""SR-owned target-framework port sources."""

from pathlib import Path


def default_template_root() -> Path:
    """Return the directory containing SR target templates."""

    return Path(__file__).resolve().parent / "templates"


def default_template_dir(target: str) -> Path:
    """Return the SR template directory for one target framework."""

    return default_template_root() / target


__all__ = ["default_template_dir", "default_template_root"]
