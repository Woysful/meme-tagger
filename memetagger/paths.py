"""User directories and default configuration files."""
from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

APP = "memetagger"
DATA = Path(__file__).resolve().parent / "data"


def _xdg(var: str, fallback: str) -> Path:
    p = Path(os.environ.get(var) or Path.home() / fallback) / APP
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share")


def user_config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config")


def bundled_models_dir() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "models"


def config_file(name: str) -> Path:
    """User copy of a config file; the packaged default (data/<stem>_default<ext>) is copied on first use."""
    dst = user_config_dir() / name
    if not dst.exists():
        p = Path(name)
        src = DATA / f"{p.stem}_default{p.suffix}"
        if src.exists():
            shutil.copy(src, dst)
    return dst
