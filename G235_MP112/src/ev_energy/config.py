"""Settings and small helpers shared by every module."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: str | Path = PARAMS_PATH) -> dict:
    """Read params.yaml with yaml.safe_load."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative: str | Path) -> Path:
    """Turn a path from params.yaml into an absolute path inside the repo."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def file_sha256(path: str | Path) -> str:
    """SHA-256 of the original file bytes (logged as data_sha256 in MLflow)."""
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
