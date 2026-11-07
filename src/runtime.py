"""Shared runtime helpers: seeds, device, experiment folders."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.prep.config import PREPROCESSING_VERSION, PROJECT_ROOT, SCHEMA_VERSION, SPEC_VERSION


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def experiment_dir(root: Path | None, name: str | None, seed: int) -> Path:
    root = root or (PROJECT_ROOT / "experiments")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = root / f"{name or 'run'}_{stamp}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def experiment_manifest(extra: dict | None = None) -> dict:
    payload = {
        "spec_version": SPEC_VERSION,
        "schema_version": SCHEMA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "quaternion_order": "x,y,z,w",
        "sample_rate_hz": 100,
        "context_samples": 400,
        "horizon_samples": 20,
        "horizon_seconds": 0.2,
        "official_metric": "mean geodesic degrees at predicted sample 20, window-weighted",
    }
    if extra:
        payload.update(extra)
    return payload
