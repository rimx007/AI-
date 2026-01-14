"""YAML configuration loader.

All hyperparameters and paths should be controlled via config/config.yaml to ensure reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config into a Python dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
