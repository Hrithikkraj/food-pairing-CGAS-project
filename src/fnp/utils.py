"""Utility functions for config loading, reproducibility, logging, and path resolution."""

import os
import sys
import yaml
import json
import random
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # src/fnp/utils.py -> parents[2] is project root
    return Path(__file__).resolve().parents[2]


def load_params(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    root = get_project_root()
    if config_path is None:
        path = root / "config" / "params.yaml"
    else:
        path = Path(config_path)
        if not path.is_absolute():
            path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


def compute_params_hash(params: Dict[str, Any]) -> str:
    """Generate an MD5 hash of the parameters dictionary for provenance."""
    serialized = json.dumps(params, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:8]


def set_seed(seed: int = 42) -> None:
    """Set global seeds across random and numpy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name: str = "fnp", log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return a standard logger."""
    root = get_project_root()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file is not None:
            log_path = Path(log_file)
            if not log_path.is_absolute():
                log_path = root / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def load_dv_table(dv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the reference daily values table."""
    root = get_project_root()
    if dv_path is None:
        path = root / "config" / "dv_table.csv"
    else:
        path = Path(dv_path)
        if not path.is_absolute():
            path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Daily value table not found at {path}")

    df = pd.read_csv(path)
    return df
