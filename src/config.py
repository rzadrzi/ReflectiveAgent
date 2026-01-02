# src/config.py

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = BASE_DIR / "configs" / "core.yaml"
PROMPTS_FILE = BASE_DIR / "configs" / "prompt.yaml"
DATA_DIR = BASE_DIR / "data"


def _load_yaml_config(filename) -> dict:
    """
    Load the core YAML configuration file.

    Returns a dictionary parsed from `filename`
    
    if it exists, otherwise returns an empty dict.
    """

    if filename.exists():
        with open(filename, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get(filename, path: str, default=None) -> Any:
    """
    Retrieve a value from the loaded YAML config using a dotted path.

    Example: `_get("rag.top_k")` will look for `{"rag": {"top_k": ...}}`
    in the parsed YAML and return the value if present, otherwise return
    the provided `default`.
    """
    _yaml_cfg = _load_yaml_config(filename)


    parts = path.split(".")
    cur = _yaml_cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

