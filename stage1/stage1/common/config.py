"""Load YAML/JSON config."""

import json
from pathlib import Path
from typing import Any

import yaml

from stage1.common.paths import CONFIG_DIR, DATA_DIR, config_file


def load_defaults() -> dict[str, Any]:
    with open(config_file("defaults.yaml")) as f:
        return yaml.safe_load(f)


def load_proxy_defaults() -> dict[str, Any]:
    with open(config_file("proxy_defaults.yaml")) as f:
        cfg = yaml.safe_load(f)
    faithful = load_defaults()
    for key in ("model", "default_layer", "enable_thinking", "dtype", "n_layers", "hidden_dim"):
        cfg.setdefault(key, faithful.get(key))
    cfg["activations_dir"] = DATA_DIR / cfg.get("activations_subdir", "activations_proxy")
    cfg["axis_path"] = DATA_DIR / cfg["axis_output"]
    cfg["manifest_path"] = DATA_DIR / cfg["manifest_output"]
    cfg["auroc_path"] = DATA_DIR / cfg["auroc_output"]
    cfg["plot_path"] = DATA_DIR / cfg["plot_output"]
    cfg["icrl_path"] = DATA_DIR / cfg["icrl_default"]
    return cfg


def load_criteria() -> list[dict]:
    with open(config_file("criteria.json")) as f:
        return json.load(f)


def load_split() -> dict[str, list[str]]:
    with open(config_file("split.json")) as f:
        return json.load(f)


def criteria_by_id() -> dict[str, dict]:
    return {c["id"]: c for c in load_criteria()}
