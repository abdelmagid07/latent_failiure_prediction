"""Load Stage 2 YAML config."""

from typing import Any

import yaml

from stage2.common.paths import CONFIG_DIR, config_file, resolve_axis_path


def load_defaults() -> dict[str, Any]:
    with open(config_file("defaults.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["axis_path"] = resolve_axis_path(cfg["axis_path"])
    if "axis_manifest_path" in cfg:
        cfg["axis_manifest_path"] = resolve_axis_path(cfg["axis_manifest_path"])
    return cfg


def load_proxy_defaults() -> dict[str, Any]:
    with open(config_file("proxy_defaults.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["axis_path"] = resolve_axis_path(cfg["axis_path"])
    if "axis_manifest_path" in cfg:
        cfg["axis_manifest_path"] = resolve_axis_path(cfg["axis_manifest_path"])
    return cfg
