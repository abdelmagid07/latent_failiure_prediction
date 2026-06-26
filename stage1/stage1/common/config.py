"""Load YAML/JSON config."""

import json
from pathlib import Path
from typing import Any

import yaml

from stage1.common.paths import CONFIG_DIR, config_file


def load_defaults() -> dict[str, Any]:
    with open(config_file("defaults.yaml")) as f:
        return yaml.safe_load(f)


def load_criteria() -> list[dict]:
    with open(config_file("criteria.json")) as f:
        return json.load(f)


def load_split() -> dict[str, list[str]]:
    with open(config_file("split.json")) as f:
        return json.load(f)


def criteria_by_id() -> dict[str, dict]:
    return {c["id"]: c for c in load_criteria()}
