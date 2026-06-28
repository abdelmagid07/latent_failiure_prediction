"""Offline tests for mini instance catalog (no model API)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from stage2.mini.catalog import MINI_INSTANCES, list_instance_ids
from stage2.mini.evaluate import evaluate_instance
from stage2.mini.sandbox import materialize_repo


def test_catalog_has_instances():
    assert len(list_instance_ids()) >= 10


def test_buggy_repos_fail_pytest():
    for iid in list_instance_ids():
        instance = MINI_INSTANCES[iid]
        with tempfile.TemporaryDirectory() as tmp:
            repo = materialize_repo(instance, Path(tmp))
            assert not evaluate_instance(instance, repo), iid


def test_fixed_repo_passes():
    """Sanity: applying the obvious fix makes tests pass."""
    fixes = {
        "mini_add_001": ("calc.py", "def add(a, b):\n    return a + b\n"),
        "mini_sign_002": (
            "stats.py",
            "def mean(values):\n"
            "    total = 0\n"
            "    for v in values:\n"
            "        total += v\n"
            "    return total / len(values)\n",
        ),
        "mini_offby_003": (
            "loops.py",
            "def count_up_to(n: int) -> list[int]:\n"
            "    return list(range(1, n + 1))\n",
        ),
    }
    for iid, (rel, content) in fixes.items():
        instance = MINI_INSTANCES[iid]
        with tempfile.TemporaryDirectory() as tmp:
            repo = materialize_repo(instance, Path(tmp))
            (repo / rel).write_text(content, encoding="utf-8")
            assert evaluate_instance(instance, repo), iid
