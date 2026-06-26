#!/usr/bin/env python
"""Orchestrate mock/full Stage 1 pipeline: mock data -> extract -> build -> eval -> gate."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from stage1.common.config import load_defaults
from stage1.common.paths import ACTIVATIONS_DIR, data_file
from stage1.icrl.mock_data import write_mock_icrl
from stage1.pipeline.build_axis import build_axis
from stage1.pipeline.eval_auroc import eval_auroc, plot_auroc
from stage1.pipeline.extract_activations import run as extract_run


def check_gate(auroc_by_layer: dict, gate_layers: list[int], threshold: float) -> bool:
    ok = True
    for layer in gate_layers:
        val = auroc_by_layer.get(str(layer), float("nan"))
        if np.isnan(val) or val < threshold:
            ok = False
    return ok


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--icrl", type=Path, default=None, help="ICRL JSON (default: mock_icrl.json)")
    ap.add_argument("--skip-extract", action="store_true", help="Use cached activations only")
    ap.add_argument("--skip-mock", action="store_true", help="Do not regenerate mock_icrl.json")
    ap.add_argument("--model", default=defaults["model"])
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--mock-only", action="store_true", help="Skip gate failure exit (smoke test)")
    args = ap.parse_args()

    icrl_path = args.icrl or data_file("mock_icrl.json")

    if not args.skip_mock and args.icrl is None:
        write_mock_icrl(icrl_path)
        print(f"Wrote mock ICRL -> {icrl_path}", flush=True)

    if not args.skip_extract:
        print("=== extract_activations ===", flush=True)
        extract_run(
            icrl_path,
            model_name=args.model,
            n_layers=defaults["n_layers"],
            enable_thinking=defaults["enable_thinking"],
            dtype=defaults["dtype"],
            force=args.force_extract,
        )

    from stage1.common.config import load_split

    split = load_split()
    train_criteria = set(split["train"])
    held_out = set(split["held_out"])

    print("=== build_axis ===", flush=True)
    axis, meta = build_axis(ACTIVATIONS_DIR, train_criteria, defaults["n_layers"])
    axis_path = data_file("value_axis.npy")
    np.save(axis_path, axis)

    manifest_path = data_file("axis_manifest.json")
    from datetime import datetime, timezone

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(axis_path),
        "shape": list(axis.shape),
        **meta,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved axis -> {axis_path}", flush=True)

    print("=== eval_auroc ===", flush=True)
    results = eval_auroc(axis, ACTIVATIONS_DIR, held_out, defaults["n_layers"])
    auroc_path = data_file("auroc_by_layer.json")
    out = {
        **results,
        "paper_targets": {str(k): v for k, v in defaults["paper_targets"].items()},
        "gate_threshold": defaults["gate_threshold"],
        "gate_layers": defaults["gate_layers"],
    }
    with open(auroc_path, "w") as f:
        json.dump(out, f, indent=2)
    plot_auroc(results["auroc_by_layer"], data_file("auroc_by_layer.png"), defaults["paper_targets"])

    passed = check_gate(
        results["auroc_by_layer"],
        defaults["gate_layers"],
        defaults["gate_threshold"],
    )

    print("\n=== GATE RESULT ===", flush=True)
    for layer in defaults["gate_layers"]:
        val = results["auroc_by_layer"].get(str(layer), float("nan"))
        status = "PASS" if not np.isnan(val) and val >= defaults["gate_threshold"] else "FAIL"
        print(f"  L{layer}: {val:.4f}  [{status}]  (threshold {defaults['gate_threshold']})", flush=True)

    if passed:
        print("GATE PASSED — axis frozen.", flush=True)
        sys.exit(0)
    if args.mock_only:
        print("GATE FAILED (expected on mock data — pipeline smoke test OK).", flush=True)
        sys.exit(0)
    print("GATE FAILED — debug before SWE-bench projection.", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
