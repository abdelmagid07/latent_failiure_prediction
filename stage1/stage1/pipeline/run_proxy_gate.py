#!/usr/bin/env python
"""Proxy Stage 1 gate: loose 0.75 threshold, separate artifact files for de-risk track."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage1.common.config import load_proxy_defaults, load_split
from stage1.common.paths import data_file
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
    proxy = load_proxy_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--icrl",
        type=Path,
        default=proxy["icrl_path"],
        help="Proxy ICRL JSON (default: data/icrl_proxy.json)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=proxy["gate_threshold"],
        help="Loose proxy gate threshold (default 0.75)",
    )
    ap.add_argument("--skip-extract", action="store_true", help="Use cached activations only")
    ap.add_argument("--model", default=proxy["model"])
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()

    activations_dir: Path = proxy["activations_dir"]
    icrl_path = args.icrl

    if not icrl_path.exists():
        raise SystemExit(
            f"Proxy ICRL not found: {icrl_path}\n"
            "Run: python -m stage1.icrl_gen.generate --n 100 --backend local_qwen "
            "--output data/icrl_proxy.json --resume"
        )

    if not args.skip_extract:
        print("=== extract_activations (proxy) ===", flush=True)
        extract_run(
            icrl_path,
            model_name=args.model,
            n_layers=proxy["n_layers"],
            enable_thinking=proxy["enable_thinking"],
            dtype=proxy["dtype"],
            force=args.force_extract,
            activations_dir=activations_dir,
        )

    split = load_split()
    train_criteria = set(split["train"])
    held_out = set(split["held_out"])

    print("=== build_axis (proxy) ===", flush=True)
    axis, meta = build_axis(activations_dir, train_criteria, proxy["n_layers"])
    axis_path: Path = proxy["axis_path"]
    np.save(axis_path, axis)

    results = eval_auroc(axis, activations_dir, held_out, proxy["n_layers"])

    passed = check_gate(
        results["auroc_by_layer"],
        proxy["gate_layers"],
        args.threshold,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(axis_path),
        "shape": list(axis.shape),
        "axis_type": proxy["axis_type"],
        "generator": proxy["generator"],
        "purpose": proxy["purpose"],
        "gate_threshold": args.threshold,
        "gate_layers": proxy["gate_layers"],
        "gate_passed": passed,
        "auroc_by_layer": results["auroc_by_layer"],
        "n_held_out_conversations": results.get("n_held_out_conversations"),
        "icrl_path": str(icrl_path),
        "activations_dir": str(activations_dir),
        **meta,
    }
    manifest_path: Path = proxy["manifest_path"]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved proxy axis -> {axis_path}", flush=True)
    print(f"Saved proxy manifest -> {manifest_path}", flush=True)

    auroc_out = {
        **results,
        "gate_threshold": args.threshold,
        "gate_layers": proxy["gate_layers"],
        "axis_type": proxy["axis_type"],
        "generator": proxy["generator"],
    }
    with open(proxy["auroc_path"], "w") as f:
        json.dump(auroc_out, f, indent=2)
    plot_auroc(
        results["auroc_by_layer"],
        proxy["plot_path"],
        paper_targets=None,
    )

    print("\n=== PROXY GATE RESULT ===", flush=True)
    print(f"  axis_type: {proxy['axis_type']}", flush=True)
    print(f"  generator: {proxy['generator']}", flush=True)
    print(f"  purpose:   {proxy['purpose']}", flush=True)
    for layer in proxy["gate_layers"]:
        val = results["auroc_by_layer"].get(str(layer), float("nan"))
        status = "PASS" if not np.isnan(val) and val >= args.threshold else "FAIL"
        print(f"  L{layer}: {val:.4f}  [{status}]  (proxy threshold {args.threshold})", flush=True)

    if passed:
        print("PROXY GATE PASSED — axis frozen for noise de-risk only.", flush=True)
        print("NOTE: This is NOT faithful Stage 1. Do not use for transfer claims.", flush=True)
        sys.exit(0)

    print("PROXY GATE FAILED — regenerate proxy ICRL or improve Qwen generation.", flush=True)
    print(f"Faithful artifacts ({data_file('value_axis.npy')}) were NOT modified.", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
