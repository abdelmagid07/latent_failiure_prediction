#!/usr/bin/env python
"""Run all three de-risking analyses and write report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stage2.analyze.final_step import final_step_plot
from stage2.analyze.noise_by_token_type import noise_by_token_type
from stage2.analyze.signal_to_noise import headline_late_bin_snr, signal_to_noise_by_position
from stage2.common.config import load_defaults, load_proxy_defaults
from stage2.common.paths import data_file


def load_axis_manifest(manifest_path: Path | None) -> dict:
    if manifest_path is None or not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        return json.load(f)


def build_proxy_metadata(manifest: dict) -> dict:
    if not manifest or manifest.get("axis_type") != "proxy":
        return {}

    auroc = manifest.get("auroc_by_layer", {})
    l21 = auroc.get("21") if "21" in auroc else auroc.get(21)
    l22 = auroc.get("22") if "22" in auroc else auroc.get(22)
    threshold = manifest.get("gate_threshold", 0.75)
    l21_s = f"{l21:.3f}" if l21 is not None else "N/A"
    l22_s = f"{l22:.3f}" if l22 is not None else "N/A"

    return {
        "experiment_type": "proxy_noise_feasibility",
        "axis_type": manifest.get("axis_type", "proxy"),
        "axis_generator": manifest.get("generator", "qwen3-8b-local"),
        "proxy_single_turn_auroc_l21": l21,
        "proxy_single_turn_auroc_l22": l22,
        "proxy_gate_threshold": threshold,
        "proxy_gate_passed": manifest.get("gate_passed"),
        "disclaimer": (
            "Proxy axis built from Qwen3-8B-local ICRL for noise measurement only. "
            "Faithful Stage 1 (Opus/authors' data, 0.93 gate) is pending."
        ),
        "jonas_summary_template": (
            f"Noise feasibility probe using a proxy axis (Qwen-local ICRL, "
            f"L21 AUROC={l21_s}, L22 AUROC={l22_s}, proxy gate>{threshold}). "
            "This tests trajectory noise readability, not value-axis transfer."
        ),
    }


def interpret_report(
    *,
    late_snr: float | None,
    separability: float,
    reasoning_std: float | None,
    tool_std: float | None,
) -> str:
    lines = []

    if late_snr is not None and late_snr >= 1.0:
        lines.append(
            "Late-bin separation-to-noise is at or above ~1: class gap rivals within-class jitter."
        )
    elif late_snr is not None:
        lines.append(
            "Late-bin separation-to-noise is below ~1: noise may dominate at this sample size."
        )
    else:
        lines.append("Insufficient data for late-bin SNR (need >=2 per class per bin).")

    if reasoning_std is not None and tool_std is not None and reasoning_std < tool_std:
        lines.append(
            f"Reasoning-token std ({reasoning_std:.3f}) is lower than tool-output std ({tool_std:.3f}), "
            "supporting the extraction discipline."
        )
    elif reasoning_std is not None and tool_std is not None:
        lines.append(
            f"Reasoning-token std ({reasoning_std:.3f}) is not clearly lower than "
            f"tool-output std ({tool_std:.3f}) at this N."
        )

    return " ".join(lines)


def run_derisk(
    projections_path: Path,
    *,
    n_bins: int = 5,
    output_dir: Path | None = None,
    axis_manifest_path: Path | None = None,
) -> dict:
    output_dir = output_dir or data_file("").parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(projections_path)

    snr_df = signal_to_noise_by_position(df, n_bins=n_bins)
    snr_csv = output_dir / "snr_by_position.csv"
    snr_df.to_csv(snr_csv, index=False)

    final_results = final_step_plot(
        df,
        output_dir / "final_step_separation.png",
    )

    token_summary = noise_by_token_type(
        df,
        output_dir / "noise_by_token_type.png",
    )

    late_snr = headline_late_bin_snr(snr_df)

    std_map = {
        row["token_type"]: float(row["std"])
        for _, row in token_summary.iterrows()
    }
    reasoning_std = std_map.get("reasoning")
    tool_std = std_map.get("tool_output")
    std_ratio = (
        reasoning_std / tool_std
        if reasoning_std is not None and tool_std and tool_std > 0
        else None
    )

    n_traj = df["trajectory_id"].nunique()
    n_success = df.groupby("trajectory_id")["outcome"].first().eq(1).sum()
    n_failure = n_traj - n_success

    report = {
        "n_trajectories": int(n_traj),
        "n_success": int(n_success),
        "n_failure": int(n_failure),
        "n_projection_rows": len(df),
        "late_bin_separation_to_noise": late_snr,
        "final_step_auroc": final_results["auroc"],
        "final_step_separability": final_results["separability"],
        "reasoning_projection_std": reasoning_std,
        "tool_output_projection_std": tool_std,
        "reasoning_to_tool_std_ratio": std_ratio,
        "snr_by_position_csv": str(snr_csv),
        "final_step_plot": final_results["plot_path"],
        "noise_by_token_type_plot": str(output_dir / "noise_by_token_type.png"),
        "interpretation": interpret_report(
            late_snr=late_snr,
            separability=final_results.get("separability", float("nan")),
            reasoning_std=reasoning_std,
            tool_std=tool_std,
        ),
        "guardrails": [
            "N is tiny; this is directional, not conclusive.",
            "This is a measurement check, not a transfer result.",
            "Separation-to-noise ratio is the first look at signal vs noise floor.",
        ],
    }

    manifest = load_axis_manifest(axis_manifest_path)
    proxy_meta = build_proxy_metadata(manifest)
    if proxy_meta:
        report.update(proxy_meta)

    report_path = output_dir / "derisk_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2), flush=True)
    print(f"\nReport written to {report_path}", flush=True)
    return report


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--projections",
        type=Path,
        default=data_file("projections.parquet"),
    )
    ap.add_argument("--n-bins", type=int, default=defaults["n_bins"])
    ap.add_argument("--output-dir", type=Path, default=data_file("").parent)
    ap.add_argument(
        "--proxy",
        action="store_true",
        help="Load proxy axis manifest for Jonas labeling",
    )
    ap.add_argument(
        "--axis-manifest",
        type=Path,
        default=None,
        help="Path to axis_manifest_proxy.json (default: from proxy_defaults if --proxy)",
    )
    args = ap.parse_args()

    if not args.projections.exists():
        raise SystemExit(f"Projections file not found: {args.projections}")

    manifest_path = args.axis_manifest
    if args.proxy and manifest_path is None:
        proxy_cfg = load_proxy_defaults()
        manifest_path = proxy_cfg.get("axis_manifest_path")

    run_derisk(
        args.projections,
        n_bins=args.n_bins,
        output_dir=args.output_dir,
        axis_manifest_path=manifest_path,
    )


if __name__ == "__main__":
    main()
