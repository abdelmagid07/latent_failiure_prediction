#!/usr/bin/env python
"""Ingest a batch of SWE-agent .traj files and write normalized JSON + manifest."""

import argparse
import json
from pathlib import Path

from stage2.common.paths import NORMALIZED_DIR, TRAJ_DIR, data_file
from stage2.trajectories.parse_swe_traj import load_results_map, parse_swe_traj
from stage2.trajectories.schema import save_trajectory


def ingest_batch(
    traj_dir: Path,
    *,
    results_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    output_dir = output_dir or NORMALIZED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    outcome_map = load_results_map(results_path) if results_path else {}

    traj_files = sorted(traj_dir.glob("*.traj"))
    if not traj_files:
        raise FileNotFoundError(f"No .traj files found in {traj_dir}")

    records = []
    for traj_path in traj_files:
        iid = traj_path.stem
        outcome = outcome_map.get(iid)
        record = parse_swe_traj(traj_path, trajectory_id=iid, outcome=outcome)
        out_path = output_dir / f"{iid}.json"
        save_trajectory(record, out_path)
        records.append(record)
        print(f"  {iid}: {record.n_steps} steps, outcome={record.outcome}", flush=True)

    n_success = sum(1 for r in records if r.outcome == 1)
    n_failure = len(records) - n_success
    mean_steps = sum(r.n_steps for r in records) / len(records) if records else 0.0

    manifest = {
        "n_trajectories": len(records),
        "n_success": n_success,
        "n_failure": n_failure,
        "success_rate": n_success / len(records) if records else 0.0,
        "mean_steps": mean_steps,
        "trajectory_ids": [r.trajectory_id for r in records],
        "normalized_dir": str(output_dir),
    }

    manifest_path = data_file("trajectories_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {manifest_path}", flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--traj-dir",
        type=Path,
        default=TRAJ_DIR,
        help="Directory containing *.traj files",
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Optional SWE-bench results.json for outcome labels",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=NORMALIZED_DIR,
        help="Where to write normalized trajectory JSON",
    )
    args = ap.parse_args()

    results = args.results
    if results is None:
        candidate = args.traj_dir / "results.json"
        if candidate.exists():
            results = candidate

    print(f"Ingesting from {args.traj_dir}...", flush=True)
    ingest_batch(args.traj_dir, results_path=results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
