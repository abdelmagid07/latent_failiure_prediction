#!/usr/bin/env bash
# Offline smoke test: parse fixture -> mock projections -> de-risking analyses
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Installing packages ==="
pip install -e ../stage1 -q
pip install -e . -q

echo "=== Parse sample.traj fixture ==="
python - <<'PY'
from pathlib import Path
from stage2.trajectories.parse_swe_traj import parse_swe_traj

sample = Path("tests/fixtures/sample.traj")
record = parse_swe_traj(sample, outcome=1)
assert record.n_steps == 3, record.n_steps
assert record.steps[0].assistant_response
print(f"Parsed {sample.name}: {record.trajectory_id}, {record.n_steps} steps")
PY

echo "=== Write smoke normalized trajectories ==="
python - <<'PY'
from pathlib import Path
from stage2.trajectories.mock_data import write_smoke_fixtures

out = Path("data/normalized_smoke")
paths = write_smoke_fixtures(
    out,
    sample_traj=Path("tests/fixtures/sample.traj"),
)
print(f"Wrote {len(paths)} normalized trajectories to {out}")
PY

echo "=== Mock projections (no GPU) ==="
python -m stage2.extract.mock_projections \
  --traj-dir data/normalized_smoke \
  --output data/projections_smoke.parquet

echo "=== De-risking analyses ==="
python -m stage2.analyze.run_derisk \
  --projections data/projections_smoke.parquet \
  --output-dir data/smoke_report

echo "=== Smoke test complete ==="
test -f data/smoke_report/derisk_report.json
test -f data/smoke_report/final_step_separation.png
test -f data/smoke_report/noise_by_token_type.png
echo "All artifacts present."
