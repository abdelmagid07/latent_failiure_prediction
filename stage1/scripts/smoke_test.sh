#!/usr/bin/env bash
# Offline smoke test (no GPU): mock ICRL -> fake activations -> build -> eval
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Installing package (editable) ==="
pip install -e . -q

echo "=== Mock ICRL ==="
python -m stage1.icrl.mock_data

echo "=== Fake activations (CPU) ==="
python -m stage1.pipeline.fake_activations

echo "=== Full gate (mock-only, skip extract) ==="
python -m stage1.pipeline.run_gate --skip-extract --skip-mock --mock-only

echo "=== Smoke test complete ==="
