#!/usr/bin/env bash
# Week 1 proxy de-risk track: Qwen-local ICRL -> proxy axis -> Stage 2 de-risk
#
# Phase A (single GPU pod): proxy axis — no Anthropic API key required
# Phase B: SWE-agent trajectories (needs vLLM + Docker) — run manually after Phase A
#
# Usage:
#   bash scripts/run_proxy_week1.sh              # Phase A only
#   bash scripts/run_proxy_week1.sh --pilot     # 10 convos for quick test
#   bash scripts/run_proxy_week1.sh --phase-b TRAJ_DIR  # Phase B after SWE-agent

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PILOT=false
PHASE_B=""
N=100
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pilot) PILOT=true; N=10; shift ;;
    --phase-b) PHASE_B="${2:-}"; shift 2 ;;
    --n) N="${2:-100}"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== Install packages ==="
pip install -e stage1 -q
pip install -e stage2 -q

if [[ -n "$PHASE_B" ]]; then
  echo "=== Phase B: Stage 2 de-risk (proxy axis) ==="
  cd stage2
  python -m stage2.trajectories.ingest_batch \
    --traj-dir "$PHASE_B" \
    --results "$PHASE_B/results.json"
  python -m stage2.extract.project_steps \
    --traj-dir data/normalized \
    --output data/projections.parquet \
    --axis-path ../stage1/data/value_axis_proxy.npy
  python -m stage2.analyze.run_derisk \
    --projections data/projections.parquet \
    --output-dir data/proxy_derisk_report \
    --proxy
  echo "=== Phase B complete ==="
  echo "Report: stage2/data/proxy_derisk_report/derisk_report.json"
  exit 0
fi

echo "=== Phase A: Proxy axis (Qwen local ICRL) ==="
cd stage1

OUTPUT="data/icrl_proxy.json"
if $PILOT; then
  OUTPUT="data/icrl_proxy_pilot.json"
fi

python -m stage1.icrl_gen.generate \
  --n "$N" \
  --backend local_qwen \
  --output "$OUTPUT" \
  --resume \
  --max-turn-retries 8

python -m stage1.pipeline.extract_activations \
  --icrl "$OUTPUT" \
  --activations-dir data/activations_proxy \
  --force

python -m stage1.pipeline.run_proxy_gate \
  --icrl "$OUTPUT" \
  --skip-extract

echo ""
echo "=== Phase A complete ==="
echo "Proxy axis:  stage1/data/value_axis_proxy.npy"
echo "Manifest:    stage1/data/axis_manifest_proxy.json"
echo ""
echo "Next (Phase B — requires vLLM + Docker for SWE-agent):"
echo "  1. Start vLLM: vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 32768 \\"
echo "       --chat-template-kwargs '{\"enable_thinking\": false}'"
echo "  2. cd stage2 && bash scripts/run_pilot_batch.sh"
echo "  3. bash scripts/run_proxy_week1.sh --phase-b stage2/data/trajectories/run_<timestamp>"
