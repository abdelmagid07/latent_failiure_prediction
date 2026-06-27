#!/usr/bin/env bash
# Run SWE-bench Lite pilot batch with Qwen3-8B served by vLLM.
#
# Prerequisites:
#   1. vLLM: vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 32768 \
#        --chat-template-kwargs '{"enable_thinking": false}'
#   2. Docker (SWE-agent execution environment)
#   3. pip install sweagent  (or: pip install -e ".[swe]")
#
# Environment:
#   VLLM_URL   default http://localhost:8000/v1
#   OUTPUT_DIR default data/trajectories/run_<timestamp>

set -euo pipefail
cd "$(dirname "$0")/.."

VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
INSTANCES_FILE="${1:-config/pilot_instances.txt}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-data/trajectories/run_${TIMESTAMP}}"

if ! command -v sweagent >/dev/null 2>&1; then
  echo "ERROR: sweagent not found. Install with: pip install sweagent"
  exit 1
fi

if [[ ! -f "$INSTANCES_FILE" ]]; then
  echo "ERROR: instances file not found: $INSTANCES_FILE"
  exit 1
fi

# Build filter regex from instance IDs (one line each)
FILTER="$(paste -sd'|' "$INSTANCES_FILE")"

mkdir -p "$OUTPUT_DIR"

echo "=== SWE-agent pilot batch ==="
echo "vLLM URL:     $VLLM_URL"
echo "Instances:    $INSTANCES_FILE ($(wc -l < "$INSTANCES_FILE") IDs)"
echo "Output dir:   $OUTPUT_DIR"
echo ""

export VLLM_URL

# Patch api_base in config via env substitution is not built-in; user should edit
# config/swe_agent_qwen.yaml api_base if not localhost:8000.

sweagent run-batch \
  --config config/swe_agent_qwen.yaml \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split test \
  --instances.filter "$FILTER" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Done ==="
echo "Trajectories in: $OUTPUT_DIR"
echo "Next: python -m stage2.trajectories.ingest_batch --traj-dir $OUTPUT_DIR"
