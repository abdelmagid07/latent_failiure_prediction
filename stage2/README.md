# Stage 2: SWE-bench Trajectory Logging + De-risking Analyses

Measurement-reliability check from [EXPERIMENT.md](../EXPERIMENT.md): can we read the frozen value axis through long agentic trajectories?

**Prerequisite:** Stage 1 gate passed → `../stage1/data/value_axis.npy` exists.

**Proxy de-risk track:** use `../stage1/data/value_axis_proxy.npy` instead (loose 0.75 gate, Qwen-local ICRL). See [proxy defaults](config/proxy_defaults.yaml).

```bash
python -m stage2.extract.project_steps \
  --axis-path ../stage1/data/value_axis_proxy.npy \
  --traj-dir data/normalized

python -m stage2.analyze.run_derisk --proxy \
  --output-dir data/proxy_derisk_report
```

The `--proxy` flag adds Jonas labeling fields from `axis_manifest_proxy.json`.

## Deliverables (for Jonas)

1. **Plot** — final-step success vs failure distributions
2. **Number** — late-bin separation-to-noise ratio
3. **Bar chart** — reasoning vs tool_output projection noise

## Quick start (offline smoke test)

```bash
cd stage2
bash scripts/smoke_test.sh
```

No GPU, no SWE-agent. Parses `tests/fixtures/sample.traj`, builds mock trajectories, runs all three analyses.

## Install

```bash
pip install -e ../stage1
pip install -e .
# Optional for trajectory generation:
pip install -e ".[swe]"
```

## Full pipeline

### Phase 0 — Stage 1 gate

Ensure `../stage1/data/value_axis.npy` exists (Colab notebook or `stage1.pipeline.run_gate`).

### Phase 1 — Generate trajectories (local Docker → remote A100 inference)

Split the work so the laptop never runs the model: **SWE-agent + Docker run locally
(WSL2)** and do all the CPU-heavy container/test execution, while **Qwen3-8B
inference runs on a remote Colab A100** behind an OpenAI-compatible API. SWE-agent
points at that remote endpoint.

```
  WSL2 (local)                              Colab A100 (remote)
  ┌─────────────────────────┐    HTTPS     ┌──────────────────────────┐
  │ SWE-agent + Docker       │ ───────────▶ │ vLLM serve Qwen3-8B       │
  │ (repos, tests, patches)  │  /v1 tunnel  │ + cloudflared tunnel      │
  └─────────────────────────┘              └──────────────────────────┘
```

**On the A100 (server side).** Open
[`notebooks/serve_qwen_colab.ipynb`](notebooks/serve_qwen_colab.ipynb) in Colab with
an **A100** runtime and run all cells. It launches vLLM
(`--chat-template-kwargs '{"enable_thinking": false}'`, which **must** match the
projection step) and opens a Cloudflare tunnel, then prints the `MODEL_API_BASE` to
export locally.

**On your machine (WSL2).** Point the batch runner at the printed URL — no config
edit needed; the script renders a resolved config from these env vars:

```bash
cd stage2
export MODEL_API_BASE="https://<your-tunnel>.trycloudflare.com/v1"
export MODEL_API_KEY="EMPTY"            # or the token you set on the server
export MODEL_NAME="hosted_vllm/Qwen3-8B"
bash scripts/run_pilot_batch.sh config/pilot_instances.txt
```

The runner preflights `${MODEL_API_BASE}/models` and checks Docker before launching,
so a stale tunnel URL fails fast instead of deep inside a container.

Target ~15–25 SWE-bench **Lite** instances; aim for ~5–10 successes and ~5–10
failures. Stop early once balanced.

For an **all-local** run (vLLM on the same box), omit the env vars — the defaults
point at `http://localhost:8000/v1`.

### Phase 2 — Ingest trajectories

```bash
python -m stage2.trajectories.ingest_batch \
  --traj-dir data/trajectories/run_<timestamp> \
  --results data/trajectories/run_<timestamp>/results.json
```

Writes normalized JSON to `data/normalized/` and `data/trajectories_manifest.json`.

### Phase 3 — Extract projections (GPU — run on the A100, not locally)

This step needs **raw residual-stream activations**, which the OpenAI-compatible
API cannot provide — so it loads Qwen3-8B with HuggingFace directly and must run on
the GPU (the same Colab A100). Upload the normalized trajectories + frozen axis to
the A100 and run:

```bash
python -m stage2.extract.project_steps \
  --traj-dir data/normalized \
  --output data/projections.parquet \
  --layer 21
```

Replays each step through Qwen3-8B (HF) and projects onto the frozen axis:
- **reasoning** — last token of assistant response at step *i*
- **tool_output** — last token of step *i* observation in step *i+1* context

Use `--mock-axis` only for smoke tests without Stage 1.

### Phase 4 — De-risking analyses (CPU)

```bash
python -m stage2.analyze.run_derisk \
  --projections data/projections.parquet
```

Outputs in `data/`:
- `snr_by_position.csv`
- `final_step_separation.png`
- `noise_by_token_type.png`
- `derisk_report.json`

## Pilot instance selection

[`config/pilot_instances.txt`](config/pilot_instances.txt) lists ~20 SWE-bench Lite IDs. Criteria:
- Diverse repos (avoid all sympy/django)
- Mix historically easy/hard
- Swap instances if Qwen3-8B solve rate is too low

**Upgrade path:** for the full study, switch to SWE-bench Verified in `swe_agent_qwen.yaml` and expand instance list.

## Locked choices

| Decision | Value |
|----------|-------|
| Model | `Qwen/Qwen3-8B` |
| Scaffold | SWE-agent + Docker local (WSL2); Qwen3-8B inference on remote A100 via OpenAI-compatible API |
| Activation read | HF teacher-forced replay, `enable_thinking=False` |
| Layer | 21 (override from Stage 1 manifest if gate picks L22) |
| Benchmark (pilot) | SWE-bench Lite |

## Project layout

```
stage2/
  config/           defaults, SWE-agent config, pilot instances
  data/             trajectories, projections, reports (gitignored)
  stage2/
    common/         paths, projection helpers
    trajectories/   parse .traj, ingest batch, mock data
    extract/        project_steps, token_spans, mock_projections
    analyze/        SNR, final-step plot, token-type noise
  scripts/          smoke_test.sh, run_pilot_batch.sh
  tests/fixtures/   sample.traj
```

## Interpretation guardrails

- **N is tiny** — directional only, not a transfer claim
- **Measurement check** — tests readability, not whether the axis tracks agentic outcomes
- Report separation-to-noise alongside any null result (see [CONTEXT.md](../CONTEXT.md) §7)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Missing axis | Complete Stage 1 gate first |
| vLLM/HF template mismatch | Same `transformers`, `enable_thinking=False` on both |
| Low solve rate | Easier Lite instances or SWE-bench Verified |
| OOM on replay | Shorter pilot instances; reduce `max-model-len` |
| Empty tool_output rows | Check observation matching in `token_spans.py` logs |
