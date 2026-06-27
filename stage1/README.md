# Stage 1: Value Axis Reproduction

Self-contained pipeline to reconstruct the value axis from ICRL conversations ([PAPER.tex](../PAPER.tex) Appendix A) and verify held-out AUROC before SWE-bench transfer experiments.

## Gate criteria

| Metric | Threshold | Paper target |
|--------|-----------|--------------|
| Layer 21 AUROC | >= 0.93 | 0.954 |
| Layer 22 AUROC | >= 0.93 | 0.947 |

Exit code 0 from `run_gate.py` means the axis is frozen in `data/value_axis.npy`.

## Quick start (offline smoke test, no GPU)

```bash
cd stage1
pip install -e .
bash scripts/smoke_test.sh
```

This runs: mock ICRL -> synthetic activations -> axis build -> AUROC eval. Gate failure is expected on mock data; the script verifies the pipeline wiring.

## Full pipeline (cloud GPU)

### 1. Environment

```bash
cd stage1
pip install -e .

# HuggingFace login if Qwen3-8B requires it
huggingface-cli login
```

**Recommended cloud:** RunPod or Lambda — 1x A100 40GB or A10G 24GB. Qwen3-8B in bf16 needs ~16GB VRAM.

### 2. Generate mock data (Phase 1) or real ICRL (Phase 2)

```bash
# Phase 1: mock conversations (~6 convos, train + held-out criteria)
python -m stage1.icrl.mock_data

# Phase 2 (TODO): python -m stage1.icrl_gen.generate --n 300
```

### 3. Extract activations

```bash
python -m stage1.pipeline.extract_activations --icrl data/mock_icrl.json
```

Caches per-conversation `.npz` files in `data/activations/`.

### 4. Build axis + evaluate + gate

```bash
python -m stage1.pipeline.run_gate --skip-mock --skip-extract
# Or all-in-one with real extraction:
python -m stage1.pipeline.run_gate --icrl data/mock_icrl.json
```

Outputs:

- `data/value_axis.npy` — shape `(37, 4096)`
- `data/axis_manifest.json` — construction metadata
- `data/auroc_by_layer.json` — per-layer AUROC
- `data/auroc_by_layer.png` — plot vs paper targets

## Locked implementation choices

| Decision | Value |
|----------|-------|
| Model | `Qwen/Qwen3-8B` |
| Hook | `model.model.layers[L]` output |
| Dtype | bfloat16 |
| Chat template | `enable_thinking=False` |
| Axis | post-mean minus pre-mean (Eq. 1) |
| Eval | token-level AUROC, `max(auc, 1-auc)` |
| Split | `config/split.json` (seed 42, 25/25) |

## Token boundary rule (v1)

1. Strip `<thinking>...</thinking>` from assistant turns
2. `satisfying_char_start` = char offset in paragraph where criterion is satisfied
3. Pre tokens: strictly before that offset; post tokens: from offset onward (inclusive)
4. Skip conversation if either set is empty

## Integration with authors' repo

[`value-axis/common/paths.py`](../value-axis/common/paths.py) points at `stage1/data/`. Once the gate passes, their task scripts can load `value_axis.npy` via:

```python
from paths import data_file, DEFAULT_LAYER
probe = data_file("value_axis.npy")
```

Do not proceed to SWE-bench projection until the gate passes on held-out criteria.

## Proxy de-risk track (Qwen local — NOT faithful Stage 1)

Fast path for Jonas noise feasibility: build a **proxy axis** from Qwen3-8B-generated ICRL on the same GPU pod. No Anthropic API key.

| | Faithful Stage 1 | Proxy de-risk |
|--|------------------|---------------|
| ICRL generator | Opus (Anthropic) | Qwen3-8B local |
| N conversations | 300 | 100 |
| Gate threshold | 0.93 | **0.75** |
| Output axis | `value_axis.npy` | `value_axis_proxy.npy` |
| Purpose | Reproduce paper | Noise measurement only |

### Quick start (single GPU pod)

```bash
# From repo root
bash scripts/run_proxy_week1.sh --pilot   # 10 convos test
bash scripts/run_proxy_week1.sh           # full 100 + extract + proxy gate
```

Or step-by-step:

```bash
cd stage1
pip install -e .

# 1. Generate proxy ICRL (GPU, hours for n=100)
python -m stage1.icrl_gen.generate --n 100 --backend local_qwen \
  --output data/icrl_proxy.json --resume --max-turn-retries 8

# 2. Extract + proxy gate (separate activation cache)
python -m stage1.pipeline.extract_activations \
  --icrl data/icrl_proxy.json --activations-dir data/activations_proxy --force
python -m stage1.pipeline.run_proxy_gate --icrl data/icrl_proxy.json --skip-extract
```

**Colab:** [`notebooks/stage1_proxy_gpu_colab.ipynb`](notebooks/stage1_proxy_gpu_colab.ipynb)

**Success:** L21/L22 AUROC ≥ 0.75 → `data/value_axis_proxy.npy` frozen for Stage 2 de-risk.

**Important:** Proxy artifacts never overwrite `value_axis.npy`. Label all Jonas deliverables as "proxy axis, noise feasibility only."

## Phase 2 — ICRL generation (Anthropic API — faithful track)

Generate ~300 conversations per paper Appendix A using Wikipedia seeds + Opus.

### Setup

```bash
pip install -e .
cp .env.example .env   # add ANTHROPIC_API_KEY
```

### Pilot (recommended first)

```bash
python -m stage1.icrl_gen.generate --n 10 --output data/icrl_pilot.json
```

Review `data/icrl_pilot.json`, then scale up:

```bash
python -m stage1.icrl_gen.generate --n 300 --output data/icrl.json --resume
```

`--resume` skips conversations already present in the output file.

### Full pipeline after generation

**Colab (GPU):** open [`notebooks/stage1_gpu_colab.ipynb`](notebooks/stage1_gpu_colab.ipynb) — upload `icrl.json`, run extract + gate on Qwen3-8B.

**CLI (cloud GPU / local):**

```bash
python -m stage1.pipeline.extract_activations --icrl data/icrl.json
python -m stage1.pipeline.run_gate --icrl data/icrl.json --skip-mock
```

### Cost / time estimate

- ~300 conversations × ~5–8 API calls each ≈ 1500–2400 Opus calls
- Rough order of magnitude: **$50–150** depending on model and paragraph length
- Wikipedia fetches are free (no API key)

### Files

- `stage1/icrl_gen/generate.py` — main generator
- `stage1/icrl_gen/wikipedia.py` — random paragraph fetcher
- `stage1/icrl_gen/verify.py` — syntactic checks + semantic LLM judge
- `data/wrong_hypotheses.json` — cached wrong hypotheses per criterion

## Project layout

```
stage1/
  config/criteria.json    # 50 criteria from paper Table 2
  config/split.json       # 25 train / 25 held-out (seed 42)
  data/                   # artifacts (gitignored)
  stage1/common/          # paths, hooks, chat template
  stage1/icrl/            # schema, boundaries, mock data
  stage1/pipeline/        # extract, build, eval, run_gate
  stage1/icrl_gen/        # Wikipedia + Opus ICRL generation
  scripts/smoke_test.sh
```

## Debugging gate failure (full ICRL)

1. Token boundaries / `satisfying_char_start`
2. Chat template / thinking tags
3. Layer hook index
4. Held-out split sensitivity
5. ICRL generation quality

Do not proceed to SWE-bench projection until the gate passes on held-out criteria.
