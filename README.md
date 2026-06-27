# Latent Failure Prediction in Long-Horizon Language Agents

Research project: test whether the **value axis** from [Jiang et al. (2026)](https://arxiv.org/abs/2606.17056) transfers to on-policy SWE-bench agent trajectories (Qwen3-8B).

## Repository layout

| Path | Description |
|------|-------------|
| [`CONTEXT.md`](CONTEXT.md) | Project context, locked decisions, null-result logic |
| [`EXPERIMENT.md`](EXPERIMENT.md) | De-risking experiment spec (Jonas) |
| [`PAPER.tex`](PAPER.tex) | Value axis anchor paper (full text) |
| [`stage1/`](stage1/) | **Stage 1 pipeline** — ICRL gen, activation extract, axis build, AUROC gate |
| [`stage2/`](stage2/) | **Stage 2 de-risking** — SWE-bench trajectories, projections, noise analyses |
| [`value-axis/`](value-axis/) | Upstream authors' code (fork) + `common/paths.py` shim |

## Quick start

### Stage 1 — value axis reconstruction

See [`stage1/README.md`](stage1/README.md).

1. **Generate ICRL data** (Anthropic API): `python -m stage1.icrl_gen.generate --n 300 --output data/icrl.json --resume`
2. **GPU extract + gate** (Colab or cloud): [`stage1/notebooks/stage1_gpu_colab.ipynb`](stage1/notebooks/stage1_gpu_colab.ipynb)

Gate: layer 21/22 held-out AUROC ≥ 0.93 → freeze `stage1/data/value_axis.npy`.

### Stage 2 — de-risking experiment

See [`stage2/README.md`](stage2/README.md) and [`EXPERIMENT.md`](EXPERIMENT.md).

**Proxy track (Week 1, recommended first):** Qwen-local ICRL → proxy axis (0.75 gate) → SWE-bench noise check. No Opus API. See [`scripts/run_proxy_week1.sh`](scripts/run_proxy_week1.sh).

**Faithful track:** Opus ICRL → 0.93 gate → same Stage 2 pipeline with `value_axis.npy`.

### Full paper (after de-risking greenlight)

## References

- Value axis paper: arXiv:2606.17056
- Authors' code: https://github.com/nickjiang2378/value-axis
