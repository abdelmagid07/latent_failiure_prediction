# HANDOFF — Latent Failure Prediction (value-axis transfer de-risk)

> Read this first, then [CONTEXT.md](CONTEXT.md) and [EXPERIMENT.md](EXPERIMENT.md) for the
> full scientific framing. This file captures live state, decisions, and the exact next
> actions as of **2026-06-28**.

---

## 1. What this project is (30-second version)

Testing whether the **value axis** (a single linear direction in Qwen3-8B's residual
stream encoding "am I on track?", from Jiang/Kauvar/Lindsey 2026, arXiv:2606.17056)
survives transfer from short single-turn tasks to **long-horizon SWE-bench agent
trajectories**. Deliverable: NeurIPS-workshop paper.

**Current focus is the DE-RISK experiment**, not the full transfer test. The de-risk is a
**measurement-reliability check**: can the value-axis projection be read off long agentic
trajectories cleanly enough that real signal wouldn't drown in noise? It is NOT a transfer
claim (see [EXPERIMENT.md](EXPERIMENT.md) and [CONTEXT.md](CONTEXT.md) §7 on the two kinds
of null result).

---

## 2. Two tracks — don't confuse them

| | **Proxy track (current)** | **Faithful track (later)** |
|--|--|--|
| ICRL generator | Qwen3-8B local | Claude Opus (API) |
| Gate threshold | AUROC ≥ **0.75** | ≥ **0.93** |
| Axis file | `value_axis_proxy.npy` | `value_axis.npy` |
| Purpose | noise-feasibility only | reproduce paper; required before any transfer claim |

We are doing the **proxy track** to answer the noise question cheaply. The faithful track
is still required before any transfer claim but is NOT today's work.

---

## 3. Status

### ✅ Stage 1 (proxy) — COMPLETE
- `value_axis_proxy.npy` + `axis_manifest_proxy.json` + `auroc_by_layer_proxy.png` built
  and **downloaded locally by the user** (artifacts are gitignored, not in the repo).
- Built from **67 conversations** (target was 100; Colab runtime disconnected overnight,
  but 67 syntactic-only convos were enough).
- **Gate PASSED** (both L21 and L22 ≥ 0.75). Pilot N=10 was L21=0.884 / L22=0.905; exact
  N=67 numbers are in the downloaded `axis_manifest_proxy.json`. The AUROC-by-layer plot
  peaks at L21/22 exactly as the paper predicts.
- Generated with **`--syntactic-only`** mode (15 programmatic criteria, no LLM judge) —
  see §5 for why this was necessary.

### ⏳ Stage 2 — CODE READY, NOT STARTED on real data
- All code + two Colab notebooks exist and pass offline smoke tests.
- **Mini local-agent track added** (`scripts/run_mini_batch.sh`) — no Docker, 12 bug-fix
  tasks, SWE-agent-compatible `.traj` output. Good partial de-risk while Docker/WSL2 is
  being set up.
- Full SWE-bench path still blocked on local Docker/WSL2 setup (see §6).

### ❌ Faithful Stage 1 — not started (needs Anthropic API key; deferred).

---

## 4. Environment / logistics

- **Repo:** https://github.com/abdelmagid07/latent_failiure_prediction (branch `main`,
  public). Note the typo "failiure" is part of the real repo name.
- **Git HEAD at handoff:** `4d9c4b6`. Working tree clean except this file.
- **GPU:** all GPU work runs on a **Colab A100** (the user's local machine is Windows 11,
  torch is CPU-only). Colab notebooks `git clone` from `main`, so **any code fix must be
  committed + pushed before it reaches Colab.**
- **Colab disconnects** during multi-hour runs — generation and extraction are slow; save
  intermediate artifacts (e.g. `icrl_proxy.json`) and resume.
- `.gitattributes` forces LF on `*.sh` (WSL2 needs it; `core.autocrlf=true` locally).
- The user commits/pushes themselves OR asks Claude to; **do not put Claude's name in
  commit messages** (explicit user preference). Confirm before pushing.

### Model facts (locked)
- `Qwen/Qwen3-8B`, **36 decoder layers** (indices 0–35) — the original config said 37,
  which was wrong and caused an `IndexError`. Now configs say 36 and the code auto-detects.
- `enable_thinking=False` everywhere (generation, extraction, projection) — must match or
  activations won't align.
- Value axis read at **layer 21** (L22 also strong).

---

## 5. Key things we learned / fixed (so you don't re-discover them)

1. **Qwen3-8B has 36 layers, not 37.** Fixed in configs + `build_axis`/`eval_auroc`/
   `extract_activations` now auto-detect layer count from the model/activation shape.
2. **The Qwen self-judge is unreliable on *semantic* criteria** — it rubber-stamped
   `satisfies=true` on paragraphs that clearly didn't satisfy. Two causes:
   (a) a real bug — `bool("false")` is `True` in Python (fixed via `_coerce_bool` in
   `verify.py`); (b) Qwen-8B is just a weak semantic judge. **Solution: proxy axis uses
   `--syntactic-only`** (15 programmatically-checked criteria, zero LLM judge). This is why
   Stage 1 passed cleanly.
3. **JSON-parse crashes** killed long runs — hardened with tolerant parsing + retry
   (`json_utils.py`) and per-conversation skip-and-continue in `generate.py`.
4. **Wikipedia 429 rate-limiting** at scale — fixed with batched fetches (≤20/request via
   `grnlimit`+`exlimit=max`) + exponential backoff in `wikipedia.py`.
5. **bf16 activations** need `.cpu().float()` before numpy (already in `hooks.py`).
6. **Some syntactic criteria have high skip rates** (c001 digits, c006 parens, c008 `;`,
   c009 dash) because Wikipedia text already contains those chars, so the "before-discovery
   must NOT satisfy" turn is hard. This lowers yield (~67/100) but does NOT corrupt the
   axis. Easy criteria (emoji, `!`, `$`, `...`) succeed readily.
7. **Extraction is SLOW** (~1–2 h for 67 convos), not fast — full 36-layer forward pass
   over each long context. (Claude wrongly called it "fast" once; it isn't.)
8. **SWE-bench: "submitted" ≠ "resolved".** Real success labels come ONLY from the
   SWE-bench evaluation harness (`resolved_ids`). `parse_swe_traj` now parses that report
   format and `ingest_batch` warns loudly if labels are missing or single-class.

---

## 6. NEXT ACTIONS — Stage 2 (this is where to resume)

Full runbook is in [stage2/README.md](stage2/README.md) Phase 1–4 + the notebooks. Summary:

### Option A — Mini local-agent (Colab only, no Docker)

Three notebooks in order — see [stage2/README.md](stage2/README.md):

1. `stage2/notebooks/serve_qwen_colab.ipynb` (A100)
2. `stage2/notebooks/mini_agent_colab.ipynb` (CPU) — batch + ingest + download `normalized.zip`
3. `stage2/notebooks/project_and_analyze_colab.ipynb` (A100)

Offline sanity (optional): `pytest stage2/tests/test_mini_catalog.py`.

### Option B — Full SWE-bench (Docker + WSL2)

**Architecture:** SWE-agent + Docker run **locally (WSL2)**; Qwen3-8B inference runs on a
**remote Colab A100** (vLLM + cloudflared tunnel); SWE-agent points at the tunnel.
Projection (`project_steps.py`) needs raw activations so it also runs **on the A100**.

**Phase 0 — local prerequisites (GATE — confirm before anything else):**
Docker Desktop + WSL2 integration; `pip install sweagent swebench`; ~50–100 GB disk; repo
cloned in WSL2. **Open question for the user: is this set up yet, or is it the first task?**

**Phase 1 — serve model:** run `stage2/notebooks/serve_qwen_colab.ipynb` on an A100; copy
the printed `MODEL_API_BASE` tunnel URL; keep the tab alive.

**Phase 2 — generate (local):**
`export MODEL_API_BASE=...` then `bash scripts/run_pilot_batch.sh config/pilot_instances.txt`.
Start with 2–3 instances to confirm the tunnel works end-to-end, then scale to ~20.

**Phase 3 — REAL labels (local, do NOT skip):** run the SWE-bench evaluation harness on the
predictions to get a report with `resolved_ids`. Exact `swebench` CLI flags vary by version
— verify against the installed version.

**Phase 4 — ingest (local):**
`python -m stage2.trajectories.ingest_batch --traj-dir <dir> --results <report>.json`.
Confirm a mix of success/failure (ideally ~5–10 each). Zip `data/normalized/` → `normalized.zip`.

**Phase 5 — project + analyze (A100):** run
`stage2/notebooks/project_and_analyze_colab.ipynb`; upload `value_axis_proxy.npy` +
`axis_manifest_proxy.json` + `normalized.zip`; it projects + runs the 3 analyses + downloads.

**Phase 6 — DONE:** `derisk_report.json` + `final_step_separation.png` +
`noise_by_token_type.png` + `snr_by_position.csv` = the Jonas deliverable (one plot, one
number, one bar chart).

### Biggest Stage 2 risk
**Qwen3-8B's SWE-bench solve rate.** It may resolve very few instances → too few "success"
trajectories → no class balance. If after Phase 4 there are ~0–1 successes, adapt (easier
instances, or scope what the de-risk can claim). The analysis code returns `nan` AUROC on a
single class rather than crashing — that's honest, not a bug.

---

## 7. Colab quick links
- Stage 1 proxy (done): `stage1/notebooks/stage1_proxy_gpu_colab.ipynb`
- Stage 2 serve model: `stage2/notebooks/serve_qwen_colab.ipynb`
- Stage 2 mini agent batch (CPU): `stage2/notebooks/mini_agent_colab.ipynb`
- Stage 2 project+analyze: `stage2/notebooks/project_and_analyze_colab.ipynb`

Open any via `https://colab.research.google.com/github/abdelmagid07/latent_failiure_prediction/blob/main/<path>`

**Mini track order:** serve (A100) → mini_agent (CPU) → project_and_analyze (A100).

---

## 8. Working agreements with the user
- Be honest about uncertainty; don't claim something is "fast"/"done" without basis.
- Verify code against real-data reality — offline smoke tests hid real bugs in both stages.
- Commit messages: no Claude attribution line. Confirm before pushing to the public repo.
- Prefer concrete recommendations over option-dumps; the user is action-oriented.
