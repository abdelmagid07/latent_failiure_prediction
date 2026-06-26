# context.md — Latent Failure Prediction in Long-Horizon Language Agents

> Context file for AI coding agents (Cursor, etc.). Describes the research project, the
> proposal, key technical decisions, and the immediate next task to implement. Read this
> before writing code. Prefer the facts here over assumptions; if something is unspecified,
> ask rather than inventing.

---

## 1. One-paragraph project summary

We are testing whether the **value axis** — a single linear direction in an LLM's residual
stream that encodes whether the model believes it is "on track" — still works when ported
from short single-turn tasks (where it was established) to **long-horizon agentic
trajectories** (where it has never been tested). Concretely: we **freeze** the value axis
constructed in the original single-turn setting and project it onto activations from
multi-step SWE-bench coding-agent trajectories, testing whether it separates eventually-
successful runs from eventually-failed ones. The deliverable is a NeurIPS-workshop paper.

## 2. The anchor result we build on

- **Paper:** Jiang, Kauvar, Lindsey (2026), "The Value Axis: Language Models Encode Whether
  They're on the Right Track." arXiv: **2606.17056**.
- **Model used there (and by us):** **Qwen3-8B**.
- **Construction:** difference-in-means between residual-stream activations *after* vs.
  *before* a "criterion discovery" moment → a single linear direction (the value axis).
- **Projection:** dot/cosine alignment of an activation with that direction → scalar
  ("high = on track").
- **Key reported facts:** held-out AUROC > ~0.9; strongest, most generalizable separation in
  **middle-to-late layers** (handoff notes referenced layers ~21–22 as best — re-verify, do
  not hardcode).
- **Their stated limitation (our entire opening):** only validated on short-horizon,
  single-turn tasks (AIME, LeetCode, single chat prompts). They did **not** test long agentic
  trajectories.
- **Their code:** public on GitHub (search "value-axis" / nickjiang2378). Use it for Stage 1
  reconstruction.

## 3. Research question

> Does the value axis persist, remain identifiable, and track eventual success or failure in
> long-horizon agentic trajectories?

Framed as a **transfer test**, not a discovery task: take a *known* signal with a published
construction and check whether it survives a change of regime. Designed so all three outcomes
are publishable:
- **Clean transfer** → signal persists into agentic regime (early-warning use case).
- **Partial transfer** → characterize where/when it degrades.
- **No transfer** → scopes the original (single-turn-only) result. Valid negative result.

## 4. Method (4 stages)

All quantities are measured as a function of **relative trajectory position** (fraction of
total trajectory length, NOT absolute step index), so trajectories of different lengths share
a common axis.

1. **Stage 1 — Reconstruct & verify.** Rebuild the value axis on Qwen3-8B from the original
   single-turn ICRL data using the authors' released code. Confirm held-out AUROC matches
   their report. This is a checkpoint/gate: if it doesn't reproduce, STOP and debug before
   anything downstream. Once verified, the axis is **frozen** (no refitting on SWE-bench).

2. **Stage 2 — Final-step transfer.** Project the frozen axis onto agentic-trajectory
   activations. First test: does the projection at the **final step** separate eventually-
   successful from eventually-failed runs? Report AUROC vs. a **majority-class baseline**
   (NOT 0.5 — classes are imbalanced). Cheapest, least-noisy, most-favorable probe; gates the
   rest.

3. **Stage 3 — Trajectory evolution + prediction baseline.** Trace the projection across the
   full trajectory, binned by relative position; compare mean projection of success vs.
   failure runs. Also elicit the agent's **own step-wise estimate** of P(eventual success)
   (single number in [0,1], by prompting it directly; optionally also a separate LLM judge),
   and compare both the internal projection and the elicited estimate against the ground-truth
   outcome. The elicited estimate is a **baseline**, not the headline.

4. **Stage 4 — Localize.** Layer-wise projection to find which layers carry the signal post-
   transfer (same middle-to-late layers as single-turn, or shifted?). Held-out generalization
   on trajectories held out from layer selection.

## 5. Locked technical decisions (do not silently change these)

- **Model:** `Qwen3-8B` (open weights; need full activation access). Same model as the anchor
  paper — switching models would confound regime-transfer with architecture-transfer.
- **Benchmark:** **SWE-bench** (primary). ~12–15 step trajectories; objective pass/fail labels
  from test suites. (Handoff originally said SWE-bench Lite; latest proposal text says
  "SWE-bench" — confirm which subset before large runs.)
- **Trajectories must be ON-POLICY.** Qwen3-8B generates its own trajectories in an agent
  scaffold. Do NOT reuse trajectories generated by a different model — the value axis only
  means anything when read off the *generating* model's own activations. Off-policy = invalid
  measurement.
- **Activation extraction position:** the **final token of the agent's own generated output at
  each step** (the point where the model has integrated everything it has seen). This is an
  explicit, stated assumption (single-turn → multi-turn format shift). Read at the verified
  axis layer(s).
- **Axis is frozen** after Stage 1. No retraining/refitting on agentic data.
- **Baseline:** majority-class, not assumed-0.5 chance.
- **Independent variable:** relative trajectory position (fraction), not absolute step index.
- **Scaffold:** use an existing agent scaffold (e.g. SWE-agent) rather than building one;
  this is the main engineering lift.

## 6. Known risks / open issues (account for these in code)

- **Class imbalance:** Qwen3-8B solves only a fraction of SWE-bench → successes are the
  minority class. Always evaluate against majority-class baseline; consider SWE-bench Verified
  or an easier subset if successes are too sparse.
- **Trajectory noise (flagged by mentor Jonas):** long trajectories carry incidental
  activation variance (tool outputs, stack traces, boilerplate) that can drown the signal.
  This risks a *false* no-transfer result (noise, not genuine absence). Mitigations: token-
  position discipline (read reasoning tokens, not echoed tool-output tokens), averaging across
  trajectories per position bin, optional smoothing over adjacent positions, and an explicit
  within-class vs. between-class variance check (see §8).
- **Sample size:** on-policy generation bounds N to ~a few hundred trajectories within the
  program timeline. Thinnest signal at earliest relative positions.
- **Projection sign:** the projection may separate classes while pointing "the wrong way"
  (AUROC < 0.5). Report `max(auroc, 1-auroc)` as separability and track the sign.

## 7. Interpreting null results (CRITICAL — do not conflate two different nulls)

A null result is NOT automatically the publishable "scoping" result. There are two distinct
nulls and they must be separated, because only one is a valid scientific claim:

- **Genuine no-signal (publishable):** "We measured cleanly, and the frozen value axis does
  not separate eventual outcomes in long-horizon agentic trajectories." → scopes the original
  single-turn result. Valid.
- **Noise-dominated non-result (NOT publishable as scoping):** "We could not read the axis
  reliably enough to tell whether it separates outcomes." → a statement about our measurement,
  not about the value axis. Says nothing about the hypothesis.

**The standard that licenses a genuine no-signal claim:**
> A no-signal result is only valid if the noise floor is demonstrably low enough that a real
> signal of plausible size (e.g. the single-turn effect size) WOULD have been detected.

This is a power/sensitivity argument: "if the axis tracked outcomes the way it does
single-turn, our measurement was sensitive enough to see it; we looked, and it is not there."

**To distinguish the two in practice, all must hold before claiming genuine no-signal:**
1. **Stage 1 passed** — axis reconstructed at reported AUROC on single-turn data (instrument
   is faithful in principle).
2. **Readable at an agentic anchor point** — the final-step projection shows coherent
   structure (not pure static). If even the final step is pure noise, the two nulls are
   indistinguishable and there is NO publishable null.
3. **Signal-to-noise diagnostic reported** — within-class variance is bounded relative to the
   (absent) between-class gap. If within-class variance is enormous, a lack of separation is
   uninterpretable; if it is bounded, overlapping classes mean the signal genuinely does not
   separate them.

**The hard middle zone (be honest about it):** if the noise floor is *moderate* — not low
enough to confidently detect a plausible signal, not high enough to obviously invalidate — a
null is **inconclusive**, and must be reported as "inconclusive at our measurement
sensitivity," NOT as a clean scoping result. All noise mitigations (token discipline,
per-position averaging, smoothing, more trajectories at key positions) exist to keep us out of
this zone. The de-risking experiment in §8 is what produces the noise-floor evidence that
later licenses (or forbids) a genuine no-signal claim.

**Code implication:** any analysis that reports a null/lack-of-separation MUST also report the
within-class vs. between-class variance (the signal-to-noise number) alongside it. Never
present "no separation" without the accompanying noise-floor evidence.

## 8. IMMEDIATE NEXT TASK — de-risking experiment (implement this first)

Goal: a small, fast **measurement-reliability check** (NOT a transfer test) to show whether
trajectory noise will prevent reading the signal. ~15–25 trajectories (aim 5–10 successes,
5–10 failures). This is the current priority and what code work should target now.

Per-step row schema to log:
```python
{
    "trajectory_id": str,
    "outcome": int,        # 1 = eventually solved, 0 = failed (ground-truth SWE-bench)
    "step_index": int,
    "n_steps": int,
    "rel_pos": float,      # step_index / (n_steps - 1), in [0, 1]
    "projection": float,   # value-axis projection at chosen token (final token of agent output)
    "token_type": str,     # "reasoning" (model's own output) vs "tool_output" (echoed external)
    "layer": int,          # verified best layer
}
```

Three analyses to produce (the deliverable = one plot, one number, one bar chart):
1. **Signal-to-noise ratio per position bin:** `|mean_succ - mean_fail| / pooled_within_class_std`.
   >~1 in late bins = noise survivable; <<1 = noise dominates.
2. **Final-step distributions:** histogram of final-step projection, success vs. failure +
   final-step AUROC (report `max(auroc, 1-auroc)`).
3. **Noise by token type:** projection std on `reasoning` vs `tool_output` tokens. Lower std on
   reasoning tokens empirically justifies the extraction choice.

Full spec with code skeletons: `derisking_experiment_spec.md`.

## 9. References (verify URLs before citing in code comments/docs)

- [1] Jiang, Kauvar, Lindsey (2026). The Value Axis. arXiv:2606.17056
- [2] Demircan et al. (2024). SAEs Reveal Temporal Difference Learning in an LLM. arXiv:2410.01280
- [3] Tatsat et al. (2026). Beyond the Black Box: Interpretability of Agentic AI Tool Use. arXiv:2605.06890
- [4] Zhang, Xiong, Wu (2026). Agentic Confidence Calibration. arXiv:2601.15778
- [5] Sanyal et al. (2026). Confidence is Not Competence. arXiv:2510.24772  (withdrawn from ICLR 2026 but on arXiv; closest internal-representation competitor; differentiate on the trajectory dimension)
- [6] Arike et al. (2025). Evaluating Goal Drift in Language Model Agents. arXiv:2505.02709
- [7] Yao et al. (2023). ReAct. arXiv:2210.03629

## 10. Related files in this workspace

- `value_axis_transfer_proposal.tex` — the full proposal (LaTeX, NeurIPS 2026 template; needs `neurips_2026.sty` to compile).
- `derisking_experiment_spec.md` — the immediate experiment spec with runnable analysis code skeletons.
- `proposal_plain_summary.md/.pdf` — plain-language summary.
- `proposal_defense_QA.md/.pdf` — Q&A defense doc.

## 11. Glossary (quick)

- **Value axis:** single linear direction in the residual stream; projection onto it ≈ "is the
  model on track."
- **Residual stream:** the running per-token vector that flows through the transformer; where
  activations are read.
- **On-policy:** trajectories generated by the same model whose activations we read.
- **Transfer test:** checking whether a known signal survives a change of setting (here:
  single-turn → long-horizon agentic).
- **Relative position:** step_index / (n_steps − 1); the normalized x-axis for all curves.
- **Majority-class baseline:** accuracy of always predicting the more common class; the honest
  floor under class imbalance.