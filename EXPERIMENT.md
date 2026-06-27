# De-Risking Experiment: Can We Read the Value Axis Through Trajectory Noise?

**Purpose.** Answer one question with data before committing the team to the full project: *can the value-axis projection be read off long agentic trajectories cleanly enough that genuine signal wouldn't be drowned out by noise?* This is a **measurement-reliability check**, not a transfer test. It directly addresses Jonas's concern that long, unpredictable trajectories produce too much interference to find reliable results.

**What this is NOT.** It is not a test of whether the value axis transfers. It is not powered for a statistical claim. It is a fast, small-N diagnostic that tells us whether noise is survivable, and whether our token-position mitigation actually works.

---

## Scope and timeline (~4–5 days)

| Step | What | Time |
|------|------|------|
| 1 | Reconstruct the value axis on Qwen3-8B, confirm reported held-out AUROC | ~1 day |
| 2 | Generate ~15–25 SWE-bench trajectories with Qwen3-8B (aim for 5–10 successes, 5–10 failures) | ~1–2 days |
| 3 | Extract projections at every step with token-type tags | ~1 day |
| 4 | Run the three analyses below, make the plots | ~1 day |

Small N is intentional. This is a noise check; a few dozen trajectories is enough to see whether within-class jitter swamps between-class separation.

**Two tracks:** The **proxy track** (Qwen-local ICRL, ~100 convos, 0.75 axis gate) answers the noise question cheaply this week. **Faithful Stage 1** (Opus 300, 0.93 gate) remains required before any transfer claim — see [`scripts/run_proxy_week1.sh`](scripts/run_proxy_week1.sh).

---

## What to log per step

For every step in every trajectory, store one row:

```python
{
    "trajectory_id": str,          # which run
    "outcome": int,                # 1 = eventually solved, 0 = eventually failed (ground-truth SWE-bench label)
    "step_index": int,             # absolute step number
    "n_steps": int,                # total steps in this trajectory (for relative position)
    "rel_pos": float,              # step_index / (n_steps - 1), in [0, 1]
    "projection": float,           # value-axis projection at the chosen token (see below)
    "token_type": str,             # "reasoning" (model's own generated output) or "tool_output" (echoed/parsed external content)
    "layer": int,                  # which layer the activation was read from (use the verified best layer)
}
```

The `token_type` tag is what powers Analysis 3 and validates the mitigation — don't skip it.

---

## The three analyses (these ARE the deliverable)

What you bring to Jonas: **one plot, one number, one bar chart**, plus an honest interpretation.

### Analysis 1 — Signal-to-noise ratio (the number that answers Jonas directly)

At each relative-position bin, compare how much the projection varies *within* an outcome class against how far *apart* the two classes are. If between-class separation is visible above within-class jitter, noise is survivable. If within-class variance swamps it, Jonas was right — and you learned it cheaply.

```python
import numpy as np
import pandas as pd

def signal_to_noise_by_position(df, n_bins=5):
    """
    For each relative-position bin, compute a separation-to-noise ratio:
        |mean_success - mean_failure| / pooled_within_class_std
    Values >~1 mean the class gap is comparable to or larger than the noise.
    Values <<1 mean noise dominates and we likely can't read signal at that position.
    """
    df = df.copy()
    df["bin"] = pd.cut(df["rel_pos"], bins=n_bins, labels=False)
    rows = []
    for b in sorted(df["bin"].dropna().unique()):
        sub = df[df["bin"] == b]
        succ = sub[sub["outcome"] == 1]["projection"]
        fail = sub[sub["outcome"] == 0]["projection"]
        if len(succ) < 2 or len(fail) < 2:
            continue
        between = abs(succ.mean() - fail.mean())
        # pooled within-class standard deviation
        within = np.sqrt(((succ.var(ddof=1) * (len(succ) - 1)) +
                          (fail.var(ddof=1) * (len(fail) - 1))) /
                         (len(succ) + len(fail) - 2))
        ratio = between / within if within > 0 else np.nan
        rows.append({"bin": b, "rel_pos_mid": (b + 0.5) / n_bins,
                     "between_class_gap": between, "within_class_std": within,
                     "separation_to_noise": ratio,
                     "n_succ": len(succ), "n_fail": len(fail)})
    return pd.DataFrame(rows)

# Headline number: the separation-to-noise ratio, especially in the late bins.
snr = signal_to_noise_by_position(df)
print(snr[["rel_pos_mid", "separation_to_noise", "n_succ", "n_fail"]])
```

**How to read it.** A ratio comfortably above ~1 in the later bins is encouraging: the class gap rivals the noise. A ratio far below 1 everywhere means noise dominates. Expect it to be weakest early (genuinely ambiguous) and strongest late — that pattern itself is informative.

### Analysis 2 — Final-step distributions (the proof-of-life plot)

The final step is the cleanest, lowest-noise case — the trajectory has resolved and activations are most about the outcome. If signal is readable anywhere, it's here. This is your headline plot.

```python
import matplotlib.pyplot as plt

def final_step_plot(df, out_path="final_step_separation.png"):
    finals = df.sort_values("step_index").groupby("trajectory_id").tail(1)
    succ = finals[finals["outcome"] == 1]["projection"]
    fail = finals[finals["outcome"] == 0]["projection"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.histogram_bin_edges(finals["projection"], bins=12)
    ax.hist(fail, bins=bins, alpha=0.6, label=f"Failed (n={len(fail)})")
    ax.hist(succ, bins=bins, alpha=0.6, label=f"Succeeded (n={len(succ)})")
    ax.axvline(fail.mean(), linestyle="--", linewidth=1)
    ax.axvline(succ.mean(), linestyle="--", linewidth=1)
    ax.set_xlabel("Value-axis projection at final step")
    ax.set_ylabel("Count")
    ax.set_title("Final-step projection: success vs. failure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    # Cheap quantitative companion: AUROC of final-step projection vs. outcome.
    from sklearn.metrics import roc_auc_score
    finals_sorted = finals.dropna(subset=["projection"])
    auroc = roc_auc_score(finals_sorted["outcome"], finals_sorted["projection"])
    # If projection points the "wrong" way, AUROC < 0.5; report max(auroc, 1-auroc)
    # as separability and note the sign.
    print(f"Final-step AUROC: {auroc:.3f}  (separability: {max(auroc, 1-auroc):.3f})")
    return auroc
```

**How to read it.** *Any* visible gap between the two distributions — even weak — is proof of life: the signal exists and is readable through the noise. Nothing even here is itself an honest early result (and a hint of genuine no-transfer, not just noise).

### Analysis 3 — Noise by token type (validates the mitigation)

This is the one that turns Jonas's concern into a result *in your favor*. If the model's own reasoning tokens are measurably cleaner than tool-output tokens, you've empirically justified the token-position discipline you proposed as the mitigation.

```python
def noise_by_token_type(df, out_path="noise_by_token_type.png"):
    """
    Compare projection variance on the model's own reasoning tokens vs.
    on tool-output / echoed tokens. Lower variance on reasoning tokens
    justifies reading the projection there.
    """
    g = df.groupby("token_type")["projection"]
    summary = g.agg(["mean", "std", "count"]).reset_index()
    print(summary)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(summary["token_type"], summary["std"])
    ax.set_ylabel("Projection std (noise proxy)")
    ax.set_title("Within-token-type projection noise")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return summary
```

**How to read it.** If `reasoning` std is clearly lower than `tool_output` std, your extraction choice is validated with data — you can tell Jonas "we don't just hope the token choice helps; here's the variance reduction it buys."

---

## What you walk into Jonas's office with

1. **One plot** — final-step success vs. failure distributions (Analysis 2).
2. **One number** — the separation-to-noise ratio, especially in the late bins (Analysis 1).
3. **One bar chart** — reasoning-token vs. tool-token noise (Analysis 3).

Plus a two-line interpretation: *"Noise is real and we measured it. Here's how much there is, here's that signal is/isn't readable above it at the final step, and here's that our token discipline measurably reduces it."*

---

## Intellectual-honesty guardrails (say these before Jonas does)

- **N is tiny.** This is directional, not conclusive. A clean separation on ~10 trajectories is encouraging, not proof of transfer.
- **This is a measurement check, not a transfer result.** We are testing readability, not whether the value axis genuinely tracks agentic outcomes.
- **It can't perfectly separate "no signal" from "signal we can't yet read"** at this scale — but the separation-to-noise ratio is the right first look at exactly that question.

## The two outcomes, both useful

- **Separation visible at the final step + reasoning tokens cleaner than tool tokens** → de-risked. Noise is present but bounded, and our extraction controls it. Reasonable to greenlight the full study.
- **No separation + noise swamps everything, including on reasoning tokens** → we likely saved the team months. That is the experiment doing its job, not a failure.