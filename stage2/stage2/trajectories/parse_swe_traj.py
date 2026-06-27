"""Parse SWE-agent .traj JSON files into normalized trajectory records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stage2.trajectories.schema import TrajectoryRecord, TrajectoryStep


def _normalize_message(msg: dict[str, Any]) -> dict[str, str]:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    return {"role": role, "content": content}


def _extract_response(step: dict[str, Any]) -> str:
    if step.get("response"):
        return str(step["response"])
    parts = []
    if step.get("thought"):
        parts.append(str(step["thought"]))
    if step.get("action"):
        parts.append(str(step["action"]))
    return "\n".join(parts) if parts else ""


def _outcome_from_info(info: dict[str, Any] | None) -> int | None:
    if not info:
        return None
    for key in ("exit_status", "submission_result", "resolved", "success"):
        if key not in info:
            continue
        val = info[key]
        if isinstance(val, bool):
            return 1 if val else 0
        if isinstance(val, str):
            low = val.lower()
            if low in ("submitted", "resolved", "success", "passed", "pass"):
                return 1
            if low in ("failed", "failure", "error", "not_submitted"):
                return 0
    return None


def parse_swe_traj(
    traj_path: Path,
    *,
    trajectory_id: str | None = None,
    outcome: int | None = None,
) -> TrajectoryRecord:
    """Parse a single SWE-agent .traj file."""
    with open(traj_path) as f:
        data = json.load(f)

    tid = trajectory_id or traj_path.stem
    if tid.endswith(".traj"):
        tid = tid[:-5]

    steps_raw = data.get("trajectory") or data.get("history") or []
    if not steps_raw:
        raise ValueError(f"No trajectory steps in {traj_path}")

    resolved_outcome = outcome
    if resolved_outcome is None:
        resolved_outcome = _outcome_from_info(data.get("info"))
    if resolved_outcome is None:
        resolved_outcome = 0

    steps: list[TrajectoryStep] = []
    for i, step in enumerate(steps_raw):
        query = step.get("query") or step.get("messages") or []
        messages = [_normalize_message(m) for m in query]
        response = _extract_response(step)
        observation = step.get("observation")
        if observation is not None:
            observation = str(observation)
        steps.append(
            TrajectoryStep(
                step_index=i,
                messages_before_gen=messages,
                assistant_response=response,
                observation=observation,
            )
        )

    n_steps = len(steps)
    return TrajectoryRecord(
        trajectory_id=tid,
        outcome=resolved_outcome,
        n_steps=n_steps,
        steps=steps,
    )


def load_results_map(results_path: Path) -> dict[str, int]:
    """Load instance_id -> outcome from SWE-bench results.json."""
    if not results_path.exists():
        return {}

    with open(results_path) as f:
        data = json.load(f)

    outcome_map: dict[str, int] = {}

    if isinstance(data, dict):
        if "resolved" in data and isinstance(data["resolved"], dict):
            for iid, val in data["resolved"].items():
                outcome_map[iid] = 1 if val else 0
            return outcome_map

        if "results" in data and isinstance(data["results"], dict):
            data = data["results"]

        for iid, entry in data.items():
            if isinstance(entry, bool):
                outcome_map[iid] = 1 if entry else 0
            elif isinstance(entry, dict):
                resolved = entry.get("resolved", entry.get("success", entry.get("passed")))
                if resolved is not None:
                    outcome_map[iid] = 1 if resolved else 0

    return outcome_map
