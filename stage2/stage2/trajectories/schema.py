"""Normalized SWE-agent trajectory schema."""

from dataclasses import dataclass, field
from typing import Any
import json


@dataclass
class TrajectoryStep:
    step_index: int
    messages_before_gen: list[dict[str, str]]
    assistant_response: str
    observation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "messages_before_gen": self.messages_before_gen,
            "assistant_response": self.assistant_response,
            "observation": self.observation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrajectoryStep":
        return cls(
            step_index=d["step_index"],
            messages_before_gen=d["messages_before_gen"],
            assistant_response=d["assistant_response"],
            observation=d.get("observation"),
        )


@dataclass
class TrajectoryRecord:
    trajectory_id: str
    outcome: int
    n_steps: int
    steps: list[TrajectoryStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "outcome": self.outcome,
            "n_steps": self.n_steps,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrajectoryRecord":
        steps = [TrajectoryStep.from_dict(s) for s in d["steps"]]
        return cls(
            trajectory_id=d["trajectory_id"],
            outcome=int(d["outcome"]),
            n_steps=int(d["n_steps"]),
            steps=steps,
        )


def save_trajectory(record: TrajectoryRecord, path) -> None:
    with open(path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)


def load_trajectory(path) -> TrajectoryRecord:
    with open(path) as f:
        return TrajectoryRecord.from_dict(json.load(f))


def load_trajectories_from_dir(directory) -> list[TrajectoryRecord]:
    from pathlib import Path

    records = []
    for p in sorted(Path(directory).glob("*.json")):
        records.append(load_trajectory(p))
    return records
