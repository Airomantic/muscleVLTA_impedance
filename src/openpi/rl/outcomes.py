"""Episode outcomes of LeRobot v3.0 recordings and the value targets derived from them.

RL needs to know how every episode ended. ``<dataset>/rl/outcomes.json`` (``scripts/ar3_rl.py label``) holds one label
per episode, "success" or "failure", and a default for the episodes it does not list (demonstrations: "success"):

    {"default": "success", "completion": "release", "episodes": {"3": "failure"}, "completion_frames": {"7": 212}}

As in RECAP (pi*0.6) the reward is -1 per step until the task is complete, so the value of a state is minus the
number of steps left, and a failed episode ends with a penalty that puts its states at the lowest value. The AR3
recordings have a fixed length and continue after the task is done (the arm retreats), so the completion frame of a
successful episode is read from the data: with ``completion: "release"`` it is the end of the last opening of the
gripper command, the release that ends an insertion or a pick-and-place; "end" is the last frame.
``completion_frames`` overrides it per episode. Values are divided by the longest time to completion of the task, to
[-1, 0].
"""

from collections.abc import Iterable, Sequence
import dataclasses
import json
import pathlib

import numpy as np

OUTCOMES_PATH = pathlib.Path("rl") / "outcomes.json"
SUCCESS, FAILURE = "success", "failure"
COMPLETION_MODES = ("release", "end")
# Gripper command steps (SDK units, 0 closed ... 1000 open) below this are not a release or a grasp.
GRIPPER_DEADBAND = 5.0


@dataclasses.dataclass(frozen=True)
class Outcomes:
    """Outcome labels of the episodes of one dataset."""

    default: str | None = None
    episodes: dict[int, str] = dataclasses.field(default_factory=dict)
    completion: str = "release"
    completion_frames: dict[int, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        labels = [*self.episodes.values(), *([] if self.default is None else [self.default])]
        if unknown := sorted(set(labels) - {SUCCESS, FAILURE}):
            raise ValueError(f"Unknown outcomes {unknown}: use '{SUCCESS}' or '{FAILURE}'")
        if self.completion not in COMPLETION_MODES:
            raise ValueError(f"Unknown completion mode '{self.completion}': use one of {COMPLETION_MODES}")

    def success(self, episode_index: int) -> bool:
        label = self.episodes.get(int(episode_index), self.default)
        if label is None:
            raise KeyError(f"Episode {episode_index} has no outcome label and there is no default")
        return label == SUCCESS

    def to_json(self) -> dict:
        return {
            "default": self.default,
            "completion": self.completion,
            "episodes": {str(index): label for index, label in sorted(self.episodes.items())},
            "completion_frames": {str(index): int(frame) for index, frame in sorted(self.completion_frames.items())},
        }


def load(root: str | pathlib.Path) -> Outcomes:
    path = pathlib.Path(root) / OUTCOMES_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}: label the episodes first, e.g. for demonstrations "
            f"`python scripts/ar3_rl.py label --dataset {root} --default success`"
        )
    data = json.loads(path.read_text())
    return Outcomes(
        default=data.get("default"),
        episodes={int(index): label for index, label in data.get("episodes", {}).items()},
        completion=data.get("completion", "release"),
        completion_frames={int(index): int(frame) for index, frame in data.get("completion_frames", {}).items()},
    )


def save(root: str | pathlib.Path, outcomes: Outcomes) -> pathlib.Path:
    path = pathlib.Path(root) / OUTCOMES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(outcomes.to_json(), indent=2) + "\n")
    return path


def parse_episodes(specs: Iterable[str]) -> list[int]:
    """Episode indices of specs like "3", "0-4" or "1,5-7"."""
    episodes: list[int] = []
    for spec in specs:
        for part in str(spec).split(","):
            if not part.strip():
                continue
            first, _, last = part.strip().partition("-")
            episodes.extend(range(int(first), int(last or first) + 1))
    return episodes


def completion_frame(gripper_target: np.ndarray, mode: str = "release") -> int:
    """Frame at which a successful episode is complete (module docstring): the end of the last opening of the gripper
    command. The last frame if the episode ends with a closing command (the object is still held) or the command
    never changes."""
    last = len(gripper_target) - 1
    if mode == "end":
        return last
    steps = np.diff(np.asarray(gripper_target, dtype=np.float64))
    changes = np.flatnonzero(np.abs(steps) > GRIPPER_DEADBAND)
    if not len(changes) or steps[changes[-1]] < 0:
        return last
    return int(changes[-1]) + 1


def step_costs(length: int, *, success: bool, completion: int, norm: float) -> np.ndarray:
    """Normalized reward [length] of every step: -1 / norm until the task is complete, then 0. A failed episode never
    completes, so all its steps cost time, and so would the steps after its last frame."""
    steps = np.arange(length)
    return np.where(steps < (completion if success else length), -1.0 / norm, 0.0)


def value_targets(length: int, *, success: bool, completion: int, norm: float, fail_penalty: float = 1.0) -> np.ndarray:
    """Normalized return to go [length] in [-1, 0]: minus the steps left until ``completion`` over ``norm`` for a
    success (0 from the completion frame on); for a failure the steps left in the episode and ``fail_penalty``."""
    costs = step_costs(length, success=success, completion=completion, norm=norm)
    to_go = costs[::-1].cumsum()[::-1]
    if not success:
        to_go = to_go - fail_penalty
    return np.maximum(to_go, -1.0).astype(np.float32)


def task_norm(times: Sequence[int]) -> float:
    """Normalizer of a task: its longest time to completion (successes) or episode (failures), in steps."""
    return float(max(max(times, default=1), 1))
