"""Advantages of the recorded action chunks, their labels, and the advantage-conditioned prompt.

``n_step_advantage`` scores the chunk executed from frame t with the critic's values V (``openpi.rl.critic``):

    A_t = r_t + ... + r_{t+n-1} + V_{t+n} - V_t        (the outcome replaces V_{t+n} beyond the end of the episode)

A chunk that advances the task as fast as the data does on average from similar states has no advantage; a faster one
is positive. Chunks that lead into a failure are strongly negative, and so are the steps spent after the failure has
become certain (time passes while the value stays at the bottom). ``label_frames`` turns the advantages into the
binary improvement indicator of RECAP, per task and data source: the best ``demo_fraction`` of the demonstration
frames and the best ``rollout_fraction`` of the policy-controlled frames are positive. Human corrections during
rollouts are always positive (RECAP), and so are the frames after the task is complete, so that the policy keeps its
ending. The labels are stored next to the dataset in ``rl/advantage_<tag>.npz``.

Training (``DataConfig.advantage_tag``): ``AttachAdvantage`` adds the label of the sampled frame and
``AdvantagePrompt`` writes it into the prompt, in one of two ways (``marking``):

* "both", as in RECAP: positive chunks get "Advantage: positive", negative ones "Advantage: negative", and a share
  ``dropout`` of the samples keeps the plain prompt, so that the policy also learns the unconditioned behaviour.
* "negative": only negative chunks are marked; positive chunks keep the prompt the policy was trained with. The
  served prompt is then the one the policy already knows, and fine-tuning moves the policy towards the positive
  chunks from where it is. No label may be dropped: a negative chunk would be trained as a positive one.

At inference no label arrives and ``AdvantagePrompt`` applies its ``inference`` indicator, "positive" by default.
"""

import dataclasses
import json
import os
import pathlib

import numpy as np

from openpi import transforms
from openpi.policies import ar3_lift_policy as _ar3

ADVANTAGE_KEY = _ar3.ADVANTAGE_KEY
NEGATIVE, POSITIVE, UNCONDITIONED = 0, 1, -1
INDICATORS = {"positive": POSITIVE, "negative": NEGATIVE, "none": UNCONDITIONED}
# Which labels are written into the prompt (module docstring).
MARKINGS = ("negative", "both")
_TEXT = {POSITIVE: "Advantage: positive", NEGATIVE: "Advantage: negative"}


def advantage_path(root: str | pathlib.Path, tag: str) -> pathlib.Path:
    return pathlib.Path(root) / "rl" / f"advantage_{tag}.npz"


def n_step_advantage(
    values: np.ndarray, targets: np.ndarray, costs: np.ndarray, last: np.ndarray, n: int
) -> np.ndarray:
    """Advantages [F] of the chunks starting at every frame (module docstring). ``last`` is the row of the last frame
    of the episode of every row; ``costs`` the reward of every step and ``targets`` the return to go. A chunk that
    reaches the end of its episode bootstraps from the outcome (the value target of the last frame) instead of V,
    and the steps beyond the end cost what the last step does: nothing after a success, time after a failure."""
    if n < 1:
        raise ValueError(f"The lookahead must be at least one step, got {n}")
    rows = np.arange(len(values))
    inside = rows + n <= last
    ahead = np.where(inside, rows + n, last)
    cumulative = np.concatenate(([0.0], np.cumsum(costs, dtype=np.float64)))
    spent = cumulative[ahead] - cumulative[rows] + (rows + n - ahead) * costs[last]
    bootstrap = np.where(inside, values[ahead], targets[ahead])
    return (spent + bootstrap - values).astype(np.float32)


def label_frames(
    advantage: np.ndarray,
    *,
    task: np.ndarray,
    rollout: np.ndarray,
    forced: np.ndarray,
    demo_fraction: float,
    rollout_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Labels [F] (``POSITIVE`` / ``NEGATIVE``) and the advantage threshold of every (task, source) group.

    ``rollout`` marks policy-controlled frames; the others are demonstration frames. ``forced`` frames are positive
    and do not count towards the fractions.
    """
    labels = np.where(forced, POSITIVE, NEGATIVE).astype(np.int8)
    thresholds = {}
    for task_index in np.unique(task):
        for source, selected, fraction in (
            ("demonstration", ~rollout, demo_fraction),
            ("rollout", rollout, rollout_fraction),
        ):
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"The positive fraction of {source} frames must be in [0, 1], got {fraction}")
            group = np.flatnonzero((task == task_index) & selected & ~forced)
            if not len(group):
                continue
            threshold = float(np.quantile(advantage[group], 1.0 - fraction)) if fraction < 1.0 else -np.inf
            labels[group] = np.where(advantage[group] > threshold, POSITIVE, NEGATIVE)
            thresholds[f"{int(task_index)}/{source}"] = threshold
    return labels, thresholds


def save_labels(
    root: str | pathlib.Path,
    tag: str,
    *,
    lengths: dict[int, int],
    arrays: dict[str, np.ndarray],
    info: dict,
) -> pathlib.Path:
    """Writes ``rl/advantage_<tag>.npz``: per-frame ``arrays`` in dataset order (at least ``label``), the episode
    lengths they belong to and ``info`` (settings, thresholds)."""
    path = advantage_path(root, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(lengths.values())
    if any(len(value) != total for value in arrays.values()):
        raise ValueError(f"Per-frame arrays do not match the {total} frames of {root}")
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(
        temporary,
        episodes=np.asarray(list(lengths), dtype=np.int64),
        lengths=np.asarray(list(lengths.values()), dtype=np.int64),
        info=np.asarray(json.dumps(info)),
        **arrays,
    )
    os.replace(temporary, path)
    return path


class AdvantageLabels:
    """The labels of one dataset (``save_labels``), looked up by episode and frame."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Missing advantage labels {self.path}: compute them with "
                f"`python scripts/ar3_rl.py fit --dataset <datasets> --tag <tag>`"
            )
        with np.load(self.path) as data:
            self.arrays = {key: data[key] for key in data.files if key not in ("episodes", "lengths", "info")}
            self.info = json.loads(str(data["info"]))
            lengths = data["lengths"].tolist()
            episodes = data["episodes"].tolist()
        self.lengths = dict(zip(episodes, lengths, strict=True))
        self._offsets = dict(zip(episodes, np.concatenate(([0], np.cumsum(lengths)))[:-1].tolist(), strict=True))
        self._labels = self.arrays["label"].astype(np.int8)

    @classmethod
    def load(cls, root: str | pathlib.Path, tag: str) -> "AdvantageLabels":
        return cls(advantage_path(root, tag))

    def check_episodes(self, lengths: dict[int, int]) -> None:
        """Raises unless every given episode is labelled with the given number of frames."""
        for index, length in lengths.items():
            if self.lengths.get(int(index)) != int(length):
                raise ValueError(
                    f"{self.path} does not match episode {index} ({length} frames): the dataset changed, "
                    "run scripts/ar3_rl.py fit again"
                )

    def label(self, episode_index: int, frame_index: int) -> int:
        if not 0 <= frame_index < self.lengths[int(episode_index)]:
            raise IndexError(f"Frame {frame_index} is outside episode {episode_index}")
        return int(self._labels[self._offsets[int(episode_index)] + int(frame_index)])


_RNG: tuple[int, np.random.Generator] | None = None


def _rng() -> np.random.Generator:
    """One generator per process: data loader workers must not repeat each other's draws."""
    global _RNG  # noqa: PLW0603
    if _RNG is None or _RNG[0] != os.getpid():
        _RNG = (os.getpid(), np.random.default_rng())
    return _RNG[1]


@dataclasses.dataclass(frozen=True)
class AttachAdvantage(transforms.DataTransformFn):
    """Training: adds the advantage label of the sampled frame of a LeRobot v3.0 dataset under ``ADVANTAGE_KEY``,
    replaced by ``UNCONDITIONED`` with probability ``dropout`` (only for prompts with both indicators)."""

    labels: AdvantageLabels
    dropout: float = 0.0

    def __call__(self, data: dict) -> dict:
        label = self.labels.label(int(data["episode_index"]), int(data["frame_index"]))
        if self.dropout > 0 and _rng().random() < self.dropout:
            label = UNCONDITIONED
        return {**data, ADVANTAGE_KEY: np.int8(label)}


def conditioned_prompt(prompt: str, label: int, marking: str = "both") -> str:
    """The prompt of a chunk with ``label``: the advantage indicator is appended if ``marking`` writes that label
    ("both": positive and negative, "negative": only negative); otherwise the prompt is unchanged."""
    if label == UNCONDITIONED or (label == POSITIVE and marking == "negative"):
        return prompt
    return f"{prompt.strip().rstrip('.;,')}. {_TEXT[int(label)]}"


@dataclasses.dataclass(frozen=True)
class AdvantagePrompt(transforms.DataTransformFn):
    """Writes the advantage label into the prompt (module docstring): the label attached by ``AttachAdvantage`` in
    training, the ``inference`` indicator ("positive", "negative" or "none") for requests without a label."""

    inference: str = "positive"
    marking: str = "negative"

    def __post_init__(self):
        if self.inference not in INDICATORS:
            raise ValueError(f"Unknown advantage indicator '{self.inference}': use one of {sorted(INDICATORS)}")
        if self.marking not in MARKINGS:
            raise ValueError(f"Unknown advantage marking '{self.marking}': use one of {MARKINGS}")

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        label = int(data.pop(ADVANTAGE_KEY, INDICATORS[self.inference]))
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            data["prompt"] = conditioned_prompt(str(np.asarray(prompt).item()), label, self.marking)
        return data


def positive_fractions(labels: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, float]:
    """Share of positive labels in every named group of frames (empty groups are left out)."""
    return {name: float((labels[rows] == POSITIVE).mean()) for name, rows in groups.items() if np.any(rows)}
