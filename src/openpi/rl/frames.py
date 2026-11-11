"""Frame tables of LeRobot v3.0 AR3 recordings for the RL critic.

A ``FrameTable`` holds what the critic reads of one dataset, for all frames in dataset order (episode, frame): the
TCP pose and gripper, the arm force of the policy (TCP wrench + joint torques), the Tac G2 fingers (finger wrenches,
and the indentation maps of the tactile cache) and, optionally, cached vision features
(``<dataset>/rl/vision_<name>.npy``, ``scripts/ar3_rl.py features``). Recordings without Tac G2 data load with the
fingers flagged missing, as the policy reads them.
"""

import dataclasses
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

from openpi.data import lerobot_v3 as _lerobot_v3
from openpi.data import tactile_cache as _tactile_cache
from openpi.policies import ar3_lift_policy as _ar3

CONTROL_FLAG_KEY = "control_flag"
HUMAN_FLAG = -1


def vision_path(root: str | pathlib.Path, name: str) -> pathlib.Path:
    return pathlib.Path(root) / "rl" / f"vision_{name}.npy"


def vision_is_current(root: str | pathlib.Path, name: str) -> bool:
    """Whether the cached vision features exist and cover exactly the episodes of the dataset (rollout datasets
    grow between rounds)."""
    path = vision_path(root, name)
    if not (path.is_file() and path.with_suffix(".json").is_file()):
        return False
    cached = json.loads(path.with_suffix(".json").read_text())["episodes"]
    episodes = _lerobot_v3.LeRobotV3Metadata(root).episodes
    return {int(index): int(length) for index, length in cached.items()} == {
        int(index): int(episode["length"]) for index, episode in episodes.items()
    }


@dataclasses.dataclass
class FrameTable:
    root: pathlib.Path
    fps: float
    tasks: dict[int, str]
    # Episode index -> number of frames, in table order.
    lengths: dict[int, int]
    episode: np.ndarray  # [F]
    frame: np.ndarray  # [F]
    task: np.ndarray  # [F] task index
    pose: np.ndarray  # [F, 6] TCP xyz [m] + rotation vector [rad]
    gripper: np.ndarray  # [F] measured opening, 0 closed ... 1000 open
    gripper_target: np.ndarray  # [F] commanded opening
    arm: np.ndarray  # [F, 13] TCP wrench + joint torques
    finger_wrench: np.ndarray  # [F, 12]
    finger_valid: np.ndarray  # [F, fingers] bool
    human: np.ndarray  # [F] bool: teleoperated frame (demonstration or correction)
    depth: np.ndarray | None  # [F, fingers, h, w] float16 indentation maps
    vision: np.ndarray | None  # [F, cameras, D] float16

    def __len__(self) -> int:
        return len(self.episode)

    def bounds(self) -> dict[int, tuple[int, int]]:
        """Row range [start, end) of every episode."""
        ends = np.cumsum(list(self.lengths.values()))
        return {
            index: (int(end - length), int(end))
            for (index, length), end in zip(self.lengths.items(), ends, strict=True)
        }


def load(root: str | pathlib.Path, *, vision: str | None = None) -> FrameTable:
    meta = _lerobot_v3.LeRobotV3Metadata(root)
    finger_keys = [key for key in _ar3.GRIPPER_WRENCH_KEYS if key in meta.features]
    has_flag = CONTROL_FLAG_KEY in meta.features
    keys = [
        "episode_index",
        "frame_index",
        "task_index",
        _ar3.TCP_POSE_KEY,
        _ar3.GRIPPER_POSITION_KEY,
        _ar3.TCP_WRENCH_KEY,
        _ar3.TAU_M_KEY,
        _ar3.ACTIONS_KEY,
        *finger_keys,
        *([CONTROL_FLAG_KEY] if has_flag else []),
    ]
    columns: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for path in dict.fromkeys(meta.data_file(index) for index in meta.episodes):
        table = pq.read_table(path, columns=keys)
        for key in keys:
            columns[key].append(_lerobot_v3.column_to_numpy(table.column(key)))
    arrays = {key: np.concatenate(values) for key, values in columns.items()}
    order = np.lexsort((arrays["frame_index"].reshape(-1), arrays["episode_index"].reshape(-1)))
    arrays = {key: value[order] for key, value in arrays.items()}

    episode = arrays["episode_index"].reshape(-1).astype(np.int64)
    frame = arrays["frame_index"].reshape(-1).astype(np.int64)
    indices, counts = np.unique(episode, return_counts=True)
    lengths = {int(index): int(count) for index, count in zip(indices, counts, strict=True)}
    for index, length in lengths.items():
        if int(meta.episodes[index]["length"]) != length:
            raise ValueError(f"{meta.root}: episode {index} has {length} frames, meta/episodes says otherwise")
    if not np.array_equal(frame, np.concatenate([np.arange(length) for length in lengths.values()])):
        raise ValueError(f"{meta.root}: frame indices are not consecutive from 0 in every episode")

    total = len(episode)
    finger_wrench = np.zeros((total, _ar3.TACTILE_WRENCH_DIM), dtype=np.float32)
    finger_valid = np.zeros((total, len(_ar3.GRIPPER_WRENCH_KEYS)), dtype=bool)
    depth = None
    cache_dir = _tactile_cache.default_cache_dir(meta.root)
    # As in the policy input, fingers count only with both their wrench and their map.
    if len(finger_keys) == len(_ar3.GRIPPER_WRENCH_KEYS) and (cache_dir / "meta.json").is_file():
        cache = _tactile_cache.TactileCache(cache_dir)
        cache.check_episodes(lengths)
        rows = np.concatenate([cache.episodes[index][0] + np.arange(length) for index, length in lengths.items()])
        depth = np.ascontiguousarray(np.load(cache_dir / "depth.npy", mmap_mode="r")[rows])
        finger_valid = np.ascontiguousarray(np.load(cache_dir / "valid.npy", mmap_mode="r")[rows])
        finger_wrench = _ar3.concat_force([arrays[key] for key in finger_keys])

    features = None
    if vision is not None:
        path = vision_path(meta.root, vision)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}: compute the vision features with "
                f"`python scripts/ar3_rl.py features --dataset {meta.root}`"
            )
        info = json.loads(path.with_suffix(".json").read_text())
        if {int(index): int(length) for index, length in info["episodes"].items()} != lengths:
            raise ValueError(
                f"{path} does not match the episodes of {meta.root}: run `python scripts/ar3_rl.py features "
                f"--dataset {meta.root}` again"
            )
        features = np.load(path, mmap_mode="r")

    return FrameTable(
        root=meta.root,
        fps=meta.fps,
        tasks=dict(meta.tasks),
        lengths=lengths,
        episode=episode,
        frame=frame,
        task=arrays["task_index"].reshape(-1).astype(np.int64),
        pose=arrays[_ar3.TCP_POSE_KEY].astype(np.float32),
        gripper=arrays[_ar3.GRIPPER_POSITION_KEY].reshape(-1).astype(np.float32),
        gripper_target=arrays[_ar3.ACTIONS_KEY][:, 6].astype(np.float32),
        arm=_ar3.concat_force([arrays[_ar3.TCP_WRENCH_KEY], arrays[_ar3.TAU_M_KEY]]),
        finger_wrench=finger_wrench,
        finger_valid=finger_valid,
        human=(
            np.isclose(arrays[CONTROL_FLAG_KEY].reshape(-1), HUMAN_FLAG) if has_flag else np.ones(total, dtype=bool)
        ),
        depth=depth,
        vision=features,
    )
