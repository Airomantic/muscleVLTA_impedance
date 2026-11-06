"""Native reader for LeRobot v3.0 datasets.

openpi pins the LeRobot v2.1 library (``lerobot.common.datasets``), which cannot read the v3.0 layout written by
recent LeRobot recorders (episode metadata stored as parquet, possibly several episodes per parquet / mp4 file).
This module reads v3.0 datasets directly with pyarrow and PyAV:

* ``meta/info.json``, ``meta/tasks.parquet`` and ``meta/episodes/**/*.parquet``
* ``data/chunk-XXX/file-XXX.parquet``: the frame table (may contain several episodes)
* ``videos/<video_key>/chunk-XXX/file-XXX.mp4``: an episode starts at ``videos/<video_key>/from_timestamp``

Items follow the LeRobot v2.1 conventions used by openpi: the current frame of every numeric feature,
``sequence_length`` consecutive frames for the sequence keys (clamped at the episode end, with a ``<key>_is_pad``
flag, like ``delta_timestamps`` in LeRobot v2.1), decoded uint8 HWC images for the video keys and the task string
under ``"task"``. Sequence keys listed as history keys also hold the ``history_length`` frames before the current one
(clamped at the episode start, also flagged in ``<key>_is_pad``), and ``LeRobotV3Options.tactile_cache`` adds the
frame-aligned Tac G2 indentation maps (``openpi.data.tactile_cache``) over the same frames.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
import dataclasses
import json
import logging
import os
import pathlib
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from openpi.shared.lerobot_reader import resolve_lerobot_dataset_root

logger = logging.getLogger(__name__)

CODEBASE_VERSION = "v3.0"

_NON_NUMERIC_DTYPES = frozenset({"video", "image", "string"})
_BOOKKEEPING_KEYS = ("episode_index", "frame_index", "timestamp", "task_index", "index")
_PLACEHOLDER_IMAGE_SHAPE = (16, 16, 3)

# Item keys of the tactile cache: float16 [frames, sensors, h, w] maps, bool [frames, sensors] validity and the
# bool [frames] episode padding flags.
TACTILE_DEPTH_KEY = "observation.tactile_depth"
TACTILE_VALID_KEY = "observation.tactile_valid"
TACTILE_PAD_KEY = f"{TACTILE_DEPTH_KEY}_is_pad"


@dataclasses.dataclass(frozen=True)
class LeRobotV3Options:
    """Options of the native LeRobot v3.0 reader."""

    # Video features to decode. ``None`` decodes every video feature listed in ``meta/info.json``.
    video_keys: tuple[str, ...] | None = None
    # If false, video features are returned as small black placeholders without decoding (e.g. to compute
    # normalization statistics, which do not depend on the images).
    decode_videos: bool = True
    # Behaviour for missing video files: "error" raises, "black" returns black frames (only for debugging).
    missing_video: Literal["error", "black"] = "error"
    # Skip frames whose ``*_valid`` features (e.g. ``observation.state_valid``) are not all 1.
    require_valid: bool = True
    # Optional subset of episode indices. ``None`` uses all episodes.
    episodes: tuple[int, ...] | None = None
    # Number of video files kept open per (worker) process.
    max_open_videos: int = 16
    # DAgger corrections (LIFT ``--online-intervention-only``): keep only samples whose whole chunk of
    # ``sequence_length`` frames (clamped at the episode end) was recorded under human control, i.e. every
    # ``intervention_key`` value equals ``intervention_value`` (control_flag: -1 human, +1 policy).
    intervention_only: bool = False
    intervention_key: str = "control_flag"
    intervention_value: float = -1.0
    # Frame-aligned Tac G2 indentation maps (scripts/build_ar3_tactile_cache.py): a cache directory, "auto" for the
    # default cache of the dataset (<dataset>/tactile_cache/depth_48x64) or None. Items then hold the maps of the
    # frames of the history keys (``TACTILE_DEPTH_KEY``, ``TACTILE_VALID_KEY``, ``TACTILE_PAD_KEY``).
    tactile_cache: str | None = None


def is_lerobot_v3_dataset(repo_id_or_root: str | os.PathLike[str]) -> bool:
    """Returns True if the dataset can be resolved locally and its ``meta/info.json`` declares codebase v3.0."""
    try:
        root = resolve_lerobot_dataset_root(repo_id_or_root)
    except (FileNotFoundError, ValueError):
        return False
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return False
    return json.loads(info_path.read_text()).get("codebase_version") == CODEBASE_VERSION


def _safe_path(root: pathlib.Path, relative: str) -> pathlib.Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset metadata points outside of the dataset root {root}: {relative}")
    return path


def column_to_numpy(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """Converts a (chunked) arrow column of scalars or fixed-width lists to a numpy array of shape [N] or [N, D]."""
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if pa.types.is_fixed_size_list(array.type):
        values = array.flatten().to_numpy(zero_copy_only=False)
        return values.reshape(len(array), array.type.list_size)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        lengths = pc.list_value_length(array).to_numpy(zero_copy_only=False)
        if len(lengths) and np.any(lengths != lengths[0]):
            raise ValueError(f"Variable-length list column of type {array.type} is not supported")
        values = array.flatten().to_numpy(zero_copy_only=False)
        return values.reshape(len(array), int(lengths[0]) if len(lengths) else 0)
    return array.to_numpy(zero_copy_only=False)


class LeRobotV3Metadata:
    """Metadata (``meta/``) of a local LeRobot v3.0 dataset."""

    def __init__(self, repo_id_or_root: str | os.PathLike[str]):
        self.root = resolve_lerobot_dataset_root(repo_id_or_root)
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata file: {info_path}")
        self.info: dict[str, Any] = json.loads(info_path.read_text())
        version = self.info.get("codebase_version")
        if version != CODEBASE_VERSION:
            raise ValueError(f"Expected a LeRobot {CODEBASE_VERSION} dataset at {self.root}, got {version}")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"Invalid fps in {info_path}: {self.info['fps']}")
        self.features: dict[str, dict[str, Any]] = self.info["features"]
        self.tasks = self._load_tasks(self.root / "meta" / "tasks.parquet")
        self.episodes = self._load_episodes(self.root / "meta" / "episodes")

    @property
    def video_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature.get("dtype") == "video"]

    @property
    def numeric_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature.get("dtype") not in _NON_NUMERIC_DTYPES]

    def data_file(self, episode_index: int) -> pathlib.Path:
        episode = self.episodes[episode_index]
        relative = self.info["data_path"].format(
            chunk_index=int(episode["data/chunk_index"]), file_index=int(episode["data/file_index"])
        )
        return _safe_path(self.root, relative)

    def video_file(self, video_key: str, episode_index: int) -> pathlib.Path:
        episode = self.episodes[episode_index]
        relative = self.info["video_path"].format(
            video_key=video_key,
            chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
            file_index=int(episode[f"videos/{video_key}/file_index"]),
        )
        return _safe_path(self.root, relative)

    def video_start(self, video_key: str, episode_index: int) -> float:
        return float(self.episodes[episode_index][f"videos/{video_key}/from_timestamp"])

    @staticmethod
    def _load_tasks(path: pathlib.Path) -> dict[int, str]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing LeRobot tasks file: {path}")
        table = pq.read_table(path)
        names = table.column_names
        # The task string is stored as the (named) pandas index of the dataframe written by LeRobot.
        task_column = "task" if "task" in names else "__index_level_0__"
        if "task_index" not in names or task_column not in names:
            raise ValueError(f"Unexpected columns in {path}: {names}")
        indices = table.column("task_index").to_pylist()
        texts = table.column(task_column).to_pylist()
        return {int(index): str(text) for index, text in zip(indices, texts, strict=True)}

    @staticmethod
    def _load_episodes(directory: pathlib.Path) -> dict[int, dict[str, Any]]:
        files = sorted(directory.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode metadata found under {directory}")
        episodes: dict[int, dict[str, Any]] = {}
        for path in files:
            schema = pq.read_schema(path)
            # Skip the (large) per-episode statistics.
            columns = [name for name in schema.names if not name.startswith("stats/")]
            for record in pq.read_table(path, columns=columns).to_pylist():
                index = int(record["episode_index"])
                if index in episodes:
                    raise ValueError(f"Duplicate metadata for episode {index} in {directory}")
                episodes[index] = record
        return dict(sorted(episodes.items()))


class _VideoDecoder:
    """Decodes single frames with PyAV, keeping a small LRU cache of open containers."""

    def __init__(self, max_open: int):
        self._max_open = max(1, max_open)
        self._containers: collections.OrderedDict[pathlib.Path, Any] = collections.OrderedDict()

    def _open(self, path: pathlib.Path):
        container = self._containers.pop(path, None)
        if container is None:
            import av

            container = av.open(str(path))
            while len(self._containers) >= self._max_open:
                _, oldest = self._containers.popitem(last=False)
                oldest.close()
        self._containers[path] = container
        return container

    def decode(self, path: pathlib.Path, timestamp: float, fps: float) -> np.ndarray:
        container = self._open(path)
        stream = container.streams.video[0]
        # Seek to the last keyframe before the requested time and decode forward to the closest frame.
        start = max(0.0, timestamp - 1.0 / fps)
        container.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
        previous = None
        current = None
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError(f"Video frame without presentation timestamp in {path}")
            previous, current = current, frame
            if frame.time >= timestamp:
                break
        candidates = [frame for frame in (previous, current) if frame is not None]
        if not candidates:
            raise ValueError(f"No video frames decoded from {path} around {timestamp:.4f}s")
        best = min(candidates, key=lambda frame: abs(frame.time - timestamp))
        if abs(best.time - timestamp) > 0.5 / fps + 1e-3:
            raise ValueError(
                f"No video frame near {timestamp:.4f}s in {path} (closest is {best.time:.4f}s). "
                "The video does not match the frame table."
            )
        return best.to_ndarray(format="rgb24")

    def close(self) -> None:
        while self._containers:
            _, container = self._containers.popitem(last=False)
            container.close()


class LeRobotV3Dataset:
    """Random-access frame dataset over a local LeRobot v3.0 dataset (map-style, picklable for worker processes)."""

    def __init__(
        self,
        repo_id_or_root: str | os.PathLike[str],
        *,
        sequence_keys: Sequence[str] = (),
        sequence_length: int = 1,
        history_keys: Sequence[str] = (),
        history_length: int = 0,
        options: LeRobotV3Options | None = None,
    ):
        self.meta = LeRobotV3Metadata(repo_id_or_root)
        self.options = options or LeRobotV3Options()
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if history_length < 0:
            raise ValueError(f"history_length must not be negative, got {history_length}")
        self.sequence_length = int(sequence_length)
        self.history_length = int(history_length)

        episodes = list(self.meta.episodes) if self.options.episodes is None else list(self.options.episodes)
        if missing_episodes := sorted(set(episodes) - set(self.meta.episodes)):
            raise ValueError(f"Episodes {missing_episodes} are not part of the dataset {self.meta.root}")
        if not episodes:
            raise ValueError(f"No episodes selected from {self.meta.root}")
        self.episode_indices = sorted(episodes)

        numeric_keys = self.meta.numeric_keys
        if unknown := sorted(set(sequence_keys) - set(numeric_keys)):
            raise ValueError(f"Unknown sequence keys {unknown}. Numeric features: {numeric_keys}")
        self.sequence_keys = tuple(sequence_keys)
        if unknown := sorted(set(history_keys) - set(self.sequence_keys)):
            raise ValueError(f"History keys {unknown} are not sequence keys {list(self.sequence_keys)}")
        self.history_keys = tuple(history_keys)
        self.video_keys = tuple(self.meta.video_keys if self.options.video_keys is None else self.options.video_keys)
        if unknown := sorted(set(self.video_keys) - set(self.meta.video_keys)):
            raise ValueError(f"Unknown video keys {unknown}. Video features: {self.meta.video_keys}")

        # Host-clock synchronisation columns (``sync.*``) are recorder bookkeeping, not policy inputs.
        self.frame_keys = tuple(
            key for key in numeric_keys if not key.startswith("sync.") and key not in self.sequence_keys
        )
        self._arrays = self._load_frames([*self.frame_keys, *self.sequence_keys, *_BOOKKEEPING_KEYS])

        episode_column = self._arrays["episode_index"]
        self._episode_bounds: dict[int, tuple[int, int]] = {}
        for episode_index in self.episode_indices:
            start = int(np.searchsorted(episode_column, episode_index, side="left"))
            end = int(np.searchsorted(episode_column, episode_index, side="right"))
            length = int(self.meta.episodes[episode_index]["length"])
            if end - start != length:
                raise ValueError(f"Episode {episode_index}: expected {length} frames, found {end - start}")
            if not np.array_equal(self._arrays["frame_index"][start:end], np.arange(length)):
                raise ValueError(f"Episode {episode_index}: frame indices are not consecutive from 0")
            self._episode_bounds[episode_index] = (start, end)
        # First and last row (inclusive) of the episode of every row, used to clamp sequences to the episode.
        self._episode_first_row = np.empty(len(episode_column), dtype=np.int64)
        self._episode_last_row = np.empty(len(episode_column), dtype=np.int64)
        for start, end in self._episode_bounds.values():
            self._episode_first_row[start:end] = start
            self._episode_last_row[start:end] = end - 1

        self._tactile = None
        if self.options.tactile_cache is not None:
            from openpi.data import tactile_cache as _tactile_cache

            directory = self.options.tactile_cache
            if directory == "auto":
                directory = _tactile_cache.default_cache_dir(self.meta.root)
            self._tactile = _tactile_cache.TactileCache(directory)
            self._tactile.check_episodes({index: end - start for index, (start, end) in self._episode_bounds.items()})

        valid = np.ones(len(episode_column), dtype=bool)
        if self.options.require_valid:
            for key in numeric_keys:
                if key.endswith("_valid") and key in self._arrays:
                    flags = self._arrays[key]
                    valid &= np.all(flags.reshape(len(flags), -1) >= 0.5, axis=-1)
        if self.options.intervention_only:
            valid &= self._intervention_chunks()
        self._samples = np.flatnonzero(valid)
        if not len(self._samples):
            suffix = (
                " (intervention_only: no chunk is entirely human-controlled)" if self.options.intervention_only else ""
            )
            raise ValueError(f"No valid frames in {self.meta.root}{suffix}")
        if skipped := len(valid) - len(self._samples):
            logger.info(
                "Skipping %d of %d frames (unset *_valid flags%s)",
                skipped,
                len(valid),
                ", or chunks not entirely human-controlled" if self.options.intervention_only else "",
            )

        self._warned_missing_videos: set[str] = set()
        self._video_decoder: _VideoDecoder | None = None

        logger.info(
            "Loaded LeRobot v3.0 dataset %s: %d episodes, %d frames (%d samples), sequences=%s x %d, "
            "history=%s x %d, tactile=%s, videos=%s",
            self.meta.root,
            len(self.episode_indices),
            len(episode_column),
            len(self._samples),
            list(self.sequence_keys),
            self.sequence_length,
            list(self.history_keys),
            self.history_length,
            self._tactile.directory if self._tactile is not None else None,
            list(self.video_keys) if self.options.decode_videos else "placeholders",
        )

    def __getstate__(self) -> dict[str, Any]:
        # Open video containers are per-process resources.
        state = self.__dict__.copy()
        state["_video_decoder"] = None
        return state

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def num_frames(self) -> int:
        return len(self._arrays["episode_index"])

    def episode_bounds(self, episode_index: int) -> tuple[int, int]:
        """Returns the [start, end) row range of an episode in the frame table of this dataset."""
        return self._episode_bounds[episode_index]

    def _intervention_chunks(self) -> np.ndarray:
        key = self.options.intervention_key
        if key not in self._arrays:
            raise ValueError(f"intervention_only needs the '{key}' feature, which {self.meta.root} does not provide")
        flags = self._arrays[key]
        human = np.all(np.isclose(flags.reshape(len(flags), -1), self.options.intervention_value), axis=-1)
        # Non-human rows before each row: rows [row, last] are all human iff the count does not change over them.
        before = np.concatenate(([0], np.cumsum(~human)))
        rows = np.arange(len(human))
        last = np.minimum(rows + self.sequence_length - 1, self._episode_last_row)
        return before[last + 1] == before[rows]

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if not -len(self) <= index < len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}")
        row = int(self._samples[index])
        episode_index = int(self._arrays["episode_index"][row])
        # Copies, so that in-place transforms cannot modify the in-memory frame table.
        item: dict[str, Any] = {key: np.array(self._arrays[key][row]) for key in self.frame_keys}
        for key in _BOOKKEEPING_KEYS:
            item.setdefault(key, self._arrays[key][row])

        last_row = int(self._episode_last_row[row])
        rows = np.arange(row, row + self.sequence_length)
        is_pad = rows > last_row
        rows = np.minimum(rows, last_row)
        for key in self.sequence_keys:
            if key not in self.history_keys:
                item[key] = self._arrays[key][rows]
                item[f"{key}_is_pad"] = is_pad
        if self.history_keys or self._tactile is not None:
            first_row = int(self._episode_first_row[row])
            window = np.arange(row - self.history_length, row + self.sequence_length)
            window_pad = (window < first_row) | (window > last_row)
            window = np.clip(window, first_row, last_row)
            for key in self.history_keys:
                item[key] = self._arrays[key][window]
                item[f"{key}_is_pad"] = window_pad
            if self._tactile is not None:
                # Frame indices are consecutive from 0 at the first row of an episode (checked in __init__).
                item[TACTILE_DEPTH_KEY], item[TACTILE_VALID_KEY] = self._tactile.frames(
                    episode_index, window - first_row
                )
                item[TACTILE_PAD_KEY] = window_pad

        timestamp = float(self._arrays["timestamp"][row])
        for key in self.video_keys:
            item[key] = self._decode_video(key, episode_index, timestamp)

        item["task"] = self.meta.tasks[int(self._arrays["task_index"][row])]
        return item

    def _load_frames(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        keys = list(dict.fromkeys(keys))
        files = list(dict.fromkeys(self.meta.data_file(index) for index in self.episode_indices))
        selected = pa.array(self.episode_indices, type=pa.int64())
        columns: dict[str, list[np.ndarray]] = collections.defaultdict(list)
        for path in files:
            names = set(pq.read_schema(path).names)
            if missing := sorted(set(keys) - names):
                raise ValueError(f"Missing columns {missing} in {path}")
            table = pq.read_table(path, columns=keys)
            table = table.filter(pc.is_in(table.column("episode_index").cast(pa.int64()), value_set=selected))
            for key in keys:
                columns[key].append(column_to_numpy(table.column(key)))
        arrays = {key: np.concatenate(values) for key, values in columns.items()}
        order = np.lexsort((arrays["frame_index"], arrays["episode_index"]))
        return {key: value[order] for key, value in arrays.items()}

    def _decode_video(self, key: str, episode_index: int, timestamp: float) -> np.ndarray:
        if not self.options.decode_videos:
            return np.zeros(_PLACEHOLDER_IMAGE_SHAPE, dtype=np.uint8)
        path = self.meta.video_file(key, episode_index)
        if not path.is_file():
            if self.options.missing_video != "black":
                raise FileNotFoundError(f"Missing video file {path}")
            if key not in self._warned_missing_videos:
                logger.warning("Video %s is missing, using black frames for '%s'", path, key)
                self._warned_missing_videos.add(key)
            return np.zeros(tuple(self.meta.features[key]["shape"]), dtype=np.uint8)
        if self._video_decoder is None:
            self._video_decoder = _VideoDecoder(self.options.max_open_videos)
        return self._video_decoder.decode(path, self.meta.video_start(key, episode_index) + timestamp, self.meta.fps)
