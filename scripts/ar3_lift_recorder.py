"""LeRobot v3.0 writer for AR3 LIFT robot rollouts (policy inference and DAgger corrections).

``scripts/ar3_lift_robot_client.py --record-dir DIR`` records every episode with the layout and features of the
training dataset written by the lerobot AR3 recorder (``record_lift.json``, e.g.
/data/jiang/ar3_eef_forceVisuoTactaile_front_wrist): 10 Hz frames with

* ``observation.images.front`` / ``observation.images.wrist``: h264 videos (yuv420p, GOP 2, CRF 30, preset fast),
* ``observation.tcp_pose`` (xyz [m] + rotation vector [rad]), ``observation.gripper_position`` (0 closed ... 1000
  open), ``observation.tcp_wrench`` (``tauExt_inBase``) and ``observation.tau_m`` (7 measured joint torques),
* ``actions``: the absolute accepted TCP target (``accepted_command``, xyz + rotation vector) and the gripper target,
* ``control_flag``: -1 while a human (teleoperation) controls the robot, +1 while the policy does. The offline
  demonstrations are all -1, and LIFT trains on chunks whose flags are all -1 (``intervention_only``),
* the ``sync.*`` host-clock bookkeeping columns of the lerobot recorder.

The tactile gripper wrenches of the lerobot recordings are not recorded (LIFT does not use them). Every episode gets
its own data / video files, so recordings can be appended across sessions and are read by ``openpi.data.lerobot_v3``.
pyarrow and PyAV are only imported when recording.
"""

from __future__ import annotations

import dataclasses
import fractions
import json
import os
import pathlib
import queue
import threading
from typing import Any

import numpy as np

CODEBASE_VERSION = "v3.0"
ROBOT_TYPE = "rokae_ar3"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
CAMERAS = {"front": "observation.images.front", "wrist": "observation.images.wrist"}
HUMAN_FLAG = -1
POLICY_FLAG = 1
POSE_NAMES = ("x_m", "y_m", "z_m", "rotvec_x_rad", "rotvec_y_rad", "rotvec_z_rad")
BOOKKEEPING_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
# Image statistics (LeRobot convention: per channel in 0..1) from every 3rd frame, subsampled 4x in both axes.
IMAGE_STATS_FRAME_STRIDE = 3
IMAGE_STATS_PIXEL_STRIDE = 4
VIDEO_CODEC = "libx264"
VIDEO_OPTIONS = {"g": "2", "crf": "30", "preset": "fast"}
_ARROW_TYPES = {"float32": "float32", "float64": "float64", "int64": "int64"}


class RecorderError(RuntimeError):
    pass


def _vector(names: list[str] | tuple[str, ...], dtype: str = "float32") -> dict[str, Any]:
    return {"dtype": dtype, "shape": [len(names)], "names": list(names)}


def _video_feature(height: int, width: int, fps: int) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
            "video.g": int(VIDEO_OPTIONS["g"]),
            "video.crf": int(VIDEO_OPTIONS["crf"]),
            "video.preset": VIDEO_OPTIONS["preset"],
            "video.fast_decode": 0,
            "video.video_backend": "pyav",
            "video.extra_options": {},
            "is_depth_map": False,
        },
    }


def lift_features(height: int = 480, width: int = 640, fps: int = 10) -> dict[str, dict[str, Any]]:
    """Features of the lerobot ``record_lift.json`` recordings, without the tactile gripper wrenches."""
    receipt = ["host_receipt", "ros_stamp"]
    scalar_int = {"dtype": "int64", "shape": [1], "names": None}
    features: dict[str, dict[str, Any]] = {
        "observation.tcp_pose": _vector(POSE_NAMES),
        "observation.gripper_position": _vector(["position_sdk"]),
        "observation.tcp_wrench": _vector(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]),
        "actions": _vector([*(f"tcp_target.{name}" for name in POSE_NAMES), "gripper.target_position_sdk"]),
        "control_flag": _vector(["control_flag"], "int64"),
        "observation.tau_m": _vector([f"tau_m.joint{joint}_Nm" for joint in range(1, 8)]),
        "sync.target_host_s": _vector(["monotonic"], "float64"),
        "sync.tcp_pose.timing_s": _vector(receipt, "float64"),
        "sync.tcp_wrench.timing_s": _vector(receipt, "float64"),
        "sync.actions.timing_s": _vector([*receipt, "command_horizon"], "float64"),
        "sync.gripper_position.timing_s": _vector(receipt, "float64"),
        "sync.gripper_action.timing_s": _vector(receipt, "float64"),
        "sync.control_flag.timing_s": _vector(receipt, "float64"),
        "sync.tau_m.timing_s": _vector(receipt, "float64"),
    }
    for camera, key in CAMERAS.items():
        features[key] = _video_feature(height, width, fps)
        features[f"sync.{camera}.timing_s"] = _vector(
            ["host_receipt", "device_timestamp", "offset_from_target"], "float64"
        )
        features[f"sync.{camera}.frame_index"] = dict(scalar_int)
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    for key in BOOKKEEPING_KEYS[1:]:
        features[key] = dict(scalar_int)
    return features


def frame_keys(features: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    """Per-frame values supplied by the robot client (everything except the videos and the bookkeeping)."""
    return tuple(key for key, spec in features.items() if spec["dtype"] != "video" and key not in BOOKKEEPING_KEYS)


def _write_atomic(path: pathlib.Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    os.replace(temporary, path)


def _write_json(path: pathlib.Path, data: Any) -> None:
    _write_atomic(path, lambda temporary: temporary.write_text(json.dumps(data, indent=4) + "\n"))


def _write_parquet(path: pathlib.Path, table) -> None:
    import pyarrow.parquet as pq

    _write_atomic(path, lambda temporary: pq.write_table(table, temporary))


def _arrow_column(values: np.ndarray, spec: dict[str, Any]):
    import pyarrow as pa

    dtype = _ARROW_TYPES[spec["dtype"]]
    flat = pa.array(np.ascontiguousarray(values, dtype=dtype).reshape(-1), type=getattr(pa, dtype)())
    width = int(np.prod(spec["shape"]))
    # Like LeRobot: shape [1] features are scalars, wider ones fixed-size lists.
    return flat if width == 1 else pa.FixedSizeListArray.from_arrays(flat, width)


def numeric_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    stats = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
    }
    quantiles = np.quantile(values, list(QUANTILES.values()), axis=0)
    stats.update(zip(QUANTILES, quantiles, strict=True))
    result: dict[str, list] = {name: value.tolist() for name, value in stats.items()}
    result["count"] = [len(values)]
    return result


def image_stats(samples: list[np.ndarray]) -> dict[str, list]:
    """Per-channel statistics in 0..1, shaped [3, 1, 1] like LeRobot's image statistics."""
    pixels = np.stack(samples).reshape(-1, 3).astype(np.float64) / 255.0
    stats = numeric_stats(pixels)
    result = {name: [[[value]] for value in values] for name, values in stats.items() if name != "count"}
    result["count"] = [len(samples)]
    return result


def aggregate_stats(old: dict[str, dict[str, list]], new: dict[str, dict[str, list]]) -> dict[str, dict[str, list]]:
    """Merges dataset statistics; quantiles are count-weighted averages (as in LeRobot's aggregate_stats)."""
    merged = dict(old)
    for key, b in new.items():
        a = old.get(key)
        if a is None:
            merged[key] = b
            continue
        count_a, count_b = float(np.sum(a["count"])), float(np.sum(b["count"]))
        weight_a, weight_b = count_a / (count_a + count_b), count_b / (count_a + count_b)
        mean_a, mean_b = np.asarray(a["mean"]), np.asarray(b["mean"])
        mean = weight_a * mean_a + weight_b * mean_b
        second = weight_a * (np.asarray(a["std"]) ** 2 + mean_a**2) + weight_b * (np.asarray(b["std"]) ** 2 + mean_b**2)
        result = {
            "min": np.minimum(np.asarray(a["min"]), np.asarray(b["min"])),
            "max": np.maximum(np.asarray(a["max"]), np.asarray(b["max"])),
            "mean": mean,
            "std": np.sqrt(np.maximum(second - mean**2, 0.0)),
        }
        for name in QUANTILES:
            result[name] = weight_a * np.asarray(a[name]) + weight_b * np.asarray(b[name])
        merged[key] = {name: value.tolist() for name, value in result.items()}
        merged[key]["count"] = [int(count_a + count_b)]
    return merged


def _read_tasks(path: pathlib.Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    column = "task" if "task" in table.column_names else "__index_level_0__"
    return {
        str(task): int(index)
        for task, index in zip(table.column(column).to_pylist(), table.column("task_index").to_pylist(), strict=True)
    }


def _write_tasks(path: pathlib.Path, tasks: dict[str, int]) -> None:
    import pyarrow as pa

    try:
        import pandas as pd
    except ImportError:
        table = pa.table({"task_index": pa.array(list(tasks.values()), pa.int64()), "task": list(tasks)})
    else:
        # LeRobot stores the task strings as the dataframe index named "task".
        frame = pd.DataFrame({"task_index": list(tasks.values())}, index=pd.Index(list(tasks), name="task"))
        table = pa.Table.from_pandas(frame)
    _write_parquet(path, table)


class _VideoWriter:
    """Encodes the frames of one camera of one episode to mp4 in a background thread (pts = frame index / fps)."""

    def __init__(self, path: pathlib.Path, fps: int, height: int, width: int):
        import av  # noqa: F401  Fail early (in the caller's thread) without PyAV.

        self.path = path
        self.frames = 0
        self._fps = fps
        self._size = (height, width)
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._error: BaseException | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name=f"encode-{path.parent.parent.name}", daemon=True)
        self._thread.start()

    def put(self, image: np.ndarray) -> None:
        if self._error is not None:
            raise RecorderError(f"encoding {self.path} failed: {self._error}") from self._error
        self._queue.put(image)

    def _run(self) -> None:
        import av

        finished = False
        container = None
        try:
            container = av.open(str(self.path), mode="w", format="mp4")
            stream = container.add_stream(VIDEO_CODEC, rate=self._fps, options=dict(VIDEO_OPTIONS))
            stream.height, stream.width = self._size
            stream.pix_fmt = "yuv420p"
            time_base = fractions.Fraction(1, self._fps)
            while True:
                image = self._queue.get()
                if image is None:
                    finished = True
                    break
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = self.frames
                frame.time_base = time_base
                container.mux(stream.encode(frame))
                self.frames += 1
            container.mux(stream.encode(None))
        except BaseException as exc:  # Reported to the recording thread.
            self._error = exc
            while not finished:
                finished = self._queue.get() is None
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception as exc:
                    self._error = self._error or exc

    def finish(self) -> int:
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RecorderError(f"encoding {self.path} failed: {self._error}") from self._error
        return self.frames

    def abort(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self.path.unlink(missing_ok=True)


@dataclasses.dataclass
class _Episode:
    index: int
    writers: dict[str, _VideoWriter]
    rows: list[dict[str, np.ndarray]] = dataclasses.field(default_factory=list)
    image_samples: dict[str, list[np.ndarray]] = dataclasses.field(default_factory=dict)


class EpisodeRecorder:
    """Appends episodes to a (new or existing) LeRobot v3.0 dataset. Not thread-safe: use one recording thread."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        task: str,
        fps: int = 10,
        image_size: tuple[int, int] = (480, 640),
        capture_info: dict[str, Any] | None = None,
    ):
        self.root = pathlib.Path(root).expanduser().resolve()
        self.fps = int(fps)
        self.task = task
        self.features = lift_features(*image_size, self.fps)
        self.frame_keys = frame_keys(self.features)
        self._capture_info = capture_info
        self._episode: _Episode | None = None
        info_path = self.root / "meta" / "info.json"
        if info_path.is_file():
            self.info = json.loads(info_path.read_text())
            self._check_compatible()
            self.tasks = _read_tasks(self.root / "meta" / "tasks.parquet")
            stats_path = self.root / "meta" / "stats.json"
            self.stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise RecorderError(f"{self.root} exists, is not empty and has no meta/info.json")
            self.info = {
                "codebase_version": CODEBASE_VERSION,
                "robot_type": ROBOT_TYPE,
                "total_episodes": 0,
                "total_frames": 0,
                "total_tasks": 0,
                "chunks_size": CHUNKS_SIZE,
                "data_files_size_in_mb": 100,
                "video_files_size_in_mb": 200,
                "fps": self.fps,
                "splits": {},
                "data_path": DATA_PATH,
                "video_path": VIDEO_PATH,
                "features": self.features,
            }
            self.tasks = {}
            self.stats = {}

    def _check_compatible(self) -> None:
        info = self.info
        if info.get("codebase_version") != CODEBASE_VERSION:
            raise RecorderError(f"{self.root} is not a LeRobot {CODEBASE_VERSION} dataset")
        if int(info.get("fps", 0)) != self.fps:
            raise RecorderError(f"{self.root} was recorded at {info.get('fps')} fps, not {self.fps}")
        if info.get("data_path") != DATA_PATH or info.get("video_path") != VIDEO_PATH:
            raise RecorderError(f"{self.root} uses a different file layout")
        existing = {key: (spec["dtype"], list(spec["shape"])) for key, spec in info["features"].items()}
        expected = {key: (spec["dtype"], list(spec["shape"])) for key, spec in self.features.items()}
        if existing != expected:
            differing = sorted(
                key for key in existing.keys() | expected.keys() if existing.get(key) != expected.get(key)
            )
            raise RecorderError(
                f"{self.root} has different features (or image sizes): {differing}; use a new directory"
            )

    @property
    def num_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def recording(self) -> bool:
        return self._episode is not None

    @property
    def num_frames(self) -> int:
        """Frames of the episode being recorded."""
        return 0 if self._episode is None else len(self._episode.rows)

    def start_episode(self) -> int:
        if self._episode is not None:
            raise RecorderError("an episode is already being recorded")
        index = self.num_episodes
        chunk_index, file_index = divmod(index, CHUNKS_SIZE)
        height, width = self.features[CAMERAS["front"]]["shape"][:2]
        writers = {}
        try:
            for camera, key in CAMERAS.items():
                path = self.root / VIDEO_PATH.format(video_key=key, chunk_index=chunk_index, file_index=file_index)
                writers[camera] = _VideoWriter(path.with_name(path.name + ".tmp"), self.fps, height, width)
        except BaseException:
            for writer in writers.values():
                writer.abort()
            raise
        self._episode = _Episode(index, writers, image_samples={camera: [] for camera in CAMERAS})
        return index

    def add_frame(self, values: dict[str, Any], images: dict[str, np.ndarray]) -> int:
        """Adds one frame: ``values`` maps every key of ``frame_keys``, ``images`` the cameras to HxWx3 uint8 RGB."""
        episode = self._episode
        if episode is None:
            raise RecorderError("no episode is being recorded")
        row = {}
        for key in self.frame_keys:
            if key not in values:
                raise RecorderError(f"frame is missing {key}")
            spec = self.features[key]
            value = np.asarray(values[key], dtype=np.float64).reshape(-1)
            if value.shape != (int(np.prod(spec["shape"])),) or not np.all(np.isfinite(value)):
                raise RecorderError(f"invalid {key}: {values[key]!r}")
            row[key] = value.astype(spec["dtype"])
        frame_index = len(episode.rows)
        for camera, key in CAMERAS.items():
            image = images.get(camera)
            if image is None or image.dtype != np.uint8 or list(image.shape) != self.features[key]["shape"]:
                shape = None if image is None else (image.shape, image.dtype)
                raise RecorderError(f"{camera} image must be uint8 {self.features[key]['shape']}, got {shape}")
        for camera in CAMERAS:
            episode.writers[camera].put(images[camera])
            if frame_index % IMAGE_STATS_FRAME_STRIDE == 0:
                stride = IMAGE_STATS_PIXEL_STRIDE
                episode.image_samples[camera].append(np.array(images[camera][::stride, ::stride]))
        episode.rows.append(row)
        return frame_index

    def discard_episode(self) -> None:
        episode, self._episode = self._episode, None
        if episode is not None:
            for writer in episode.writers.values():
                writer.abort()

    def save_episode(self) -> int:
        """Writes the episode (videos, frames, metadata, statistics) and returns its index."""
        import pyarrow as pa

        episode, self._episode = self._episode, None
        if episode is None:
            raise RecorderError("no episode is being recorded")
        try:
            length = len(episode.rows)
            if length == 0:
                raise RecorderError("cannot save an empty episode")
            for camera, writer in episode.writers.items():
                if (frames := writer.finish()) != length:
                    raise RecorderError(f"{camera} video has {frames} frames, expected {length}")
        except BaseException:
            for writer in episode.writers.values():
                writer.abort()
            raise

        index = episode.index
        chunk_index, file_index = divmod(index, CHUNKS_SIZE)
        location = {"chunk_index": chunk_index, "file_index": file_index}
        task_index = self.tasks.setdefault(self.task, len(self.tasks))
        from_index = int(self.info["total_frames"])
        frames = np.arange(length)
        columns: dict[str, np.ndarray] = {
            "timestamp": (frames / self.fps).astype(np.float32),
            "frame_index": frames,
            "episode_index": np.full(length, index),
            "index": from_index + frames,
            "task_index": np.full(length, task_index),
        }
        for key in self.frame_keys:
            columns[key] = np.stack([row[key] for row in episode.rows])
        table = pa.table(
            {key: _arrow_column(columns[key], spec) for key, spec in self.features.items() if spec["dtype"] != "video"}
        )
        _write_parquet(self.root / DATA_PATH.format(**location), table)
        for camera, writer in episode.writers.items():
            os.replace(writer.path, self.root / VIDEO_PATH.format(video_key=CAMERAS[camera], **location))

        stats = {key: numeric_stats(value) for key, value in columns.items()}
        stats.update({CAMERAS[camera]: image_stats(samples) for camera, samples in episode.image_samples.items()})
        record: dict[str, Any] = {
            "episode_index": index,
            "tasks": [self.task],
            "length": length,
            "data/chunk_index": chunk_index,
            "data/file_index": file_index,
            "dataset_from_index": from_index,
            "dataset_to_index": from_index + length,
        }
        for key in CAMERAS.values():
            record.update(
                {
                    f"videos/{key}/chunk_index": chunk_index,
                    f"videos/{key}/file_index": file_index,
                    f"videos/{key}/from_timestamp": 0.0,
                    f"videos/{key}/to_timestamp": length / self.fps,
                }
            )
        for key, values in stats.items():
            record.update({f"stats/{key}/{name}": value for name, value in values.items()})
        record.update({"meta/episodes/chunk_index": chunk_index, "meta/episodes/file_index": file_index})
        _write_parquet(self.root / EPISODES_PATH.format(**location), pa.Table.from_pylist([record]))

        _write_tasks(self.root / "meta" / "tasks.parquet", self.tasks)
        self.stats = aggregate_stats(self.stats, stats)
        _write_json(self.root / "meta" / "stats.json", self.stats)
        if self._capture_info is not None and not (self.root / "meta" / "ar3_capture.json").is_file():
            _write_json(self.root / "meta" / "ar3_capture.json", self._capture_info)
        self.info.update(
            total_episodes=index + 1,
            total_frames=from_index + length,
            total_tasks=len(self.tasks),
            splits={"train": f"0:{index + 1}"},
        )
        _write_json(self.root / "meta" / "info.json", self.info)
        return index

    def close(self) -> None:
        """Discards an unsaved episode."""
        self.discard_episode()
