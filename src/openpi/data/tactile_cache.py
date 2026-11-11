"""Frame-aligned cache of the Tac G2 indentation maps of an AR3 LeRobot v3.0 dataset.

The AR3 recorder stores the tactile streams natively (``tactile/episode-XXXXXX/tactile_depth.<sensor>.parquet``: ~26 Hz
288x384 float32 maps, ~0.5 GB per episode), next to the 10 Hz frame table. For every frame, training needs the newest
map received at or before the frame's host time ``sync.target_host_s``: the map the robot client holds at that control
step. ``build_cache`` aligns and area-pools the maps once (``openpi_client.tactile.pool_depth``) into::

    <dataset>/tactile_cache/depth_48x64/
        meta.json   sensors, map shape, maximum sample age, episode -> {offset, length}
        depth.npy   float16 [frames, sensors, 48, 64] in dataset order (episode, frame)
        valid.npy   bool [frames, sensors]: a map at most ``max_age_s`` old existed (else the map is zero)

``TactileCache`` memory-maps the arrays in the data loader workers.

The recorder writes the sidecars at ~26 Hz while the Tac G2 topics arrive at ~118 Hz, so the cached map of a frame is
on average ~20 ms (at most ~50 ms) older than the newest one, e.g. the finger wrench of the frame table; the robot
client uses the newest map, a negligible difference at 10 Hz control.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import concurrent.futures
import json
import logging
import os
import pathlib
import shutil
from typing import Any

import numpy as np
from openpi_client import tactile as _tactile
import pyarrow.parquet as pq

from openpi.data import lerobot_v3 as _lerobot_v3

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
DEFAULT_MAX_AGE_S = 0.2
_TARGET_TIME_KEY = "sync.target_host_s"


def default_cache_dir(
    dataset_root: str | os.PathLike[str], shape: Sequence[int] = _tactile.DEPTH_SHAPE
) -> pathlib.Path:
    height, width = (int(size) for size in shape)
    return pathlib.Path(dataset_root) / "tactile_cache" / f"depth_{height}x{width}"


def align_causal(sample_times: np.ndarray, target_times: np.ndarray, max_age_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Index of the newest sample at or before every target time, and whether it is at most ``max_age_s`` old."""
    sample_times = np.asarray(sample_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if np.any(np.diff(sample_times) < 0):
        raise ValueError("Tactile sample times are not sorted")
    indices = np.searchsorted(sample_times, target_times, side="right") - 1
    valid = indices >= 0
    clipped = np.maximum(indices, 0)
    if len(sample_times):
        valid &= target_times - sample_times[clipped] <= max_age_s
    else:
        valid[:] = False
    return clipped, valid


def depth_sidecar(dataset_root: pathlib.Path, episode_index: int, sensor: str) -> pathlib.Path:
    return dataset_root / "tactile" / f"episode-{episode_index:06d}" / f"tactile_depth.{sensor}.parquet"


def _read_pooled_rows(
    path: pathlib.Path, rows: np.ndarray, raw_shape: Sequence[int], shape: Sequence[int]
) -> dict[int, np.ndarray]:
    """Pooled maps of the given sidecar rows, reading only the row groups that contain them."""
    file = pq.ParquetFile(path)
    sizes = [file.metadata.row_group(index).num_rows for index in range(file.metadata.num_row_groups)]
    starts = np.concatenate(([0], np.cumsum(sizes)))
    wanted = np.unique(rows)
    pooled: dict[int, np.ndarray] = {}
    for group in np.unique(np.searchsorted(starts, wanted, side="right") - 1):
        column = file.read_row_group(int(group), columns=["value"]).column("value").combine_chunks()
        local = wanted[(wanted >= starts[group]) & (wanted < starts[group + 1])]
        maps = column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), -1)[local - starts[group]]
        if maps.shape[-1] != int(np.prod(raw_shape)):
            raise ValueError(f"{path}: depth maps hold {maps.shape[-1]} values, expected {tuple(raw_shape)}")
        for row, value in zip(local, _tactile.pool_depth(maps.reshape(-1, *raw_shape), shape), strict=True):
            pooled[int(row)] = value
    return pooled


def _raw_shape(dataset_root: pathlib.Path, episode_index: int, sensor: str) -> tuple[int, ...]:
    """Recorded map shape from ``tactile/episode-XXXXXX/info.json`` (the native 288x384 when absent)."""
    path = dataset_root / "tactile" / f"episode-{episode_index:06d}" / "info.json"
    if path.is_file():
        stream = json.loads(path.read_text()).get("streams", {}).get(f"tactile_depth.{sensor}", {})
        if stream.get("shape"):
            return tuple(int(size) for size in stream["shape"])
    return _tactile.RAW_DEPTH_SHAPE


def _episode_maps(
    dataset_root: pathlib.Path,
    episode_index: int,
    target_times: np.ndarray,
    sensors: Sequence[str],
    shape: Sequence[int],
    max_age_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.zeros((len(target_times), len(sensors), *shape), dtype=np.float16)
    valid = np.zeros((len(target_times), len(sensors)), dtype=bool)
    for position, sensor in enumerate(sensors):
        path = depth_sidecar(dataset_root, episode_index, sensor)
        if not path.is_file():
            continue
        sample_times = pq.read_table(path, columns=["host_timestamp_s"]).column(0).to_numpy()
        indices, fresh = align_causal(sample_times, target_times, max_age_s)
        if not fresh.any():
            continue
        pooled = _read_pooled_rows(path, indices[fresh], _raw_shape(dataset_root, episode_index, sensor), shape)
        for frame in np.flatnonzero(fresh):
            depth[frame, position] = pooled[int(indices[frame])]
        valid[:, position] = fresh
    return depth, valid


def _frame_target_times(meta: _lerobot_v3.LeRobotV3Metadata) -> dict[int, np.ndarray]:
    """Host time of every frame (``sync.target_host_s``) per episode, in frame order."""
    if _TARGET_TIME_KEY not in meta.features:
        raise ValueError(f"{meta.root} has no '{_TARGET_TIME_KEY}' feature to align the tactile streams to")
    times: dict[int, np.ndarray] = {}
    for path in dict.fromkeys(meta.data_file(index) for index in meta.episodes):
        table = pq.read_table(path, columns=["episode_index", "frame_index", _TARGET_TIME_KEY])
        episodes = _lerobot_v3.column_to_numpy(table.column("episode_index")).reshape(-1)
        frames = _lerobot_v3.column_to_numpy(table.column("frame_index")).reshape(-1)
        targets = _lerobot_v3.column_to_numpy(table.column(_TARGET_TIME_KEY)).reshape(-1)
        for episode_index in np.unique(episodes):
            if int(episode_index) not in meta.episodes:
                continue
            selected = episodes == episode_index
            order = np.argsort(frames[selected])
            if not np.array_equal(frames[selected][order], np.arange(selected.sum())):
                raise ValueError(f"Episode {episode_index}: frame indices are not consecutive from 0")
            times[int(episode_index)] = targets[selected][order]
    for episode_index, episode in meta.episodes.items():
        if episode_index not in times or len(times[episode_index]) != int(episode["length"]):
            raise ValueError(f"Episode {episode_index}: frame table does not match meta/episodes")
    return times


def _capture_max_age(dataset_root: pathlib.Path) -> float:
    path = dataset_root / "meta" / "ar3_capture.json"
    if path.is_file():
        value = json.loads(path.read_text()).get("robot", {}).get("max_tactile_age_s")
        if value is not None:
            return float(value)
    return DEFAULT_MAX_AGE_S


def _run_jobs(jobs: dict[int, tuple], workers: int) -> Iterator[tuple[int, tuple[np.ndarray, np.ndarray]]]:
    """Yields ``(episode, _episode_maps(*args))`` as the episodes finish; ``workers <= 1`` runs in this process."""
    if workers <= 1:
        for episode_index, args in jobs.items():
            yield episode_index, _episode_maps(*args)
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_episode_maps, *args): episode_index for episode_index, args in jobs.items()}
        for future in concurrent.futures.as_completed(futures):
            yield futures[future], future.result()


def build_cache(
    dataset: str | os.PathLike[str],
    out_dir: str | os.PathLike[str] | None = None,
    *,
    sensors: Sequence[str] = _tactile.SENSORS,
    shape: Sequence[int] = _tactile.DEPTH_SHAPE,
    max_age_s: float | None = None,
    workers: int = 8,
    overwrite: bool = False,
) -> pathlib.Path:
    """Builds the tactile cache of a LeRobot v3.0 dataset (see the module docstring). Returns its directory.

    Episodes without tactile sidecars are cached as invalid (zero maps), so mixtures with such data still load.
    """
    meta = _lerobot_v3.LeRobotV3Metadata(dataset)
    out_dir = pathlib.Path(out_dir) if out_dir is not None else default_cache_dir(meta.root, shape)
    shape = tuple(int(size) for size in shape)
    max_age_s = _capture_max_age(meta.root) if max_age_s is None else float(max_age_s)
    if (out_dir / "meta.json").is_file() and not overwrite:
        cache = TactileCache(out_dir)
        if (cache.sensors, cache.shape, cache.max_age_s) != (tuple(sensors), shape, max_age_s):
            raise ValueError(
                f"{out_dir} holds {cache.sensors} {cache.shape} maps aligned with max_age_s={cache.max_age_s}; "
                "rebuild it with overwrite=True"
            )
        try:
            cache.check_episodes({index: int(episode["length"]) for index, episode in meta.episodes.items()})
        except ValueError:
            # Episodes were added since (rollouts recorded between RL rounds, DAgger rounds).
            logger.info("Tactile cache %s does not cover the episodes of the dataset any more: rebuilding", out_dir)
        else:
            logger.info("Tactile cache %s is up to date", out_dir)
            return out_dir

    times = _frame_target_times(meta)
    offsets: dict[int, tuple[int, int]] = {}
    total = 0
    for episode_index, targets in times.items():
        offsets[episode_index] = (total, len(targets))
        total += len(targets)

    staging = out_dir.with_name(out_dir.name + ".building")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    depth = np.lib.format.open_memmap(
        staging / "depth.npy", mode="w+", dtype=np.float16, shape=(total, len(sensors), *shape)
    )
    valid = np.lib.format.open_memmap(staging / "valid.npy", mode="w+", dtype=bool, shape=(total, len(sensors)))
    missing: list[int] = []
    jobs = {index: (meta.root, index, targets, tuple(sensors), shape, max_age_s) for index, targets in times.items()}
    for done, (episode_index, (episode_depth, episode_valid)) in enumerate(_run_jobs(jobs, workers), start=1):
        offset, length = offsets[episode_index]
        depth[offset : offset + length] = episode_depth
        valid[offset : offset + length] = episode_valid
        if not episode_valid.any():
            missing.append(episode_index)
        logger.info(
            "episode %d (%d/%d): %d frames, valid %s",
            episode_index,
            done,
            len(jobs),
            length,
            np.round(episode_valid.mean(axis=0), 3).tolist(),
        )
    depth.flush()
    valid.flush()
    del depth, valid
    if missing:
        logger.warning("Episodes without tactile depth maps (cached as invalid): %s", sorted(missing))
    info: dict[str, Any] = {
        "version": CACHE_VERSION,
        "sensors": list(sensors),
        "shape": list(shape),
        "max_age_s": max_age_s,
        "alignment": f"newest sample with host_timestamp_s <= {_TARGET_TIME_KEY}",
        "num_frames": total,
        "episodes": {str(index): {"offset": offset, "length": length} for index, (offset, length) in offsets.items()},
    }
    (staging / "meta.json").write_text(json.dumps(info, indent=2))
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(out_dir)
    logger.info("Wrote tactile cache %s (%d frames)", out_dir, total)
    return out_dir


class TactileCache:
    """Read access to a cache written by ``build_cache``; the arrays are memory-mapped lazily (per process)."""

    def __init__(self, directory: str | os.PathLike[str]):
        self.directory = pathlib.Path(directory)
        meta_path = self.directory / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Missing tactile cache {self.directory}. Build it with: "
                "python scripts/build_ar3_tactile_cache.py --dataset <dataset root>"
            )
        meta = json.loads(meta_path.read_text())
        if meta.get("version") != CACHE_VERSION:
            raise ValueError(f"Unsupported tactile cache version {meta.get('version')} in {meta_path}")
        self.sensors = tuple(meta["sensors"])
        self.shape = tuple(int(size) for size in meta["shape"])
        self.max_age_s = float(meta["max_age_s"])
        self.episodes = {
            int(index): (int(entry["offset"]), int(entry["length"])) for index, entry in meta["episodes"].items()
        }
        self._depth: np.ndarray | None = None
        self._valid: np.ndarray | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_depth"] = state["_valid"] = None
        return state

    def check_episodes(self, lengths: dict[int, int]) -> None:
        """Raises unless every given episode is cached with the given number of frames."""
        for episode_index, length in lengths.items():
            cached = self.episodes.get(int(episode_index))
            if cached is None or cached[1] != int(length):
                raise ValueError(
                    f"Tactile cache {self.directory} does not match episode {episode_index} ({length} frames): "
                    "rebuild it with scripts/build_ar3_tactile_cache.py --overwrite"
                )

    def frames(self, episode_index: int, frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Maps float16 [T, sensors, h, w] and validity bool [T, sensors] of the given frames of an episode."""
        offset, length = self.episodes[int(episode_index)]
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if np.any(frame_indices < 0) or np.any(frame_indices >= length):
            raise IndexError(f"Frames {frame_indices} out of range for episode {episode_index} ({length} frames)")
        if self._depth is None:
            self._depth = np.load(self.directory / "depth.npy", mmap_mode="r")
            self._valid = np.load(self.directory / "valid.npy", mmap_mode="r")
        rows = offset + frame_indices
        return np.asarray(self._depth[rows]), np.asarray(self._valid[rows])
