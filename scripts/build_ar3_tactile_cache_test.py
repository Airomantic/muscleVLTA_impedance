import json
import pickle

import numpy as np
from openpi_client import tactile
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.data import tactile_cache
from scripts import ar3_lift_recorder as recorder_lib

RAW_SHAPE, SHAPE, MAX_AGE = (4, 6), (2, 3), 0.12
# Frame host times of the two episodes (10 fps) and the tactile sample times of episode 0 per sensor.
STARTS = (100.0, 200.0)
SAMPLES = {"sensor_0": [99.95, 100.02, 100.06, 100.15, 100.31, 100.33, 100.41, 100.5], "sensor_2": [100.25, 100.45]}
# Expected aligned sample row of every frame of episode 0 (-1: none at most MAX_AGE old).
EXPECTED_ROWS = {"sensor_0": [0, 2, 3, -1, 5, 7], "sensor_2": [-1, -1, -1, 0, -1, 1]}


def _raw_map(sensor: str, row: int) -> np.ndarray:
    return np.arange(np.prod(RAW_SHAPE), dtype=np.float32).reshape(RAW_SHAPE) * 0.01 + row + (sensor == "sensor_2")


def _write_dataset(root) -> None:
    recorder = recorder_lib.EpisodeRecorder(root, task="Insert the plug", fps=10, image_size=(16, 16))
    for start, length in zip(STARTS, (6, 4), strict=True):
        recorder.start_episode()
        for step in range(length):
            values = {key: np.zeros(recorder.features[key]["shape"]) for key in recorder.frame_keys}
            values["sync.target_host_s"] = np.array([start + 0.1 * step])
            recorder.add_frame(values, {camera: np.zeros((16, 16, 3), np.uint8) for camera in recorder_lib.CAMERAS})
        recorder.save_episode()
    # Episode 1 has no tactile recording; episode 0 stores the maps in row groups of 3 samples.
    episode_dir = root / "tactile" / "episode-000000"
    episode_dir.mkdir(parents=True)
    streams = {f"tactile_depth.{sensor}": {"shape": list(RAW_SHAPE), "dtype": "float32"} for sensor in SAMPLES}
    (episode_dir / "info.json").write_text(json.dumps({"episode_index": 0, "streams": streams}))
    for sensor, times in SAMPLES.items():
        values = np.stack([_raw_map(sensor, row).reshape(-1) for row in range(len(times))])
        table = pa.table(
            {
                "episode_index": np.zeros(len(times), np.int64),
                "sample_index": np.arange(len(times)),
                "host_timestamp_s": np.asarray(times),
                "source_timestamp_s": np.asarray(times) - 0.01,
                "sequence": np.arange(len(times)),
                "value": pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1]),
            }
        )
        pq.write_table(table, episode_dir / f"tactile_depth.{sensor}.parquet", row_group_size=3)


def test_cache_holds_the_newest_fresh_map_of_every_frame(tmp_path):
    root = tmp_path / "plug_in"
    _write_dataset(root)
    out = tactile_cache.build_cache(root, shape=SHAPE, max_age_s=MAX_AGE, workers=1)
    assert out == root / "tactile_cache" / "depth_2x3"
    cache = tactile_cache.TactileCache(out)
    assert (cache.sensors, cache.shape, cache.episodes) == (tactile.SENSORS, SHAPE, {0: (0, 6), 1: (6, 4)})

    depth, valid = cache.frames(0, np.arange(6))
    assert (depth.dtype, depth.shape, valid.shape) == (np.float16, (6, 2, *SHAPE), (6, 2))
    for position, sensor in enumerate(tactile.SENSORS):
        rows = EXPECTED_ROWS[sensor]
        np.testing.assert_array_equal(valid[:, position], [row >= 0 for row in rows])
        for frame, row in enumerate(rows):
            expected = tactile.pool_depth(_raw_map(sensor, row), SHAPE) if row >= 0 else np.zeros(SHAPE)
            np.testing.assert_array_equal(depth[frame, position], expected.astype(np.float16))
    depth, valid = cache.frames(1, np.array([3, 0, 0]))
    assert not valid.any()
    assert not depth.any()
    with pytest.raises(IndexError):
        cache.frames(1, np.array([4]))
    # Data loader workers receive the cache without its memory maps.
    assert pickle.loads(pickle.dumps(cache))._depth is None

    # Up-to-date caches are reused; other settings need an explicit rebuild, here by worker processes.
    assert tactile_cache.build_cache(root, shape=SHAPE, max_age_s=MAX_AGE) == out
    with pytest.raises(ValueError, match="overwrite"):
        tactile_cache.build_cache(root, shape=SHAPE, max_age_s=0.2)
    tactile_cache.build_cache(root, shape=SHAPE, max_age_s=0.2, workers=2, overwrite=True)
    rebuilt = tactile_cache.TactileCache(out)
    assert rebuilt.frames(0, np.arange(6))[1].sum(axis=0).tolist() == [6, 3]
    with pytest.raises(ValueError, match="does not match episode 1"):
        rebuilt.check_episodes({0: 6, 1: 5})
    with pytest.raises(FileNotFoundError, match="build_ar3_tactile_cache.py"):
        tactile_cache.TactileCache(tmp_path / "missing")


def test_align_causal_matches_samples_at_or_before_each_target():
    indices, valid = tactile_cache.align_causal(np.array([1.0, 2.0]), np.array([0.5, 1.0, 1.9, 2.3]), 0.5)
    np.testing.assert_array_equal(indices, [0, 0, 0, 1])
    np.testing.assert_array_equal(valid, [False, True, False, True])
    _, valid = tactile_cache.align_causal(np.array([]), np.array([1.0]), 0.5)
    assert not valid.any()
    with pytest.raises(ValueError, match="not sorted"):
        tactile_cache.align_causal(np.array([2.0, 1.0]), np.array([1.0]), 0.5)
