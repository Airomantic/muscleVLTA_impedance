import dataclasses
import json

import numpy as np
from openpi_client import tactile
import pytest

from openpi.data import lerobot_v3
from openpi.data import tactile_cache
from openpi.policies import ar3_lift_policy as ar3
from openpi.rl import frames as rl_frames
from openpi.rl import outcomes as rl_outcomes
from openpi.training import config as _config
from openpi.training import data_loader
from scripts import ar3_lift_recorder as recorder_lib

HEIGHT, WIDTH, FPS, TASK = 48, 64, 10, "Unplug the white charging plug"


def _images(step: int) -> dict[str, np.ndarray]:
    return {
        "front": np.full((HEIGHT, WIDTH, 3), (step * 20 % 250, 40, 200), dtype=np.uint8),
        "wrist": np.full((HEIGHT, WIDTH, 3), (10, 200, step * 20 % 250), dtype=np.uint8),
    }


def _record(recorder: recorder_lib.EpisodeRecorder, flags: list[int], *, save: bool = True) -> int:
    recorder.start_episode()
    for step, flag in enumerate(flags):
        values = {key: np.full(recorder.features[key]["shape"], step + 0.25) for key in recorder.frame_keys}
        values["control_flag"] = flag
        recorder.add_frame(values, _images(step))
    if save:
        return recorder.save_episode()
    recorder.discard_episode()
    return -1


def _recorder(root, **kwargs) -> recorder_lib.EpisodeRecorder:
    return recorder_lib.EpisodeRecorder(root, task=TASK, fps=FPS, image_size=(HEIGHT, WIDTH), **kwargs)


def test_recordings_are_readable_by_the_training_reader(tmp_path):
    root = tmp_path / "dagger_round1"
    human, policy = recorder_lib.HUMAN_FLAG, recorder_lib.POLICY_FLAG
    recorder = _recorder(root, capture_info={"control_flag": "-1 human"})
    assert _record(recorder, [policy] * 6 + [human] * 6) == 0
    assert _record(recorder, [human] * 3, save=False) == -1
    assert _record(recorder, [human] * 5) == 1
    # A later session appends to the same dataset.
    assert _record(_recorder(root), [policy] * 4) == 2
    assert not list(root.rglob("*.tmp"))

    info = json.loads((root / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["splits"]) == (3, 21, {"train": "0:3"})
    stats = json.loads((root / "meta" / "stats.json").read_text())
    assert stats["control_flag"]["count"] == [21]
    assert stats["control_flag"]["min"] == [-1.0]
    assert stats["control_flag"]["max"] == [1.0]
    assert np.asarray(stats["observation.images.front"]["mean"]).shape == (3, 1, 1)

    dataset = lerobot_v3.LeRobotV3Dataset(root, sequence_keys=("actions",), sequence_length=4)
    assert len(dataset) == 21
    item = dataset[7]
    assert (item["episode_index"], item["frame_index"], item["task"]) == (0, 7, TASK)
    np.testing.assert_allclose(item["observation.tcp_pose"], np.full(6, 7.25))
    np.testing.assert_allclose(item["actions"][:, 0], [7.25, 8.25, 9.25, 10.25])
    for camera, key in recorder_lib.CAMERAS.items():
        assert item[key].shape == (HEIGHT, WIDTH, 3)
        np.testing.assert_allclose(item[key].mean(axis=(0, 1)), _images(7)[camera][0, 0], atol=6)

    # LIFT --online-intervention-only: chunks (clamped to the episode end) recorded entirely under human control.
    filtered = lerobot_v3.LeRobotV3Dataset(
        root,
        sequence_keys=("actions",),
        sequence_length=4,
        options=lerobot_v3.LeRobotV3Options(decode_videos=False, intervention_only=True),
    )
    rows = [(int(filtered[i]["episode_index"]), int(filtered[i]["frame_index"])) for i in range(len(filtered))]
    assert rows == [(0, frame) for frame in range(6, 12)] + [(1, frame) for frame in range(5)]


def test_recorder_rejects_incompatible_datasets_and_frames(tmp_path):
    root = tmp_path / "recording"
    _record(_recorder(root), [1, 1])
    with pytest.raises(recorder_lib.RecorderError, match="different features"):
        recorder_lib.EpisodeRecorder(root, task=TASK, fps=FPS, image_size=(HEIGHT * 2, WIDTH))
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "file").write_text("x")
    with pytest.raises(recorder_lib.RecorderError, match="no meta/info.json"):
        _recorder(tmp_path / "other")

    recorder = _recorder(root)
    recorder.start_episode()
    values = {key: np.zeros(recorder.features[key]["shape"]) for key in recorder.frame_keys}
    with pytest.raises(recorder_lib.RecorderError, match="front image"):
        recorder.add_frame(values, {"front": np.zeros((HEIGHT, WIDTH + 2, 3), np.uint8), "wrist": _images(0)["wrist"]})
    with pytest.raises(recorder_lib.RecorderError, match="invalid observation.tau_m"):
        recorder.add_frame({**values, "observation.tau_m": [np.nan] * 7}, _images(0))
    with pytest.raises(recorder_lib.RecorderError, match="missing actions"):
        recorder.add_frame({key: value for key, value in values.items() if key != "actions"}, _images(0))
    recorder.close()
    assert not list(root.rglob("*.tmp"))
    assert recorder.num_episodes == 1


def test_dagger_corrections_are_mixed_one_to_one_with_offline_data(tmp_path):
    human, policy = recorder_lib.HUMAN_FLAG, recorder_lib.POLICY_FLAG
    offline = _recorder(tmp_path / "offline")
    _record(offline, [human] * 12)
    _record(offline, [human] * 12)
    _record(_recorder(tmp_path / "dagger"), [policy] * 6 + [human] * 6)
    data_config = _config.DataConfig(
        repo_id=str(tmp_path / "offline"), prompt_from_task=True, dagger_repo_ids=(str(tmp_path / "dagger"),)
    )
    dataset = data_loader.create_torch_dataset(data_config, 4, None, decode_videos=False)
    assert (len(dataset), dataset.primary_size, dataset.secondary_size) == (30, 24, 6)
    item = dataset[24]
    assert (item["control_flag"], item["frame_index"], item["prompt"]) == (human, 6, TASK)

    sampler = data_loader.MixtureSampler(24, 6, batch_size=8, ratio=0.5, seed=3)
    first, second = list(sampler), list(sampler)
    assert len(first) == len(sampler) == 48
    batches = np.asarray(first).reshape(6, 8)
    assert np.all((batches >= 24).sum(axis=1) == 4)
    assert sorted(batches[batches < 24]) == list(range(24))
    assert first != second
    assert list(data_loader.MixtureSampler(24, 6, batch_size=8, ratio=0.5, seed=3)) == first
    with pytest.raises(ValueError, match="ratio"):
        data_loader.MixtureSampler(24, 6, batch_size=8, ratio=0.01)


def test_offline_sessions_are_read_as_one_dataset(tmp_path):
    human = recorder_lib.HUMAN_FLAG
    # A later session with longer episodes (e.g. 35 s instead of 30 s) and its own task string.
    _record(_recorder(tmp_path / "plug_in"), [human] * 12)
    longer = recorder_lib.EpisodeRecorder(
        tmp_path / "plug_in_35", task="Insert the plug", fps=FPS, image_size=(HEIGHT, WIDTH)
    )
    _record(longer, [human] * 14)
    _record(longer, [human] * 14)
    data_config = _config.DataConfig(
        repo_id=str(tmp_path / "plug_in"), extra_repo_ids=(str(tmp_path / "plug_in_35"),), prompt_from_task=True
    )
    dataset = data_loader.create_torch_dataset(data_config, 4, None, decode_videos=False)
    assert len(dataset) == 12 + 2 * 14
    rows = [(dataset[i]["episode_index"], dataset[i]["frame_index"], dataset[i]["prompt"]) for i in (11, 12, 39)]
    assert rows == [(0, 11, TASK), (0, 0, "Insert the plug"), (1, 13, "Insert the plug")]
    np.testing.assert_array_equal(dataset[39]["actions_is_pad"], [False, True, True, True])

    # DAgger post-training mixes its corrections with all offline sessions.
    _record(_recorder(tmp_path / "dagger"), [human] * 6)
    mixture = data_loader.create_torch_dataset(
        dataclasses.replace(data_config, dagger_repo_ids=(str(tmp_path / "dagger"),)), 4, None, decode_videos=False
    )
    assert (mixture.primary_size, mixture.secondary_size) == (40, 6)


def test_validation_split_holds_out_every_nth_episode(tmp_path):
    recorder = _recorder(tmp_path / "plug_in")
    for length in (4, 5, 6, 7):
        _record(recorder, [recorder_lib.HUMAN_FLAG] * length)
    data_config = _config.DataConfig(repo_id=str(tmp_path / "plug_in"), holdout_every=2)

    def episodes(config):
        dataset = data_loader.create_torch_dataset(config, 4, None, decode_videos=False)
        return [int(dataset[index]["episode_index"]) for index in range(len(dataset))]

    training, held_out = episodes(data_config), episodes(dataclasses.replace(data_config, holdout=True))
    assert (sorted(set(training)), len(training)) == ([0, 2], 4 + 6)
    assert (sorted(set(held_out)), len(held_out)) == ([1, 3], 5 + 7)


def test_offline_sessions_must_be_recorded_alike(tmp_path):
    _record(_recorder(tmp_path / "plug_in"), [1] * 4)

    def read_with(other, **changes):
        data_config = _config.DataConfig(
            repo_id=str(tmp_path / "plug_in"),
            extra_repo_ids=(str(other),),
            action_sequence_keys=("actions", "observation.tau_m"),
            **changes,
        )
        return data_loader.create_torch_dataset(data_config, 4, None, decode_videos=False)

    fast = tmp_path / "fast"
    _record(recorder_lib.EpisodeRecorder(fast, task=TASK, fps=15, image_size=(HEIGHT, WIDTH)), [1] * 4)
    with pytest.raises(ValueError, match="15 fps"):
        read_with(fast)
    other = tmp_path / "other"
    _record(_recorder(other), [1] * 4)
    assert len(read_with(other)) == 8
    with pytest.raises(ValueError, match="belongs to one dataset"):
        read_with(other, tactile_cache=str(tmp_path / "cache"))
    info_path = other / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    tau_m = info["features"].pop("observation.tau_m")
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="records observation.tau_m but"):
        read_with(other)
    info["features"]["observation.tau_m"] = tau_m
    info["features"]["observation.tcp_pose"]["shape"] = [7]
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match=r"observation.tcp_pose has shape \[6\]"):
        read_with(other)


def _record_tactile(recorder: recorder_lib.EpisodeRecorder, depth: np.ndarray, valid: np.ndarray, start: float) -> int:
    recorder.start_episode()
    for step in range(len(depth)):
        values = {key: np.full(recorder.features[key]["shape"], step + 0.25) for key in recorder.frame_keys}
        values["control_flag"] = recorder_lib.POLICY_FLAG
        values["sync.target_host_s"] = start + step / FPS
        recorder.add_frame(values, _images(step), (depth[step], valid[step]))
    return recorder.save_episode()


def test_tactile_rollouts_train_with_the_fingers_of_every_frame(tmp_path):
    # The recorder runs on the robot PC without openpi and keeps its own copy of the finger layout.
    assert (recorder_lib.TACTILE_SENSORS, recorder_lib.TACTILE_DEPTH_SHAPE) == (tactile.SENSORS, tactile.DEPTH_SHAPE)
    root = tmp_path / "rollouts"
    depth = np.random.default_rng(0).uniform(0.0, 0.3, size=(8, 2, *recorder_lib.TACTILE_DEPTH_SHAPE))
    depth = depth.astype(np.float16)
    valid = np.ones((8, 2), dtype=bool)
    valid[3, 0] = False  # one finger drops out for a frame,
    valid[[0, 6, 7], 1] = False  # the other one starts late and stops early
    recorder = _recorder(root, tactile=True, capture_info={"source": "test"})
    assert _record_tactile(recorder, depth, valid, start=100.0) == 0
    recorder.label_episode(0, "failure")
    cache_dir = tactile_cache.build_cache(root, workers=1)

    # A later session appends an episode in which only one finger ever reads; its outcome is added to the labels.
    one_finger = np.zeros((5, 2), dtype=bool)
    one_finger[:, 0] = True
    recorder = _recorder(root, tactile=True)
    assert _record_tactile(recorder, depth[:5], one_finger, start=300.0) == 1
    recorder.label_episode(1, "success")
    with pytest.raises(recorder_lib.RecorderError, match="no episode 2"):
        recorder.label_episode(2, "success")
    with pytest.raises(recorder_lib.RecorderError, match="unknown outcome"):
        recorder.label_episode(1, "ok")
    assert not (root / "tactile" / "episode-000001" / "tactile_depth.sensor_2.parquet").exists()
    capture = json.loads((root / "meta" / "ar3_capture.json").read_text())
    assert capture["source"] == "test"
    assert 0 < capture["robot"]["max_tactile_age_s"] < 1 / FPS

    # The cache is rebuilt for the new episode and gives every frame exactly the fingers it was recorded with:
    # a missing finger is not filled with its map of the frame before.
    assert tactile_cache.build_cache(root, workers=1) == cache_dir
    cache = tactile_cache.TactileCache(cache_dir)
    assert cache.episodes == {0: (0, 8), 1: (8, 5)}
    cached_depth, cached_valid = cache.frames(0, np.arange(8))
    np.testing.assert_array_equal(cached_valid, valid)
    np.testing.assert_array_equal(cached_depth, np.where(valid[:, :, None, None], depth, 0))
    np.testing.assert_array_equal(cache.frames(1, np.arange(5))[1], one_finger)

    # The policy trains on these fingers (wrench columns + maps), ...
    sensor_keys = (*ar3.FORCE_KEYS_WRENCH_TAU_M, *ar3.HAND_KEYS)
    data_config = _config.DataConfig(
        repo_id=str(root),
        action_sequence_keys=("actions", *sensor_keys),
        history_keys=sensor_keys,
        history_length=2,
        tactile_cache="auto",
    )
    dataset = data_loader.create_torch_dataset(data_config, 4, None, decode_videos=False)
    inputs = ar3.AR3LiftRepack(history_length=2, tactile=True)(dataset[3])
    np.testing.assert_array_equal(inputs["tactile_valid"], valid[1:7])
    np.testing.assert_array_equal(inputs["tactile_depth"], cached_depth[1:7])
    np.testing.assert_array_equal(inputs["tactile_wrench"][:, 0], np.arange(1, 7) + 0.25)

    # ... and offline RL reads them and how the episodes ended.
    table = rl_frames.load(root)
    np.testing.assert_array_equal(table.finger_valid, np.concatenate([valid, one_finger]))
    np.testing.assert_array_equal(table.depth[:8], cached_depth)
    assert not table.human.any()
    outcomes = rl_outcomes.load(root)
    assert [outcomes.success(index) for index in table.lengths] == [False, True]

    # Recordings with and without the fingers do not mix.
    with pytest.raises(recorder_lib.RecorderError, match="different features"):
        _recorder(root)
    plain = _recorder(tmp_path / "plain")
    plain.start_episode()
    values = {key: np.zeros(plain.features[key]["shape"]) for key in plain.frame_keys}
    with pytest.raises(recorder_lib.RecorderError, match="tactile=True"):
        plain.add_frame(values, _images(0), (depth[0], valid[0]))
    plain.close()
    recorder.start_episode()
    values = {key: np.zeros(recorder.features[key]["shape"]) for key in recorder.frame_keys}
    with pytest.raises(recorder_lib.RecorderError, match="missing the Tac G2 reading"):
        recorder.add_frame(values, _images(0))
    with pytest.raises(recorder_lib.RecorderError, match="Tac G2 maps"):
        recorder.add_frame(values, _images(0), (depth[0, :1], valid[0]))
    recorder.close()
    (root / "meta" / "ar3_capture.json").write_text(json.dumps({"robot": {"max_tactile_age_s": 0.2}}))
    with pytest.raises(recorder_lib.RecorderError, match="one Tac G2 map per frame"):
        _recorder(root, tactile=True)
