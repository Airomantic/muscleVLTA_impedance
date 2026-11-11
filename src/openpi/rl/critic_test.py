import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.rl import critic
from openpi.rl import frames
from openpi.rl import outcomes

TINY = {"history": 3, "bins": 21, "width": 16, "hidden": 32, "tactile_features": 8, "dropout": 0.0}


def _table(
    completions: list[int], length: int = 24, *, task: str = "insert", human: bool = True, sensors: bool = True
) -> frames.FrameTable:
    """Episodes whose gripper closes at frame 3 and releases at the completion frame; the TCP x coordinate counts
    the steps that are left until then."""
    rng = np.random.default_rng(len(completions))
    total = length * len(completions)
    steps = np.tile(np.arange(length), len(completions))
    completion = np.repeat(completions, length)
    pose = rng.normal(scale=0.01, size=(total, 6)).astype(np.float32)
    pose[:, 0] += np.maximum(completion - steps, 0) / length
    target = np.where((steps >= 3) & (steps < completion), 300.0, 1000.0).astype(np.float32)
    return frames.FrameTable(
        root=pathlib.Path(f"/data/{task}"),
        fps=10.0,
        tasks={0: task},
        lengths=dict.fromkeys(range(len(completions)), length),
        episode=np.repeat(np.arange(len(completions)), length),
        frame=steps,
        task=np.zeros(total, dtype=np.int64),
        pose=pose,
        gripper=np.full(total, 500.0, dtype=np.float32),
        gripper_target=target,
        arm=rng.normal(size=(total, 13)).astype(np.float32),
        finger_wrench=rng.normal(size=(total, 12)).astype(np.float32),
        finger_valid=np.full((total, 2), sensors),
        human=np.full(total, human),
        depth=rng.uniform(0, 0.3, size=(total, 2, 8, 8)).astype(np.float16) if sensors else None,
        vision=rng.normal(size=(total, 2, 12)).astype(np.float16) if sensors else None,
    )


def test_two_hot_targets_keep_the_value():
    targets = jnp.asarray([-1.0, -0.63, -0.5, 0.0])
    encoded = critic.two_hot(targets, 21)
    np.testing.assert_allclose(encoded.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(encoded @ jnp.linspace(-1.0, 0.0, 21), targets, atol=1e-6)
    assert int((encoded[1] > 0).sum()) == 2


def test_value_targets_follow_the_outcome_and_the_source_of_every_episode():
    demos = _table([12, 20])
    # A rollout of the same task: the policy acts, a human takes over episode 0 from frame 10; episode 1 fails.
    rollout = _table([16, 16], human=False)
    rollout.human[10:24] = True
    data = critic.build_data(
        [demos, rollout], [outcomes.Outcomes(default="success"), outcomes.Outcomes("success", {1: "failure"})]
    )

    assert data.task_names == ["insert"]
    assert data.offsets == [0, 48]
    # Normalized by the longest time of the task: the failed rollout runs through its 24 frames.
    np.testing.assert_allclose(data.norm, 23.0)
    np.testing.assert_allclose(data.targets[:13], -np.arange(12, -1, -1) / 23.0, atol=1e-6)
    assert data.done[:24].tolist() == [False] * 12 + [True] * 12
    np.testing.assert_allclose(data.targets[72:], -1.0)
    assert not data.done[72:].any()
    assert data.success.tolist() == [True] * 72 + [False] * 24
    # Only the first dataset is teleoperated throughout; the takeover is a correction, not a demonstration.
    assert data.demonstration.tolist() == [True] * 48 + [False] * 48
    assert data.human[48:72].tolist() == [False] * 10 + [True] * 14
    # Sensor windows never reach into the previous episode.
    assert data.arrays["first"][[0, 23, 24, 47, 48, 95]].tolist() == [0, 0, 24, 24, 48, 72]
    assert data.last[[0, 24, 95]].tolist() == [23, 47, 95]

    # A rollout saved without an outcome names the dataset and the command that labels it.
    with pytest.raises(ValueError, match=r"episode 1 has no outcome label.*ar3_rl.py label"):
        critic.build_data([rollout], [outcomes.Outcomes(episodes={0: "success"})])


def test_cross_fitted_critic_scores_every_episode_without_having_seen_it(tmp_path):
    data = critic.build_data([_table([12, 20, 15, 18, 10, 22, 14, 17])], [outcomes.Outcomes(default="success")])
    config = critic.CriticConfig(streams=("pose",), **TINY)
    values, params, stats = critic.cross_fit(config, data, folds=2, steps=400, batch_size=64, learning_rate=3e-3)

    assert len(params) == 2
    # The TCP x coordinate tells how many steps are left, also in the episodes of the other fold.
    assert np.abs(values - data.targets).mean() < 0.06
    # Each fold's critic is the one that scored its held-out episodes.
    tables = critic.device_tables(config, data, stats)
    for fold in range(2):
        rows = np.flatnonzero(data.episode % 2 == fold)
        np.testing.assert_allclose(critic.predict(config, params[fold], tables, rows), values[rows], atol=1e-5)

    critic.save(tmp_path, config, params, stats, data.task_names)
    loaded_config, loaded_params, loaded_stats, task_names = critic.load(tmp_path)
    assert loaded_config == config
    assert task_names == ["insert"]
    rows = np.arange(len(data))
    np.testing.assert_allclose(
        critic.predict(loaded_config, loaded_params[0], critic.device_tables(config, data, loaded_stats), rows),
        critic.predict(config, params[0], tables, rows),
        atol=1e-5,
    )


def test_critic_reads_all_streams_and_names_the_inputs_it_misses():
    data = critic.build_data([_table([12, 20, 15, 18])], [outcomes.Outcomes(default="success")])
    config = critic.CriticConfig(**TINY)
    values, _, stats = critic.cross_fit(config, data, folds=2, steps=3, batch_size=8)
    assert values.shape == (96,)
    assert np.isfinite(values).all()
    assert np.all((values >= -1) & (values <= 0))
    assert stats["vision_mean"].shape == (2, 12)

    # History rows before the episode start are invalid and clamped to its first frame.
    tables = critic.device_tables(config, data, stats)
    batch = critic.gather(tables, jnp.asarray([24, 26]), config.history)
    assert batch["valid"].tolist() == [[False, False, True], [True, True, True]]
    np.testing.assert_array_equal(batch["arm"][0, 0], batch["arm"][0, 2])
    assert not batch["finger_valid"][0, :2].any()

    blind = critic.build_data([_table([12, 20], sensors=False)], [outcomes.Outcomes(default="success")])
    with pytest.raises(ValueError, match="ar3_rl.py features"):
        critic.cross_fit(critic.CriticConfig(streams=("vision",), **TINY), blind, folds=2, steps=1)
    with pytest.raises(ValueError, match="build_ar3_tactile_cache.py"):
        critic.cross_fit(critic.CriticConfig(streams=("touch",), **TINY), blind, folds=2, steps=1)
    with pytest.raises(ValueError, match="smell"):
        critic.CriticConfig(streams=("smell",))
