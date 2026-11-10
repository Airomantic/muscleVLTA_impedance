import dataclasses

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from scripts import ar3_sensor_attribution as attribution

HISTORY, HORIZON = 2, 4


def _observation(*, history: int, tactile: bool):
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        reactive_action_expert_variant="dummy",
        cross_attention_config="dummy",
        action_horizon=HORIZON,
        action_dim=8,
        reactive_in_use=True,
        cross_attention_latency=1,
        wrench_dim=13,
        sensor_history=history,
        tactile_in_use=tactile,
        tactile_depth_shape=(8, 8),
        tactile_feature_dim=16,
    )
    observation = config.fake_obs(batch_size=3)
    return dataclasses.replace(observation, wrench=jax.random.normal(jax.random.key(0), observation.wrench.shape))


def test_counterfactuals_replace_one_stream_at_a_time():
    observation = _observation(history=HISTORY, tactile=True)
    variants = attribution.counterfactuals(observation, HISTORY)
    wrench = np.asarray(observation.wrench)
    start = wrench[:, HISTORY : HISTORY + 1]

    # The changes of this sample at the level of the next one, and this sample's level without its changes.
    swapped = np.asarray(variants["arm level swapped"].wrench)
    np.testing.assert_allclose(swapped - swapped[:, HISTORY : HISTORY + 1], wrench - start, atol=1e-5)
    np.testing.assert_allclose(swapped[:, HISTORY], np.roll(wrench, 1, axis=0)[:, HISTORY], atol=1e-5)
    np.testing.assert_array_equal(variants["arm changes removed"].wrench, np.broadcast_to(start, wrench.shape))

    assert not np.asarray(variants["hand removed"].tactile_valid).any()
    np.testing.assert_array_equal(variants["hand removed"].wrench, wrench)
    valid = np.asarray(variants["history removed"].sensor_valid)
    assert not valid[:, :HISTORY].any()
    assert valid[:, HISTORY:].all()
    assert not np.asarray(variants["all sensors removed"].wrench_mask).any()

    # LIFT: no hand stream and no history to remove.
    lift = attribution.counterfactuals(_observation(history=0, tactile=False), 0)
    assert set(lift) == {"arm swapped", "arm level swapped", "arm changes removed", "all sensors removed"}


def test_summary_reports_change_and_error_per_chunk_type():
    recorded = np.random.default_rng(0).normal(size=(6, HORIZON, 7)).astype(np.float32)
    contact = np.array([True, True, False, False, False, False])
    base = recorded + 0.1
    shifted = base.copy()
    shifted[contact, :, :3] += 2.0 * recorded.reshape(-1, 7).std(axis=0)[:3]

    summary = attribution.summarize({attribution.RECORDED: base, "hand removed": shifted}, recorded, contact)

    expected = {"xyz": 0.0, "rotation": 0.0, "gripper": 0.0, "error": 0.01, "chunks": 4}
    assert summary[attribution.RECORDED]["free"] == pytest.approx(expected, abs=1e-5)
    row = summary["hand removed"]
    assert row["contact"]["xyz"] == pytest.approx(2.0, rel=1e-4)
    assert (row["contact"]["rotation"], row["free"]["xyz"], row["contact"]["chunks"]) == (0.0, 0.0, 2)
    assert row["contact"]["error"] > row["free"]["error"] == pytest.approx(0.01, abs=1e-5)
    assert "contact (2 chunks)" in attribution.format_summary(summary)


def test_checkpoint_statistics_datasets_and_validation_split_reach_the_data_config(tmp_path):
    stats = tmp_path / "assets" / "plug_in+plug_in_35"
    stats.mkdir(parents=True)
    (stats / "norm_stats.json").write_text("{}")
    args = attribution.parse_args(
        ["--checkpoint", str(tmp_path), "--dataset", "/data/a:/data/b", "--holdout-every", "8"]
    )

    config = attribution._train_config(args)

    assert (config.data.repo_id, config.data.base_config.extra_repo_ids) == ("/data/a", ("/data/b",))
    assert (config.data.base_config.holdout_every, config.data.base_config.holdout) == (8, True)
    assert (config.data.assets.assets_dir, config.data.assets.asset_id) == (str(stats.parent), stats.name)
    # Without --holdout-every all episodes are read.
    everything = attribution._train_config(attribution.parse_args(["--checkpoint", str(tmp_path)]))
    assert (everything.data.base_config.holdout_every, everything.data.base_config.holdout) == (0, False)
