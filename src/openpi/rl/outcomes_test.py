import numpy as np
import pytest

from openpi.rl import outcomes


def test_completion_is_the_end_of_the_last_gripper_opening():
    # Open, grasp at frames 3-4, hold, release at frames 8-9, then retreat with the gripper open.
    target = np.array([1000, 1000, 1000, 600, 300, 300, 300, 300, 700, 1000, 1000, 1000], dtype=np.float32)
    assert outcomes.completion_frame(target) == 9
    assert outcomes.completion_frame(target, "end") == 11
    # The episode ends while the object is still held: the task is complete at the last frame at the earliest.
    assert outcomes.completion_frame(target[:8]) == 7
    assert outcomes.completion_frame(np.full(5, 1000.0)) == 4
    # Command noise within the deadband is not a release.
    assert outcomes.completion_frame(target + np.where(np.arange(12) == 11, 3.0, 0.0)) == 9


def test_value_targets_count_the_steps_to_completion_and_put_failures_at_the_bottom():
    success = outcomes.value_targets(6, success=True, completion=4, norm=8.0)
    np.testing.assert_allclose(success, [-0.5, -0.375, -0.25, -0.125, 0.0, 0.0])
    np.testing.assert_allclose(outcomes.step_costs(6, success=True, completion=4, norm=8.0), [-0.125] * 4 + [0.0] * 2)

    failure = outcomes.value_targets(6, success=False, completion=4, norm=8.0)
    np.testing.assert_allclose(failure, -1.0)
    # A smaller penalty keeps the time left visible: 6 steps and the penalty at the first frame.
    mild = outcomes.value_targets(6, success=False, completion=4, norm=20.0, fail_penalty=0.5)
    np.testing.assert_allclose(mild, [-0.8, -0.75, -0.7, -0.65, -0.6, -0.55])
    # Every step of a failed episode costs time, also its last.
    np.testing.assert_allclose(outcomes.step_costs(6, success=False, completion=4, norm=20.0), [-0.05] * 6)


def test_outcomes_round_trip_and_reject_unknown_or_missing_labels(tmp_path):
    with pytest.raises(FileNotFoundError, match="ar3_rl.py label"):
        outcomes.load(tmp_path)
    labels = outcomes.Outcomes(default="success", episodes={3: "failure"}, completion_frames={7: 212})
    outcomes.save(tmp_path, labels)
    assert outcomes.load(tmp_path) == labels
    assert labels.success(0)
    assert not labels.success(3)

    with pytest.raises(KeyError, match="Episode 5"):
        outcomes.Outcomes(episodes={3: "failure"}).success(5)
    with pytest.raises(ValueError, match="aborted"):
        outcomes.Outcomes(episodes={1: "aborted"})
    assert outcomes.parse_episodes(["3", "0-2", "7,9-10"]) == [3, 0, 1, 2, 7, 9, 10]
