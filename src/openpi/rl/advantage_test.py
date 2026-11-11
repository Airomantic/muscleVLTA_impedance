import numpy as np
import pytest

from openpi.rl import advantage


def test_n_step_advantage_bootstraps_inside_the_episode_and_uses_the_return_at_its_end():
    # Two episodes of 5 frames; a step costs 0.1 until the last frame, so the return to go is -0.4 ... 0.
    costs = np.tile([-0.1, -0.1, -0.1, -0.1, 0.0], 2)
    targets = np.tile([-0.4, -0.3, -0.2, -0.1, 0.0], 2).astype(np.float32)
    last = np.repeat([4, 9], 5)

    # A critic that predicts the return exactly sees no advantage anywhere.
    np.testing.assert_allclose(advantage.n_step_advantage(targets, targets, costs, last, 2), 0.0, atol=1e-6)

    # The critic expects the second episode to take two steps longer than it does: the chunks that reach the end of
    # the episode get the full credit, the others only what the critic concedes after n steps.
    values = targets.copy()
    values[5:] -= 0.2
    result = advantage.n_step_advantage(values, targets, costs, last, 2)
    np.testing.assert_allclose(result[:5], 0.0, atol=1e-6)
    np.testing.assert_allclose(result[5:], [0.0, 0.0, 0.0, 0.2, 0.2], atol=1e-6)
    # Frame 3 looks 2 steps ahead to frame 5 of the next episode: it must use its own episode's return instead.
    assert result[3] == pytest.approx(0.0, abs=1e-6)

    # A failed episode whose critic already expects the failure: every further step is lost time, up to the end.
    failed = advantage.n_step_advantage(np.full(5, -1.0), np.full(5, -1.0), np.full(5, -0.1), last[:5], 2)
    np.testing.assert_allclose(failed, -0.2, atol=1e-6)


def test_labels_take_the_best_share_of_every_task_and_source_and_keep_forced_frames_positive():
    values = np.arange(20, dtype=np.float32)
    task = np.repeat([0, 1], 10)
    rollout = np.tile([False] * 6 + [True] * 4, 2)
    forced = np.zeros(20, dtype=bool)
    forced[[0, 19]] = True  # the worst demonstration frame of task 0 and the best policy frame of task 1
    labels, thresholds = advantage.label_frames(
        values, task=task, rollout=rollout, forced=forced, demo_fraction=0.4, rollout_fraction=0.5
    )
    assert labels[0] == labels[19] == advantage.POSITIVE
    # Task 0: demonstration frames 1-5 (best 40%: 4, 5), policy frames 6-9 (best half: 8, 9).
    assert labels[:10].tolist() == [1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    # Task 1: demonstration frames 10-15 (best 40%: 14, 15), policy frames 16-18 without the forced one (best: 18).
    assert labels[10:].tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    assert set(thresholds) == {"0/demonstration", "0/rollout", "1/demonstration", "1/rollout"}

    everything, _ = advantage.label_frames(
        values, task=task, rollout=rollout, forced=forced, demo_fraction=1.0, rollout_fraction=0.0
    )
    assert everything[~rollout].all()
    assert not everything[rollout & ~forced].any()


def test_labels_are_stored_per_dataset_and_checked_against_its_episodes(tmp_path):
    with pytest.raises(FileNotFoundError, match="ar3_rl.py fit"):
        advantage.AdvantageLabels.load(tmp_path, "round1")
    lengths = {2: 3, 5: 2}
    label = np.array([1, 0, 0, 1, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="5 frames"):
        advantage.save_labels(tmp_path, "round1", lengths=lengths, arrays={"label": label[:4]}, info={})
    path = advantage.save_labels(
        tmp_path, "round1", lengths=lengths, arrays={"label": label, "value": np.zeros(5)}, info={"n_step": 10}
    )
    assert path == tmp_path / "rl" / "advantage_round1.npz"

    labels = advantage.AdvantageLabels.load(tmp_path, "round1")
    assert labels.info == {"n_step": 10}
    assert [labels.label(2, frame) for frame in range(3)] == [1, 0, 0]
    assert [labels.label(5, frame) for frame in range(2)] == [1, 1]
    with pytest.raises(IndexError):
        labels.label(5, 2)
    labels.check_episodes(lengths)
    with pytest.raises(ValueError, match="episode 5"):
        labels.check_episodes({2: 3, 5: 4})
    with pytest.raises(ValueError, match="episode 6"):
        labels.check_episodes({6: 1})


def test_training_samples_get_their_label_and_requests_the_served_indicator(tmp_path):
    advantage.save_labels(tmp_path, "t", lengths={0: 2}, arrays={"label": np.array([1, 0], dtype=np.int8)}, info={})
    labels = advantage.AdvantageLabels.load(tmp_path, "t")
    samples = [
        {"episode_index": np.int64(0), "frame_index": np.int64(frame), "prompt": "insert the plug."} for frame in (0, 1)
    ]

    positive, negative = (advantage.AttachAdvantage(labels)(sample) for sample in samples)
    assert positive[advantage.ADVANTAGE_KEY] == advantage.POSITIVE
    assert negative[advantage.ADVANTAGE_KEY] == advantage.NEGATIVE
    assert advantage.ADVANTAGE_KEY not in samples[1]
    dropped = advantage.AttachAdvantage(labels, dropout=1.0)(samples[1])
    assert dropped[advantage.ADVANTAGE_KEY] == advantage.UNCONDITIONED

    # Only negative chunks are marked: positive chunks and robot requests keep the prompt of the policy.
    prompt = advantage.AdvantagePrompt()
    conditioned = prompt(negative)
    assert conditioned["prompt"] == "insert the plug. Advantage: negative"
    assert advantage.ADVANTAGE_KEY not in conditioned
    assert prompt(positive)["prompt"] == "insert the plug."
    assert prompt({"prompt": "insert the plug."})["prompt"] == "insert the plug."
    assert advantage.AdvantagePrompt("negative")({"prompt": np.asarray("insert")})["prompt"] == (
        "insert. Advantage: negative"
    )

    # Both indicators, as RECAP: a dropped label leaves the plain prompt, requests are served as positive.
    recap = advantage.AdvantagePrompt(marking="both")
    assert recap(positive)["prompt"] == "insert the plug. Advantage: positive"
    assert recap(negative)["prompt"] == "insert the plug. Advantage: negative"
    assert recap(dropped)["prompt"] == "insert the plug."
    assert recap({"prompt": "insert the plug"})["prompt"] == "insert the plug. Advantage: positive"
    assert advantage.AdvantagePrompt("none", "both")({"prompt": "insert the plug"})["prompt"] == "insert the plug"
    with pytest.raises(ValueError, match="good"):
        advantage.AdvantagePrompt("good")
    with pytest.raises(ValueError, match="marking"):
        advantage.AdvantagePrompt(marking="positive")
