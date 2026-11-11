import json

import numpy as np
import pytest

from openpi.policies import ar3_lift_policy as ar3
from openpi.rl import advantage
from openpi.rl import outcomes
from openpi.training import config as _config
from openpi.training import data_loader
from scripts import ar3_lift_recorder as recorder_lib
from scripts import ar3_rl

HUMAN, POLICY = recorder_lib.HUMAN_FLAG, recorder_lib.POLICY_FLAG
LENGTH = 16


def _record(root, episodes: list[tuple[int | None, list[int]]], task: str = "insert the plug") -> str:
    """Episodes (completion frame, control flags): the gripper closes at frame 2 and releases at the completion
    frame, and the TCP x coordinate counts the steps left until then. Without a completion frame the episode fails:
    the plug jams off the socket (TCP y) and is never released."""
    recorder = recorder_lib.EpisodeRecorder(root, task=task, fps=10, image_size=(48, 64))
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    for completion, flags in episodes:
        recorder.start_episode()
        for step, flag in enumerate(flags):
            values = {key: np.zeros(recorder.features[key]["shape"]) for key in recorder.frame_keys}
            held = step >= 2 and (completion is None or step < completion)
            target = 300.0 if held else 1000.0
            pose = [0.5, 1.0] if completion is None else [max(completion - step, 0) / LENGTH, 0.0]
            values["observation.tcp_pose"] = np.array([*pose, 0, 0, 0, 0])
            values["observation.gripper_position"] = [target]
            values["actions"] = np.array([0, 0, 0, 0, 0, 0, target])
            values["control_flag"] = flag
            recorder.add_frame(values, {"front": image, "wrist": image})
        recorder.save_episode()
    recorder.close()
    return str(root)


def test_a_round_labels_demonstrations_rollouts_and_corrections(tmp_path, capsys):
    demos = _record(tmp_path / "demos", [(completion, [HUMAN] * LENGTH) for completion in (8, 12, 10, 14)])
    rollouts = _record(
        tmp_path / "rollouts",
        [
            (10, [POLICY] * LENGTH),
            (None, [POLICY] * LENGTH),
            (12, [POLICY] * 6 + [HUMAN] * 10),  # the operator takes over and finishes
            (9, [POLICY] * LENGTH),
            (None, [POLICY] * LENGTH),
            (11, [POLICY] * LENGTH),
        ],
    )
    with pytest.raises(SystemExit, match="episode 0 has no label"):
        ar3_rl.main(["label", "--dataset", rollouts, "--failure", "1"])
    ar3_rl.main(["label", "--dataset", demos, "--default", "success"])
    ar3_rl.main(["label", "--dataset", rollouts, "--default", "success", "--failure", "1", "4"])
    assert outcomes.load(rollouts) == outcomes.Outcomes(default="success", episodes={1: "failure", 4: "failure"})
    assert "4 successes, 2 failures [1, 4]" in capsys.readouterr().out

    arguments = ["--dataset", f"{demos}:{rollouts}", "--n-step", "4", "--rollout-fraction", "0.5"]
    critic_dir = tmp_path / "critic"
    fit = ["fit", *arguments, "--tag", "round1", "--out", str(critic_dir), "--vision", "none", "--streams", "pose"]
    ar3_rl.main([*fit, "--folds", "2", "--steps", "300", "--batch-size", "32", "--learning-rate", "3e-3"])

    report = json.loads((critic_dir / "report.json").read_text())
    task = report["critic"]["insert the plug"]
    assert (task["episodes"], task["failed_episodes"]) == (10, 2)
    # The pose tells the steps that are left, and the critic of each fold recognizes the failure it has not seen.
    assert task["mae_seconds"] < 0.4
    assert task["outcome_auc"] > 0.9
    assert report["labels"]["positive"]["human corrections"] == 1.0
    assert report["labels"]["positive"]["policy frames of failed episodes"] == 0.0
    assert report["labels"]["positive"]["demonstration frames"] == pytest.approx(0.5, abs=0.1)

    labels = advantage.AdvantageLabels.load(rollouts, "round1")
    assert labels.lengths == dict.fromkeys(range(6), LENGTH)
    assert labels.info["n_step"] == 4
    assert labels.info["critic"] == str(critic_dir)
    rows = labels.arrays["label"].reshape(6, LENGTH)
    # Corrections and the frames after the task is complete are positive whatever their advantage.
    assert rows[2, 6:].all()
    assert rows[0, 10:].all()
    assert rows[3, 9:].all()
    # Time spent in a failed episode is never an improvement, up to its last frame.
    assert not rows[[1, 4]].any()
    assert rows[[0, 3, 5], :9].mean() > 0.5

    # Another positive share reuses the critic's values.
    ar3_rl.main(["relabel", *arguments, "--from-tag", "round1", "--tag", "strict", "--demo-fraction", "0.2"])
    strict = advantage.AdvantageLabels.load(demos, "strict")
    np.testing.assert_array_equal(
        strict.arrays["value"], advantage.AdvantageLabels.load(demos, "round1").arrays["value"]
    )
    assert strict.arrays["label"].sum() < advantage.AdvantageLabels.load(demos, "round1").arrays["label"].sum()

    # Training reads the label of every sampled chunk and writes it into the prompt: only negative chunks are marked.
    data_config = _config.DataConfig(repo_id=rollouts, prompt_from_task=True, advantage_tag="round1")
    dataset = data_loader.create_torch_dataset(data_config, 4, None, decode_videos=False)
    prompts = set()
    for index in (LENGTH + 15, 2 * LENGTH + 8):
        item = dataset[index]
        expected = labels.label(int(item["episode_index"]), int(item["frame_index"]))
        assert item[ar3.ADVANTAGE_KEY] == expected
        repacked = ar3.AR3LiftRepack(force_keys=())(item)
        prompt = advantage.AdvantagePrompt()(repacked)["prompt"]
        assert prompt == ("insert the plug" if expected else "insert the plug. Advantage: negative")
        prompts.add(prompt)
    assert len(prompts) == 2
    with pytest.raises(FileNotFoundError, match="advantage_round2.npz"):
        data_loader.create_torch_dataset(
            _config.DataConfig(repo_id=rollouts, advantage_tag="round2"), 4, None, decode_videos=False
        )


def test_datasets_are_given_as_arguments_or_as_one_list():
    args = ar3_rl.parse_args(["fit", "--dataset", "/data/a:/data/b", "/data/c", "--tag", "t"])
    assert ar3_rl.dataset_roots(args.dataset) == ["/data/a", "/data/b", "/data/c"]
    assert (args.n_step, args.folds, args.streams) == (50, 4, ["vision", "pose", "arm", "touch"])
