import re

import numpy as np
import pytest

from scripts import ar3_lift_recorder as recorder_lib
from scripts import ar3_smoothness as smoothness

FPS = 10.0


def _reach(distance: float = 0.3, duration: float = 6.0) -> np.ndarray:
    """A minimum-jerk straight reach along x, sampled at FPS."""
    phase = np.arange(0.0, duration + 1e-9, 1.0 / FPS) / duration
    path = np.zeros((len(phase), 3))
    path[:, 0] = distance * (10 * phase**3 - 15 * phase**4 + 6 * phase**5)
    return path


def _jittery(path: np.ndarray) -> np.ndarray:
    """The path with a 4 mm, 2 Hz oscillation that fades in and out with the movement."""
    time = np.arange(len(path)) / FPS
    wobble = 0.004 * np.sin(2 * np.pi * 2.0 * time) * np.sin(np.pi * time / time[-1])
    return path + np.stack([wobble, np.zeros_like(wobble), np.zeros_like(wobble)], axis=1)


def test_metrics_rank_a_jittery_path_below_a_smooth_one():
    smooth, jittery = _reach(), _jittery(_reach())

    def sparc(path):
        return smoothness.sparc(smoothness.speed_profile(path, FPS), FPS)

    assert sparc(jittery) < sparc(smooth) < 0
    assert smoothness.log_dimensionless_jerk(jittery, FPS) < smoothness.log_dimensionless_jerk(smooth, FPS)
    assert smoothness.rms_jerk(jittery, FPS) > 5 * smoothness.rms_jerk(smooth, FPS)
    # SPARC and LDLJ ignore the amplitude of the movement; RMS jerk scales with it.
    assert sparc(2 * smooth) == pytest.approx(sparc(smooth))
    assert smoothness.log_dimensionless_jerk(2 * smooth, FPS) == pytest.approx(
        smoothness.log_dimensionless_jerk(smooth, FPS)
    )
    assert smoothness.rms_jerk(2 * smooth, FPS) == pytest.approx(2 * smoothness.rms_jerk(smooth, FPS))


def test_command_metrics_find_chunk_restarts_and_gripper_chatter():
    time = np.arange(60)
    targets = np.zeros((60, 3))
    targets[:, 0] = 0.1 * np.sin(0.05 * time)
    assert smoothness.boundary_ratio(targets, 10) == pytest.approx(1.0, abs=0.2)
    # Every new chunk restarts 2 mm away from where the previous one ended.
    restarting = targets.copy()
    restarting[:, 0] += 0.002 * (time // 10)
    assert smoothness.boundary_ratio(restarting, 10) > 10

    steady = np.concatenate([np.full(20, 1000.0), np.linspace(1000, 300, 20), np.full(20, 300.0)])
    chatter = steady + np.where(time % 2 == 0, 15.0, -15.0) * (time >= 40)
    assert smoothness.gripper_reversals(steady, FPS) == 0
    # The closing ramp turns into 20 alternating steps: 20 direction changes in 6 s.
    assert smoothness.gripper_reversals(chatter, FPS) == pytest.approx(20 / 6.0)


def _record(recorder: recorder_lib.EpisodeRecorder, path: np.ndarray, flags: list[int]) -> None:
    recorder.start_episode()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    for position, flag in zip(path, flags, strict=True):
        values = {key: np.zeros(recorder.features[key]["shape"]) for key in recorder.frame_keys}
        values["observation.tcp_pose"] = np.concatenate([position, np.zeros(3)])
        values["actions"] = np.concatenate([position, np.zeros(3), [1000.0]])
        values["control_flag"] = flag
        recorder.add_frame(values, {"front": image, "wrist": image})
    recorder.save_episode()


def test_rollout_recordings_are_summarised_on_their_policy_frames(tmp_path):
    human, policy = recorder_lib.HUMAN_FLAG, recorder_lib.POLICY_FLAG
    recorder = recorder_lib.EpisodeRecorder(tmp_path / "rollouts", task="insert", fps=int(FPS), image_size=(48, 64))
    reach = _reach()
    # The operator moves the arm away for 3 s before the policy takes over; only the policy part is the rollout.
    approach = np.linspace([0.0, 0.2, 0.0], [0.0, 0.0, 0.0], 30)
    _record(recorder, np.concatenate([approach, _jittery(reach)]), [human] * 30 + [policy] * len(reach))
    _record(recorder, reach, [policy] * len(reach))

    result = smoothness.summarize_run([str(tmp_path / "rollouts")], "auto", 10)

    assert result["episodes"] == 2
    jittery, smooth = result["per_episode"]
    assert jittery["sparc"] < smooth["sparc"]
    assert jittery["duration"] <= len(reach) / FPS
    assert smooth["peak_speed"] == pytest.approx(0.3 * 1.875 / 6.0, rel=0.02)
    low, high = result["summary"]["sparc"]["ci95"]
    assert low <= result["summary"]["sparc"]["mean"] <= high
    # The human part of the recording is another movement.
    by_hand = smoothness.summarize_run([str(tmp_path / "rollouts")], "human", 10)
    assert by_hand["episodes"] == 1
    assert by_hand["per_episode"][0]["peak_speed"] == pytest.approx(0.2 / 2.9, rel=0.02)

    figure = smoothness.draw({"policy": result, "human": by_hand}).svg()
    assert set(re.findall(r"<([a-z]+)", figure)) <= {"xml", "svg", "rect", "path", "circle", "text"}
    assert "n = 2" in figure
