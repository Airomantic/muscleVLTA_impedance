"""Offline motion/state-machine tests: only fake clocks, feedback and services; no ROS or robot access."""

import contextlib
import itertools
import json
from pathlib import Path
import types

import numpy as np
import pytest

from scripts import ar3_lift_robot_client as client


def copy_pose(pose):
    return tuple(value.copy() for value in pose)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.on_sleep = lambda _dt: None

    def __call__(self):
        return self.now

    def sleep(self, dt):
        assert dt >= 0
        self.now += dt
        self.on_sleep(dt)


class HomingRobot:
    def __init__(self, clock, *, follow_after=0.0, fault_at=None, limits=None):
        self.clock = clock
        self.limits = limits or client.Limits(workspace_radius=0.45)
        self.measured = client.pose_from_vector([0.0] * 6)
        self.command = copy_pose(self.measured)
        self.target = copy_pose(self.measured)
        self.follow_after = follow_after
        self.fault_at = fault_at
        self.in_home_mode = False
        self.mode_events = []
        self.targets = []
        self.history = []
        self.heartbeats = []
        self.last_heartbeat = clock()
        clock.on_sleep = self.advance

    @contextlib.contextmanager
    def return_home_mode(self):
        self.in_home_mode = True
        self.mode_events.append("enter")
        try:
            yield client.home_limits(self.limits)
        finally:
            self.in_home_mode = False
            self.mode_events.append("exit")

    def heartbeat(self):
        if self.fault_at is not None and self.clock() >= self.fault_at:
            raise client.RobotError("controller stopping: tau_ext_base raw channel 5=10.878 exceeds abs limit=10.000")
        self.last_heartbeat = self.clock()
        self.heartbeats.append(self.clock())

    def poses(self):
        return copy_pose(self.measured), copy_pose(self.command), copy_pose(self.target)

    def set_target(self, position, rotation, *, clamp=True):
        assert not clamp, "home waypoints must be validated rather than silently clamped"
        self.target = (np.asarray(position).copy(), np.asarray(rotation).copy())
        self.command = copy_pose(self.target)
        self.targets.append((self.clock(), copy_pose(self.target)))
        return False

    def advance(self, _dt):
        if self.clock() - self.last_heartbeat > self.limits.target_timeout + 1e-9:
            raise client.RobotError("policy target is stale during home/gripper wait")
        if self.clock() >= self.follow_after:
            self.measured = copy_pose(self.command)
        self.history.append((self.clock(), copy_pose(self.measured), copy_pose(self.command)))


class HomingGripper:
    def __init__(self, clock, *, open_after=0.0):
        self.clock = clock
        self.open_after = open_after
        self.commanded_at = None
        self.commands = []

    def command(self, value):
        self.commands.append(value)
        if self.commanded_at is None:
            self.commanded_at = self.clock()

    def read(self):
        return 1000.0 if self.commanded_at is not None and self.clock() - self.commanded_at >= self.open_after else 0.0

    def wait_position(self, *_args, **_kwargs):
        pytest.fail("blocking gripper wait starves the arm heartbeat")


def test_home_follows_a_smooth_speed_bounded_path_and_settles_before_success():
    clock = Clock()
    limits = client.Limits(workspace_radius=0.45, max_linear_speed=0.015, max_angular_speed=0.08)
    robot = HomingRobot(clock, limits=limits)
    goal = [0.06, 0.0, 0.0, 0.0, 0.0, 0.2]
    client.move_to_start(robot, None, goal, timeout=20.0, clock=clock, sleep=clock.sleep)
    assert robot.mode_events == ["enter", "exit"]
    assert not robot.in_home_mode
    np.testing.assert_allclose(robot.measured[0], goal[:3], atol=1e-9)
    np.testing.assert_allclose(robot.measured[1], client.rpy_matrix(goal[3:]), atol=1e-9)
    assert len(robot.targets) > 20
    for (before, previous), (now, target) in itertools.pairwise(robot.targets):
        distance, angle = client.pose_error(previous, target)
        assert distance <= limits.max_linear_speed * (now - before) + 1e-9
        assert angle <= limits.max_angular_speed * (now - before) + 1e-7
    first_arrival = next(t for t, _, command in robot.history if max(client.pose_error(command, robot.target)) < 1e-6)
    assert clock() - first_arrival >= 0.29


@pytest.mark.parametrize("goal", [[0.059, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0.07]])
def test_stationary_feedback_outside_home_tolerance_does_not_count_as_arrival(goal):
    clock = Clock()
    robot = HomingRobot(clock, follow_after=float("inf"))
    with pytest.raises(client.RobotError):
        client.move_to_start(robot, None, goal, timeout=8.0, clock=clock, sleep=clock.sleep)
    assert clock() >= 8.0
    assert robot.mode_events == ["enter", "exit"]


def test_home_pauses_on_feedback_lag_and_resumes_without_accumulating_lead():
    clock = Clock()
    robot = HomingRobot(clock, follow_after=8.0)
    goal = [0.15, 0.0, 0.0, 0.0, 0.0, 0.0]
    client.move_to_start(robot, None, goal, timeout=30.0, clock=clock, sleep=clock.sleep)
    frozen = [(t, measured, command) for t, measured, command in robot.history if t < 8.0]
    assert max(client.pose_error(measured, command)[0] for _, measured, command in frozen) <= 0.06 + 1e-9
    repeats = sum(
        client.pose_error(previous[2], current[2])[0] < 1e-12 for previous, current in itertools.pairwise(frozen)
    )
    assert repeats > 20
    np.testing.assert_allclose(robot.measured[0], goal[:3], atol=1e-9)


def test_force_guard_stops_homing_and_restores_normal_mode():
    clock = Clock()
    robot = HomingRobot(clock, fault_at=0.5)
    with pytest.raises(client.RobotError, match="raw channel 5=10.878"):
        client.move_to_start(robot, None, [0.1, 0, 0, 0, 0, 0], timeout=20.0, clock=clock, sleep=clock.sleep)
    assert clock() < 0.6
    assert robot.mode_events == ["enter", "exit"]


def test_real_streamer_restores_normal_lead_after_home_mode_exception():
    limits = client.Limits(workspace_radius=0.45)
    robot = client.Ar3Robot(execute=False, limits=limits)
    robot._streaming = True
    robot._state = np.zeros(client.STATE_SIZE)
    robot._state_at = robot._target_at = 1.0
    robot._origin = np.zeros(3)
    robot._last_command = client.pose_from_vector([0.0] * 6)
    robot._target = client.pose_from_vector([0, 0, 0, 0, 0, 0.5])

    def interrupted_home():
        with robot.return_home_mode():
            for _ in range(150):
                command = robot.next_command(1.0, 0.01)
            assert client.pose_error(robot._last_command, client.pose_from_vector([0.0] * 6))[1] == pytest.approx(0.1)
            np.testing.assert_array_equal(command[6:12], np.zeros(6))
            raise client.RobotError("simulated home stop")

    with pytest.raises(client.RobotError, match="simulated home stop"):
        interrupted_home()
    for _ in range(150):
        robot.next_command(1.0, 0.01)
    assert client.pose_error(robot._last_command, client.pose_from_vector([0.0] * 6))[1] == pytest.approx(0.05)
    assert limits.max_tracking_error_rad == 0.15
    assert limits.max_lead_rad == 0.05


def test_slow_gripper_open_keeps_the_robot_heartbeat_alive():
    clock = Clock()
    robot = HomingRobot(clock)
    gripper = HomingGripper(clock, open_after=4.0)
    client.move_to_start(robot, gripper, [0.001, 0, 0, 0, 0, 0], timeout=15.0, clock=clock, sleep=clock.sleep)
    assert clock() >= 4.0
    assert gripper.commands
    assert all(value == 1000.0 for value in gripper.commands)
    assert max(np.diff(robot.heartbeats)) < robot.limits.target_timeout


def test_gripper_that_never_opens_fails_homing_instead_of_warning_and_continuing():
    clock = Clock()
    robot = HomingRobot(clock)
    gripper = HomingGripper(clock, open_after=float("inf"))
    with pytest.raises((client.RobotError, client.GripperError)):
        client.move_to_start(robot, gripper, [0.001, 0, 0, 0, 0, 0], timeout=15.0, clock=clock, sleep=clock.sleep)
    assert max(np.diff(robot.heartbeats)) < robot.limits.target_timeout
    assert not robot.in_home_mode


@pytest.fixture
def session_setup(monkeypatch):
    events = []
    args = client.parse_args(["--execute", "--move-to-start", "--start-pose", *["0"] * 6])
    session = client.Session(args)
    session.robot = types.SimpleNamespace(
        start=lambda **_kwargs: events.append("start"),
        wait_ready=lambda *_args, **_kwargs: events.append("ready"),
        activate=lambda **_kwargs: events.append("activate"),
        poses=lambda: (client.pose_from_vector([0.0] * 6), None, None),
    )
    gripper = types.SimpleNamespace(read=lambda: 1000.0, configure=lambda: events.append("configure_gripper"))
    monkeypatch.setattr(client.DMGripper, "connect", lambda *_args, **_kwargs: gripper)
    policy = types.SimpleNamespace(
        get_server_metadata=lambda: {"reactive": True, "wrench_dim": 13, "action_horizon": 10}
    )
    monkeypatch.setattr(client, "connect_policy", lambda *_args: policy)
    monkeypatch.setattr(client, "ChunkPlanner", lambda *_args, **_kwargs: types.SimpleNamespace(horizon=10, latency=0))
    monkeypatch.setattr(session, "_configure_limits", lambda: events.append("limits"))
    monkeypatch.setattr(session, "_reanchor", lambda: events.append("reanchor"))
    monkeypatch.setattr(client, "warm_up", lambda *_args: events.append("warmup"))
    monkeypatch.setattr(client, "move_to_start", lambda *_args, **_kwargs: events.append("home_succeeded"))
    monkeypatch.setattr(client, "run_chunks", lambda *_args: events.append("chunks") or 1)
    return session, events


def test_session_executes_policy_actions_only_after_home_success(session_setup):
    session, events = session_setup
    assert session.run() == 0
    relevant = [event for event in events if event in {"warmup", "activate", "home_succeeded", "reanchor", "chunks"}]
    assert relevant == ["warmup", "activate", "home_succeeded", "reanchor", "chunks"]


@pytest.mark.parametrize("failure_stage", ["warmup", "home"])
def test_session_failure_never_enters_policy_execution(session_setup, monkeypatch, failure_stage):
    session, events = session_setup

    def fail(*_args, **_kwargs):
        raise client.RobotError(f"{failure_stage} failed")

    monkeypatch.setattr(client, "warm_up" if failure_stage == "warmup" else "move_to_start", fail)
    with pytest.raises(client.RobotError, match=f"{failure_stage} failed"):
        session.run()
    assert "chunks" not in events
    assert "reanchor" not in events
    if failure_stage == "warmup":
        assert "activate" not in events
        assert "configure_gripper" not in events


def test_impossible_home_timeout_is_rejected_before_enabling_motion(session_setup):
    session, events = session_setup
    session.args.move_timeout = 2.0
    with pytest.raises(client.RobotError, match="timeout"):
        session.run()
    assert "warmup" in events
    assert not {"activate", "configure_gripper", "home_succeeded", "reanchor", "chunks"}.intersection(events)


@pytest.fixture
def saved_home(tmp_path):
    path = tmp_path / "config" / "cartesian_home.json"
    path.parent.mkdir()
    pose = [0.4, -0.3, 0.8, 0.2, -0.1, 0.3]
    path.write_text(
        json.dumps(
            {"schema_version": 1, "frame_id": client.BASE_FRAME, "source": "measured_tcp_pose", "tcp_pose": pose}
        )
    )
    return path, pose


def test_default_home_comes_from_the_configured_arm_workspace(saved_home, monkeypatch):
    path, pose = saved_home
    monkeypatch.setenv("INFER_AR3_WS", str(path.parent.parent))
    args = client.parse_args(["--execute", "--move-to-start"])
    assert args.home_file == path
    assert args.start_pose == pose


def test_explicit_home_file_overrides_workspace(saved_home, tmp_path, monkeypatch):
    path, pose = saved_home
    monkeypatch.setenv("INFER_AR3_WS", str(tmp_path / "missing-workspace"))
    args = client.parse_args(["--home-file", str(path)])
    assert args.home_file == path
    assert args.start_pose == pose


def test_explicit_start_pose_does_not_require_a_home_file(tmp_path, monkeypatch):
    monkeypatch.setenv("INFER_AR3_WS", str(tmp_path / "missing-workspace"))
    args = client.parse_args(["--start-pose", *["0"] * 6])
    assert args.home_file is None
    assert args.start_pose == [0.0] * 6


def test_home_file_and_start_pose_cannot_be_combined(saved_home):
    path, _pose = saved_home
    with pytest.raises(SystemExit) as error:
        client.parse_args(["--home-file", str(path), "--start-pose", *["0"] * 6])
    assert error.value.code == 2


def test_invalid_home_file_stops_before_a_session_can_start(saved_home, capsys):
    path, _pose = saved_home
    data = json.loads(path.read_text())
    data["frame_id"] = "different_robot_base"
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit) as error:
        client.parse_args(["--execute", "--move-to-start", "--home-file", str(path)])
    assert error.value.code == 2
    assert "home frame differs" in capsys.readouterr().err


def test_vscode_execute_configuration_passes_the_requested_saved_home_to_the_client():
    launch = json.loads((Path(__file__).resolve().parents[1] / ".vscode" / "launch.json").read_text())
    configuration = next(config for config in launch["configurations"] if "--execute" in config["args"])
    args = configuration["args"]
    client_args = args[args.index("--") + 1 :]
    assert "--move-to-start" in client_args
    assert client_args[client_args.index("--home-file") + 1] == (
        "${workspaceFolder}/../ar3_CartesianImpedanceControl/config/cartesian_home.json"
    )
