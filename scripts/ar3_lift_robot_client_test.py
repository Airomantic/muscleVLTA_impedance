import array
import dataclasses
import json
import math
import threading
import time
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.policies import ar3_lift_policy as ar3
from scripts import ar3_lift_robot_client as client

HORIZON = 10


def _rotations(rng: np.random.Generator) -> list[np.ndarray]:
    random = list(Rotation.random(20, random_state=1).as_matrix())
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    special = [Rotation.from_rotvec(axis * angle).as_matrix() for angle in (0.0, 1e-9, 1e-4, 2.5, np.pi - 1e-6)]
    return random + special


def test_rotation_conversions_match_scipy():
    rng = np.random.default_rng(0)
    for matrix in _rotations(rng):
        expected = Rotation.from_matrix(matrix).as_rotvec()
        np.testing.assert_allclose(client.matrix_rotvec(matrix), expected, atol=1e-9)
        np.testing.assert_allclose(client.rotvec_matrix(expected), matrix, atol=1e-9)
        np.testing.assert_allclose(client.rpy_matrix(client.matrix_rpy(matrix)), matrix, atol=1e-9)


def test_state_and_targets_match_training_transforms():
    rng = np.random.default_rng(1)
    for _ in range(20):
        pose6 = np.concatenate([rng.uniform(-0.5, 0.5, 3), rng.uniform(-np.pi, np.pi, 3) * [1, 0.45, 1]])
        state = client.policy_state(pose6, 640.0)
        expected = np.concatenate([pose6[:3], ar3.rpy_to_rotvec(pose6[3:]), [640.0]]).astype(np.float32)
        np.testing.assert_allclose(state, expected, atol=1e-6)

        action = np.concatenate([rng.normal(0, 0.03, 3), rng.normal(0, 0.1, 3), [300.0]])
        position, rotation = client.absolute_target(action, state)
        reference = ar3.relative_to_absolute_actions(action[None], state)[0]
        np.testing.assert_allclose(position, reference[:3], atol=1e-9)
        np.testing.assert_allclose(rotation, Rotation.from_rotvec(reference[3:6]).as_matrix(), atol=1e-9)


def test_rgb_image_drops_row_padding_and_converts_bgr():
    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    padded = np.zeros((2, 12), dtype=np.uint8)
    padded[:, :9] = pixels.reshape(2, 9)
    message = types.SimpleNamespace(encoding="rgb8", height=2, width=3, step=12, data=padded.tobytes())
    np.testing.assert_array_equal(client.rgb_image(message), pixels)
    message.encoding = "bgr8"
    np.testing.assert_array_equal(client.rgb_image(message), pixels[..., ::-1])
    message.encoding = "mono8"
    with pytest.raises(client.RobotError):
        client.rgb_image(message)


# Policy server fake with the reactive semantics of ReactiveSession: action r only sees force rows <= r - latency,
# missing rows repeat the newest reading.
class FakePolicy:
    def __init__(self, latency: int = 3, gate: threading.Event | None = None, *, fail: bool = False):
        self.latency = latency
        self.gate = gate
        self.fail = fail
        self.calls = []
        self.observation = None

    def actions(self, wrench: np.ndarray, tag: float = 0.0) -> np.ndarray:
        padded = ar3.pad_force_sequence(wrench, HORIZON)
        rows = []
        for r in range(HORIZON):
            seen = padded[: max(r - self.latency, 0) + 1, 0].sum()
            rows.append([0.001 * r, 1e-4 * seen, tag, 0.0, 0.0, 0.001 * r, 500.0 + r])
        return np.asarray(rows)

    def reactive_replan(self, observation, *, k):
        self.calls.append(("replan", k, None, len(observation["wrench"])))
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError("server down")
        self.observation = observation
        actions = self.actions(observation["wrench"], observation.get("tag", 0.0))
        return {"actions": actions[:k], "start_step": 0, "next_step": k, "policy_timing": {"infer_ms": 1.0}}

    def reactive_step(self, readings, *, k, current_step):
        self.calls.append(("step", k, current_step, len(readings["wrench"])))
        actions = self.actions(readings["wrench"], self.observation.get("tag", 0.0))
        end = min(current_step + k, HORIZON)
        return {"actions": actions[current_step:end], "start_step": current_step, "next_step": end}

    def infer(self, observation):
        self.calls.append(("infer", None, None, len(observation["wrench"])))
        return {"actions": self.actions(observation["wrench"])}


def _readings(count: int = HORIZON) -> list[np.ndarray]:
    return [np.full(13, 2.0**i, dtype=np.float32) for i in range(count)]


def test_planner_requests_each_action_once_its_readings_exist():
    policy = FakePolicy()
    planner = client.ChunkPlanner(policy, reactive=True, horizon=HORIZON, latency=3)
    readings = _readings()
    planner.begin_chunk({"tag": 1.0}, readings[0])
    got = [planner.wait_action(0, 2)]
    for i in range(1, HORIZON):
        planner.add_reading(readings[i])
        planner.wait_action(min(i + 3, HORIZON - 1), 2)  # deterministic request pattern
        got.append(planner.wait_action(i, 2))
    planner.close()
    np.testing.assert_array_equal(np.stack(got), policy.actions(np.stack(readings), 1.0))
    assert policy.calls == [("replan", 4, None, 1)] + [("step", 1, s, s - 2) for s in range(4, HORIZON)]


def test_planner_batches_and_stays_exact_when_readings_arrive_quickly():
    policy = FakePolicy(latency=0)
    planner = client.ChunkPlanner(policy, reactive=True, horizon=HORIZON, latency=0)
    readings = _readings()
    planner.begin_chunk({}, readings[0])
    for reading in readings[1:]:
        planner.add_reading(reading)
    got = [planner.wait_action(i, 2) for i in range(HORIZON)]
    planner.close()
    np.testing.assert_array_equal(np.stack(got), policy.actions(np.stack(readings)))
    assert sum(k for _, k, _, _ in policy.calls) == HORIZON


def test_planner_discards_stale_chunk_and_reports_errors():
    gate = threading.Event()
    policy = FakePolicy(gate=gate)
    planner = client.ChunkPlanner(policy, reactive=True, horizon=HORIZON, latency=3)
    planner.begin_chunk({"tag": 1.0}, _readings(1)[0])
    deadline = time.monotonic() + 2
    while not policy.calls and time.monotonic() < deadline:
        time.sleep(0.001)
    planner.begin_chunk({"tag": 2.0}, _readings(1)[0])  # while the first replan is in flight
    gate.set()
    assert planner.wait_action(0, 2)[2] == 2.0
    with pytest.raises(client.PolicyError, match="not ready"):
        planner.wait_action(HORIZON - 1, 0.05)
    planner.close()

    failing = client.ChunkPlanner(FakePolicy(fail=True), reactive=True, horizon=HORIZON, latency=3)
    failing.begin_chunk({}, _readings(1)[0])
    with pytest.raises(client.PolicyError, match="server down"):
        failing.wait_action(0, 2)


def test_planner_infer_mode_makes_one_call_per_chunk():
    policy = FakePolicy()
    planner = client.ChunkPlanner(policy, reactive=False, horizon=HORIZON, latency=0)
    reading = _readings(1)[0]
    planner.begin_chunk({}, reading)
    got = [planner.wait_action(i, 2) for i in range(HORIZON)]
    planner.close()
    np.testing.assert_array_equal(np.stack(got), policy.actions(reading[None]))
    assert policy.calls == [("infer", None, None, 1)]


# Gripper fakes (gRPC multicallables / futures).
class FakeFuture:
    def __init__(self, result=None, error=None, *, done=True):
        self._result, self._error, self._done = result, error, done

    def done(self):
        return self._done

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result


class RpcError(Exception):
    pass


class FakeOperation:
    def __init__(self, name, log):
        self.name, self.log = name, log
        self.next_future = None

    def __call__(self, request, timeout):
        self.log.append((self.name, request))
        return types.SimpleNamespace(success=True, message="")

    def future(self, request, timeout):
        self.log.append((self.name, request))
        future, self.next_future = self.next_future, None
        return future or FakeFuture(types.SimpleNamespace(success=True, message=""))


def _gripper_state(**overrides):
    state = {
        "valid": True,
        "init_state": 1,
        "grip_state": 0,
        "position": 1000,
        "error": 0,
        "update_age_ms": 10.0,
        "stale": False,
        "connected": True,
    }
    state.update(overrides)
    return types.SimpleNamespace(**state)


def _gripper(state=None):
    log = []
    operations = {
        name: FakeOperation(name, log)
        for name in ("Enable", "SetTargetSpeed", "SetTargetForce", "SetTargetPosition", "Stop")
    }
    get_state = types.SimpleNamespace(future=lambda request, timeout: FakeFuture(state or _gripper_state()))
    gripper = client.DMGripper(
        get_state=get_state, operations=operations, encode_value=lambda value: value, rpc_error=RpcError
    )
    return gripper, operations, log


def test_gripper_feedback_validation():
    assert client.gripper_position(_gripper_state(position=420), max_age=0.5) == 420.0
    for bad in ({"init_state": 0}, {"stale": True}, {"error": 3}, {"update_age_ms": 900.0}, {"position": 1200}):
        with pytest.raises(client.GripperError):
            client.gripper_position(_gripper_state(**bad), max_age=0.5)


def test_gripper_commands_are_non_blocking_with_deadband():
    gripper, operations, log = _gripper()
    gripper.stop()
    assert log == []  # never touched: no Stop
    pending = FakeFuture(types.SimpleNamespace(success=True, message=""), done=False)
    operations["SetTargetPosition"].next_future = pending
    gripper.command(1000.4)
    gripper.command(500)  # previous write still in flight: skipped
    pending._done = True
    gripper.command(999)  # within the deadband
    gripper.command(-5)
    assert log == [("SetTargetPosition", 1000), ("SetTargetPosition", 0)]

    operations["SetTargetPosition"].next_future = FakeFuture(error=RpcError("unavailable"))
    gripper.command(600)
    with pytest.raises(client.GripperError, match="unavailable"):
        gripper.command(100)
    with pytest.raises(client.GripperError):
        gripper.command(float("nan"))
    gripper.stop()
    assert log[-1] == ("Stop", b"")


# Streamer / controller session with injected ROS objects.
class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(np.asarray(message.data))


class FakeStopClient:
    def __init__(self, robot=None, replies=("stopped: RT motion ended",)):
        self.robot, self.replies, self.calls = robot, list(replies), 0
        self.srv_name = "/ar3/cartesian_impedance/stop"

    def service_is_ready(self):
        return True

    def wait_for_service(self, timeout_sec):
        return True

    def call_async(self, request):
        self.calls += 1
        if self.robot is not None:  # the controller reports leaving the RT session while stopping
            self.robot._on_status(types.SimpleNamespace(data="monitoring: RT 1ms state"))
        message = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return FakeFuture(types.SimpleNamespace(success=True, message=message))


class FakeNode:
    def __init__(self, publishers=1):
        self.publishers, self.nodes = publishers, ["cartesian_impedance_node", "ar3_lift_robot_client"]

    def count_publishers(self, topic):
        assert topic == "/ar3/cartesian_impedance/command"
        return self.publishers

    def get_node_names_and_namespaces(self):
        return [(name, "/") for name in self.nodes]


class FakeServiceClient:
    def __init__(self, replies, *, ready=True):
        self.replies, self.ready, self.calls = list(replies), ready, 0
        self.srv_name = "/ar3/cartesian_impedance/recenter_workspace"

    def service_is_ready(self):
        return self.ready

    def wait_for_service(self, timeout_sec):
        return self.ready

    def call_async(self, request):
        self.calls += 1
        success, message = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return FakeFuture(types.SimpleNamespace(success=success, message=message))


CONTRACT = {**client.RT_CONTRACT, "pose": client.RPY_CONVENTION, "workspace_recenter": client.WORKSPACE_RECENTER}


def _robot(*, execute=True, **limits):
    robot = client.Ar3Robot(execute=execute, limits=client.Limits(**{"workspace_radius": 0.45, **limits}))
    state = np.concatenate([client.DATASET_START_POSE, np.arange(6) + 1.0, np.arange(18) + 100.0])
    measured = client.pose_from_vector(state[:6])
    robot._state, robot._state_at = state, time.monotonic()
    robot._target, robot._target_at = measured, time.monotonic()
    robot._last_command, robot._origin = measured, measured[0].copy()
    robot._streaming, robot._armed, robot._sdk_active = True, execute, execute
    robot._ros = {
        "message": lambda data: types.SimpleNamespace(data=data),
        "preview": FakePublisher(),
        "command": FakePublisher(),
        "stop": FakeStopClient(),
        "Trigger": types.SimpleNamespace(Request=lambda: None),
        "node": FakeNode(publishers=int(execute)),
    }
    return robot, measured


def test_streamer_applies_teleoperation_lead_slew_and_session_parameters():
    robot, measured = _robot()
    target = (measured[0] + [0.2, 0.0, 0.0], Rotation.from_rotvec([0.5, 0, 0]).as_matrix() @ measured[1])
    robot.set_target(*target)
    previous = measured
    for _ in range(3000):
        values = robot.next_command(time.monotonic(), 0.01)
        command = robot._last_command
        step = client.pose_error(command, previous)
        assert step[0] <= 0.04 * 0.01 + 1e-12
        assert step[1] <= 0.15 * 0.01 + 1e-9
        previous = command
    lead = client.pose_error(command, measured)
    np.testing.assert_allclose(lead, (0.06, 0.05), atol=1e-6)
    np.testing.assert_array_equal(values[6:12], 0.0)
    np.testing.assert_array_equal(values[12:], robot._state[12:])
    np.testing.assert_allclose(client.rpy_matrix(values[3:6]), command[1], atol=1e-12)


def test_streamer_rejects_stale_inputs_and_tracking_errors():
    robot, measured = _robot()
    with pytest.raises(ValueError, match="state is stale"):
        robot.next_command(time.monotonic() + 1.0, 0.01)
    now = time.monotonic()
    robot._state_at, robot._target_at = now, now - 1.1
    with pytest.raises(ValueError, match="target is stale"):
        robot.next_command(now, 0.01)
    robot._state_at = robot._target_at = time.monotonic()
    robot._last_command = (measured[0] + [0.0, 0.0, 0.09], measured[1])
    with pytest.raises(ValueError, match="tracking limits"):
        robot.next_command(time.monotonic(), 0.01)


def test_targets_are_clamped_into_the_session_workspace():
    robot, measured = _robot()
    far = measured[0] + [0.6, 0.0, 0.0]
    assert robot.set_target(far, measured[1])
    np.testing.assert_allclose(robot._target[0], measured[0] + [0.43, 0.0, 0.0])
    with pytest.raises(client.RobotError, match="outside"):
        robot.set_target(far + 0.1, measured[1], clamp=False)
    np.testing.assert_allclose(robot._target[0], measured[0] + [0.43, 0.0, 0.0])


def test_tick_publishes_preview_only_in_dry_run_and_halts_once_on_violations():
    robot, _ = _robot(execute=False)
    robot._tick()
    assert len(robot._ros["preview"].messages) == 1
    assert not robot._ros["command"].messages

    robot, measured = _robot(execute=True)
    robot._tick()
    assert len(robot._ros["command"].messages) == 1
    robot._state_at -= 1.0
    robot._tick()
    robot._on_status(types.SimpleNamespace(data="fault: lost connection"))
    assert "stale" in robot._fault
    assert robot._ros["stop"].calls == 1
    with pytest.raises(client.RobotError, match="stale"):
        robot.heartbeat()


def test_controller_status_halts_foreign_stops_but_not_own_stop():
    robot, _ = _robot()
    robot._on_status(types.SimpleNamespace(data="stopped: controller idle confirmed"))
    assert robot._fault is not None
    assert robot._ros["stop"].calls == 1

    robot, _ = _robot()
    robot._ros["stop"] = FakeStopClient(robot, replies=("stopping: waiting", "stopped: RT motion ended"))
    robot.stop()
    assert robot._fault is None
    assert not robot._armed
    assert robot._ros["stop"].calls == 2


def test_wait_ready_requires_the_controller_command_contract():
    robot, _ = _robot()
    now = time.monotonic()
    robot._status, robot._images = "active: RT session 1", {"front": (None, now), "wrist": (None, now)}
    with pytest.raises(client.RobotError, match="metadata"):
        robot.wait_ready(0.0, need_tau_m=False)
    robot._on_metadata(types.SimpleNamespace(data=json.dumps(CONTRACT)))
    robot.wait_ready(0.0, need_tau_m=False)
    for data in (json.dumps({**CONTRACT, "pose": "TCP xyz[m],rpy[rad];Rx*Ry*Rz"}), "not json"):
        robot._on_metadata(types.SimpleNamespace(data=data))
        with pytest.raises(client.RobotError, match="contract"):
            robot.wait_ready(0.0, need_tau_m=False)


def test_other_command_sources_block_activation_and_halt_the_stream():
    robot, _ = _robot()
    robot.heartbeat()
    robot._ros["node"].nodes.append("ar3_cartesian_impedance_teleop")
    robot._ownership_checked_at = -math.inf
    with pytest.raises(client.RobotError, match="ar3_cartesian_impedance_teleop"):
        robot.heartbeat()
    assert not robot._streaming
    assert robot._ros["stop"].calls == 1

    robot, _ = _robot()
    robot._ros["node"].publishers = 2
    robot._status = "monitoring: RT 1ms state"
    with pytest.raises(client.RobotError, match="1 other publisher"):
        robot.activate(service_timeout=0.1)

    robot, _ = _robot(execute=False)
    robot._ros["node"].nodes.append("ar3_cartesian_impedance_teleop")
    robot.heartbeat()


def test_workspace_recenter_waits_for_the_rt_hold_and_moves_the_origin():
    robot, measured = _robot()
    robot._metadata = dict(CONTRACT)
    origin = [*(measured[0] + [0.001, 0.0, 0.0]), 0.1, 0.2, 0.3]
    recentered = json.dumps({"state": "recentered", "session": 2, "origin": origin})
    robot._ros["recenter"] = FakeServiceClient([(True, '{"state":"pending"}'), (True, recentered)])
    assert robot.recenter_workspace(timeout=1.0, service_timeout=0.5)
    assert robot._ros["recenter"].calls == 2
    np.testing.assert_allclose(robot._origin, origin[:3])

    robot._ros["recenter"] = FakeServiceClient([(True, '{"state":"pending"}')])
    assert not robot.recenter_workspace(timeout=0.0, service_timeout=0.5)
    robot._ros["recenter"] = FakeServiceClient([(False, "workspace recenter requires fresh active RT hold")])
    with pytest.raises(client.RobotError, match="fresh active RT hold"):
        robot.recenter_workspace(timeout=1.0, service_timeout=0.5)

    robot._metadata = {**CONTRACT, "workspace_recenter": None}
    robot._ros["recenter"] = FakeServiceClient([(True, recentered)])
    assert not robot.recenter_workspace(timeout=1.0, service_timeout=0.5)
    assert robot._ros["recenter"].calls == 0


def test_controller_parameters_bound_the_command_shaping():
    velocity = types.SimpleNamespace(type=8, double_array_value=array.array("d", [0.04] * 3 + [0.1] * 3))
    parameters = {
        "enable_motion": True,
        "workspace_radius_m": 0.45,
        "speed_mm_s": 30.0,
        "max_velocity": client.parameter_value(velocity),
        "command_timeout_s": 0.5,
    }
    session = client.Session(client.parse_args([]))
    session.robot.controller_parameters = lambda names, timeout: {name: parameters.get(name) for name in names}
    session._configure_limits()
    assert session.limits.max_linear_speed == pytest.approx(0.03)
    assert session.limits.max_angular_speed == pytest.approx(0.1)
    assert session.limits.workspace_radius == 0.45
    parameters["command_timeout_s"] = 0.015
    with pytest.raises(client.RobotError, match="command_timeout_s"):
        session._configure_limits()


@dataclasses.dataclass
class FakeRobot:
    readings: list
    pose: np.ndarray = dataclasses.field(default_factory=lambda: np.asarray(client.DATASET_START_POSE))
    targets: list = dataclasses.field(default_factory=list)
    reading_index: int = 0

    def observe(self, *, wrench_dim, max_age, max_image_age):
        self.reading_index = 1
        image = np.zeros((480, 640, 3), np.uint8)
        return client.Observation(pose=self.pose, reading=self.readings[0], front=image, wrist=image)

    def force_reading(self, *, wrench_dim, max_age):
        self.reading_index += 1
        return self.readings[self.reading_index - 1]

    def set_target(self, position, rotation, *, clamp=True):
        self.targets.append((position, rotation))
        return False

    def heartbeat(self):
        pass


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_run_chunks_executes_exact_reactive_actions_at_control_rate():
    readings = _readings()
    robot, policy, clock = FakeRobot(readings), FakePolicy(), FakeClock()
    planner = client.ChunkPlanner(policy, reactive=True, horizon=HORIZON, latency=3)
    gripper = types.SimpleNamespace(read=lambda: 1000.0, commands=[])
    gripper.command = gripper.commands.append
    config = client.RunConfig(max_chunks=2, resize=0, command_gripper=True)
    assert client.run_chunks(robot, planner, gripper, config, client.Trace(None), clock=clock, sleep=clock.sleep) == 2
    planner.close()

    expected = policy.actions(np.stack(readings))
    state = client.policy_state(robot.pose, 1000.0)
    assert len(robot.targets) == 2 * HORIZON
    assert gripper.commands == list(expected[:, 6]) * 2
    for (position, rotation), action in zip(robot.targets, np.concatenate([expected, expected]), strict=True):
        reference = client.absolute_target(action, state)
        np.testing.assert_allclose(position, reference[0])
        np.testing.assert_allclose(rotation, reference[1])
    assert clock.now == pytest.approx(2 * HORIZON * config.period)
    assert policy.observation["observation/state"][6] == 1000.0
