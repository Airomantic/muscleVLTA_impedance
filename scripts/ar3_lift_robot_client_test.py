import array
import dataclasses
import io
import json
import logging
import math
import threading
import time
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.data import tactile_cache
from openpi.models import model as _model
from openpi.policies import ar3_lift_policy as ar3
from scripts import ar3_lift_recorder
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


def _tactile_messages(value: float):
    """Tac G2 bridge messages of one finger: a native 32FC1 map filled with ``value`` and a wrench with Fz = value."""
    raw = np.full(client.tactile_io().RAW_DEPTH_SHAPE, value, dtype="<f4")
    depth = types.SimpleNamespace(
        encoding="32FC1", height=raw.shape[0], width=raw.shape[1], step=4 * raw.shape[1], is_bigendian=False
    )
    depth.data = raw.tobytes()
    force, torque = types.SimpleNamespace(x=1.0, y=2.0, z=value), types.SimpleNamespace(x=4.0, y=5.0, z=6.0)
    return depth, types.SimpleNamespace(wrench=types.SimpleNamespace(force=force, torque=torque))


def test_tactile_messages_decode_like_the_recorder():
    pixels = np.arange(6, dtype=np.float32).reshape(2, 3)
    padded = np.zeros((2, 4), dtype=">f4")
    padded[:, :3] = pixels
    message = types.SimpleNamespace(
        encoding="32FC1", height=2, width=3, step=16, is_bigendian=True, data=list(padded.tobytes())
    )
    np.testing.assert_array_equal(client.depth_map(message), pixels)
    for change in ({"encoding": "16UC1"}, {"step": 8}, {"data": padded.tobytes()[:-4]}):
        with pytest.raises(client.RobotError):
            client.depth_map(types.SimpleNamespace(**{**vars(message), **change}))
    message.data = np.full((2, 4), np.nan, dtype=">f4").tobytes()
    with pytest.raises(client.RobotError, match="non-finite"):
        client.depth_map(message)
    np.testing.assert_array_equal(client.finger_wrench(_tactile_messages(3.0)[1]), [1, 2, 3, 4, 5, 6])


def test_hand_reading_sends_stale_or_malformed_fingers_as_missing(monkeypatch, caplog):
    tactile = client.tactile_io()
    robot = client.Ar3Robot(execute=False, limits=client.Limits())
    clock = FakeClock()
    clock.now = 10.0
    monkeypatch.setattr(client.time, "monotonic", clock)
    for index, sensor in enumerate(tactile.SENSORS):
        depth, wrench = _tactile_messages(0.1 * (index + 1))
        robot._tactile.update({("depth", sensor): (depth, 9.95), ("wrench", sensor): (wrench, 9.9)})
    depth, valid, wrench = robot.hand_reading()
    assert depth.dtype == np.float16
    assert depth.shape == (len(tactile.SENSORS), *tactile.DEPTH_SHAPE)
    np.testing.assert_allclose(depth.max(axis=(1, 2)), [0.1, 0.2], rtol=1e-3)
    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(wrench, [1, 2, 0.1, 4, 5, 6, 1, 2, 0.2, 4, 5, 6], rtol=1e-6)

    # The recorder reads the same fingers with their receipt times, also while the operator has taken over.
    robot._takeover_at = 9.0
    with pytest.raises(client.HumanTakeover):
        robot.hand_reading()
    snapshot = robot.hand_snapshot()
    for recorded, sent in zip(snapshot[:3], (depth, valid, wrench), strict=True):
        np.testing.assert_array_equal(recorded, sent)
    assert snapshot[3].tolist() == [9.9, 9.9]
    robot._takeover_at = None

    robot._tactile["wrench", "sensor_2"] = (robot._tactile["wrench", "sensor_2"][0], 9.7)  # older than 0.2 s
    robot._tactile["depth", "sensor_0"][0].encoding = "16UC1"
    with caplog.at_level(logging.WARNING, logger=client.logger.name):
        depth, valid, wrench = robot.hand_reading()
    assert valid.tolist() == [False, False]
    assert not depth.any()
    assert not wrench.any()
    assert "sensor_0: Tac G2 depth must be" in caplog.text
    assert "sensor_2: samples 0.30 s old" in caplog.text


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
        # Node names of the publishers on {prefix}/command; other topics: topic_nodes.
        self.publishers = ["ar3_lift_robot_client"] * publishers
        self.nodes = ["cartesian_impedance_node", "ar3_lift_robot_client"]
        self.topic_nodes = {}

    def get_publishers_info_by_topic(self, topic):
        names = self.publishers if topic == "/ar3/cartesian_impedance/command" else self.topic_nodes.get(topic, [])
        return [types.SimpleNamespace(node_name=name) for name in names]

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


def _robot(*, execute=True, dagger=False, record=False, **limits):
    robot = client.Ar3Robot(
        execute=execute,
        limits=client.Limits(**{"workspace_radius": 0.45, **limits}),
        dagger=dagger,
        record=record,
    )
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
    if dagger:
        robot._ros["Bool"] = lambda data: types.SimpleNamespace(data=data)
        robot._ros["human_control"] = FakePublisher()
    return robot, measured


@pytest.mark.parametrize("stale", ["state", "tau_m", "front", "wrist"])
def test_wait_ready_waits_for_new_samples_at_requested_age_limits(monkeypatch, stale):
    robot = client.Ar3Robot(execute=False, limits=client.Limits())
    robot._status, robot._metadata = "monitoring:", CONTRACT
    robot._state_at = robot._tau_m_at = 9.9
    robot._images = {"front": (None, 9.9), "wrist": (None, 9.9)}
    clock = FakeClock()
    clock.now = 10.0
    sleeps = []

    def set_sample(timestamp):
        if stale in ("state", "tau_m"):
            setattr(robot, f"_{stale}_at", timestamp)
        else:
            robot._images[stale] = (None, timestamp)

    def fresh_sample(seconds):
        sleeps.append(seconds)
        clock.sleep(seconds)
        set_sample(clock())

    set_sample(9.7)
    monkeypatch.setattr(client.time, "monotonic", clock)
    monkeypatch.setattr(client.time, "sleep", fresh_sample)
    robot.wait_ready(1.0, need_tau_m=True)  # Backward-compatible initial discovery accepts ages below 1 s.
    assert not sleeps
    robot.wait_ready(1.0, need_tau_m=True, max_observation_age=0.25, max_image_age=0.2)
    assert sleeps == [0.05]
    assert not robot._armed
    assert not robot._streaming


def test_wait_ready_times_out_when_fresh_camera_frames_never_arrive(monkeypatch):
    robot = client.Ar3Robot(execute=False, limits=client.Limits())
    robot._status, robot._metadata = "monitoring:", CONTRACT
    robot._state_at = robot._tau_m_at = 10.0
    robot._images = {"front": (None, 9.7), "wrist": (None, 10.0)}
    clock = FakeClock()
    clock.now = 10.0
    monkeypatch.setattr(client.time, "monotonic", clock)
    monkeypatch.setattr(client.time, "sleep", clock.sleep)
    with pytest.raises(client.RobotError, match="/front/color/image_raw"):
        robot.wait_ready(0.1, need_tau_m=True, max_observation_age=0.25, max_image_age=0.2)
    assert clock.now < 10.2
    assert not robot._armed
    assert not robot._streaming


def test_wait_ready_requires_fresh_decodable_fingers_for_tactile_policies(monkeypatch):
    robot = client.Ar3Robot(execute=False, limits=client.Limits())
    robot._status, robot._metadata = "monitoring:", CONTRACT
    robot._state_at = robot._tau_m_at = 10.0
    robot._images = {"front": (None, 10.0), "wrist": (None, 10.0)}
    clock = FakeClock()
    clock.now = 10.0
    monkeypatch.setattr(client.time, "monotonic", clock)
    monkeypatch.setattr(client.time, "sleep", clock.sleep)
    with pytest.raises(client.RobotError, match="tactile=True"):
        robot.wait_ready(0.1, need_tau_m=True, need_tactile=True)
    sensors = client.tactile_io().SENSORS
    robot._tactile_topics = {
        (kind, sensor): f"{client.TACTILE_PREFIX}/{sensor}/{kind}" for sensor in sensors for kind in ("depth", "wrench")
    }
    with pytest.raises(client.RobotError, match="sensor_0/depth.*ar3_tactile_bridge"):
        robot.wait_ready(0.1, need_tau_m=True, need_tactile=True)
    for sensor in sensors:
        depth, wrench = _tactile_messages(0.1)
        robot._tactile.update({("depth", sensor): (depth, clock.now), ("wrench", sensor): (wrench, clock.now)})
    robot.wait_ready(0.1, need_tau_m=True, need_tactile=True, max_tactile_age=0.2)
    robot._tactile["depth", "sensor_2"][0].height = 5  # the payload no longer matches the image size
    with pytest.raises(client.RobotError, match=r"unusable \(sensor_2: Tac G2 depth has an invalid row step"):
        robot.wait_ready(0.1, need_tau_m=True, need_tactile=True, max_tactile_age=0.2)
    robot.wait_ready(0.1, need_tau_m=True)  # LIFT policies do not read the fingers


@pytest.mark.parametrize(("check", "tactile"), [(False, False), (True, False), (False, True)])
def test_session_refreshes_observations_after_setup_before_report_or_warmup(monkeypatch, check, tactile):
    session = client.Session(client.parse_args([*(["--check"] if check else []), "--start-pose", *["0"] * 6]))
    events = []
    waits = []
    starts = []

    def wait_ready(timeout, **kwargs):
        waits.append(kwargs)
        events.append("wait")

    session.robot = types.SimpleNamespace(start=lambda **kwargs: starts.append(kwargs), wait_ready=wait_ready)
    monkeypatch.setattr(session, "_configure_limits", lambda: events.append("parameters"))
    gripper = types.SimpleNamespace(read=lambda: events.append("gripper") or 1000.0)
    monkeypatch.setattr(client.DMGripper, "connect", lambda *args, **kwargs: gripper)
    metadata = {"reactive": True, "wrench_dim": 13, "action_horizon": 10}
    if tactile:
        metadata.update(sensor_history=10, tactile_in_use=True)
    policy = types.SimpleNamespace(get_server_metadata=lambda: metadata)
    monkeypatch.setattr(client, "connect_policy", lambda *args: policy)
    monkeypatch.setattr(client, "ChunkPlanner", lambda *args, **kwargs: types.SimpleNamespace(horizon=10, latency=0))

    def observe_after_setup(*args):
        assert events == ["wait", "parameters", "gripper", "wait"]
        # --check subscribes to the fingers to report them; only tactile policies require them.
        assert [start["tactile"] for start in starts] == [tactile or check]
        assert waits == [
            {"need_tau_m": True, "need_tactile": tactile},
            {
                "need_tau_m": True,
                "need_tactile": tactile,
                "max_observation_age": 0.25,
                "max_image_age": 0.2,
                "max_tactile_age": client.MAX_TACTILE_AGE,
            },
        ]
        if args:  # warm_up(robot, planner, gripper, config)
            assert (args[3].sensor_history, args[3].tactile) == ((10, True) if tactile else (0, False))
        if not check:
            raise RuntimeError("stop before activation")

    monkeypatch.setattr(session, "_report", observe_after_setup)
    monkeypatch.setattr(client, "warm_up", observe_after_setup)
    if check:
        assert session.run() == 0
    else:
        with pytest.raises(RuntimeError, match="stop before activation"):
            session.run()


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
    robot._ros["node"].publishers.append("another_client")
    robot._status = "monitoring: RT 1ms state"
    with pytest.raises(client.RobotError, match="1 other publisher"):
        robot.activate(service_timeout=0.1)

    robot, _ = _robot(execute=False)
    robot._ros["node"].nodes.append("ar3_cartesian_impedance_teleop")
    robot.heartbeat()

    # DAgger: the AR3 teleoperation (only) may run next to the client.
    robot, _ = _robot(dagger=True)
    robot._ros["node"].nodes.append(client.TELEOP_NODE)
    robot._ros["node"].publishers.append(client.TELEOP_NODE)
    assert robot.command_sources() == (0, [])
    robot._ros["node"].nodes.append("rokae_driver7")
    assert robot.command_sources() == (0, ["rokae_driver7"])


def test_dagger_clutch_hands_the_session_to_the_teleoperation_without_stopping_it():
    robot, _ = _robot(dagger=True, record=True)
    robot._tick()
    own = robot._ros["command"].messages[-1]
    robot._on_clutch(types.SimpleNamespace(data=True))
    assert (robot._streaming, robot._armed) == (False, False)
    assert robot._ros["stop"].calls == 0
    assert robot._ros["human_control"].messages[-1]
    with pytest.raises(client.HumanTakeover):
        robot.heartbeat()
    robot._tick()
    assert len(robot._ros["command"].messages) == 1
    # Only the teleoperation's commands count, not echoes of this client's own.
    robot._on_command(types.SimpleNamespace(data=own.tolist()))
    robot._on_command(types.SimpleNamespace(data=[0.5] * client.STATE_SIZE))
    pressed_at, released_at, commands = robot.takeover()
    assert pressed_at is not None
    assert (released_at, commands) == (None, 1)

    # The release ends the takeover; the teleoperation's stop that follows is not a fault of this client.
    robot._on_clutch(types.SimpleNamespace(data=False))
    robot._on_status(types.SimpleNamespace(data="stopped: controller idle confirmed"))
    assert (robot._fault, robot._ros["stop"].calls) == (None, 0)
    assert robot.takeover()[1] is not None
    assert not robot._ros["human_control"].messages[-1]
    robot.end_takeover()
    assert robot.takeover() == (None, None, 0)
    assert [sample.value for sample in robot._streams["control"]._samples] == [client.HUMAN_FLAG, client.POLICY_FLAG]

    # A clutch held before the episode starts keeps the policy from enabling the controller.
    robot._clutch = True
    with pytest.raises(client.HumanTakeover):
        robot.activate(service_timeout=0.1)


def test_dagger_needs_the_live_teleoperation_and_a_gated_gripper_teleoperation():
    robot, _ = _robot(dagger=True)
    with pytest.raises(client.RobotError, match="no AR3 teleoperation status"):
        robot.check_teleoperation(0.0)
    status = {"dry_run": True, "clutch_pressed": False}
    robot._on_teleop_status(types.SimpleNamespace(data=json.dumps(status)))
    with pytest.raises(client.RobotError, match="dry run"):
        robot.check_teleoperation(0.0)
    robot._on_teleop_status(types.SimpleNamespace(data=json.dumps({**status, "dry_run": False})))
    node = robot._ros["node"]
    node.nodes.append("v3_gripper_teleop")  # V3_Gripper's teleop_gripper.py, even without its telemetry
    with pytest.raises(client.RobotError, match="ungated"):
        robot.check_teleoperation(0.0)
    node.nodes[-1] = client.GRIPPER_GATE_NODE
    node.topic_nodes[client.GRIPPER_ACTION_TOPIC] = [client.GRIPPER_GATE_NODE, "v3_gripper_recording_telemetry"]
    with pytest.raises(client.RobotError, match="v3_gripper_recording_telemetry"):
        robot.check_teleoperation(0.0)
    node.topic_nodes[client.GRIPPER_ACTION_TOPIC] = [client.GRIPPER_GATE_NODE]
    robot.check_teleoperation(0.0)


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
    session = client.Session(client.parse_args(["--start-pose", *["0"] * 6]))
    session.robot.controller_parameters = lambda names, timeout: {name: parameters.get(name) for name in names}
    session._configure_limits()
    assert session.limits.max_linear_speed == pytest.approx(0.03)
    assert session.limits.max_angular_speed == pytest.approx(0.1)
    assert session.limits.workspace_radius == 0.45
    parameters["command_timeout_s"] = 0.015
    with pytest.raises(client.RobotError, match="command_timeout_s"):
        session._configure_limits()


def test_ros_shutdown_joins_spin_before_destroying_resources():
    robot = client.Ar3Robot(execute=False, limits=client.Limits())
    wake = threading.Event()
    events = []

    def spin():
        wake.wait()
        events.append("spin exited")

    thread = threading.Thread(target=spin)

    def stop_executor(*, timeout_sec):
        events.append("shutdown")
        wake.set()
        return True

    def destroy_node():
        assert not thread.is_alive()
        events.append("destroy node")

    robot._ros = {
        "executor": types.SimpleNamespace(shutdown=stop_executor),
        "thread": thread,
        "node": types.SimpleNamespace(destroy_node=destroy_node),
        "rclpy": types.SimpleNamespace(try_shutdown=lambda: events.append("shutdown context")),
    }
    thread.start()
    try:
        robot.shutdown()
        robot.shutdown()  # Repeated cleanup must not touch destroyed ROS resources.
        assert events == ["shutdown", "spin exited", "destroy node", "shutdown context"]
        assert robot._ros == {}
    finally:
        wake.set()
        thread.join(timeout=1.0)


def test_ros_shutdown_timeout_preserves_live_resources_and_reports_failure(monkeypatch):
    session = client.Session(client.parse_args(["--start-pose", *["0"] * 6]))
    robot = session.robot
    release = threading.Event()
    destroyed = []
    thread = threading.Thread(target=release.wait)
    robot._ros = {
        "executor": types.SimpleNamespace(shutdown=lambda **kwargs: True),
        "thread": thread,
        "node": types.SimpleNamespace(destroy_node=lambda: destroyed.append("node")),
        "rclpy": types.SimpleNamespace(try_shutdown=lambda: destroyed.append("context")),
    }
    shutdown = robot.shutdown
    monkeypatch.setattr(robot, "shutdown", lambda: shutdown(timeout=0.01))
    monkeypatch.setattr(client.signal, "signal", lambda *args: None)
    thread.start()
    try:
        assert not session.close()
        assert not destroyed
        assert robot._ros["thread"] is thread
    finally:
        release.set()
        thread.join(timeout=1.0)
        shutdown()
    assert destroyed == ["node", "context"]


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


def test_run_chunks_asks_before_every_chunk_whether_to_stop():
    robot, policy, clock = FakeRobot(_readings()), FakePolicy(), FakeClock()
    planner = client.ChunkPlanner(policy, reactive=True, horizon=HORIZON, latency=3)
    asked = []

    def should_stop():
        asked.append(clock.now)
        return len(asked) > 1

    config = client.RunConfig(max_chunks=0, resize=0)
    trace = client.Trace(None)
    assert client.run_chunks(
        robot, planner, None, config, trace, clock=clock, sleep=clock.sleep, should_stop=should_stop
    )
    planner.close()
    assert len(robot.targets) == HORIZON
    assert asked == pytest.approx([0.0, HORIZON * config.period])


@dataclasses.dataclass
class TactileRobot(FakeRobot):
    """Every measurement (chunk start or step) is tagged with its count m: wrench, finger maps / wrenches = m; the
    second finger is missing at odd m."""

    measured: int = 0

    def observe(self, *, wrench_dim, max_age, max_image_age):
        observation = super().observe(wrench_dim=wrench_dim, max_age=max_age, max_image_age=max_image_age)
        return dataclasses.replace(observation, reading=self.force_reading(wrench_dim=wrench_dim, max_age=max_age))

    def force_reading(self, *, wrench_dim, max_age):
        self.measured += 1
        return np.full(wrench_dim, self.measured, dtype=np.float32)

    def hand_reading(self, *, max_age):
        depth = np.full((2, 48, 64), self.measured / 100, dtype=np.float16)
        return depth, np.asarray([True, self.measured % 2 == 0]), np.full(12, self.measured, dtype=np.float32)


class SensorPolicy:
    """Reactive server fake that feeds every request, merged over the chunk-start observation like
    ``ReactiveSession``, through the server's input transform of a muscleVLTA policy."""

    def __init__(self, history_length: int):
        self.history_length = history_length
        self.inputs = ar3.AR3LiftInputs(
            model_type=_model.ModelType.PI05, action_horizon=HORIZON, history_length=history_length, tactile=True
        )
        self.requests = []
        self.observation = None

    def _respond(self, data, start, end):
        self.requests.append((start, data))
        inputs = self.inputs(data)
        assert inputs["tactile_depth"].shape == (self.history_length + HORIZON, 2, 48, 64)
        actions = np.zeros((end - start, 7))
        actions[:, 6] = 500.0
        return {"actions": actions, "start_step": start, "next_step": end}

    def reactive_replan(self, observation, *, k):
        self.observation = observation
        return self._respond(observation, 0, k)

    def reactive_step(self, readings, *, k, current_step):
        return self._respond({**self.observation, **readings}, current_step, min(current_step + k, HORIZON))


def test_run_chunks_sends_muscle_vlta_readings_with_the_episode_history():
    history = 10
    robot, policy, clock = TactileRobot([np.zeros(13, dtype=np.float32)]), SensorPolicy(history), FakeClock()
    planner = client.ChunkPlanner(policy, reactive=True, horizon=HORIZON, latency=3)
    apertures = iter(range(1000, 0, -1))  # measurement m reads the opening 1001 - m
    gripper = types.SimpleNamespace(read=lambda: float(next(apertures)))
    config = client.RunConfig(max_chunks=2, resize=0, sensor_history=history, tactile=True)
    for _ in range(2):  # every call is an episode
        assert client.run_chunks(robot, planner, gripper, config, client.Trace(None), clock=clock, sleep=clock.sleep)
    planner.close()

    chunks = []
    for start, data in policy.requests:
        if start == 0:
            chunks.append([])
        chunks[-1].append(data)
    assert len(chunks) == 4
    for index, requests in enumerate(chunks):
        base, episode_start = HORIZON * index, index % 2 == 0
        # The steps before the episode start repeat its first reading as missing; later chunks carry the readings of
        # the previous chunk.
        before = [base + 1] * history if episode_start else list(range(base - history + 1, base + 1))
        for data in requests:
            rows = len(data["wrench"])
            assert history < rows <= history + HORIZON
            assert all(len(data[key]) == rows for key in ar3.READING_KEYS)
            measured = np.asarray(before + list(range(base + 1, base + 1 + rows - history)), dtype=np.float32)
            np.testing.assert_array_equal(data["wrench"][:, 0], measured)
            np.testing.assert_array_equal(data["tactile_wrench"][:, 0], measured)
            np.testing.assert_allclose(data["tactile_depth"][:, 0, 0, 0], measured / 100, rtol=1e-3)
            np.testing.assert_array_equal(data["tactile_valid"][:, 1], measured % 2 == 0)
            np.testing.assert_array_equal(data["gripper_aperture"], 1001 - measured)
            np.testing.assert_array_equal(
                data["sensor_valid"], np.logical_or(not episode_start, np.arange(rows) >= history)
            )
        assert requests[0]["observation/state"][6] == 1000 - base
        assert len(requests[-1]["wrench"]) >= history + HORIZON - planner.latency  # last action saw its readings


def test_mask_tactile_sends_both_fingers_as_missing_and_keeps_the_other_readings():
    robot = TactileRobot([np.zeros(13, dtype=np.float32)])
    robot.measured = 2  # both fingers deliver samples
    force = np.arange(13, dtype=np.float32)
    config = client.RunConfig(sensor_history=10, tactile=True)
    seen = client.sensor_reading(robot, force, 640.0, config)
    assert seen["tactile_valid"].all()
    assert seen["tactile_depth"].any()

    masked = client.sensor_reading(robot, force, 640.0, dataclasses.replace(config, mask_tactile=True))
    assert not masked["tactile_valid"].any()
    assert not masked["tactile_depth"].any()
    assert not masked["tactile_wrench"].any()
    assert masked["tactile_depth"].dtype == seen["tactile_depth"].dtype
    np.testing.assert_array_equal(masked["wrench"], force)
    assert masked["gripper_aperture"] == seen["gripper_aperture"] == 640.0


def test_gripper_resends_targets_after_someone_else_moved_the_fingers():
    gripper, _, log = _gripper()
    gripper.command(600)
    gripper.command(600)
    assert log == [("SetTargetPosition", 600)]
    assert gripper.target == 600.0
    gripper.forget_target()
    assert gripper.target is None
    gripper.command(600)
    assert log == [("SetTargetPosition", 600)] * 2


# Recording.
IMAGE = types.SimpleNamespace(encoding="rgb8", height=2, width=2, step=6, data=bytes(12))
ACCEPTED = np.asarray(client.DATASET_START_POSE) + 0.01


def _samples(t, **overrides):
    """``Ar3Robot.recording_samples`` + the gripper poll of a policy-controlled moment with fresh inputs."""
    state = np.concatenate([client.DATASET_START_POSE, np.arange(6) + 1.0, np.zeros(18)])
    samples = {
        "state": client.Sample(t - 0.002, state),
        "tau_m": client.Sample(t - 0.001, np.arange(7.0)),
        "accepted": client.Sample(t - 0.003, ACCEPTED),
        "commanded": client.Sample(t - 0.004, ACCEPTED + 0.01),
        "gripper": client.Sample(t - 0.02, (900.0, 0.0), 5.0),
        "front": client.Sample(t + 0.01, IMAGE, 7.0, 3),
        "wrist": client.Sample(t - 0.02, IMAGE, 8.0, 4),
        "control": None,
        "teleop": None,
        "human_gripper": None,
    }
    return {**samples, **overrides}


def test_frames_are_human_only_while_the_operator_commands_with_fresh_inputs():
    t = 100.0
    values, problems = client.frame_values(_samples(t), t)
    assert problems == []
    assert values["control_flag"] == [client.POLICY_FLAG]
    np.testing.assert_allclose(values["actions"], [*client.pose_rotvec(ACCEPTED), 0.0])  # the policy's gripper target
    np.testing.assert_allclose(values["observation.tcp_wrench"], np.arange(6) + 1.0)
    assert values["sync.front.timing_s"] == pytest.approx([t + 0.01, 7.0, 0.01])
    assert values["sync.wrist.frame_index"] == [4]

    human = {
        "control": client.Sample(t - 5.0, client.HUMAN_FLAG),
        "teleop": client.Sample(t - 0.05, None),
        "human_gripper": client.Sample(t - 0.05, 300.0, 9.0),
    }
    values, problems = client.frame_values(_samples(t, **human), t)
    assert (values["control_flag"], values["actions"][6], problems) == ([client.HUMAN_FLAG], 300.0, [])
    # No teleoperation command lately (e.g. the handover) or a stale input: not trained on.
    values, _ = client.frame_values(_samples(t, **{**human, "teleop": client.Sample(t - 0.5, None)}), t)
    assert values["control_flag"] == [client.POLICY_FLAG]
    values, problems = client.frame_values(_samples(t, **human, accepted=client.Sample(t - 1.0, ACCEPTED)), t)
    assert (values["control_flag"], problems) == ([client.POLICY_FLAG], ["action"])
    np.testing.assert_allclose(values["actions"][:6], client.pose_rotvec(ACCEPTED + 0.01))
    values, problems = client.frame_values(_samples(t, **human, front=client.Sample(t - 0.3, IMAGE)), t)
    assert (values["control_flag"], problems) == ([client.POLICY_FLAG], ["front"])


def test_human_chunks_count_the_training_samples_of_intervention_only():
    # Rows whose next 3 frames (clamped to the episode end) are all teleoperated: 1, 5 and 6.
    flags = [client.POLICY_FLAG] + [client.HUMAN_FLAG] * 3 + [client.POLICY_FLAG] + [client.HUMAN_FLAG] * 2
    assert client.human_chunks(flags, 3) == 3
    assert client.human_chunks([client.POLICY_FLAG] * 4, 3) == 0


class FakeWriter:
    root = "/data/rollouts"

    def __init__(self):
        self.frames, self.tactile, self.events = [], [], []

    def start_episode(self):
        self.events.append("start")

    def add_frame(self, values, images, tactile=None):
        assert images["front"].shape == (2, 2, 3)
        self.frames.append(values)
        self.tactile.append(tactile)

    def save_episode(self):
        self.events.append("save")
        return 4

    def label_episode(self, index, outcome):
        self.events.append(("label", index, outcome))

    def discard_episode(self):
        self.events.append("discard")

    def close(self):
        self.events.append("close")


def test_rollout_recorder_records_10_hz_frames_up_to_the_end_time():
    writer = FakeWriter()
    robot = types.SimpleNamespace(recording_samples=lambda t: {k: v for k, v in _samples(t).items() if k != "gripper"})
    gripper = types.SimpleNamespace(read=lambda: 900.0, target=None)
    recorder = client.RolloutRecorder(robot, writer, gripper, horizon=2)
    recorder.start()
    until = time.monotonic() + 0.25
    summary = recorder.stop(until)
    recorder.close()
    assert summary == {"frames": 3, "human": 0, "chunks": 0, "stale": {}}
    times = [values["sync.target_host_s"][0] for values in writer.frames]
    np.testing.assert_allclose(np.diff(times), 0.1)
    assert times[-1] <= until
    assert [values["actions"][6] for values in writer.frames] == [900.0] * 3  # no policy target yet: hold
    assert writer.tactile == [None] * 3
    assert writer.events == ["start", "discard", "close"]


def test_recorded_frames_carry_the_fingers_the_policy_is_sent():
    tactile = client.tactile_io()
    maps = np.stack([np.full(tactile.DEPTH_SHAPE, value, dtype=np.float16) for value in (0.1, 0.2)])
    wrench = np.arange(12, dtype=np.float32)
    t = 50.0
    # Polled 10 ms before the frame; the messages of sensor_2 arrived more than the age limit before it.
    hand = client.Sample(t - 0.01, (maps, np.array([True, True]), wrench, np.array([t - 0.03, t - 0.25])))
    values, (depth, valid) = client.hand_frame(hand, t, 0.2)
    assert valid.tolist() == [True, False]
    assert depth.dtype == np.float32
    np.testing.assert_array_equal(depth[0], maps[0])
    assert not depth[1].any()
    np.testing.assert_array_equal(values["observation.gripper_wrench.sensor_0"], wrench[:6])
    assert not np.any(values["observation.gripper_wrench.sensor_2"])
    assert values["sync.gripper_wrench.sensor_0.timing_s"] == [t - 0.03, 0.0]
    # Before the first poll, or for a finger that never published, the frame has no reading.
    silent = client.Sample(t - 0.01, (maps * 0, np.array([False, False]), wrench * 0, np.array([-np.inf, -np.inf])))
    for missing in (None, silent):
        values, (depth, valid) = client.hand_frame(missing, t, 0.2)
        assert not valid.any()
        assert not depth.any()
        assert values["sync.gripper_wrench.sensor_2.timing_s"] == [0.0, 0.0]


def test_rollout_recorder_adds_the_fingers_of_tactile_policies_and_labels_the_outcome():
    tactile = client.tactile_io()
    writer = FakeWriter()
    maps = np.full((2, *tactile.DEPTH_SHAPE), 0.1, dtype=np.float16)

    def hand_snapshot(*, max_age):
        assert max_age == 0.3
        received = np.array([time.monotonic() - 0.01, -np.inf])  # sensor_2 never published
        return maps, np.array([True, False]), np.arange(12, dtype=np.float32), received

    robot = types.SimpleNamespace(
        recording_samples=lambda t: {k: v for k, v in _samples(t).items() if k != "gripper"},
        hand_snapshot=hand_snapshot,
    )
    gripper = types.SimpleNamespace(read=lambda: 900.0, target=None)
    recorder = client.RolloutRecorder(robot, writer, gripper, horizon=2, tactile=True, max_tactile_age=0.3)
    time.sleep(0.1)  # as on the robot, the fingers are polled before the first episode starts
    recorder.start()
    summary = recorder.stop(time.monotonic() + 0.25)
    assert summary == {"frames": 3, "human": 0, "chunks": 0, "stale": {"tactile": 3}}
    for values, (depth, valid) in zip(writer.frames, writer.tactile, strict=True):
        assert valid.tolist() == [True, False]
        np.testing.assert_array_equal(depth[0], maps[0])
        np.testing.assert_array_equal(values["observation.gripper_wrench.sensor_0"], np.arange(6))
        received = values["sync.gripper_wrench.sensor_0.timing_s"][0]
        assert 0 < values["sync.target_host_s"][0] - received <= 0.3
    assert recorder.save("failure") == 4
    assert recorder.save() == 4
    recorder.close()
    assert writer.events == ["start", "save", ("label", 4, "failure"), "save", "discard", "close"]


def test_a_slow_gripper_request_does_not_make_the_fingers_look_missing():
    tactile = client.tactile_io()
    writer = FakeWriter()
    maps = np.full((2, *tactile.DEPTH_SHAPE), 0.1, dtype=np.float16)

    def hand_snapshot(*, max_age):
        return maps, np.array([True, True]), np.arange(12, dtype=np.float32), np.full(2, time.monotonic() - 0.01)

    def slow_read():
        time.sleep(0.25)
        return 900.0

    robot = types.SimpleNamespace(
        recording_samples=lambda t: {k: v for k, v in _samples(t).items() if k != "gripper"},
        hand_snapshot=hand_snapshot,
    )
    gripper = types.SimpleNamespace(read=slow_read, target=None)
    recorder = client.RolloutRecorder(robot, writer, gripper, horizon=2, tactile=True, max_tactile_age=0.15)
    time.sleep(0.35)  # the first gripper position has arrived
    recorder.start()
    summary = recorder.stop(time.monotonic() + 0.45)
    recorder.close()
    assert summary["frames"] == 5
    assert "tactile" not in summary["stale"]
    assert all(valid.all() for _, valid in writer.tactile)


def test_tactile_rollouts_are_written_as_a_dataset_the_tactile_cache_reads(tmp_path):
    tactile = client.tactile_io()
    image = types.SimpleNamespace(encoding="rgb8", height=48, width=64, step=192, data=bytes(48 * 64 * 3))
    maps = np.random.default_rng(0).uniform(0.0, 0.3, size=(2, *tactile.DEPTH_SHAPE)).astype(np.float16)

    def recording_samples(t):
        cameras = {"front": client.Sample(t + 0.01, image, 7.0, 3), "wrist": client.Sample(t - 0.02, image, 8.0, 4)}
        return {k: v for k, v in _samples(t, **cameras).items() if k != "gripper"}

    def hand_snapshot(*, max_age):
        received = np.array([time.monotonic() - 0.01, -np.inf])
        return maps, np.array([True, False]), np.arange(12, dtype=np.float32), received

    robot = types.SimpleNamespace(recording_samples=recording_samples, hand_snapshot=hand_snapshot)
    gripper = types.SimpleNamespace(read=lambda: 900.0, target=None)
    root = tmp_path / "rollouts"
    writer = ar3_lift_recorder.EpisodeRecorder(root, task="plug in", image_size=(48, 64), tactile=True)
    recorder = client.RolloutRecorder(robot, writer, gripper, horizon=2, tactile=True)
    time.sleep(0.1)
    recorder.start()
    assert recorder.stop(time.monotonic() + 0.25)["frames"] == 3
    assert recorder.save("success") == 0
    recorder.close()

    depth, valid = tactile_cache.TactileCache(tactile_cache.build_cache(root, workers=1)).frames(0, np.arange(3))
    assert valid.tolist() == [[True, False]] * 3
    np.testing.assert_array_equal(depth[:, 0], np.broadcast_to(maps[0], (3, *tactile.DEPTH_SHAPE)))
    assert not depth[:, 1].any()
    assert json.loads((root / "rl" / "outcomes.json").read_text())["episodes"] == {"0": "success"}


def test_the_operator_can_label_saved_episodes_for_offline_rl():
    session = client.Session(client.parse_args(["--record-dir", "unused", "--start-pose", *["0"] * 6]))
    saved, discarded = [], []
    session.recorder = types.SimpleNamespace(
        save=lambda outcome=None: saved.append(outcome) or len(saved) - 1,
        discard=lambda: discarded.append(True),
        writer=FakeWriter(),
    )
    answers = iter(["y", "n", "", "?", "d", "q"])
    session.console = types.SimpleNamespace(ask=lambda prompt, default: next(answers))
    summary = {"frames": 5, "human": 0, "chunks": 0, "stale": {}}
    assert [session._keep_episode(summary) for _ in range(5)] == [True, True, True, True, False]
    assert saved == ["success", "failure", None, None]
    assert discarded == [True]


def test_takeover_episodes_record_until_the_clutch_release_and_leave_the_stop_to_the_teleoperation(monkeypatch):
    session = client.Session(
        client.parse_args(["--execute", "--dagger", "--record-dir", "unused", "--start-pose", *["0"] * 6])
    )
    calls = []
    session.robot = types.SimpleNamespace(
        takeover=lambda: (1.0, 2.5, 7),
        status="stopped: RT motion ended",
        end_takeover=lambda: calls.append("end takeover"),
        stop=lambda: calls.append("stop arm"),
    )
    session.recorder = types.SimpleNamespace(
        start=lambda: calls.append("record"),
        stop=lambda until: calls.append(("end recording", until)) or {"frames": 3},
        abort=lambda: calls.append("abort"),
        failed=False,
    )
    session.console = client.Console(io.StringIO())

    def taken_over(*args, should_stop, **kwargs):
        assert not should_stop()
        raise client.HumanTakeover

    monkeypatch.setattr(client, "run_chunks", taken_over)
    assert session._run_episode(client.RunConfig()) == {"frames": 3}
    assert calls == ["record", ("end recording", 2.5), "end takeover"]

    calls.clear()
    monkeypatch.setattr(client, "run_chunks", lambda *args, should_stop, **kwargs: 4)
    session._run_episode(client.RunConfig())
    assert [call if isinstance(call, str) else call[0] for call in calls] == ["record", "stop arm", "end recording"]


def test_recording_and_dagger_arguments():
    home = ["--start-pose", *["0"] * 6]
    assert client.parse_args(home).episodes == 1
    assert client.parse_args([*home, "--execute", "--dagger", "--record-dir", "d"]).episodes == 0
    assert client.parse_args([*home, "--record-dir", "d", "--episodes", "3"]).episodes == 3
    assert client.parse_args([*home, "--check", "--dagger"]).dagger
    for argv in (
        ["--dagger", "--record-dir", "d"],
        ["--execute", "--dagger"],
        ["--record-dir", "d", "--no-gripper"],
        ["--episodes", "-1"],
    ):
        with pytest.raises(SystemExit):
            client.parse_args([*home, *argv])
