#!/usr/bin/env python3
"""Runs an AR3 LIFT policy on the real robot: Rokae AR3 Cartesian impedance arm + DM tac G2/V3 gripper.

Robot side of ``scripts/serve_ar3_lift.sh``. Run it on the robot PC with the ROS 2 Jazzy Python that has the DM gripper
client (``scripts/run_ar3_lift_robot.sh`` sets this up) while these run (``scripts/prepare_ar3_inference.sh``):

* ``ar3_cartesian_impedance`` ``cartesian_impedance_node`` (github.com/Airomantic/ar3_CartesianImpedanceControl) with
  ``enable_motion:=true`` and a ``workspace_radius_m`` covering the task,
* the DM V3 gripper gRPC server (github.com/Airomantic/dmrobotics_Flux_pkg_V1.2.13.1), connected and initialized,
* the front and wrist Orbbec cameras (rgb8 ``/front/color/image_raw`` and ``/wrist/color/image_raw``, 640x480),
* the policy server ``scripts/serve_ar3_lift.sh`` (on this PC or a GPU server).

Each action chunk follows the recording / training layout (10 Hz, action horizon 10, LIFT keys):

1. Chunk start: front + wrist images, ``observation/state`` = measured TCP xyz + rotation vector + gripper position
   (0 = closed ... 1000 = open) and the first force reading.
2. LIFT Algorithm 1: ``reactive_replan`` encodes the observation once; one force reading per executed step (the
   ``tauExt_inBase`` TCP wrench + the 7 joint torques ``tau_m``) is appended and ``reactive_step`` re-plans the chunk
   with all readings so far. Action ``r`` only attends to readings ``<= r - cross_attention_latency``, so a background
   thread requests every action as soon as all its readings exist: the actions equal those of LIFT's synchronous loop
   while the inference time hides behind the latency.
3. Relative actions (xyz + xyz Euler of ``inv(T_state) @ T_target``) become absolute TCP targets that are streamed at
   100 Hz to the impedance node with the command shaping of the AR3 teleoperation that recorded the data (lead
   60 mm / 0.05 rad, 40 mm/s, 0.15 rad/s, tracking stop 80 mm / 0.15 rad, session stiffness, zero desired wrench).
   The gripper target goes through gRPC (speed 80, force level 20 of 20..100: the recording teleoperation's).

muscleVLTA policies (server metadata ``sensor_history`` > 0 and / or ``tactile_in_use``; trained with
``scripts/train_ar3_muscle_vlta.sh``) read more per step, as recorded: the Tac G2 hand stream (both fingers' indentation
maps, area-pooled with ``openpi_client.tactile``, and 6D wrenches from ``/v3_gripper/tactile/sensor_{0,2}/{depth,wrench}``
of the lerobot ``scripts/ar3_tactile_bridge.py``; a finger without samples of the last 0.2 s is sent as missing) and the
gripper opening. Every request starts with the readings of the ``sensor_history`` control steps before the chunk
(``SensorHistory``; marked missing before the episode start).

Safety: without ``--execute`` it is a dry run (targets only on ``preview_command``; no enable, no gripper writes). It
never takes over a controller session it does not own, refuses to run next to another arm command source (the
teleoperation, except with ``--dagger``), and stops on stale sensors, tracking-limit violations, controller faults,
policy timeouts, gripper errors and Ctrl+C, waiting for the controller to confirm ``stopped``.

Recording and DAgger (``--record-dir DIR``, ``--dagger``): episodes (move to the start pose, policy rollout, operator
prompt to save or discard) are appended to a LeRobot v3.0 dataset with the features of the lerobot AR3 recordings
(``scripts/ar3_lift_recorder.py``: front / wrist videos, TCP pose, gripper position, TCP wrench, ``tau_m``, the
accepted TCP target + gripper target as ``actions``, ``control_flag``; for muscleVLTA policies with the Tac G2 fingers
also the finger wrenches and the pooled indentation map of every frame). Saving an episode as a success or a failure
(``y`` / ``n`` at the prompt) labels it for offline RL (``scripts/train_ar3_muscle_vlta_rl.sh``). With ``--dagger``
the AR3 teleoperation runs next to the client for LIFT's human-in-the-loop corrections: pressing its clutch takes over
at once (the client stops commanding; the teleoperation continues the same controller session from the measured TCP,
and the gated gripper teleoperation ``scripts/ar3_dagger_gripper_teleop.sh`` gets the gripper once its trigger matches
the fingers). Releasing the clutch ends the episode (the teleoperation stops the arm). Frames commanded by the operator
get ``control_flag`` -1, policy frames +1; ``scripts/train_ar3_lift_dagger_gpu012.sh`` trains on the human chunks 1:1
with the offline demonstrations.
"""

import argparse
import collections
from collections.abc import Sequence
import contextlib
import dataclasses
import functools
import json
import logging
import math
import os
import pathlib
import queue
import signal
import sys
import threading
import time
from typing import Any, NamedTuple

import numpy as np

if __package__:
    from . import ar3_lift_home as home
else:
    import ar3_lift_home as home

_CLIENT_SRC = pathlib.Path(__file__).resolve().parents[1] / "packages" / "openpi-client" / "src"
if _CLIENT_SRC.is_dir() and str(_CLIENT_SRC) not in sys.path:
    sys.path.append(str(_CLIENT_SRC))

logger = logging.getLogger("ar3_lift_robot")

DEFAULT_PROMPT = (
    "Unplug the white charging plug from the white socket and place it on the table, "
    "don't bump your fingers into other objects"
)
# Mean TCP pose of the first frame of the 33 recorded episodes: xyz [m] + SDK roll/pitch/yaw [rad] (R = Rz @ Ry @ Rx),
# gripper fully open. Spread across episodes <= 13.5 mm / 0.033 rad. The recorded motion reaches 0.43 m from it.
DATASET_START_POSE = (0.1650, -0.3853, 0.1348, -2.0216, 0.4905, -3.0251)  # Historical reference, not the homing target.
BASE_FRAME = "AR3-3_07R-W4C5C5_base"
TASK_REACH_M = 0.43
GRIPPER_OPEN = 1000.0
STATE_SIZE = 30  # controller state: pose6 (xyz + rpy) + wrench6 + K6 + D6 + M6
TAU_M_SIZE = 7
IDLE_STATUS = ("monitoring:", "stopped:")
# Command contract of the impedance node (its /metadata), as checked by the recording project's Cartesian replay.
RT_CONTRACT = {
    "schema": "pose6,wrench6,K6,D6,M6",
    "motion_backend": "rt",
    "control_mode": "cartesianImpedance",
    "session_fixed_K_wrench": True,
}
RPY_CONVENTION = "TCP xyz[m],rpy[rad];Rz*Ry*Rx"
WORKSPACE_RECENTER = "active_hold_once_v1"
TELEOP_NODE = "ar3_cartesian_impedance_teleop"
# Other arm command sources: the impedance teleoperation and the legacy joint driver.
CONFLICTING_NODES = frozenset({TELEOP_NODE, "rokae_driver7"})
GRIPPER_RPC_PREFIX = "/gripper.GripperService/"
GRIPPER_PREPARATION = (
    "Prepare the gripper first: ./infer_oneclick.sh --ready-only "
    "(add --initialize-gripper once after power-up; it moves the fingers)."
)
# Tac G2 fingers of muscleVLTA policies (lerobot scripts/ar3_tactile_bridge.py): <prefix>/<sensor>/{depth,wrench}.
TACTILE_PREFIX = "/v3_gripper/tactile"
# Freshness limit of the recordings (meta/ar3_capture.json max_tactile_age_s) and of the training tactile cache.
MAX_TACTILE_AGE = 0.2
TACTILE_PREPARATION = (
    "run ./infer_oneclick.sh --tactile (Tac G2 sensor server and lerobot scripts/ar3_tactile_bridge.py; "
    "VS Code: the muscleVLTA configurations pass --tactile)"
)
# Recording / DAgger.
RECORD_FPS = 10  # frame rate of the training data
HUMAN_FLAG, POLICY_FLAG = -1, 1  # control_flag of the recordings: teleoperated / policy-controlled frame
# Latched Bool: tells the gated gripper teleoperation (scripts/ar3_dagger_gripper_teleop.py) when the operator has
# taken over.
HUMAN_CONTROL_TOPIC = "/ar3_lift/human_control"
GRIPPER_ACTION_TOPIC = "/v3_gripper/action"  # gripper teleoperation target (JointState position, 0..1 = SDK / 1000)
GRIPPER_GATE_NODE = "ar3_dagger_gripper_gate"
# V3_Gripper's teleop_gripper.py (lerobot scripts/ar3_gripper_teleop.sh): it always follows the trigger.
UNGATED_GRIPPER_NODES = frozenset({"v3_gripper_teleop"})
# Like the lerobot recorder, a frame describes host time T = now - SYNC_DELAY, so all inputs around T have arrived.
SYNC_DELAY = 0.1
# Inputs older than this (camera frames: further from T) give the frame the policy flag (it is not trained on).
STALE_LIMITS = {"state": 0.1, "tau_m": 0.1, "action": 0.2, "gripper": 0.3, "camera": 0.1}
TELEOP_ACTIVITY = 0.1  # a teleoperated frame needs a teleoperation command within this time before T
# Sample history per input (state / tau_m / accepted_command up to 1 kHz, commands 100 Hz, cameras 30 Hz).
STREAM_SIZES = {
    "state": 4096,
    "tau_m": 4096,
    "accepted": 4096,
    "commanded": 512,
    "front": 16,
    "wrist": 16,
    "control": 64,
    "teleop": 512,
    "human_gripper": 128,
}
DAGGER_HELP = (
    "DAgger: each episode moves the arm to the start pose and runs the policy. When it goes wrong, press the "
    "teleoperation clutch: you control the arm at once (the gripper follows once its trigger matches the fingers). "
    "Finish the task and release the clutch to end the episode. Enter = end the episode without a takeover, Ctrl+C = "
    "abort. Do not press the clutch between episodes."
)


class RobotError(RuntimeError):
    """The arm, its controller or its sensors cannot (or can no longer) be used safely."""


class GripperError(RuntimeError):
    """Gripper feedback or a gripper command failed."""


class PolicyError(RuntimeError):
    """The policy server failed or was too slow."""


class HumanTakeover(Exception):  # noqa: N818
    """DAgger: the operator pressed the teleoperation clutch, so the policy no longer commands the robot."""


# Pose math. Poses are (xyz [m], 3x3 rotation matrix); the controller uses SDK roll/pitch/yaw, R = Rz @ Ry @ Rx.
# rpy_matrix, matrix_rpy, limit_pose and pose_error are copied from ar3_CartesianImpedanceControl
# (scripts/cartesian_teleop_math.py, scripts/cartesian_return_home.py): the teleoperation that recorded the data.


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    r, p, y = rpy
    sr, sp, sy = np.sin([r, p, y])
    cr, cp, cy = np.cos([r, p, y])
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    if np.hypot(rotation[0, 0], rotation[1, 0]) < 1e-8:
        return np.array([0.0, pitch, np.arctan2(-rotation[0, 1], rotation[1, 1])])
    return np.array([np.arctan2(rotation[2, 1], rotation[2, 2]), pitch, np.arctan2(rotation[1, 0], rotation[0, 0])])


def rotation_angle(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def pose_error(first, second) -> tuple[float, float]:
    """Translation [m] and rotation [rad] distance between two poses."""
    return float(np.linalg.norm(first[0] - second[0])), rotation_angle(first[1] @ second[1].T)


def limit_pose(previous, target, max_translation: float, max_rotation: float):
    """Bounds one incremental command on SE(3), without Euler wrap jumps."""
    p0, r0 = previous
    p1, r1 = target
    delta = p1 - p0
    distance = np.linalg.norm(delta)
    position = p0 + delta * min(1.0, max_translation / max(distance, 1e-12))
    rotation_delta = r1 @ r0.T
    angle = rotation_angle(rotation_delta)
    if angle <= max_rotation:
        return position, r1.copy()
    # A discontinuous half-turn is not a valid increment.
    if np.pi - angle < 1e-5:
        raise ValueError("target orientation jumped by pi")
    axis = np.array(
        [
            rotation_delta[2, 1] - rotation_delta[1, 2],
            rotation_delta[0, 2] - rotation_delta[2, 0],
            rotation_delta[1, 0] - rotation_delta[0, 1],
        ]
    ) / (2 * np.sin(angle))
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    increment = np.eye(3) + np.sin(max_rotation) * skew + (1 - np.cos(max_rotation)) * (skew @ skew)
    return position, increment @ r0


def rotvec_matrix(rotvec: Sequence[float]) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = rotvec / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def matrix_rotvec(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector exactly as scipy's ``Rotation.from_matrix(m).as_rotvec()`` (angle <= pi)."""
    m = np.asarray(rotation, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    choice = int(np.argmax([m[0, 0], m[1, 1], m[2, 2], trace]))
    quaternion = np.empty(4)  # x, y, z, w
    if choice == 3:
        quaternion[:] = (m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], 1.0 + trace)
    else:
        i, j, k = choice, (choice + 1) % 3, (choice + 2) % 3
        quaternion[i] = 1.0 - trace + 2.0 * m[i, i]
        quaternion[j] = m[j, i] + m[i, j]
        quaternion[k] = m[k, i] + m[i, k]
        quaternion[3] = m[k, j] - m[j, k]
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion
    angle = 2.0 * math.atan2(float(np.linalg.norm(quaternion[:3])), float(quaternion[3]))
    scale = 2.0 + angle**2 / 12.0 + 7.0 * angle**4 / 2880.0 if angle <= 1e-3 else angle / math.sin(angle / 2.0)
    return scale * quaternion[:3]


def pose_from_vector(pose6: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Controller pose vector (xyz + SDK rpy) -> (xyz, rotation matrix)."""
    pose6 = np.asarray(pose6, dtype=float)
    return pose6[:3].copy(), rpy_matrix(pose6[3:6])


def pose_rotvec(pose6: Sequence[float]) -> np.ndarray:
    """Controller pose (xyz + SDK rpy) -> xyz + rotation vector: the pose layout of the recordings."""
    position, rotation = pose_from_vector(pose6)
    return np.concatenate([position, matrix_rotvec(rotation)])


def policy_state(pose6: Sequence[float], gripper_position: float) -> np.ndarray:
    """``observation/state`` as recorded: TCP xyz + rotation vector (of the SDK rpy) + gripper position 0..1000."""
    return np.concatenate([pose_rotvec(pose6), [float(gripper_position)]]).astype(np.float32)


def absolute_target(action: Sequence[float], state: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Absolute TCP target of a LIFT relative action for the chunk-start state: ``T_state @ T(action)``."""
    action = np.asarray(action, dtype=float)
    state = np.asarray(state, dtype=float)
    rotation = rotvec_matrix(state[3:6])
    return state[:3] + rotation @ action[:3], rotation @ rpy_matrix(action[3:6])


def rgb_image(message: Any) -> np.ndarray:
    """Decodes an rgb8/bgr8 ``sensor_msgs/Image`` to an RGB uint8 HWC array (no cv_bridge; drops row padding)."""
    if message.encoding not in ("rgb8", "bgr8"):
        raise RobotError(f"unsupported camera encoding {message.encoding!r}; launch the cameras with rgb8 output")
    if message.height <= 0 or message.width <= 0 or message.step < message.width * 3:
        raise RobotError("camera image has invalid dimensions or row step")
    raw = message.data
    data = memoryview(bytes(raw) if isinstance(raw, list) else raw).cast("B")
    if data.nbytes != message.height * message.step:
        raise RobotError("camera image payload length does not match height * step")
    image = np.ndarray((message.height, message.width, 3), dtype=np.uint8, buffer=data, strides=(message.step, 3, 1))
    if message.encoding == "bgr8":
        image = image[..., ::-1]
    return image.copy(order="C")


@functools.cache
def tactile_io() -> Any:
    """``openpi_client.tactile``: the map pooling shared with the training data (imported for tactile policies)."""
    try:
        from openpi_client import tactile
    except ImportError as error:
        raise RobotError(f"openpi_client.tactile is not importable ({error}): use {_CLIENT_SRC}") from error
    return tactile


def depth_map(message: Any) -> np.ndarray:
    """Decodes a 32FC1 Tac G2 indentation map (``sensor_msgs/Image``) to float32 [H, W], as the AR3 recorder does."""
    if message.encoding != "32FC1" or message.height <= 0 or message.width <= 0:
        raise RobotError(f"Tac G2 depth must be a nonempty 32FC1 image, got {message.encoding!r}")
    raw = message.data
    data = memoryview(bytes(raw) if isinstance(raw, list) else raw).cast("B")
    if message.step < message.width * 4 or message.step % 4 or data.nbytes != message.height * message.step:
        raise RobotError("Tac G2 depth has an invalid row step or payload length")
    dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    values = np.frombuffer(data, dtype=dtype).reshape(message.height, message.step // 4)[:, : message.width]
    values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise RobotError("Tac G2 depth contains non-finite values")
    return values


def finger_wrench(message: Any) -> np.ndarray:
    """``geometry_msgs/WrenchStamped`` -> float32 [6] Fx Fy Fz Mx My Mz (SDK units, sensor frame), as recorded."""
    force, torque = message.wrench.force, message.wrench.torque
    values = np.asarray([force.x, force.y, force.z, torque.x, torque.y, torque.z], dtype=np.float32)
    if not np.isfinite(values).all():
        raise RobotError("Tac G2 wrench contains non-finite values")
    return values


def _describe(problems: dict[str, str]) -> str:
    return ", ".join(f"{name}: {problem}" for name, problem in problems.items())


# DM V3 gripper (gRPC). Field number/type pairs of G3_client/DM_V3Gripper/gripper.proto StateResponse.
GRIPPER_STATE_FIELDS = {
    "valid": (1, 8),
    "init_state": (2, 5),
    "grip_state": (3, 5),
    "position": (4, 5),
    "error": (7, 5),
    "update_age_ms": (9, 1),
    "stale": (10, 8),
    "connected": (14, 8),
}


def check_gripper_schema(state_type: Any) -> None:
    """Rejects an installed ``DM_V3Gripper`` whose ``StateResponse`` fields differ from the ones read here."""
    fields = state_type.DESCRIPTOR.fields_by_name
    for name, expected in GRIPPER_STATE_FIELDS.items():
        field = fields.get(name)
        if field is None or (field.number, field.type) != expected:
            raise GripperError(f"installed DM_V3Gripper StateResponse schema is incompatible ({name})")


def gripper_position(state: Any, *, max_age: float, elapsed: float = 0.0) -> float:
    """Validates a ``GetAllState`` response; returns the position (0 = closed ... 1000 = open), as recorded."""
    if not state.connected or not state.valid or state.stale:
        raise GripperError(
            f"gripper feedback unavailable (connected={state.connected}, valid={state.valid}, stale={state.stale}). "
            f"{GRIPPER_PREPARATION}"
        )
    if state.init_state != 1:
        raise GripperError(f"gripper is not initialized (init_state={state.init_state}). {GRIPPER_PREPARATION}")
    if state.error != 0 or state.grip_state not in (0, 1, 2, 3):
        raise GripperError(f"gripper reports error={state.error}, grip_state={state.grip_state}. {GRIPPER_PREPARATION}")
    age = float(state.update_age_ms) / 1000.0 + elapsed
    if not math.isfinite(age) or not 0 <= age <= max_age:
        raise GripperError(f"gripper feedback is stale ({age:.3f} s > {max_age} s). {GRIPPER_PREPARATION}")
    position = float(state.position)
    if not math.isfinite(position) or not 0 <= position <= 1000:
        raise GripperError(f"gripper position {position} is outside 0..1000")
    return position


class DMGripper:
    """DM tac G2/V3 gripper through its gRPC server. Read-only (``GetAllState``) unless write operations are bound."""

    def __init__(
        self,
        *,
        get_state: Any,
        operations: dict[str, Any],
        encode_value: Any,
        rpc_error: type[BaseException],
        max_age: float = 0.5,
        speed: int = 80,
        force: int = 20,
        deadband: int = 2,
        rpc_timeout: float = 0.2,
        close: Any = None,
    ) -> None:
        self._get_state = get_state
        self._operations = operations
        self._encode = encode_value
        self._rpc_error = rpc_error
        self._max_age = max_age
        self._speed = int(speed)
        self._force = int(force)
        self._deadband = int(deadband)
        self._rpc_timeout = rpc_timeout
        self._close = close
        self._pending: Any = None
        self._last_sent: int | None = None
        self._touched = False
        # Newest target of the policy (recorded as the gripper action of policy frames).
        self.target: float | None = None

    @classmethod
    def connect(cls, address: str, *, writable: bool, **kwargs: Any) -> "DMGripper":
        from DM_V3Gripper import gripper_pb2
        import grpc

        check_gripper_schema(gripper_pb2.StateResponse)
        channel = grpc.insecure_channel(address, options=(("grpc.enable_http_proxy", 0),))
        operations = {}
        if writable:
            operations = {
                name: channel.unary_unary(
                    GRIPPER_RPC_PREFIX + name, response_deserializer=gripper_pb2.Response.FromString
                )
                for name in ("Enable", "SetTargetSpeed", "SetTargetForce", "SetTargetPosition", "Stop")
            }
        return cls(
            get_state=channel.unary_unary(
                GRIPPER_RPC_PREFIX + "GetAllState", response_deserializer=gripper_pb2.StateResponse.FromString
            ),
            operations=operations,
            encode_value=lambda value: gripper_pb2.ValueRequest(value=value).SerializeToString(),
            rpc_error=grpc.RpcError,
            close=channel.close,
            **kwargs,
        )

    def read(self) -> float:
        started = time.monotonic()
        try:
            state = self._get_state.future(b"", timeout=self._rpc_timeout).result()
        except self._rpc_error as error:
            raise GripperError(f"gripper GetAllState failed: {error}. {GRIPPER_PREPARATION}") from error
        return gripper_position(state, max_age=self._max_age, elapsed=time.monotonic() - started)

    def configure(self) -> None:
        """Enables the motor and sets the recorded speed and force (once, before the arm moves)."""
        self._call("Enable")
        self._call("SetTargetSpeed", self._speed)
        self._call("SetTargetForce", self._force)

    def command(self, position: float) -> None:
        """Sends a target position (0..1000) without blocking; while one is in flight the newest waits a step."""
        if not math.isfinite(position):
            raise GripperError(f"non-finite gripper action {position}")
        target = round(min(max(float(position), 0.0), GRIPPER_OPEN))
        self.target = float(target)
        if self._pending is not None:
            if not self._pending.done():
                return
            self._collect_pending()
        if self._last_sent is not None and abs(target - self._last_sent) < self._deadband:
            return
        self._touched = True
        self._pending = self._operations["SetTargetPosition"].future(self._encode(target), timeout=self._rpc_timeout)
        self._last_sent = target

    def forget_target(self) -> None:
        """Someone else (the gripper teleoperation) may have moved the fingers: send the next target in any case."""
        self._last_sent = None
        self.target = None

    def wait_position(self, target: float, *, tolerance: float, timeout: float) -> float:
        deadline = time.monotonic() + timeout
        while True:
            position = self.read()
            if abs(position - target) <= tolerance or time.monotonic() > deadline:
                return position
            time.sleep(0.05)

    def stop(self) -> None:
        """Stops the fingers if this client commanded them."""
        if not self._touched:
            return
        if self._pending is not None:
            try:
                self._collect_pending()
            except GripperError as error:
                logger.warning("%s", error)
        self._call("Stop")

    def close(self) -> None:
        if self._close is not None:
            self._close()

    def _collect_pending(self) -> None:
        future, self._pending = self._pending, None
        try:
            response = future.result()
        except self._rpc_error as error:
            raise GripperError(f"gripper SetTargetPosition failed: {error}") from error
        if not response.success:
            raise GripperError(f"gripper SetTargetPosition rejected: {response.message}")

    def _call(self, name: str, value: int | None = None) -> None:
        request = b"" if value is None else self._encode(int(value))
        self._touched = True
        try:
            response = self._operations[name](request, timeout=self._rpc_timeout)
        except self._rpc_error as error:
            raise GripperError(f"gripper {name} failed: {error}") from error
        if not response.success:
            raise GripperError(f"gripper {name} rejected: {response.message}")


@dataclasses.dataclass(frozen=True)
class Observation:
    pose: np.ndarray  # [6] measured TCP xyz + SDK rpy
    reading: np.ndarray  # [6] tauExt_inBase wrench (+ [7] tau_m): one force reading
    front: np.ndarray  # HWC uint8 RGB
    wrist: np.ndarray  # HWC uint8 RGB


@dataclasses.dataclass
class Limits:
    """Command shaping of the AR3 teleoperation (its defaults), and the session workspace."""

    rate_hz: float = 100.0
    max_lead_m: float = 0.06
    max_lead_rad: float = 0.05
    max_linear_speed: float = 0.04
    max_angular_speed: float = 0.15
    max_tracking_error_m: float = 0.08
    max_tracking_error_rad: float = 0.15
    feedback_timeout: float = 0.5
    target_timeout: float = 1.0
    workspace_radius: float = 0.25
    workspace_margin: float = 0.02


class Sample(NamedTuple):
    received: float  # host monotonic time of receipt
    value: Any
    stamp: float = 0.0  # source time stamp (ROS header / wall clock), 0 if unknown
    index: int = 0  # message counter (camera frames)


class Stream:
    """Recent samples of one recorded input, in receipt order (guarded by the owner's lock)."""

    def __init__(self, maxlen: int) -> None:
        self._samples: collections.deque[Sample] = collections.deque(maxlen=maxlen)

    def add(self, sample: Sample) -> None:
        self._samples.append(sample)

    def before(self, t: float) -> Sample | None:
        """Newest sample received at or before ``t``."""
        for sample in reversed(self._samples):
            if sample.received <= t:
                return sample
        return None

    def at(self, t: float) -> Sample | None:
        """Value at ``t``: the newest sample up to ``t``, else the oldest one."""
        sample = self.before(t)
        return sample if sample is not None or not self._samples else self._samples[0]

    def nearest(self, t: float) -> Sample | None:
        return min(self._samples, key=lambda sample: abs(sample.received - t), default=None)

    def clear(self) -> None:
        self._samples.clear()


def _stamp(message: Any) -> float:
    stamp = message.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


class Ar3Robot:
    """ROS 2 side: sensors, controller status / services and the 100 Hz command streamer.

    Every tick the newest target is limited to a lead of ``max_lead`` from the measured TCP pose and slewed from the
    previous command at ``max_*_speed``; previous and new commands must stay within the tracking limits (else stop).
    Commands carry the session's stiffness / damping / mass (``state[12:]``) and zero desired wrench. All of this is
    the AR3 teleoperation's shaping, so the policy commands the arm the way the training data was produced.

    ``record`` keeps a short history of every input for ``RolloutRecorder``. ``dagger``: the AR3 teleoperation runs
    alongside; its clutch hands the active controller session over to it (``HumanTakeover``) without stopping it.
    """

    def __init__(
        self,
        *,
        execute: bool,
        limits: Limits,
        prefix: str = "/ar3/cartesian_impedance",
        dagger: bool = False,
        record: bool = False,
    ) -> None:
        self.execute = execute
        self.limits = limits
        self.prefix = prefix.rstrip("/")
        self.dagger = dagger
        self._lock = threading.RLock()
        self._state: np.ndarray | None = None
        self._state_at = -math.inf
        self._tau_m: np.ndarray | None = None
        self._tau_m_at = -math.inf
        self._images: dict[str, tuple[Any, float]] = {}
        self._camera_topics = {"front": "/front/color/image_raw", "wrist": "/wrist/color/image_raw"}
        # Tac G2 (muscleVLTA policies): newest message and its receipt time per (kind, sensor).
        self._tactile: dict[tuple[str, str], tuple[Any, float]] = {}
        self._tactile_topics: dict[tuple[str, str], str] = {}
        self._finger_problems: dict[str, str] = {}
        self._size_warned: set[str] = set()
        self._status = ""
        self._metadata: dict | None = None
        self._ownership_checked_at = -math.inf
        self._target: tuple[np.ndarray, np.ndarray] | None = None
        self._target_at = -math.inf
        self._last_command: tuple[np.ndarray, np.ndarray] | None = None
        self._session_parameters: np.ndarray | None = None
        self._origin: np.ndarray | None = None
        self._streaming = False
        self._home_limits: Limits | None = None
        self._armed = False
        self._sdk_active = False
        self._stopping = False
        self._fault: str | None = None
        self._last_tick = time.monotonic()
        self._clamp_logged_at = -math.inf
        self._ros: dict[str, Any] = {}
        self._streams = {name: Stream(size) for name, size in STREAM_SIZES.items()} if record else {}
        self._camera_counts: dict[str, int] = {}
        # DAgger takeover: clutch state, press / release host times, teleoperation commands since the press.
        self._clutch = False
        self._takeover_at: float | None = None
        self._released_at: float | None = None
        self._teleop_commands = 0
        self._published: collections.deque[tuple[float, ...]] = collections.deque(maxlen=8)
        self._teleop_status: dict | None = None

    # ROS wiring (rclpy is imported lazily so the logic is testable without ROS).
    def start(
        self,
        *,
        front_topic: str,
        wrist_topic: str,
        controller_node: str,
        clutch_topic: str = "/clutch/right",
        tactile: bool = False,
    ) -> None:
        """Subscribes to the controller, the cameras and, with ``tactile``, the Tac G2 fingers."""
        try:
            from geometry_msgs.msg import WrenchStamped
            from rcl_interfaces.srv import GetParameters
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import DurabilityPolicy
            from rclpy.qos import HistoryPolicy
            from rclpy.qos import QoSProfile
            from rclpy.qos import ReliabilityPolicy
            from rclpy.qos import qos_profile_sensor_data
            from rclpy.signals import SignalHandlerOptions
            from sensor_msgs.msg import Image
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Bool
            from std_msgs.msg import Float64MultiArray
            from std_msgs.msg import String
            from std_srvs.srv import SetBool
            from std_srvs.srv import Trigger
        except ImportError as error:
            raise RobotError(
                f"ROS 2 is not importable ({error}): source /opt/ros/jazzy/setup.bash first "
                "(scripts/run_ar3_lift_robot.sh does this)"
            ) from error
        sensors = tactile_io().SENSORS if tactile else ()

        # Ctrl+C must reach the main thread (which stops the arm), not shut ROS down underneath it.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node("ar3_lift_robot_client")
        # The Orbbec publishers are reliable. Raw 640x480 RGB frames lose fragments with best-effort DDS on
        # this PC; reliable delivery restores 30 Hz. Keep only the newest frame, with the same freshness limit.
        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE)
        node.create_subscription(Float64MultiArray, self.prefix + "/state", self._on_state, qos_profile_sensor_data)
        node.create_subscription(Float64MultiArray, self.prefix + "/tau_m", self._on_tau_m, qos_profile_sensor_data)
        node.create_subscription(
            String,
            self.prefix + "/status",
            self._on_status,
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        node.create_subscription(
            String,
            self.prefix + "/metadata",
            self._on_metadata,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._camera_topics = {"front": front_topic, "wrist": wrist_topic}
        for name, topic in self._camera_topics.items():
            node.create_subscription(Image, topic, functools.partial(self._on_image, name), image_qos)
        # The bridge publishes with the sensor-data QoS (as the recorder subscribed); maps are decoded on use.
        for sensor in sensors:
            for kind, message_type in (("depth", Image), ("wrench", WrenchStamped)):
                topic = self._tactile_topics[kind, sensor] = f"{TACTILE_PREFIX}/{sensor}/{kind}"
                node.create_subscription(
                    message_type, topic, functools.partial(self._on_tactile, (kind, sensor)), qos_profile_sensor_data
                )
        if self._streams:
            node.create_subscription(
                Float64MultiArray, self.prefix + "/accepted_command", self._on_accepted, qos_profile_sensor_data
            )
        latched = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        if self.dagger:
            # Same QoS as the teleoperation's subscriptions / publishers.
            node.create_subscription(Bool, clutch_topic, self._on_clutch, 10)
            node.create_subscription(Float64MultiArray, self.prefix + "/command", self._on_command, 10)
            node.create_subscription(JointState, GRIPPER_ACTION_TOPIC, self._on_gripper_action, 10)
            node.create_subscription(
                String, self.prefix + "/teleop/return_home_status", self._on_teleop_status, latched
            )
        self._ros = {
            "rclpy": rclpy,
            "node": node,
            "message": Float64MultiArray,
            # Dry runs must not look like a command source to the controller's other clients.
            "command": node.create_publisher(Float64MultiArray, self.prefix + "/command", 1) if self.execute else None,
            "preview": node.create_publisher(Float64MultiArray, self.prefix + "/preview_command", 1),
            "enable": node.create_client(SetBool, self.prefix + "/enable"),
            "stop": node.create_client(Trigger, self.prefix + "/stop"),
            "recenter": node.create_client(Trigger, self.prefix + "/recenter_workspace"),
            "parameters": node.create_client(GetParameters, controller_node.rstrip("/") + "/get_parameters"),
            "SetBool": SetBool,
            "Trigger": Trigger,
            "GetParameters": GetParameters,
        }
        if self.dagger:
            self._ros["Bool"] = Bool
            self._ros["human_control"] = node.create_publisher(Bool, HUMAN_CONTROL_TOPIC, latched)
            self._publish_human_control(active=False)
        node.create_timer(1.0 / self.limits.rate_hz, self._tick)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        self._ros["executor"] = executor
        self._ros["thread"] = threading.Thread(target=self._spin, name="ros", daemon=True)
        self._ros["thread"].start()

    def _spin(self) -> None:
        try:
            self._ros["executor"].spin()
        except Exception as error:
            with self._lock:
                self._halt_locked(f"ROS executor failed: {error!r}")

    def shutdown(self, timeout: float = 3.0) -> None:
        """Wake and join the executor before destroying resources used by its native wait set."""
        if not self._ros:
            return
        deadline = time.monotonic() + timeout
        stopped = self._ros["executor"].shutdown(timeout_sec=timeout)
        thread = self._ros["thread"]
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive() or not stopped:
            # Destroying the node/context while spin() is still using them can abort the Python process.
            # Keep the resources alive so the caller can report the failure and retry cleanup.
            raise RobotError(f"ROS executor did not stop within {timeout:g} s; its resources were not destroyed")
        try:
            self._publish_human_control(active=False)
            self._ros["node"].destroy_node()
        finally:
            self._ros["rclpy"].try_shutdown()
        self._ros = {}

    def _publish_human_control(self, *, active: bool) -> None:
        publisher = self._ros.get("human_control")
        if publisher is not None:
            publisher.publish(self._ros["Bool"](data=active))

    def _call(self, name: str, request: Any, timeout: float = 2.0, idle: Any = None) -> Any:
        """Calls a controller service; ``idle`` runs while waiting (e.g. ``heartbeat`` while streaming)."""
        client = self._ros[name]
        if not client.wait_for_service(timeout_sec=timeout):
            raise RobotError(f"controller service {client.srv_name} is not available")
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        try:
            while not future.done():
                if idle is not None:
                    idle()
                if time.monotonic() > deadline:
                    raise RobotError(f"controller service {client.srv_name} timed out")
                time.sleep(0.005)
        except BaseException:
            future.cancel()
            client.remove_pending_request(future)
            raise
        return future.result()

    # Subscriptions (executor thread).
    def _on_state(self, message: Any) -> None:
        values = np.asarray(message.data, dtype=float)
        with self._lock:
            if values.shape != (STATE_SIZE,) or not np.isfinite(values).all():
                self._halt_locked("invalid controller state message")
                return
            self._state, self._state_at = values, time.monotonic()
            self._add_locked("state", Sample(self._state_at, values))

    def _on_tau_m(self, message: Any) -> None:
        values = np.asarray(message.data, dtype=float)
        if values.shape == (TAU_M_SIZE,) and np.isfinite(values).all():
            with self._lock:
                self._tau_m, self._tau_m_at = values, time.monotonic()
                self._add_locked("tau_m", Sample(self._tau_m_at, values))

    def _on_image(self, name: str, message: Any) -> None:
        now = time.monotonic()
        with self._lock:
            self._images[name] = (message, now)
            if self._streams:
                index = self._camera_counts[name] = self._camera_counts.get(name, -1) + 1
                self._add_locked(name, Sample(now, message, _stamp(message), index))

    def _on_tactile(self, key: tuple[str, str], message: Any) -> None:
        now = time.monotonic()
        with self._lock:
            self._tactile[key] = (message, now)

    def _on_accepted(self, message: Any) -> None:
        values = np.asarray(message.data, dtype=float)
        if values.shape == (STATE_SIZE,) and np.isfinite(values).all():
            with self._lock:
                self._add_locked("accepted", Sample(time.monotonic(), values[:6].copy()))

    def _on_clutch(self, message: Any) -> None:
        pressed = bool(message.data)
        now = time.monotonic()
        with self._lock:
            was, self._clutch = self._clutch, pressed
            if pressed and not was:
                if self._takeover_at is None and (self._streaming or (self._armed and not self._stopping)):
                    self._take_over_locked(now)
            elif was and not pressed and self._takeover_at is not None and self._released_at is None:
                # The teleoperation now stops the arm (its clutch release ends a session it did not start).
                self._released_at = now
                self._add_locked("control", Sample(now, POLICY_FLAG))
                self._publish_human_control(active=False)
                logger.info("clutch released: the takeover ended")

    def _take_over_locked(self, now: float) -> None:
        """Stops commanding at once without stopping the controller: the teleoperation continues the session (its
        enable keeps an armed session, and both send the session stiffness with zero desired wrench)."""
        self._streaming = self._armed = self._sdk_active = self._stopping = False
        self._target = None
        self._takeover_at, self._released_at, self._teleop_commands = now, None, 0
        self._add_locked("control", Sample(now, HUMAN_FLAG))
        self._publish_human_control(active=True)
        logger.warning("clutch pressed: the operator controls the robot (DAgger takeover)")

    def _on_command(self, message: Any) -> None:
        values = tuple(message.data)
        now = time.monotonic()
        with self._lock:
            # Only teleoperation commands: not before the takeover, after the release, or this client's own.
            if self._takeover_at is None or self._released_at is not None or values in self._published:
                return
            self._teleop_commands += 1
            self._add_locked("teleop", Sample(now, None))

    def _on_gripper_action(self, message: Any) -> None:
        if not len(message.position):
            return
        position = float(message.position[0]) * GRIPPER_OPEN
        if not math.isfinite(position):
            return
        now = time.monotonic()
        with self._lock:
            if self._takeover_at is not None and self._released_at is None:
                self._add_locked("human_gripper", Sample(now, min(max(position, 0.0), GRIPPER_OPEN), _stamp(message)))

    def _on_teleop_status(self, message: Any) -> None:
        try:
            status = json.loads(message.data)
        except ValueError:
            return
        if isinstance(status, dict):
            with self._lock:
                self._teleop_status = status

    def _add_locked(self, name: str, sample: Sample) -> None:
        stream = self._streams.get(name)
        if stream is not None:
            stream.add(sample)

    def _on_metadata(self, message: Any) -> None:
        try:
            metadata = json.loads(message.data)
        except ValueError:
            metadata = None
        with self._lock:
            self._metadata = metadata if isinstance(metadata, dict) else {"unparsable": message.data}

    def _on_status(self, message: Any) -> None:
        text = message.data
        with self._lock:
            if text != self._status:
                logger.info("controller status: %s", text)
            self._status = text
            if text.startswith("stop failed:"):
                self._fault = self._fault or f"controller {text}"
                self._streaming = False
                return
            if (
                text.startswith(("fault:", "offline:"))
                or "RT monitoring restoration failed" in text
                or "monitoring unavailable" in text
            ):
                self._halt_locked(f"controller {text}")
            elif text.startswith("active:") and self._armed:
                self._sdk_active = True
            elif self._sdk_active and text.startswith(("connecting:", "monitoring:", "armed:", "initializing:")):
                self._halt_locked(f"controller left the active RT session: {text}")
            elif text.startswith(("stopping:", "stopped:")) and self._armed and not self._stopping:
                self._halt_locked(f"controller {text}")

    def _halt_locked(self, reason: str) -> None:
        """Ends streaming and requests a stop (completed by ``stop``). No-op unless this client is commanding."""
        if not (self._streaming or self._armed):
            return
        if self._fault is None:
            self._fault = reason
            logger.error("halting: %s", reason)
        self._streaming = self._sdk_active = False
        self._target = None
        if self._armed and not self._stopping and self._ros and self._ros["stop"].service_is_ready():
            self._stopping = True
            self._ros["stop"].call_async(self._ros["Trigger"].Request())

    # Command streamer (executor thread).
    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            dt = min(now - self._last_tick, 1.0 / self.limits.rate_hz)
            self._last_tick = now
            if not self._streaming:
                return
            try:
                values = self.next_command(now, dt)
            except ValueError as error:
                self._halt_locked(str(error))
                return
            message = self._ros["message"](data=values.tolist())
            self._ros["preview"].publish(message)
            if self.execute:
                self._ros["command"].publish(message)
            if self.dagger:
                self._published.append(tuple(message.data))
            self._add_locked("commanded", Sample(now, values[:6].copy()))

    def next_command(self, now: float, dt: float) -> np.ndarray:
        """Shapes the next 30-value controller command (call with the lock held); ValueError means stop."""
        limits = self._home_limits or self.limits
        if self._state is None or now - self._state_at > limits.feedback_timeout:
            raise ValueError(f"controller state is stale ({now - self._state_at:.3f} s)")
        if self._target is None or now - self._target_at > limits.target_timeout:
            raise ValueError(f"policy target is stale ({now - self._target_at:.3f} s > {limits.target_timeout} s)")
        measured = pose_from_vector(self._state[:6])
        self._check_tracking(self._last_command, measured)
        bounded = limit_pose(measured, self._target, limits.max_lead_m, limits.max_lead_rad)
        command = limit_pose(self._last_command, bounded, limits.max_linear_speed * dt, limits.max_angular_speed * dt)
        # Targets are already inside the workspace; this only matters if contact pushed the TCP outside of it.
        command = (clamp_to_sphere(command[0], self._origin, self._clamp_radius())[0], command[1])
        self._check_tracking(command, measured)
        self._last_command = command
        if self._session_parameters is None:
            self._session_parameters = np.concatenate([np.zeros(6), self._state[12:]])
        return np.concatenate([command[0], matrix_rpy(command[1]), self._session_parameters])

    def _check_tracking(self, command, measured) -> None:
        translation, rotation = pose_error(command, measured)
        if (
            translation > self.limits.max_tracking_error_m + 1e-9
            or rotation > self.limits.max_tracking_error_rad + 1e-9
        ):
            raise ValueError(
                f"measured TCP exceeded the tracking limits: {translation * 1000:.1f} mm / {rotation:.3f} rad "
                f"(limits {self.limits.max_tracking_error_m * 1000:.0f} mm / {self.limits.max_tracking_error_rad} rad)"
            )

    # Main-thread API.
    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def wait_ready(
        self,
        timeout: float,
        *,
        need_tau_m: bool,
        need_tactile: bool = False,
        max_observation_age: float = 1.0,
        max_image_age: float = 1.0,
        max_tactile_age: float = 1.0,
    ) -> None:
        """Waits for the controller, force, cameras (and Tac G2 fingers) to satisfy the requested freshness."""
        if need_tactile and not self._tactile_topics:
            raise RobotError("the policy reads the Tac G2 fingers: start the robot with tactile=True")
        deadline = time.monotonic() + timeout
        while True:
            now = time.monotonic()
            with self._lock:
                ages = {self.prefix + "/state": (now - self._state_at, max_observation_age)}
                if need_tau_m:
                    ages[self.prefix + "/tau_m"] = (now - self._tau_m_at, max_observation_age)
                for name, topic in self._camera_topics.items():
                    ages[topic] = (now - self._images.get(name, (None, -math.inf))[1], max_image_age)
                if need_tactile:
                    for key, topic in self._tactile_topics.items():
                        ages[topic] = (now - self._tactile.get(key, (None, -math.inf))[1], max_tactile_age)
                status, metadata = self._status, self._metadata
            missing = [topic for topic, (age, limit) in ages.items() if age > limit]
            if not status:
                missing.append(self.prefix + "/status")
            if metadata is None:
                missing.append(self.prefix + "/metadata")
            if not missing:
                self._check_contract(metadata)
                if need_tactile:
                    self._check_fingers(max_tactile_age)
                return
            if now > deadline:
                hint = f"; {TACTILE_PREPARATION}" if any(topic.startswith(TACTILE_PREFIX) for topic in missing) else ""
                raise RobotError(
                    f"no fresh data on {', '.join(missing)} after {timeout:.0f} s: start the impedance controller and "
                    f"the cameras (scripts/prepare_ar3_inference.sh) with the same ROS_DOMAIN_ID{hint}"
                )
            time.sleep(0.05)

    def _check_fingers(self, max_age: float) -> None:
        """Both fingers' newest samples must decode as recorded (32FC1 maps that pool to the training shape)."""
        _, valid, _ = self.hand_reading(max_age=max_age)
        if not valid.all():
            raise RobotError(f"Tac G2 fingers unusable ({_describe(self._finger_problems)}): {TACTILE_PREPARATION}")

    def _check_contract(self, metadata: dict) -> None:
        """The node must stream the RT Cartesian impedance command layout (and pose convention) this client uses."""
        mismatched = {key: metadata.get(key) for key, value in RT_CONTRACT.items() if metadata.get(key) != value}
        if metadata.get("pose", RPY_CONVENTION) != RPY_CONVENTION:
            mismatched["pose"] = metadata["pose"]
        if mismatched:
            raise RobotError(
                f"{self.prefix}/metadata does not match the RT Cartesian impedance contract of this client "
                f"(expected {RT_CONTRACT}, pose {RPY_CONVENTION!r}; got {mismatched}): update ar3_CartesianImpedanceControl"
            )

    @property
    def metadata(self) -> dict:
        with self._lock:
            return dict(self._metadata or {})

    def command_sources(self) -> tuple[int, list[str]]:
        """Other publishers on ``{prefix}/command`` and running conflicting nodes (teleoperation, legacy driver).

        With ``dagger`` the AR3 teleoperation is expected: it only commands while its clutch is pressed.
        """
        node = self._ros["node"]
        allowed = {TELEOP_NODE} if self.dagger else set()
        publishers = node.get_publishers_info_by_topic(self.prefix + "/command")
        others = sum(info.node_name not in allowed for info in publishers) - int(self.execute)
        conflicts = sorted({name for name, _ in node.get_node_names_and_namespaces()} & CONFLICTING_NODES - allowed)
        return others, conflicts

    def check_ownership(self) -> None:
        """With ``execute``: this client must be the only arm command source (no teleoperation / legacy driver)."""
        if not self.execute:
            return
        others, conflicts = self.command_sources()
        if others > 0 or conflicts:
            raise RobotError(
                f"another arm command source is running ({', '.join(conflicts) or 'unknown node'}; {others} other "
                f"publisher(s) on {self.prefix}/command): stop the teleoperation / other clients first"
            )

    def controller_parameters(self, names: Sequence[str], timeout: float) -> dict[str, Any]:
        """Reads parameters of the impedance node (None for undeclared names)."""
        response = self._call("parameters", self._ros["GetParameters"].Request(names=list(names)), timeout)
        return {name: parameter_value(value) for name, value in zip(names, response.values, strict=False)}

    def observe(self, *, wrench_dim: int, max_age: float, max_image_age: float) -> Observation:
        """Chunk-start observation: measured TCP pose, force reading and the newest front / wrist frames."""
        now = time.monotonic()
        with self._lock:
            self._raise_if_fault_locked()
            pose, reading = self._sample_locked(now, wrench_dim, max_age)
            frames = dict(self._images)
        images = {}
        for name, topic in self._camera_topics.items():
            message, received = frames.get(name, (None, -math.inf))
            if message is None or now - received > max_image_age:
                raise RobotError(
                    f"{name} camera image on {topic} is stale ({now - received:.3f} s > {max_image_age} s)"
                )
            images[name] = rgb_image(message)
            if images[name].shape[:2] != (480, 640) and name not in self._size_warned:
                self._size_warned.add(name)
                logger.warning(
                    "%s camera frames are %dx%d; the policy was trained on 640x480", name, *images[name].shape[1::-1]
                )
        return Observation(pose=pose, reading=reading, front=images["front"], wrist=images["wrist"])

    def force_reading(self, *, wrench_dim: int, max_age: float) -> np.ndarray:
        now = time.monotonic()
        with self._lock:
            self._raise_if_fault_locked()
            return self._sample_locked(now, wrench_dim, max_age)[1]

    def hand_reading(self, *, max_age: float = MAX_TACTILE_AGE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tac G2 reading of this control step, as in the training data: area-pooled indentation maps float16
        [2, 48, 64], finger validity bool [2] and finger wrenches float32 [12] (Fx Fy Fz Mx My Mz of sensor_0, then
        sensor_2).

        A finger counts when its newest map and wrench are at most ``max_age`` old (newest sample at or before this
        step, like the tactile cache) and decode; otherwise it is sent as missing with zeros, which the policy was
        trained with (finger dropout, recording gaps). Changes of the missing fingers are logged.
        """
        now = time.monotonic()
        with self._lock:
            self._raise_if_fault_locked()
            samples = dict(self._tactile)
        depth, valid, wrench, _, problems = self._read_hand(samples, now, max_age)
        if problems.keys() != self._finger_problems.keys():
            if problems:
                logger.warning("Tac G2 fingers missing (%s); the policy continues without them", _describe(problems))
            else:
                logger.info("Tac G2 fingers available again")
        self._finger_problems = problems
        return depth, valid, wrench

    def hand_snapshot(
        self, *, max_age: float = MAX_TACTILE_AGE
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``hand_reading`` for the recorder, with the host times [2] at which the fingers' newest messages arrived
        (-inf without any). It also reads during a human takeover or after a fault, and does not log."""
        now = time.monotonic()
        with self._lock:
            samples = dict(self._tactile)
        return self._read_hand(samples, now, max_age)[:4]

    @staticmethod
    def _read_hand(
        samples: dict[tuple[str, str], tuple[Any, float]], now: float, max_age: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
        """Maps, finger validity, finger wrenches, receipt times and the problems of the missing fingers."""
        tactile = tactile_io()
        depth = np.zeros((len(tactile.SENSORS), *tactile.DEPTH_SHAPE), dtype=np.float16)
        wrench = np.zeros((len(tactile.SENSORS), tactile.WRENCH_DIM), dtype=np.float32)
        valid = np.zeros(len(tactile.SENSORS), dtype=bool)
        received = np.full(len(tactile.SENSORS), -math.inf)
        problems: dict[str, str] = {}
        for index, sensor in enumerate(tactile.SENSORS):
            (depth_message, depth_at), (wrench_message, wrench_at) = (
                samples.get((kind, sensor), (None, -math.inf)) for kind in ("depth", "wrench")
            )
            received[index] = min(depth_at, wrench_at)
            age = now - received[index]
            if age > max_age:
                problems[sensor] = "no samples" if math.isinf(age) else f"samples {age:.2f} s old"
                continue
            try:
                depth[index] = tactile.pool_depth(depth_map(depth_message))
                wrench[index] = finger_wrench(wrench_message)
            except (RobotError, ValueError) as error:
                depth[index], wrench[index] = 0.0, 0.0
                problems[sensor] = str(error)
                continue
            valid[index] = True
        return depth, valid, wrench.reshape(-1), received, problems

    def poses(self) -> tuple[tuple[np.ndarray, np.ndarray], Any, Any]:
        """(measured, last streamed command, current target) poses; the last two are None when not streaming."""
        now = time.monotonic()
        with self._lock:
            self._raise_if_fault_locked()
            measured = pose_from_vector(self._sample_locked(now, 6, self.limits.feedback_timeout)[0])
            return measured, self._last_command, self._target

    @contextlib.contextmanager
    def return_home_mode(self):
        """Use the recording's homing profile while retaining the independent tracking/force stops."""
        with self._lock:
            self._raise_if_fault_locked()
            if not self._streaming or self._home_limits is not None:
                raise RobotError("return home requires an active stream with no other homing operation")
            self._home_limits = home_limits(self.limits)
        try:
            yield self._home_limits
        finally:
            with self._lock:
                self._home_limits = None

    def set_target(self, position: np.ndarray, rotation: np.ndarray, *, clamp: bool = True) -> bool:
        """Replaces the streamed TCP target, clamped into the session workspace. Returns whether it was clamped.

        With ``clamp=False`` a target outside the workspace raises instead (the current target is kept).
        """
        with self._lock:
            self._raise_if_fault_locked()
            if not self._streaming:
                raise RobotError("the command stream is not active")
            clamped_position, clamped = clamp_to_sphere(position, self._origin, self._clamp_radius())
            if clamped and not clamp:
                raise RobotError(
                    f"target is {np.linalg.norm(np.asarray(position) - self._origin):.3f} m from the session origin, "
                    f"outside the {self._clamp_radius():.3f} m workspace (workspace_radius_m - margin): move the arm "
                    "closer first (e.g. the teleoperation's return-home) or raise workspace_radius_m"
                )
            position = clamped_position
            self._target = (position, np.array(rotation, dtype=float))
            self._target_at = time.monotonic()
            if clamped and self._target_at - self._clamp_logged_at > 1.0:
                self._clamp_logged_at = self._target_at
                logger.warning(
                    "policy target clamped to the %.3f m session workspace (workspace_radius_m - margin)",
                    self._clamp_radius(),
                )
        return clamped

    def heartbeat(self) -> None:
        """Main loop liveness: keeps the current target fresh and raises if the stream halted.

        While commanding it also checks (about once per second) that no other arm command source appeared.
        """
        with self._lock:
            self._raise_if_fault_locked()
            now = time.monotonic()
            if self._streaming:
                self._target_at = now
            check = self._streaming and self.execute and now - self._ownership_checked_at >= 1.0
            if check:
                self._ownership_checked_at = now
        if check:
            try:
                self.check_ownership()
            except RobotError as error:
                with self._lock:
                    self._halt_locked(str(error))
                raise

    def activate(self, *, service_timeout: float, idle_timeout: float = 5.0, active_timeout: float = 10.0) -> None:
        """Starts streaming at the measured pose; with ``execute`` it first takes over the idle controller.

        A new controller session anchors its workspace sphere (``workspace_radius_m``) at the measured TCP pose.
        """
        if self.execute:
            if self.dagger and self.clutch_pressed:
                raise HumanTakeover
            self.check_ownership()
            deadline = time.monotonic() + idle_timeout
            while not self.status.startswith(IDLE_STATUS):
                with self._lock:
                    self._raise_if_fault_locked()
                if time.monotonic() > deadline:
                    raise RobotError(
                        f"controller is not idle ({self.status!r}): stop other clients (e.g. the teleoperation) or "
                        "recover the controller first"
                    )
                time.sleep(0.05)
        with self._lock:
            self._raise_if_fault_locked()
            now = time.monotonic()
            measured = pose_from_vector(self._sample_locked(now, 6, self.limits.feedback_timeout)[0])
            self._target = self._last_command = measured
            self._target_at = now
            self._origin = measured[0].copy()
            self._session_parameters = None
            self._sdk_active = self._stopping = False
            # As in the teleoperation: nothing is published while enable is pending, and a lost enable reply does not
            # prove that the controller stayed disarmed (so stop() still runs).
            self._armed = self.execute
            self._streaming = not self.execute
        if not self.execute:
            logger.info("dry run: streaming targets on %s/preview_command only", self.prefix)
            return
        try:
            response = self._call("enable", self._ros["SetBool"].Request(data=True), service_timeout)
        except RobotError as error:
            with self._lock:
                self._halt_locked(f"enable failed: {error}")
            raise
        with self._lock:
            # The clutch was pressed while enable was pending: the teleoperation owns the session now.
            self._raise_if_fault_locked()
        if not response.success or "existing session retained" in response.message:
            with self._lock:
                self._armed = False
                self._target = None
            if response.success:
                raise RobotError("the controller session belongs to another client; not commanding or stopping it")
            raise RobotError(f"controller refused enable: {response.message}")
        logger.info("controller %s", response.message)
        with self._lock:
            self._target_at = time.monotonic()
            self._streaming = True
        deadline = time.monotonic() + active_timeout
        while True:
            self.heartbeat()
            with self._lock:
                if self._sdk_active:
                    return
            if time.monotonic() > deadline:
                with self._lock:
                    self._halt_locked("controller did not start the RT session")
                raise RobotError(f"controller did not report active within {active_timeout:.0f} s ({self.status!r})")
            time.sleep(0.01)

    def stop(self, timeout: float = 10.0) -> None:
        """Ends the command stream; if this client armed the controller, stops it and waits for ``stopped``."""
        with self._lock:
            self._streaming = False
            self._target = None
            if not self._armed or not self._ros:
                return
            # The controller leaves the RT session now (e.g. "monitoring:"); that is not a fault of this stop.
            self._stopping, self._sdk_active = True, False
        self._confirm_stop(timeout)

    def stop_controller(self, timeout: float = 10.0) -> None:
        """Stops the controller whoever commands it (DAgger: a session the teleoperation holds after its clutch)."""
        with self._lock:
            self._streaming = False
            self._target = None
            if self._armed:
                self._stopping, self._sdk_active = True, False
        self._confirm_stop(timeout)

    def _confirm_stop(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        message = ""
        while True:
            try:
                response = self._call("stop", self._ros["Trigger"].Request(), 2.0)
                message = response.message
            except RobotError as error:
                message = str(error)
            if message.startswith("stopped"):
                with self._lock:
                    self._armed = self._sdk_active = self._stopping = False
                logger.info("controller %s", message)
                return
            if message.startswith("stop failed:"):
                raise RobotError(f"controller {message}")
            if time.monotonic() > deadline:
                raise RobotError(
                    f"controller did not confirm the stop within {timeout:.0f} s ({message}); check the robot"
                )
            time.sleep(0.2)

    def recenter_workspace(self, timeout: float, service_timeout: float) -> bool:
        """Re-centres the active session's workspace sphere at the measured TCP without restarting the session.

        Uses the node's ``recenter_workspace`` service (once per session; the RT loop accepts it after a stationary,
        zero-wrench hold of 0.3 s), as the recording project's Cartesian replay does. Returns False when the node does
        not offer it or did not confirm it within ``timeout``.
        """
        with self._lock:
            self._raise_if_fault_locked()
            if not (self.execute and self._sdk_active):
                raise RobotError("workspace recenter needs an active session of this client")
            supported = (self._metadata or {}).get("workspace_recenter") == WORKSPACE_RECENTER
        if not supported or not self._ros["recenter"].service_is_ready():
            return False
        deadline = time.monotonic() + timeout
        while True:
            self.heartbeat()
            response = self._call("recenter", self._ros["Trigger"].Request(), service_timeout, idle=self.heartbeat)
            try:
                reply = json.loads(response.message) if response.success else {}
            except ValueError:
                reply = {}
            if reply.get("state") == "recentered":
                origin = np.asarray(reply.get("origin"), dtype=float)
                if origin.shape != (6,) or not np.isfinite(origin).all():
                    raise RobotError(f"invalid workspace recenter reply: {response.message}")
                with self._lock:
                    self._origin = origin[:3].copy()
                logger.info("workspace sphere re-centred at the measured TCP %s", np.round(origin[:3], 4))
                return True
            if reply.get("state") != "pending":
                raise RobotError(f"workspace recenter failed: {response.message}")
            if time.monotonic() > deadline:
                logger.warning(
                    "workspace recenter not confirmed within %.0f s (the RT loop needs a still arm at its target)",
                    timeout,
                )
                return False
            sleep_until(time.monotonic() + 0.05, self.heartbeat)

    def _clamp_radius(self) -> float:
        return max(self.limits.workspace_radius - self.limits.workspace_margin, 0.0)

    # DAgger / recording.
    def check_teleoperation(self, timeout: float) -> None:
        """DAgger preconditions: the AR3 teleoperation runs for real with its clutch released, and a gripper
        teleoperation (if any) is the gated one (``scripts/ar3_dagger_gripper_teleop.sh``)."""
        topic = self.prefix + "/teleop/return_home_status"
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                status = self._teleop_status
            if status is not None:
                break
            if time.monotonic() > deadline:
                raise RobotError(
                    f"no AR3 teleoperation status on {topic} after {timeout:.0f} s: start the teleoperation "
                    "(scripts/prepare_ar3_inference.sh --dagger) with the same ROS_DOMAIN_ID"
                )
            time.sleep(0.05)
        if status.get("dry_run") is not False:
            raise RobotError(
                f"the AR3 teleoperation is in dry run mode, so it cannot take over: ros2 param set /{TELEOP_NODE} "
                "dry_run false (scripts/prepare_ar3_inference.sh --dagger does this)"
            )
        if status.get("clutch_pressed") or self.clutch_pressed:
            raise RobotError("the teleoperation clutch is pressed: release it before starting")
        node = self._ros["node"]
        names = {name for name, _ in node.get_node_names_and_namespaces()}
        publishers = {info.node_name for info in node.get_publishers_info_by_topic(GRIPPER_ACTION_TOPIC)}
        ungated = sorted((names & UNGATED_GRIPPER_NODES) | (publishers - {GRIPPER_GATE_NODE}))
        if ungated:
            raise RobotError(
                f"an ungated gripper teleoperation runs ({', '.join(ungated)}): it would fight the policy. Stop it "
                "(e.g. lerobot's scripts/ar3_gripper_teleop.sh) and run scripts/ar3_dagger_gripper_teleop.sh instead"
            )
        if GRIPPER_GATE_NODE not in names:
            logger.warning(
                "no gated gripper teleoperation (scripts/ar3_dagger_gripper_teleop.sh) is running: during takeovers "
                "the gripper keeps the policy's last target"
            )

    @property
    def clutch_pressed(self) -> bool:
        with self._lock:
            return self._clutch

    def takeover(self) -> tuple[float | None, float | None, int]:
        """Host times of the current takeover's clutch press and release, and the teleoperation commands since."""
        with self._lock:
            return self._takeover_at, self._released_at, self._teleop_commands

    def end_takeover(self) -> None:
        """Hands the robot back to the policy after a takeover (call once the controller is idle again)."""
        now = time.monotonic()
        with self._lock:
            if self._takeover_at is None:
                return
            if self._released_at is None:
                self._add_locked("control", Sample(now, POLICY_FLAG))
            self._takeover_at = self._released_at = None
            self._teleop_commands = 0
            if "human_gripper" in self._streams:
                self._streams["human_gripper"].clear()
        self._publish_human_control(active=False)

    def recording_samples(self, t: float) -> dict[str, Sample | None]:
        """The input samples describing host time ``t`` (``record``)."""
        with self._lock:
            streams = self._streams
            samples = {name: streams[name].at(t) for name in ("state", "tau_m", "accepted", "commanded")}
            samples.update({name: streams[name].before(t) for name in ("control", "teleop", "human_gripper")})
            samples.update({name: streams[name].nearest(t) for name in ("front", "wrist")})
        return samples

    def _raise_if_fault_locked(self) -> None:
        if self._fault is not None:
            raise RobotError(self._fault)
        if self._takeover_at is not None:
            raise HumanTakeover

    def _sample_locked(self, now: float, wrench_dim: int, max_age: float) -> tuple[np.ndarray, np.ndarray]:
        """Measured pose [6] and force reading [wrench_dim]: TCP wrench (+ ``tau_m``), as recorded."""
        if self._state is None or now - self._state_at > max_age:
            raise RobotError(f"controller state is stale ({now - self._state_at:.3f} s > {max_age} s)")
        parts = [self._state[6:12]]
        if wrench_dim == 6 + TAU_M_SIZE:
            if self._tau_m is None or now - self._tau_m_at > max_age:
                raise RobotError(f"{self.prefix}/tau_m is stale ({now - self._tau_m_at:.3f} s > {max_age} s)")
            parts.append(self._tau_m)
        elif wrench_dim != 6:
            raise RobotError(f"unsupported policy wrench_dim {wrench_dim}: AR3 readings have 6 (wrench) or 13 values")
        return self._state[:6].copy(), np.concatenate(parts).astype(np.float32)


def clamp_to_sphere(position: Sequence[float], center: np.ndarray, radius: float) -> tuple[np.ndarray, bool]:
    position = np.asarray(position, dtype=float)
    offset = position - center
    distance = float(np.linalg.norm(offset))
    if distance <= radius:
        return position.copy(), False
    return center + offset * (radius / distance), True


def parameter_value(value: Any) -> Any:
    """rcl_interfaces/ParameterValue -> Python value (None when the parameter is not set)."""
    fields = {
        1: "bool_value",
        2: "integer_value",
        3: "double_value",
        4: "string_value",
        6: "bool_array_value",
        7: "integer_array_value",
        8: "double_array_value",
        9: "string_array_value",
    }
    if value.type not in fields:
        return None
    result = getattr(value, fields[value.type])
    return list(result) if value.type >= 6 else result


def _reading(reading: Any) -> Any:
    return dict(reading) if isinstance(reading, dict) else np.asarray(reading, dtype=np.float32)


def stack_readings(rows: Sequence[Any]) -> dict[str, np.ndarray]:
    """Per-step readings -> request keys [T, ...]: force readings -> ``wrench`` (LIFT); dicts -> one array per key."""
    if isinstance(rows[0], dict):
        return {key: np.stack([np.asarray(row[key]) for row in rows]) for key in rows[0]}
    return {"wrench": np.stack(rows)}


def sensor_reading(robot: Any, force: np.ndarray, gripper_position: float, config: "RunConfig") -> Any:
    """One per-step reading of the client protocol (``ar3_lift_policy``): the force reading for LIFT policies; for
    muscleVLTA policies the dict of its reading keys, with the Tac G2 fingers and the gripper opening if used."""
    if not (config.sensor_history or config.tactile):
        return force
    reading = {"wrench": np.asarray(force, dtype=np.float32), "sensor_valid": np.True_}
    if config.tactile:
        depth, valid, wrench = robot.hand_reading(max_age=config.max_tactile_age)
        if config.mask_tactile:
            depth, valid, wrench = np.zeros_like(depth), np.zeros_like(valid), np.zeros_like(wrench)
        reading.update(
            tactile_depth=depth,
            tactile_valid=valid,
            tactile_wrench=wrench,
            gripper_aperture=np.float32(gripper_position),
        )
    return reading


def trace_reading(reading: Any) -> dict:
    """Trace fields of a reading: the force reading, and the Tac G2 values with only the peak of each map."""
    if not isinstance(reading, dict):
        return {"reading": reading}
    fields = {"reading": reading["wrench"]}
    if "tactile_depth" in reading:
        fields["tactile_peak"] = reading["tactile_depth"].max(axis=(-2, -1))
    fields.update(
        (key, reading[key]) for key in ("tactile_valid", "tactile_wrench", "gripper_aperture") if key in reading
    )
    return fields


def tactile_summary(reading: Any) -> str:
    """Chunk log suffix with the last Tac G2 reading: indentation peak per finger (- = missing) and gripper opening."""
    if not isinstance(reading, dict) or "tactile_depth" not in reading:
        return ""
    peaks = reading["tactile_depth"].max(axis=(-2, -1))
    fingers = " ".join(
        f"{peak:.2f}" if valid else "-" for peak, valid in zip(peaks, reading["tactile_valid"], strict=True)
    )
    return f" | Tac G2 peak [{fingers}] at gripper {float(reading['gripper_aperture']):.0f}"


class SensorHistory:
    """Readings of the last ``length`` control steps of the episode: the history rows of muscleVLTA requests."""

    def __init__(self, length: int) -> None:
        self.length = length
        self._rows: collections.deque[dict] = collections.deque(maxlen=length)

    def add(self, reading: Any) -> None:
        if self.length:
            self._rows.append(reading)

    def rows(self, current: Any) -> list[dict]:
        """The ``length`` readings before ``current``. Steps before the episode start repeat the oldest reading marked
        missing (``sensor_valid`` False), as training clamps history frames to the episode start."""
        if not self.length:
            return []
        rows = list(self._rows)
        padding = {**(rows[0] if rows else current), "sensor_valid": np.False_}
        return [padding] * (self.length - len(rows)) + rows


class ChunkPlanner:
    """Fetches the actions of the current chunk from the policy server; a worker thread owns the websocket.

    ``reactive`` (LIFT Algorithm 1): ``reactive_replan`` sends the chunk-start observation with force reading 0 and
    returns actions 0..latency. Reactive action ``r`` only attends to readings ``<= r - latency``, so every new
    reading ``n`` makes actions up to ``n + latency`` computable; they are requested with ``reactive_step`` and all
    readings of the chunk so far. Each action therefore sees exactly the readings of LIFT's synchronous loop while
    ``latency`` control steps hide the inference time (a slow server automatically requests several actions at once).
    ``infer``: one ``infer`` call per chunk (non-reactive models, or a reactive model without step-wise re-planning).
    """

    def __init__(self, policy: Any, *, reactive: bool, horizon: int, latency: int) -> None:
        if horizon < 1 or not 0 <= latency < horizon:
            raise PolicyError(f"invalid action horizon {horizon} / cross-attention latency {latency}")
        self._policy = policy
        self.reactive = reactive
        self.horizon = horizon
        self.latency = latency
        self.last_infer_ms = math.nan
        self.requests = 0
        self._cond = threading.Condition()
        self._chunk = 0
        self._observation: dict | None = None
        self._history: list[dict] = []
        self._readings: list[Any] = []
        self._actions: list[np.ndarray] = []
        self._sent_first = False
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="policy", daemon=True)
        self._thread.start()

    def begin_chunk(self, observation: dict, reading: Any, history: Sequence[dict] = ()) -> None:
        """Starts a new chunk. ``reading``: chunk-start reading (None for models without force input): a force reading,
        or a dict of the muscleVLTA reading keys. ``history``: the readings of the steps before the chunk
        (``SensorHistory.rows``), sent first in every request."""
        with self._cond:
            self._raise_locked()
            self._chunk += 1
            self._observation = observation
            self._history = list(history)
            self._readings = [] if reading is None else [_reading(reading)]
            self._actions = []
            self._sent_first = False
            self._cond.notify_all()

    def add_reading(self, reading: Any) -> None:
        """Adds the reading measured at the next step of the chunk."""
        with self._cond:
            self._raise_locked()
            if self._readings and len(self._readings) < self.horizon:
                self._readings.append(_reading(reading))
                self._cond.notify_all()

    def wait_action(self, index: int, timeout: float, idle: Any = None) -> np.ndarray:
        """Action ``index`` of the current chunk; ``idle()`` runs while waiting (e.g. the robot heartbeat)."""
        deadline = time.monotonic() + timeout
        while True:
            with self._cond:
                self._raise_locked()
                if len(self._actions) > index:
                    return self._actions[index]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PolicyError(f"policy action {index} of the chunk was not ready within {timeout:.2f} s")
                self._cond.wait(min(remaining, 0.02))
            if idle is not None:
                idle()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _raise_locked(self) -> None:
        if self._error is not None:
            raise PolicyError(f"policy server request failed: {self._error!r}")
        if self._closed:
            raise PolicyError("the policy planner is closed")

    def _next_request_locked(self) -> tuple[Any, int] | None:
        if self._observation is None:
            return None
        if not self._sent_first:
            self._sent_first = True
            observation = dict(self._observation)
            if self._readings:
                observation.update(stack_readings(self._history + self._readings[:1]))
            if self.reactive:
                return functools.partial(self._policy.reactive_replan, observation, k=self.latency + 1), 0
            return functools.partial(self._policy.infer, observation), 0
        known = len(self._actions)
        if not self.reactive or not self._readings or known == 0:
            return None
        computable = min(self.horizon - 1, len(self._readings) - 1 + self.latency)
        if known > computable:
            return None
        readings = stack_readings(self._history + self._readings)
        return functools.partial(
            self._policy.reactive_step, readings, k=computable - known + 1, current_step=known
        ), known

    def _run(self) -> None:
        while True:
            with self._cond:
                request = None
                while not self._closed and (request := self._next_request_locked()) is None:
                    self._cond.wait()
                if self._closed:
                    return
                chunk = self._chunk
            call, start = request
            try:
                response = call()
                actions = np.asarray(response["actions"], dtype=float)
                if actions.ndim != 2 or len(actions) == 0 or actions.shape[1] < 7 or not np.isfinite(actions).all():
                    raise ValueError(f"invalid actions of shape {actions.shape} (expected [k >= 1, 7], finite)")
                if int(response.get("start_step", start)) != start:
                    raise ValueError(f"server returned step {response.get('start_step')}, expected {start}")
            except Exception as error:
                with self._cond:
                    self._error = error
                    self._cond.notify_all()
                return
            with self._cond:
                self.requests += 1
                self.last_infer_ms = float(response.get("policy_timing", {}).get("infer_ms", math.nan))
                if chunk == self._chunk:  # else the chunk ended while the request was in flight
                    self._actions.extend(actions[: self.horizon - start])
                    self._cond.notify_all()


@functools.cache
def _image_tools() -> Any:
    try:
        from openpi_client import image_tools
    except ImportError as error:  # PIL missing: the server resizes the full frames identically
        logger.warning("client-side resize disabled (%s); sending full frames", error)
        return None
    return image_tools


def policy_observation(observation: Observation, gripper: float, prompt: str, resize: int) -> dict:
    """LIFT client keys (``AR3LiftInputs``): state, wrist -> left wrist image, front -> base image, prompt."""
    front, wrist = observation.front, observation.wrist
    tools = _image_tools() if resize else None
    if tools is not None:
        front, wrist = (tools.resize_with_pad(image, resize, resize) for image in (front, wrist))
    return {
        "observation/state": policy_state(observation.pose, gripper),
        "observation/left_wrist_image": wrist,
        "observation/base_image": front,
        "prompt": prompt,
    }


class Trace:
    """Optional JSON-lines log of every executed step."""

    def __init__(self, path: str | None) -> None:
        self._file = None
        if path:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def write(self, **record: Any) -> None:
        if self._file is not None:
            self._file.write(json.dumps(record, default=_json_default) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def sleep_until(deadline: float, idle: Any, *, clock: Any = time.monotonic, sleep: Any = time.sleep) -> None:
    while True:
        idle()
        remaining = deadline - clock()
        if remaining <= 0:
            return
        sleep(min(remaining, 0.02))


@dataclasses.dataclass
class RunConfig:
    prompt: str = DEFAULT_PROMPT
    wrench_dim: int = 13  # force reading width of the model (0: no force input)
    max_chunks: int = 30
    period: float = 0.1
    resize: int = 224
    first_action_timeout: float = 2.0
    policy_timeout: float = 1.0
    max_observation_age: float = 0.25
    max_image_age: float = 0.2
    command_gripper: bool = False
    assumed_gripper: float = GRIPPER_OPEN
    sensor_history: int = 0  # muscleVLTA: readings of the steps before each chunk sent with it (0: LIFT requests)
    tactile: bool = False  # muscleVLTA: every reading carries the Tac G2 fingers and the gripper opening
    max_tactile_age: float = MAX_TACTILE_AGE
    # Ablation: the policy sees both fingers as missing (as under finger dropout); the gripper opening stays.
    mask_tactile: bool = False


def run_chunks(
    robot: Any,
    planner: ChunkPlanner,
    gripper: Any,
    config: RunConfig,
    trace: Trace,
    *,
    clock: Any = time.monotonic,
    sleep: Any = time.sleep,
    should_stop: Any = None,
) -> int:
    """Closed loop at ``1 / period`` Hz, one action chunk at a time (as recorded). Returns the completed chunks.

    ``should_stop()`` is asked before every chunk (e.g. whether the operator ended the episode). Every call is one
    episode: the sensor history of muscleVLTA policies starts empty.
    """
    chunks = 0
    history = SensorHistory(config.sensor_history)
    # Readings are measured every step when the policy reacts to them or keeps them as the next chunks' history.
    measure = bool(config.wrench_dim) and (planner.reactive or config.sensor_history > 0)
    while config.max_chunks <= 0 or chunks < config.max_chunks:
        if should_stop is not None and should_stop():
            break
        observation = robot.observe(
            wrench_dim=config.wrench_dim or 6, max_age=config.max_observation_age, max_image_age=config.max_image_age
        )
        gripper_now = gripper.read() if gripper is not None else config.assumed_gripper
        request = policy_observation(observation, gripper_now, config.prompt, config.resize)
        state = request["observation/state"]
        reading = observation.reading
        if config.wrench_dim:
            reading = sensor_reading(robot, reading, gripper_now, config)
            planner.begin_chunk(request, reading, history.rows(reading))
            history.add(reading)
        else:
            planner.begin_chunk(request, None)
        relative = planner.wait_action(0, config.first_action_timeout, robot.heartbeat)
        t0 = clock()
        clamped_steps = 0
        for step in range(planner.horizon):
            if step:
                sleep_until(t0 + step * config.period, robot.heartbeat, clock=clock, sleep=sleep)
                reading = None
                if measure:
                    force = robot.force_reading(wrench_dim=config.wrench_dim, max_age=config.max_observation_age)
                    aperture = gripper.read() if config.tactile and gripper is not None else config.assumed_gripper
                    reading = sensor_reading(robot, force, aperture, config)
                    history.add(reading)
                    if planner.reactive:
                        planner.add_reading(reading)
                relative = planner.wait_action(step, config.policy_timeout, robot.heartbeat)
            position, rotation = absolute_target(relative, state)
            clamped = robot.set_target(position, rotation)
            clamped_steps += clamped
            if config.command_gripper and gripper is not None:
                gripper.command(float(relative[6]))
            trace.write(
                chunk=chunks,
                step=step,
                t=clock(),
                state=state,
                action=relative[:7],
                target=np.concatenate([position, matrix_rpy(rotation), relative[6:7]]),
                clamped=clamped,
                **trace_reading(reading),
                infer_ms=planner.last_infer_ms,
            )
        sleep_until(t0 + planner.horizon * config.period, robot.heartbeat, clock=clock, sleep=sleep)
        chunks += 1
        logger.info(
            "chunk %d: state xyz [%.3f %.3f %.3f] gripper %.0f -> %.0f | %d policy requests, last %.0f ms%s%s",
            chunks,
            *state[:3],
            state[6],
            relative[6],
            planner.requests,
            planner.last_infer_ms,
            f" | {clamped_steps} targets clamped to the workspace" if clamped_steps else "",
            tactile_summary(reading),
        )
    return chunks


def home_limits(limits: Limits) -> Limits:
    """The recording's separate homing profile, bounded by this controller's tracking and speed limits."""
    return dataclasses.replace(
        limits,
        max_lead_m=min(limits.max_lead_m, 0.06),
        max_lead_rad=min(home.HOME_ROTATION_LEAD, 0.8 * limits.max_tracking_error_rad),
        max_linear_speed=min(limits.max_linear_speed, home.HOME_LINEAR_SPEED),
        max_angular_speed=min(limits.max_angular_speed, home.HOME_ANGULAR_SPEED),
    )


def plan_home(start, target, limits: Limits, timeout: float) -> home.ReturnHomePath:
    """Reject invalid or unreachable startup requests before enabling the controller."""
    if pose_error(start, target)[0] > limits.workspace_radius - limits.workspace_margin:
        raise RobotError("saved home is outside the available controller workspace from the current TCP")
    profile = home_limits(limits)
    try:
        path = home.ReturnHomePath(start, target, profile.max_linear_speed, profile.max_angular_speed)
    except ValueError as error:
        raise RobotError(f"invalid return-home path: {error}") from error
    if path.duration + home.HOME_SETTLE_TIME >= timeout:
        raise RobotError(
            f"return-home path needs at least {path.duration + home.HOME_SETTLE_TIME:.1f} s; "
            f"--move-timeout is only {timeout:.1f} s"
        )
    return path


def move_to_start(
    robot: Any,
    gripper: Any,
    start_pose: Sequence[float],
    *,
    timeout: float,
    clock: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> None:
    """Follow the recording's quintic path, pausing for feedback and requiring measured arrival."""
    target = pose_from_vector(start_pose)
    measured, command, _ = robot.poses()
    path = plan_home(command if command is not None else measured, target, robot.limits, timeout)
    distance, angle = pose_error(measured, target)
    logger.info(
        "returning to saved home: %.0f mm / %.3f rad away; smooth path %.1f s, timeout %.0f s; "
        "measured arrival required within %.0f mm / %.3f rad for %.1f s",
        distance * 1000,
        angle,
        path.duration,
        timeout,
        home.HOME_POSITION_TOLERANCE * 1000,
        home.HOME_ROTATION_TOLERANCE,
        home.HOME_SETTLE_TIME,
    )
    with robot.return_home_mode() as limits:
        if gripper is not None:
            gripper.command(GRIPPER_OPEN)
        started = previous = clock()
        report_at, settled_at = started, None
        period = 1.0 / limits.rate_hz
        while True:
            robot.heartbeat()
            measured, command, _ = robot.poses()
            now = clock()
            dt, previous = min(max(now - previous, 0.0), period), now
            distance, angle = pose_error(measured, target)
            if now - started >= timeout:
                raise RobotError(
                    f"saved home not reached within {timeout:.0f} s "
                    f"({distance * 1000:.1f} mm / {angle:.3f} rad left); policy execution was not started"
                )
            candidate, elapsed = path.candidate(dt)
            lead_m, lead_rad = pose_error(candidate, measured)
            paused = lead_m > limits.max_lead_m + 1e-9 or lead_rad > limits.max_lead_rad + 1e-9
            if not paused:
                robot.set_target(*candidate, clamp=False)
                path.elapsed = elapsed
            # Do not mistake a stationary arm or a sent command for arrival at the saved pose.
            arrived = (
                path.elapsed >= path.duration
                and command is not None
                and max(pose_error(command, target)) < 1e-5
                and distance <= home.HOME_POSITION_TOLERANCE
                and angle <= home.HOME_ROTATION_TOLERANCE
            )
            if not arrived:
                settled_at = None
            elif settled_at is None:
                settled_at = now
            elif now - settled_at >= home.HOME_SETTLE_TIME:
                break
            if now >= report_at:
                logger.info(
                    "return-home %.0f%%: %.1f mm / %.3f rad remaining%s",
                    100 * path.elapsed / path.duration,
                    distance * 1000,
                    angle,
                    "; waiting for measured TCP to follow" if paused else "",
                )
                report_at = now + 2.0
            sleep(period)
        if gripper is not None:
            deadline = clock() + 5.0
            while True:
                robot.heartbeat()
                gripper.command(GRIPPER_OPEN)  # Also collects the asynchronous command acknowledgement/errors.
                position = gripper.read()
                if abs(position - GRIPPER_OPEN) <= 20:
                    break
                if clock() >= deadline:
                    raise GripperError(
                        f"gripper did not open for inference (position {position:.0f}); not starting policy"
                    )
                sleep(0.05)
        distance, angle = pose_error(robot.poses()[0], target)
        if distance > home.HOME_POSITION_TOLERANCE or angle > home.HOME_ROTATION_TOLERANCE:
            raise RobotError("TCP left the saved-home tolerance while waiting for the gripper; not starting policy")
        logger.info("saved home reached: measured error %.1f mm / %.3f rad; gripper ready", distance * 1000, angle)


def frame_values(samples: dict[str, Sample | None], t: float) -> tuple[dict[str, Any], list[str]]:
    """Values of the recorded frame describing host time ``t`` (``EpisodeRecorder.add_frame``, without images).

    ``samples``: ``Ar3Robot.recording_samples(t)`` + ``gripper`` (value: position, policy target). Also returns the
    inputs that were missing or stale at ``t``. ``control_flag`` is -1 (trained on by LIFT) only while the operator
    has taken over, the teleoperation is commanding and every input is fresh.
    """
    state, gripper = samples["state"], samples["gripper"]
    if state is None or gripper is None or samples["front"] is None or samples["wrist"] is None:
        raise RobotError("recording needs the controller state, the gripper position and both cameras")

    def age(sample: Sample | None) -> float:
        return math.inf if sample is None else abs(t - sample.received)

    limits = {"state": "state", "tau_m": "tau_m", "gripper": "gripper", "front": "camera", "wrist": "camera"}
    problems = [name for name, limit in limits.items() if age(samples[name]) > STALE_LIMITS[limit]]
    # The accepted TCP target, as in the recordings; else this client's command or the measured pose.
    action = samples["accepted"]
    if age(action) > STALE_LIMITS["action"]:
        problems.append("action")
        action = samples["commanded"] if age(samples["commanded"]) <= STALE_LIMITS["action"] else state
    position, policy_target = gripper.value
    control, teleop, human_gripper = samples["control"], samples["teleop"], samples["human_gripper"]
    human = control is not None and control.value == HUMAN_FLAG
    if human and human_gripper is not None:
        target, target_sample = human_gripper.value, human_gripper
    else:
        target, target_sample = (position if policy_target is None else policy_target), gripper
    teleoperated = human and teleop is not None and t - teleop.received <= TELEOP_ACTIVITY
    tau_m = samples["tau_m"]
    values: dict[str, Any] = {
        "observation.tcp_pose": pose_rotvec(state.value[:6]),
        "observation.gripper_position": [position],
        "observation.tcp_wrench": state.value[6:12],
        "actions": [*pose_rotvec(action.value[:6]), target],
        "control_flag": [HUMAN_FLAG if teleoperated and not problems else POLICY_FLAG],
        "observation.tau_m": np.zeros(TAU_M_SIZE) if tau_m is None else tau_m.value,
        "sync.target_host_s": [t],
        "sync.tcp_pose.timing_s": [state.received, 0.0],
        "sync.tcp_wrench.timing_s": [state.received, 0.0],
        "sync.actions.timing_s": [action.received, 0.0, 0.0],
        "sync.gripper_position.timing_s": [gripper.received, gripper.stamp],
        "sync.gripper_action.timing_s": [target_sample.received, target_sample.stamp],
        "sync.control_flag.timing_s": [t if control is None else control.received, 0.0],
        "sync.tau_m.timing_s": [0.0 if tau_m is None else tau_m.received, 0.0],
    }
    for name in ("front", "wrist"):
        frame = samples[name]
        values[f"sync.{name}.timing_s"] = [frame.received, frame.stamp, frame.received - t]
        values[f"sync.{name}.frame_index"] = [frame.index]
    return values, problems


def hand_frame(hand: Sample | None, t: float, max_age: float) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    """Tac G2 part of the recorded frame describing host time ``t``: the finger wrench columns and the reading of
    ``EpisodeRecorder.add_frame`` (pooled maps [2, 48, 64], finger validity [2]).

    ``hand``: the newest ``Ar3Robot.hand_snapshot`` taken at or before ``t``. A finger whose messages arrived more
    than ``max_age`` before ``t`` is recorded as missing, with zeros, as the policy is sent it.
    """
    tactile = tactile_io()
    fingers = len(tactile.SENSORS)
    depth = np.zeros((fingers, *tactile.DEPTH_SHAPE), dtype=np.float32)
    wrench = np.zeros((fingers, tactile.WRENCH_DIM), dtype=np.float32)
    valid = np.zeros(fingers, dtype=bool)
    received = np.full(fingers, -math.inf)
    if hand is not None:
        maps, fresh, wrenches, received = hand.value
        valid = np.asarray(fresh, dtype=bool) & (t - received <= max_age)
        depth[valid] = maps[valid]
        wrench[valid] = np.reshape(wrenches, (fingers, -1))[valid]
    values: dict[str, Any] = {}
    for index, sensor in enumerate(tactile.SENSORS):
        values[f"observation.gripper_wrench.{sensor}"] = wrench[index]
        values[f"sync.gripper_wrench.{sensor}.timing_s"] = [
            received[index] if math.isfinite(received[index]) else 0.0,
            0.0,
        ]
    return values, (depth, valid)


def human_chunks(flags: Sequence[int], horizon: int) -> int:
    """Frames LIFT trains on (``intervention_only``): the next ``horizon`` flags (up to the episode end) are human."""
    count = run = 0
    for index in range(len(flags) - 1, -1, -1):
        run = run + 1 if flags[index] == HUMAN_FLAG else 0
        count += run >= min(horizon, len(flags) - index)
    return count


class RolloutRecorder:
    """Records episodes with an ``ar3_lift_recorder.EpisodeRecorder`` (``writer``).

    Frame ``k`` describes host time ``T = start + k / fps`` and is assembled ``SYNC_DELAY`` after ``T`` from the
    robot's input history and 20 Hz polls of the gripper (position + the policy's current target) and, with
    ``tactile``, of the Tac G2 fingers (``hand_frame``: the reading the policy is sent, up to a poll period older).
    """

    def __init__(
        self,
        robot: Any,
        writer: Any,
        gripper: Any,
        *,
        fps: int = RECORD_FPS,
        horizon: int = 10,
        poll_hz: float = 20.0,
        tactile: bool = False,
        max_tactile_age: float = MAX_TACTILE_AGE,
    ) -> None:
        self.writer = writer
        self._robot = robot
        self._gripper = gripper
        self._fps = fps
        self._horizon = horizon
        self._tactile = tactile
        self._max_tactile_age = max_tactile_age
        self._lock = threading.Lock()
        self._gripper_samples = Stream(64)
        self._hand_samples = Stream(64)
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._until = math.inf
        self._error: Exception | None = None
        self._flags: list[int] = []
        self._stale: collections.Counter[str] = collections.Counter()
        self._poller = threading.Thread(target=self._poll, args=(1.0 / poll_hz,), name="gripper-poll", daemon=True)
        self._poller.start()
        if tactile:
            # A thread of its own: a slow gripper request must not make the fingers look missing.
            threading.Thread(target=self._poll_hand, args=(1.0 / poll_hz,), name="hand-poll", daemon=True).start()

    @property
    def failed(self) -> bool:
        return self._error is not None

    def _poll(self, period: float) -> None:
        failing = False
        while True:
            started = time.monotonic()
            try:
                position = self._gripper.read()
            except GripperError as error:
                if not failing:
                    logger.warning("recording: %s", error)
                failing = True
            else:
                failing = False
                with self._lock:
                    self._gripper_samples.add(Sample(time.monotonic(), (position, self._gripper.target), time.time()))
            if self._closed.wait(max(period - (time.monotonic() - started), 0.0)):
                return

    def _poll_hand(self, period: float) -> None:
        while True:
            started = time.monotonic()
            reading = self._robot.hand_snapshot(max_age=self._max_tactile_age)
            with self._lock:
                # Stamped after the snapshot: everything in it had arrived by then.
                self._hand_samples.add(Sample(time.monotonic(), reading, time.time()))
            if self._closed.wait(max(period - (time.monotonic() - started), 0.0)):
                return

    def start(self) -> None:
        """Starts an episode now."""
        self.writer.start_episode()
        self._flags, self._stale, self._error, self._until = [], collections.Counter(), None, math.inf
        self._thread = threading.Thread(target=self._record, args=(time.monotonic(),), name="recorder", daemon=True)
        self._thread.start()

    def _record(self, start: float) -> None:
        index = 0
        try:
            while True:
                t = start + index / self._fps
                if t > self._until:
                    return
                remaining = t + SYNC_DELAY - time.monotonic()
                if remaining > 0:
                    time.sleep(min(remaining, 0.02))
                    continue
                self._add(t)
                index += 1
        except Exception as error:
            logger.error("recording failed: %s", error)
            self._error = error

    def _add(self, t: float) -> None:
        samples = self._robot.recording_samples(t)
        with self._lock:
            samples["gripper"] = self._gripper_samples.at(t)
            hand = self._hand_samples.before(t)
        values, problems = frame_values(samples, t)
        images = {name: rgb_image(samples[name].value) for name in ("front", "wrist")}
        if self._tactile:
            hand_values, reading = hand_frame(hand, t, self._max_tactile_age)
            values.update(hand_values)
            self.writer.add_frame(values, images, reading)
            if not reading[1].all():
                self._stale["tactile"] += 1
        else:
            self.writer.add_frame(values, images)
        self._flags.append(int(values["control_flag"][0]))
        self._stale.update(problems)

    def stop(self, until: float) -> dict[str, Any]:
        """Ends the episode with the last frame at or before host time ``until``; returns its summary."""
        self._until = until
        self._join()
        if self._error is not None:
            self.writer.discard_episode()
            raise RobotError(f"recording failed: {self._error}") from self._error
        return {
            "frames": len(self._flags),
            "human": self._flags.count(HUMAN_FLAG),
            "chunks": human_chunks(self._flags, self._horizon),
            "stale": dict(self._stale),
        }

    def abort(self) -> None:
        """Ends and discards the current episode."""
        self._until = -math.inf
        self._join()
        self.writer.discard_episode()

    def save(self, outcome: str | None = None) -> int:
        """Saves the episode; ``outcome`` ("success" / "failure") also records how it ended, for offline RL."""
        try:
            index = self.writer.save_episode()
            if outcome is not None:
                self.writer.label_episode(index, outcome)
        except Exception as error:
            raise RobotError(f"saving the episode failed: {error}") from error
        return index

    def discard(self) -> None:
        self.writer.discard_episode()

    def close(self) -> None:
        """Discards an unsaved episode and stops the gripper poll."""
        self._closed.set()
        self.abort()
        self._poller.join(timeout=2.0)
        self.writer.close()

    def _join(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None


class Console:
    """Operator input: Enter presses (``poll``) and answers (``ask``). Without a terminal, answers are the defaults."""

    def __init__(self, stream: Any = None) -> None:
        stream = sys.stdin if stream is None else stream
        self.interactive = bool(stream is not None and stream.isatty())
        self._lines: queue.Queue[str | None] = queue.Queue()
        if self.interactive:
            threading.Thread(target=self._read, args=(stream,), name="console", daemon=True).start()

    def _read(self, stream: Any) -> None:
        for line in stream:
            self._lines.put(line.strip().lower())
        self._lines.put(None)  # end of input

    def poll(self) -> bool:
        """Whether Enter was pressed since the last call."""
        pressed = False
        while True:
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                return pressed
            if line is None:
                self.interactive = False
            else:
                pressed = True

    def ask(self, prompt: str, default: str) -> str:
        self.poll()  # earlier input is no answer
        if not self.interactive:
            logger.info("%s -> %r (no terminal)", prompt, default)
            return default
        print(f"\n>>> {prompt}: ", end="", flush=True)
        while True:
            try:
                line = self._lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                self.interactive = False
                return default
            return line


CONTROLLER_PARAMETERS = (
    "enable_motion",
    "workspace_radius_m",
    "max_translation_step_m",
    "max_rotation_step_rad",
    "speed_mm_s",
    "max_velocity",
    "command_timeout_s",
)
RECENTER_TIMEOUT = 5.0
WARMUP_TIMEOUT = 300.0


def connect_policy(host: str, port: int, api_key: str | None, timeout: float) -> Any:
    """Connects to the policy server (``scripts/serve_ar3_lift.sh``), waiting up to ``timeout`` s for it to listen."""
    import socket

    try:
        from openpi_client import websocket_client_policy
    except ImportError as error:
        raise PolicyError(
            f"the policy client cannot be imported ({error}); install its dependencies into this Python: "
            f'{sys.executable} -m pip install "websockets>=11,<18" msgpack typing_extensions pillow'
        ) from error
    deadline = time.monotonic() + timeout
    while True:
        try:
            socket.create_connection((host, port), timeout=2.0).close()
            break
        except OSError as error:
            if time.monotonic() > deadline:
                raise PolicyError(
                    f"policy server {host}:{port} is not reachable ({error}): start scripts/serve_ar3_lift.sh"
                ) from error
            logger.info("waiting for the policy server at %s:%d ...", host, port)
            time.sleep(2.0)
    return websocket_client_policy.WebsocketClientPolicy(host, port, api_key)


def warm_up(robot: Any, planner: ChunkPlanner, gripper: Any, config: RunConfig) -> None:
    """Runs one replan + one step on the live observation (JIT compilation: ~15 s) before the arm is enabled."""
    started = time.monotonic()
    observation = robot.observe(
        wrench_dim=config.wrench_dim or 6, max_age=config.max_observation_age, max_image_age=config.max_image_age
    )
    gripper_now = gripper.read() if gripper is not None else config.assumed_gripper
    request = policy_observation(observation, gripper_now, config.prompt, config.resize)
    reading = None
    if config.wrench_dim:
        reading = sensor_reading(robot, observation.reading, gripper_now, config)
        planner.begin_chunk(request, reading, SensorHistory(config.sensor_history).rows(reading))
    else:
        planner.begin_chunk(request, None)
    planner.wait_action(0, WARMUP_TIMEOUT)
    if planner.reactive and config.wrench_dim and planner.latency + 1 < planner.horizon:
        planner.add_reading(reading)
        planner.wait_action(planner.latency + 1, WARMUP_TIMEOUT)
    logger.info(
        "policy warm-up done in %.1f s (last request %.0f ms)", time.monotonic() - started, planner.last_infer_ms
    )


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


class Session:
    """One client run: connects everything, runs the policy and always leaves the arm stopped."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.limits = Limits(
            rate_hz=args.rate_hz,
            max_lead_m=args.max_lead_m,
            max_lead_rad=args.max_lead_rad,
            max_linear_speed=args.max_linear_speed,
            max_angular_speed=args.max_angular_speed,
            max_tracking_error_m=args.max_tracking_error_m,
            max_tracking_error_rad=args.max_tracking_error_rad,
            feedback_timeout=args.feedback_timeout,
            target_timeout=args.target_timeout,
            workspace_margin=args.workspace_margin,
        )
        self.recording = args.record_dir is not None
        self.robot = Ar3Robot(
            execute=args.execute,
            limits=self.limits,
            prefix=args.controller_prefix,
            dagger=args.dagger,
            record=self.recording,
        )
        self.gripper: DMGripper | None = None
        self.planner: ChunkPlanner | None = None
        self.recorder: RolloutRecorder | None = None
        self.console: Console | None = None
        self.trace = Trace(args.trace)

    def run(self) -> int:
        args, robot = self.args, self.robot
        logger.info("%s run%s", "EXECUTE" if args.execute else "DRY", " (DAgger)" if args.dagger else "")
        logger.info(
            "Home target from %s: xyz[m], rpy[rad] %s",
            args.home_file or "explicit --start-pose",
            np.round(args.start_pose, 6),
        )
        metadata: dict = {}
        policy = None
        if not args.check:
            policy = connect_policy(args.host, args.port, args.api_key, args.connect_timeout)
            metadata = policy.get_server_metadata()
            logger.info("policy server metadata: %s", metadata)
        reactive_model = bool(metadata.get("reactive", False))
        wrench_dim = int(metadata.get("wrench_dim", 0)) if reactive_model else 0
        # muscleVLTA policies also take the readings of the steps before each chunk and, with tactile_in_use, the
        # Tac G2 fingers and the gripper opening at every step (sensor layout announced by scripts/serve_policy.py).
        sensor_history = int(metadata.get("sensor_history", 0)) if reactive_model else 0
        tactile = reactive_model and bool(metadata.get("tactile_in_use", False))
        mode = ("reactive" if reactive_model else "infer") if args.mode == "auto" else args.mode
        if mode == "reactive" and not reactive_model and not args.check:
            raise PolicyError("the served model is not force-reactive; use --mode infer")
        robot.start(
            front_topic=args.front_topic,
            wrist_topic=args.wrist_topic,
            controller_node=args.controller_node,
            clutch_topic=args.clutch_topic,
            tactile=tactile or args.check,
        )
        robot.wait_ready(
            args.ready_timeout,
            need_tau_m=args.check or self.recording or wrench_dim == 6 + TAU_M_SIZE,
            need_tactile=tactile,
        )
        self._configure_limits()
        if args.dagger:
            robot.check_teleoperation(args.ready_timeout)
        if not args.no_gripper:
            try:
                self.gripper = DMGripper.connect(
                    args.gripper_address,
                    writable=args.execute,
                    max_age=args.gripper_max_age,
                    speed=args.gripper_speed,
                    force=args.gripper_force,
                    deadband=args.gripper_deadband,
                )
            except ImportError as error:
                raise GripperError(
                    f"the DM gripper client is not importable ({error}): run with the gripper's ROS venv "
                    "(scripts/run_ar3_lift_robot.sh) or pass --no-gripper"
                ) from error
            logger.info("gripper position %.0f (0 = closed ... 1000 = open)", self.gripper.read())
        config = RunConfig(
            prompt=args.prompt,
            wrench_dim=wrench_dim,
            max_chunks=args.max_chunks,
            period=1.0 / args.control_hz,
            resize=args.resize,
            first_action_timeout=args.first_action_timeout,
            policy_timeout=args.policy_timeout,
            max_observation_age=args.max_observation_age,
            max_image_age=args.max_image_age,
            command_gripper=args.execute and self.gripper is not None,
            assumed_gripper=args.assumed_gripper,
            sensor_history=sensor_history,
            tactile=tactile,
            mask_tactile=args.mask_tactile,
        )
        if tactile and args.mask_tactile:
            logger.warning("--mask-tactile: the policy receives both Tac G2 fingers as missing (ablation run)")
        # Parameter/gripper setup can outlast the first camera samples. Wait for fresh observations while the
        # controller is still inactive; execution keeps the normal fail-fast sensor checks and age limits.
        robot.wait_ready(
            args.ready_timeout,
            need_tau_m=args.check or self.recording or wrench_dim == 6 + TAU_M_SIZE,
            need_tactile=tactile,
            max_observation_age=args.max_observation_age,
            max_image_age=args.max_image_age,
            max_tactile_age=config.max_tactile_age,
        )
        if args.check:
            self._report()
            return 0

        self.planner = ChunkPlanner(
            policy,
            reactive=mode == "reactive",
            horizon=int(metadata.get("action_horizon", 10)),
            latency=int(metadata.get("cross_attention_latency", 0)) if mode == "reactive" else 0,
        )
        logger.info(
            "mode %s: horizon %d, latency %d, force reading width %d%s%s, %.0f Hz, prompt %r",
            mode,
            self.planner.horizon,
            self.planner.latency,
            wrench_dim,
            f", {sensor_history} history steps" if sensor_history else "",
            ", Tac G2 fingers + gripper opening" if tactile else "",
            args.control_hz,
            args.prompt,
        )
        if args.warmup:
            warm_up(robot, self.planner, self.gripper, config)
        episodes = args.record_dir is not None or args.dagger or args.episodes != 1
        if args.execute and (args.move_to_start or episodes):
            plan_home(robot.poses()[0], pose_from_vector(args.start_pose), self.limits, args.move_timeout)
        if args.execute and self.gripper is not None:
            self.gripper.configure()
        if episodes:
            return self._run_episodes(config, metadata)
        robot.activate(service_timeout=args.service_timeout)
        if args.move_to_start and args.execute:
            move_to_start(robot, self.gripper, args.start_pose, timeout=args.move_timeout)
            if args.reanchor_at_start:
                self._reanchor()
        else:
            if args.move_to_start:
                logger.info("dry run: --move-to-start skipped")
            distance, angle = pose_error(robot.poses()[0], pose_from_vector(args.start_pose))
            if distance > 0.05 or angle > 0.15:
                logger.warning(
                    "the TCP is %.0f mm / %.2f rad from the start pose of the recordings; consider --move-to-start",
                    distance * 1000,
                    angle,
                )
        logger.info(
            "Starting policy execution from fresh observations%s",
            " after confirmed home arrival" if args.execute and args.move_to_start else "",
        )
        chunks = run_chunks(robot, self.planner, self.gripper, config, self.trace)
        logger.info("finished %d chunks", chunks)
        return 0

    # Episodes (--episodes, --record-dir, --dagger).
    def _run_episodes(self, config: RunConfig, metadata: dict) -> int:
        """Start pose -> policy (-> operator takeover) -> save / discard, until --episodes or the operator quits."""
        args = self.args
        self.console = Console()
        if self.recording:
            self._open_recorder(metadata, config)
        logger.info("%s", DAGGER_HELP if args.dagger else "Enter ends an episode early, Ctrl+C aborts.")
        if not args.execute:
            logger.warning("dry run: the arm stays where it is (episodes only exercise the policy and the recording)")
        done = 0
        while args.episodes == 0 or done < args.episodes:
            if done and self.console.ask("reset the scene; Enter = next episode, q = quit", "q") == "q":
                break
            if not self._start_episode():
                break
            summary = self._run_episode(config)
            done += 1
            if not self._keep_episode(summary):
                break
        logger.info("finished %d episode(s)", done)
        return 0

    def _open_recorder(self, metadata: dict, config: RunConfig) -> None:
        args = self.args
        try:
            if __package__:
                from . import ar3_lift_recorder
            else:
                import ar3_lift_recorder
            import av  # noqa: F401
            import pyarrow as pa  # noqa: F401
        except ImportError as error:
            raise RobotError(
                f"recording needs pyarrow and PyAV ({error}): {sys.executable} -m pip install pyarrow av"
            ) from error
        observation = self.robot.observe(
            wrench_dim=6, max_age=args.max_observation_age, max_image_age=args.max_image_age
        )
        if observation.front.shape != observation.wrist.shape:
            raise RobotError(
                f"recording needs equally sized cameras (front {observation.front.shape}, wrist "
                f"{observation.wrist.shape})"
            )
        capture_info = {
            "source": "scripts/ar3_lift_robot_client.py",
            "dagger": args.dagger,
            "control_flag": {"human": HUMAN_FLAG, "policy": POLICY_FLAG},
            "policy_server": json.loads(json.dumps(metadata, default=str)),
            "control_hz": args.control_hz,
            "gripper_speed": args.gripper_speed,
            "gripper_force": args.gripper_force,
            "sync_delay_s": SYNC_DELAY,
        }
        if config.tactile:
            # The fingers are recorded as measured, also when the policy was not sent them (--mask-tactile).
            capture_info["tactile"] = {"max_age_s": config.max_tactile_age, "masked_for_policy": config.mask_tactile}
        try:
            writer = ar3_lift_recorder.EpisodeRecorder(
                args.record_dir,
                task=args.prompt,
                fps=RECORD_FPS,
                image_size=observation.front.shape[:2],
                capture_info=capture_info,
                tactile=config.tactile,
            )
        except (ar3_lift_recorder.RecorderError, OSError, ValueError) as error:
            raise RobotError(f"cannot record to {args.record_dir}: {error}") from error
        self.recorder = RolloutRecorder(
            self.robot,
            writer,
            self.gripper,
            horizon=self.planner.horizon,
            tactile=config.tactile,
            max_tactile_age=config.max_tactile_age,
        )
        logger.info(
            "recording to %s (%d episodes so far%s)",
            writer.root,
            writer.num_episodes,
            ", with the Tac G2 fingers" if config.tactile else "",
        )

    def _start_episode(self) -> bool:
        """Activates the controller and moves to the start pose. False if the operator quits."""
        args, robot = self.args, self.robot
        while True:
            if robot.takeover()[0] is not None:
                self._await_release()
                self._finish_takeover()
            if args.dagger and not self._wait_for_controller():
                return False
            if self.gripper is not None:
                self.gripper.forget_target()
            try:
                robot.activate(service_timeout=args.service_timeout)
                if args.execute:
                    move_to_start(robot, self.gripper, args.start_pose, timeout=args.move_timeout)
                    if args.reanchor_at_start:
                        self._reanchor()
            except HumanTakeover:
                logger.warning("clutch pressed before the episode started: release it to start again")
                continue
            return True

    def _run_episode(self, config: RunConfig) -> dict | None:
        """Runs the policy until --max-chunks or Enter; after a takeover until the clutch release.

        Returns the recording summary (None without --record-dir).
        """
        robot, recorder = self.robot, self.recorder
        self.console.poll()
        if recorder is not None:
            recorder.start()
        taken_over = False
        try:
            try:
                chunks = run_chunks(
                    robot, self.planner, self.gripper, config, self.trace, should_stop=self._should_stop
                )
                until = time.monotonic()
                logger.info("episode ended after %d policy chunks", chunks)
                robot.stop()
            except HumanTakeover:
                taken_over = True
                logger.warning("takeover: finish the task, then release the clutch to end the episode")
                until = self._await_release()
            summary = None if recorder is None else recorder.stop(until)
        except BaseException:
            if recorder is not None:
                recorder.abort()
            raise
        if taken_over:
            self._finish_takeover()
        return summary

    def _should_stop(self) -> bool:
        if self.console.poll():
            logger.info("Enter: ending the episode")
            return True
        return self.recorder is not None and self.recorder.failed

    def _keep_episode(self, summary: dict | None) -> bool:
        """Saves or discards the recorded episode as the operator decides. False to quit."""
        if summary is None:
            return True
        logger.info(
            "episode: %d frames (%.1f s), %d teleoperated, %d LIFT training samples (all-teleoperated chunks)%s",
            summary["frames"],
            summary["frames"] / RECORD_FPS,
            summary["human"],
            summary["chunks"],
            f"; frames with stale inputs: {summary['stale']}" if summary["stale"] else "",
        )
        if not summary["frames"]:
            self.recorder.discard()
            logger.warning("empty episode discarded")
            return True
        while True:
            answer = self.console.ask(
                "Enter/s = save, y = save as success, n = save as failure, d = discard, "
                "q = save and quit, x = discard and quit",
                "q",
            )
            if answer in ("", "s", "q", "y", "n"):
                # Offline RL (scripts/ar3_rl.py) trains on how the episodes ended.
                outcome = {"y": "success", "n": "failure"}.get(answer)
                index = self.recorder.save(outcome)
                logger.info(
                    "saved episode %d in %s%s",
                    index,
                    self.recorder.writer.root,
                    f", labelled {outcome} (rl/outcomes.json)" if outcome else "",
                )
                return answer != "q"
            if answer in ("d", "x"):
                self.recorder.discard()
                logger.info("episode discarded")
                return answer != "x"
            print(f"unknown answer {answer!r}")

    def _await_release(self) -> float:
        """Waits for the clutch release; returns its host time."""
        robot, warned = self.robot, False
        while True:
            pressed_at, released_at, commands = robot.takeover()
            if released_at is not None:
                return released_at
            if pressed_at is None and not robot.clutch_pressed:
                return time.monotonic()
            if pressed_at is not None and not warned and not commands and time.monotonic() - pressed_at > 1.0:
                warned = True
                logger.warning("no teleoperation commands since the clutch press: is the teleoperation running?")
            time.sleep(0.02)

    def _finish_takeover(self) -> None:
        """After the release the teleoperation stops the arm; then the policy may command again."""
        deadline = time.monotonic() + 10.0
        while not self.robot.status.startswith(IDLE_STATUS) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.robot.end_takeover()

    def _wait_for_controller(self) -> bool:
        """DAgger: clutch released and controller idle. A clutch press while no episode runs starts a teleoperation
        session that it keeps after the release; the operator may stop it. False if the operator quits."""
        robot = self.robot
        if robot.clutch_pressed:
            logger.warning("release the teleoperation clutch to start the episode")
            self._await_release()
        deadline = time.monotonic() + 3.0
        while not robot.status.startswith(IDLE_STATUS):
            if time.monotonic() > deadline:
                answer = self.console.ask(
                    f"the controller is busy ({robot.status}), e.g. the teleoperation holds the arm. "
                    "Enter = stop the controller (the arm stays where it is), q = quit",
                    "q",
                )
                if answer == "q":
                    return False
                robot.stop_controller()
                return True
            time.sleep(0.05)
        return True

    def _reanchor(self) -> None:
        """Centres the controller's workspace sphere at the start pose, like in the recordings."""
        args, robot = self.args, self.robot
        if robot.recenter_workspace(timeout=RECENTER_TIMEOUT, service_timeout=args.service_timeout):
            return
        # A new controller session anchors its workspace at the measured TCP pose.
        logger.info("restarting the controller session at the start pose to centre its workspace there")
        robot.stop()
        robot.activate(service_timeout=args.service_timeout)

    def _configure_limits(self) -> None:
        args, limits = self.args, self.limits
        try:
            parameters = self.robot.controller_parameters(CONTROLLER_PARAMETERS, args.service_timeout)
        except RobotError as error:
            parameters = {}
            logger.warning("cannot read the controller parameters (%s)", error)
        logger.info("controller parameters: %s", parameters)
        if args.execute and parameters.get("enable_motion") is False:
            raise RobotError(
                "the impedance node runs with enable_motion:=false (monitoring only): restart it with "
                "enable_motion:=true (scripts/prepare_ar3_inference.sh)"
            )
        radii = [value for value in (args.workspace_radius, parameters.get("workspace_radius_m")) if value is not None]
        limits.workspace_radius = float(min(radii)) if radii else 0.25
        if parameters.get("workspace_radius_m") is None:
            logger.warning("controller workspace_radius_m unknown: assuming %.2f m", limits.workspace_radius)
        for name, limit in (
            ("max_translation_step_m", "max_tracking_error_m"),
            ("max_rotation_step_rad", "max_tracking_error_rad"),
        ):
            if parameters.get(name) is not None:
                setattr(limits, limit, min(getattr(limits, limit), float(parameters[name])))
        velocity = parameters.get("max_velocity")
        if isinstance(velocity, list) and len(velocity) == 6:
            linear = min(velocity[:3])
            if parameters.get("speed_mm_s") is not None:
                linear = min(linear, float(parameters["speed_mm_s"]) / 1000.0)
            for name, speed in (("max_linear_speed", linear), ("max_angular_speed", min(velocity[3:]))):
                if getattr(limits, name) > speed:
                    logger.info("%s lowered to the controller's %.3f", name, speed)
                    setattr(limits, name, float(speed))
        timeout = parameters.get("command_timeout_s")
        if timeout is not None and 2.0 / limits.rate_hz >= float(timeout):
            raise RobotError(
                f"--rate-hz {limits.rate_hz:g} is too slow for the controller's command_timeout_s {timeout}"
            )
        if limits.max_lead_m >= limits.max_tracking_error_m or limits.max_lead_rad >= limits.max_tracking_error_rad:
            raise RobotError(
                f"command lead {limits.max_lead_m} m / {limits.max_lead_rad} rad must stay below the tracking limits {limits.max_tracking_error_m} m / {limits.max_tracking_error_rad} rad"
            )
        if limits.workspace_radius - limits.workspace_margin < TASK_REACH_M:
            logger.warning(
                "the session workspace (workspace_radius_m %.2f - margin %.2f m) is smaller than the %.2f m the recorded "
                "motions reach from the start pose; farther targets are clamped. Start the controller with "
                "workspace_radius_m:=0.45 (the scripts/prepare_ar3_inference.sh default).",
                limits.workspace_radius,
                limits.workspace_margin,
                TASK_REACH_M,
            )

    def _report(self) -> None:
        args = self.args
        observation = self.robot.observe(
            wrench_dim=6 + TAU_M_SIZE, max_age=args.max_observation_age, max_image_age=args.max_image_age
        )
        distance, angle = pose_error(pose_from_vector(observation.pose), pose_from_vector(args.start_pose))
        logger.info("controller status: %s", self.robot.status)
        logger.info(
            "TCP xyz %s m, rpy %s rad (%.0f mm / %.2f rad from the start pose)",
            np.round(observation.pose[:3], 4),
            np.round(observation.pose[3:], 4),
            distance * 1000,
            angle,
        )
        logger.info(
            "TCP wrench %s, tau_m %s", np.round(observation.reading[:6], 2), np.round(observation.reading[6:], 2)
        )
        logger.info("front %s, wrist %s", observation.front.shape, observation.wrist.shape)
        depth, valid, wrench = self.robot.hand_reading()
        if valid.all():
            logger.info(
                "Tac G2 %s",
                "; ".join(
                    f"{sensor}: indentation peak {depth[index].max():.2f}, force {np.round(wrench[6 * index : 6 * index + 3], 2)}"
                    for index, sensor in enumerate(tactile_io().SENSORS)
                ),
            )
        else:
            logger.warning(
                "Tac G2 fingers unavailable: tactile muscleVLTA policies will not start; %s", TACTILE_PREPARATION
            )
        recenter = self.robot.metadata.get("workspace_recenter") == WORKSPACE_RECENTER
        logger.info(
            "controller contract ok; workspace recenter %s",
            "supported" if recenter else "unsupported (the session restarts at the start pose instead)",
        )
        others, conflicts = self.robot.command_sources()
        if others > 0 or conflicts:
            logger.warning(
                "other arm command sources are running (%s; %d publisher(s) on %s/command): --execute refuses to "
                "start until they are stopped",
                ", ".join(conflicts) or "unknown node",
                others,
                self.robot.prefix,
            )
        logger.info("check passed")

    def close(self) -> bool:
        """Stops the arm and the gripper; returns False if that failed. Further Ctrl+C cannot interrupt it."""
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        ok = True
        if self.recorder is not None:
            self.recorder.close()
        if self.planner is not None:
            self.planner.close()
        try:
            self.robot.stop()
        except RobotError as error:
            logger.error("%s", error)
            ok = False
        if self.gripper is not None:
            try:
                self.gripper.stop()
            except GripperError as error:
                logger.error("%s", error)
                ok = False
            self.gripper.close()
        self.trace.close()
        try:
            self.robot.shutdown()
        except RobotError as error:
            logger.error("%s", error)
            ok = False
        return ok


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs an AR3 LIFT policy on the Rokae AR3 + DM gripper (dry run unless --execute).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run = parser.add_argument_group("run")
    run.add_argument("--execute", action="store_true", help="move the arm and gripper (default: dry run, preview only)")
    run.add_argument("--check", action="store_true", help="check controller, cameras and gripper, then exit")
    run.add_argument(
        "--mask-tactile",
        action="store_true",
        help="ablation of tactile muscleVLTA policies: send both Tac G2 fingers as missing (the gripper opening "
        "stays; the fingers must still run, so that everything else is unchanged)",
    )
    run.add_argument(
        "--max-chunks",
        type=int,
        default=30,
        help="action chunks to run (nominal seconds = chunks * action_horizon / control_hz); 0 = until Ctrl+C",
    )
    run.add_argument(
        "--move-to-start",
        action="store_true",
        help="first reach the saved Cartesian home and open the gripper (episodes of --record-dir / --dagger / "
        "--episodes always do)",
    )
    start = run.add_mutually_exclusive_group()
    start.add_argument(
        "--home-file",
        type=pathlib.Path,
        help="saved measured TCP home; default: $INFER_AR3_WS/config/cartesian_home.json in the arm workspace",
    )
    start.add_argument(
        "--start-pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="explicit TCP home override (m, SDK rpy rad); mutually exclusive with --home-file",
    )
    run.add_argument("--move-timeout", type=float, default=home.HOME_TIMEOUT)
    run.add_argument(
        "--reanchor-at-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="centre the controller workspace sphere at the start pose (recenter_workspace, else a new session)",
    )
    run.add_argument("--trace", default=None, help="JSON-lines log of every executed step")
    run.add_argument("--ready-timeout", type=float, default=15.0)

    policy = parser.add_argument_group("policy server")
    policy.add_argument("--host", default="127.0.0.1")
    policy.add_argument("--port", type=int, default=8000)
    policy.add_argument("--api-key", default=None)
    policy.add_argument("--connect-timeout", type=float, default=120.0)
    policy.add_argument(
        "--task",
        "--prompt",
        dest="prompt",
        default=DEFAULT_PROMPT,
        help="task instruction sent to the policy for warm-up and every action chunk",
    )
    policy.add_argument(
        "--mode", choices=("auto", "reactive", "infer"), default="auto", help="auto: reactive for force-reactive models"
    )
    policy.add_argument("--control-hz", type=float, default=10.0, help="action rate (the dataset is 10 Hz)")
    policy.add_argument("--resize", type=int, default=224, help="client-side resize_with_pad; 0 sends 640x480 frames")
    policy.add_argument("--first-action-timeout", type=float, default=2.0)
    policy.add_argument("--policy-timeout", type=float, default=1.0)
    policy.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)

    ros = parser.add_argument_group("controller / ROS 2")
    ros.add_argument("--controller-prefix", default="/ar3/cartesian_impedance")
    ros.add_argument("--controller-node", default="/cartesian_impedance_node")
    ros.add_argument("--front-topic", default="/front/color/image_raw")
    ros.add_argument("--wrist-topic", default="/wrist/color/image_raw")
    ros.add_argument("--service-timeout", type=float, default=2.0)
    ros.add_argument("--max-observation-age", type=float, default=0.25)
    ros.add_argument("--max-image-age", type=float, default=0.2)
    ros.add_argument(
        "--workspace-radius", type=float, default=None, help="default: the controller's workspace_radius_m"
    )
    ros.add_argument("--workspace-margin", type=float, default=0.02)

    shaping = parser.add_argument_group("command shaping (AR3 teleoperation defaults)")
    shaping.add_argument("--rate-hz", type=float, default=100.0)
    shaping.add_argument("--max-lead-m", type=float, default=0.06)
    shaping.add_argument("--max-lead-rad", type=float, default=0.05)
    shaping.add_argument("--max-linear-speed", type=float, default=0.04)
    shaping.add_argument("--max-angular-speed", type=float, default=0.15)
    shaping.add_argument("--max-tracking-error-m", type=float, default=0.08)
    shaping.add_argument("--max-tracking-error-rad", type=float, default=0.15)
    shaping.add_argument("--feedback-timeout", type=float, default=0.5)
    shaping.add_argument("--target-timeout", type=float, default=1.0)

    gripper = parser.add_argument_group("DM gripper")
    gripper.add_argument("--gripper-address", default="192.168.127.10:50008")
    gripper.add_argument("--gripper-speed", type=int, default=80, help="as in the recording teleoperation")
    gripper.add_argument(
        "--gripper-force", type=int, default=20, help="device force level 20..100 (the recordings used 20)"
    )
    gripper.add_argument("--gripper-deadband", type=int, default=2)
    gripper.add_argument("--gripper-max-age", type=float, default=0.5)
    gripper.add_argument(
        "--no-gripper", action="store_true", help="run without the gripper (state uses --assumed-gripper)"
    )
    gripper.add_argument("--assumed-gripper", type=float, default=GRIPPER_OPEN)

    record = parser.add_argument_group("recording / DAgger")
    record.add_argument(
        "--record-dir",
        default=None,
        help="append the episodes the operator keeps to this LeRobot v3.0 dataset (created if missing); tactile "
        "policies also record the Tac G2 fingers, and y / n at the save prompt label the outcome for offline RL",
    )
    record.add_argument(
        "--dagger",
        action="store_true",
        help="LIFT human-in-the-loop corrections: the AR3 teleoperation's clutch takes over from the policy "
        "(needs --execute and --record-dir, and scripts/prepare_ar3_inference.sh --dagger)",
    )
    record.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="episodes to run, each from the saved home; 0 = until the operator quits "
        "(default: 0 with --record-dir / --dagger, else 1)",
    )
    record.add_argument("--clutch-topic", default="/clutch/right", help="the teleoperation clutch (std_msgs/Bool)")
    args = parser.parse_args(argv)
    for name in (
        "control_hz",
        "rate_hz",
        "first_action_timeout",
        "policy_timeout",
        "max_observation_age",
        "max_image_age",
        "service_timeout",
        "feedback_timeout",
        "target_timeout",
        "move_timeout",
        "ready_timeout",
        "max_linear_speed",
        "max_angular_speed",
        "max_lead_m",
        "max_lead_rad",
        "max_tracking_error_m",
        "max_tracking_error_rad",
    ):
        if not math.isfinite(getattr(args, name)) or not getattr(args, name) > 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.resize < 0 or args.max_chunks < 0:
        parser.error("--resize and --max-chunks must not be negative")
    if not (1 <= args.gripper_speed <= 100 and 20 <= args.gripper_force <= 100):
        parser.error("the DM gripper accepts --gripper-speed 1..100 and --gripper-force 20..100")
    if args.start_pose is None:
        if args.home_file is None:
            arm_workspace = pathlib.Path(
                os.environ.get(
                    "INFER_AR3_WS", pathlib.Path(__file__).resolve().parents[2] / "ar3_CartesianImpedanceControl"
                )
            )
            args.home_file = arm_workspace / "config/cartesian_home.json"
        args.home_file = args.home_file.expanduser().resolve()
        try:
            args.start_pose = home.load_home(args.home_file, BASE_FRAME).tolist()
        except (OSError, ValueError) as error:
            parser.error(f"cannot load saved home {args.home_file}: {error}")
    elif not np.isfinite(args.start_pose).all():
        parser.error("--start-pose requires six finite xyz[m]/RPY[rad] values")
    if args.episodes is None:
        args.episodes = 0 if args.dagger or args.record_dir is not None else 1
    if args.episodes < 0:
        parser.error("--episodes must not be negative")
    if args.dagger and not args.check and not (args.execute and args.record_dir is not None):
        parser.error("--dagger needs --execute and --record-dir")
    if args.record_dir is not None and args.no_gripper:
        parser.error("--record-dir needs the gripper (its position and target are recorded)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    signal.signal(signal.SIGTERM, _interrupt)
    session = Session(args)
    code = 1
    try:
        code = session.run()
    except KeyboardInterrupt:
        logger.warning("interrupted: stopping")
        code = 130
    except (RobotError, GripperError, PolicyError) as error:
        logger.error("%s", error)
    finally:
        if not session.close() and code == 0:
            code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
