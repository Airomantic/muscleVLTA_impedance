"""AR3 (Rokae AR3 Cartesian impedance arm + DM V3 gripper) policy I/O in the LIFT single-arm format.

The AR3 LIFT recorder (``meta/ar3_capture.json``, ``training_schema: lift``) stores LeRobot v3.0 datasets with
absolute poses in the robot base frame. Training maps them to the LIFT client keys with ``AR3LiftRepack``; at
inference the robot client sends the same keys:

* ``observation/state``: [7] measured TCP pose, xyz [m] + rotation vector [rad], then the gripper opening in native
  SDK units (0 = closed ... 1000 = open).
* ``observation/left_wrist_image``: HWC uint8 RGB wrist camera image.
* ``observation/base_image``: HWC uint8 RGB front camera image (required by configs trained with the front camera).
* ``wrench``: [action_horizon, wrench_dim] measured force sequence: the TCP wrench (``tauExt_inBase``, 6D) and, for
  ``wrench_dim=13``, the 7 measured joint torques ``tau_m`` [Nm]. Rows are aligned with the action steps, as in LIFT.
  At inference the client sends the readings measured since the chunk start ([T, wrench_dim], 1 <= T <= horizon);
  they are padded to the horizon by repeating the newest reading (see ``pad_force_sequence``).
* ``prompt``: task string.

muscleVLTA policies (``Pi0Config.sensor_history`` L > 0 and/or ``tactile_in_use``) read every per-step sensor key
(``READING_KEYS``) with L extra leading rows: the readings of the L control steps before the chunk start, then the
readings measured since the chunk start (L + 1 <= T <= L + horizon rows, all keys with the same T):

* ``wrench``: as above.
* ``sensor_valid``: [T] bool, False for history rows without a reading (before the episode start or after a gap);
  default all True.
* ``tactile_depth``: [T, 2, 48, 64] Tac G2 indentation maps of sensor_0 / sensor_2, area-pooled with
  ``openpi_client.tactile.pool_depth`` (float16 is fine).
* ``tactile_valid``: [T, 2] bool, False where a finger has no fresh map (zero map).
* ``tactile_wrench``: [T, 12] Tac G2 finger wrenches (Fx Fy Fz Mx My Mz of sensor_0, then sensor_2).
* ``gripper_aperture``: [T] gripper opening in SDK units (0 = closed ... 1000 = open).

Offline RL presets (``pi05_ar3_muscle_vlta_rl``) condition the policy on the advantage of its action chunk, written
into the prompt. Training samples carry the label of their action chunk under ``rl_advantage`` (``ADVANTAGE_KEY``, see
``openpi.rl.advantage``); the robot client sends nothing and the policy is served as for a positive advantage.

As in LIFT (``IPhoneZeroStateAndRealRelativeActions``), the policy outputs [action_horizon, 7] actions relative to
the observed TCP pose: xyz [m] + xyz Euler angles [rad] of ``inv(T_state) @ T_target``, followed by the absolute
gripper target (0..1000). ``relative_to_absolute_actions`` recovers absolute targets for the controller.
"""

from collections.abc import Sequence
import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import pose_utils

TCP_POSE_KEY = "observation.tcp_pose"
GRIPPER_POSITION_KEY = "observation.gripper_position"
TCP_WRENCH_KEY = "observation.tcp_wrench"
TAU_M_KEY = "observation.tau_m"
GRIPPER_WRENCH_KEYS = ("observation.gripper_wrench.sensor_0", "observation.gripper_wrench.sensor_2")
TACTILE_DEPTH_KEY = "observation.tactile_depth"
TACTILE_VALID_KEY = "observation.tactile_valid"
WRIST_IMAGE_KEY = "observation.images.wrist"
FRONT_IMAGE_KEY = "observation.images.front"
ACTIONS_KEY = "actions"

STATE_KEYS = (TCP_POSE_KEY, GRIPPER_POSITION_KEY)
# LIFT force input: the 6D TCP wrench, optionally followed by the 7 measured joint torques.
FORCE_KEYS_WRENCH = (TCP_WRENCH_KEY,)
FORCE_KEYS_WRENCH_TAU_M = (TCP_WRENCH_KEY, TAU_M_KEY)
FORCE_KEY_DIMS = {TCP_WRENCH_KEY: 6, TAU_M_KEY: 7}
# Hand stream of the muscleVLTA tactile policies: the two finger wrenches and the gripper opening.
HAND_KEYS = (*GRIPPER_WRENCH_KEYS, GRIPPER_POSITION_KEY)
TACTILE_WRENCH_DIM = 6 * len(GRIPPER_WRENCH_KEYS)
# Model input = indentation map / scale (Tac G2 map peaks reach ~0.55 on the plug-insertion data).
TACTILE_DEPTH_SCALE = 0.25
# Per-step sensor keys of the client protocol (module docstring), padded together at inference.
READING_KEYS = ("wrench", "sensor_valid", "tactile_depth", "tactile_valid", "tactile_wrench", "gripper_aperture")
TACTILE_READING_KEYS = ("tactile_depth", "tactile_valid", "tactile_wrench", "gripper_aperture")
# Advantage label of a training sample (offline RL presets only), consumed by openpi.rl.advantage.AdvantagePrompt.
ADVANTAGE_KEY = "rl_advantage"

ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")
GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 1000.0


def force_dim(force_keys: Sequence[str]) -> int:
    """Per-step width of the force sequence built from ``force_keys``."""
    return sum(FORCE_KEY_DIMS[key] for key in force_keys)


def concat_force(parts: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenates force signals of shape [T, d_i] (or [d_i]) along the last axis to float32 [T, sum(d_i)]."""
    arrays = [np.asarray(part, dtype=np.float32) for part in parts]
    arrays = [array[None] if array.ndim == 1 else array for array in arrays]
    return np.concatenate(arrays, axis=-1)


def make_ar3_lift_example(
    *, action_horizon: int = 10, wrench_dim: int = 13, history_length: int = 0, tactile: bool = False
) -> dict:
    """Random robot-client observation in the format expected by the AR3 LIFT (and muscleVLTA) policies."""
    rows = history_length + action_horizon
    example = {
        "observation/state": np.concatenate(
            [np.random.uniform(-0.5, 0.5, size=6), np.random.uniform(GRIPPER_CLOSED, GRIPPER_OPEN, size=1)]
        ).astype(np.float32),
        "observation/left_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/base_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "wrench": np.random.randn(rows, wrench_dim).astype(np.float32),
        "prompt": "do something",
    }
    if history_length:
        example["sensor_valid"] = np.arange(rows) >= history_length // 2
    if tactile:
        example["tactile_depth"] = np.random.uniform(0.0, 0.3, size=(rows, 2, 48, 64)).astype(np.float16)
        example["tactile_valid"] = np.ones((rows, 2), dtype=bool)
        example["tactile_wrench"] = np.random.randn(rows, TACTILE_WRENCH_DIM).astype(np.float32)
        example["gripper_aperture"] = np.random.uniform(GRIPPER_CLOSED, GRIPPER_OPEN, size=rows).astype(np.float32)
    return example


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AR3LiftRepack(transforms.DataTransformFn):
    """Training only: maps raw LeRobot v3.0 AR3 items to the LIFT client keys listed in the module docstring.

    With ``history_length`` L, the force (and hand) keys are dataset history keys holding the L frames before the
    current one first (``LeRobotV3Dataset``); ``tactile`` reads the tactile cache and the hand keys.
    """

    force_keys: tuple[str, ...] = FORCE_KEYS_WRENCH_TAU_M
    use_front_camera: bool = True
    history_length: int = 0
    tactile: bool = False

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([self._current_frame(data, key) for key in STATE_KEYS])
        inputs = {
            "observation/state": state,
            "observation/left_wrist_image": data[WRIST_IMAGE_KEY],
            "actions": np.asarray(data[ACTIONS_KEY], dtype=np.float32),
        }
        if self.use_front_camera:
            inputs["observation/base_image"] = data[FRONT_IMAGE_KEY]
        if self.force_keys:
            inputs["wrench"] = concat_force([data[key] for key in self.force_keys])
            if self.history_length:
                # History rows before the episode start have no reading; rows clamped at the episode end repeat
                # the last reading and stay valid (LIFT).
                valid = np.ones(len(inputs["wrench"]), dtype=bool)
                pad = np.asarray(data[f"{self.force_keys[0]}_is_pad"], dtype=bool)
                valid[: self.history_length] = ~pad[: self.history_length]
                inputs["sensor_valid"] = valid
        if self.tactile:
            inputs.update(self._hand(data))
        prompt = data.get("prompt", data.get("task"))
        if prompt is not None:
            inputs["prompt"] = prompt
        if ADVANTAGE_KEY in data:
            inputs[ADVANTAGE_KEY] = data[ADVANTAGE_KEY]
        return inputs

    def _current_frame(self, data: dict, key: str) -> np.ndarray:
        value = np.asarray(data[key], dtype=np.float32)
        if f"{key}_is_pad" in data:  # A sequence key: [history frames..., current frame, next frames...].
            value = value[len(value) - len(data[ACTIONS_KEY])]
        return value.reshape(-1)

    def _hand(self, data: dict) -> dict:
        depth = np.asarray(data[TACTILE_DEPTH_KEY])
        valid = np.asarray(data[TACTILE_VALID_KEY], dtype=bool)
        if all(key in data for key in GRIPPER_WRENCH_KEYS):
            finger_wrench = concat_force([data[key] for key in GRIPPER_WRENCH_KEYS])
        else:
            # Recordings without the Tac G2 wrenches (e.g. LIFT DAgger rounds): the fingers are unknown.
            finger_wrench = np.zeros((len(depth), TACTILE_WRENCH_DIM), dtype=np.float32)
            valid = np.zeros_like(valid)
        aperture = np.asarray(data[GRIPPER_POSITION_KEY], dtype=np.float32).reshape(-1)
        if not len(depth) == len(valid) == len(finger_wrench) == len(aperture):
            raise ValueError(
                f"Tactile rows {len(depth)}, finger wrench rows {len(finger_wrench)} and gripper rows "
                f"{len(aperture)} differ: {GRIPPER_POSITION_KEY} and {GRIPPER_WRENCH_KEYS} must be history keys."
            )
        return {
            "tactile_depth": depth,
            "tactile_valid": valid,
            "tactile_wrench": finger_wrench,
            "gripper_aperture": aperture,
        }


def pad_readings(values: np.ndarray, action_horizon: int, *, history_length: int = 0) -> np.ndarray:
    """Pads per-step readings [T, ...] (``history_length`` history rows, then the readings measured since the chunk
    start) to [history_length + action_horizon, ...] by repeating the newest reading.

    Reactive action r only attends to readings of steps <= r - cross_attention_latency (through causal GRUs), so the
    padding never changes actions whose inputs were measured; it only fills the shape the model was trained with.
    """
    values = np.asarray(values)
    length = history_length + action_horizon
    if values.ndim < 1 or not history_length < values.shape[0] <= length:
        raise ValueError(
            f"Expected {history_length} history readings followed by 1..{action_horizon} readings of the chunk "
            f"([T, ...] with {history_length + 1} <= T <= {length}), got {values.shape}."
        )
    if values.shape[0] < length:
        values = np.concatenate([values, np.repeat(values[-1:], length - values.shape[0], axis=0)])
    return values


def pad_force_sequence(wrench: np.ndarray, action_horizon: int, *, history_length: int = 0) -> np.ndarray:
    """Pads the force readings ([T, d], or [d] without history) to [history_length + action_horizon, d].

    See ``pad_readings``.
    """
    wrench = np.asarray(wrench, dtype=np.float32)
    if wrench.ndim == 1 and not history_length:
        wrench = wrench[None]
    if wrench.ndim != 2:
        raise ValueError(f"Expected force readings of shape [T, d], got {wrench.shape}.")
    return pad_readings(wrench, action_horizon, history_length=history_length)


@dataclasses.dataclass(frozen=True)
class AR3LiftInputs(transforms.DataTransformFn):
    """Converts LIFT client keys to model inputs (training and inference): wrist -> left wrist, front -> base.

    With ``action_horizon`` set, per-step readings holding only the readings measured so far in the chunk are padded
    to the action horizon (``pad_readings``); training samples already carry the full sequences. ``history_length``
    and ``tactile`` select the muscleVLTA sensor keys (module docstring); the tactile maps are scaled by
    ``TACTILE_DEPTH_SCALE`` and the gripper opening to [0, 1].
    """

    model_type: _model.ModelType
    use_front_camera: bool = True
    action_horizon: int | None = None
    history_length: int = 0
    tactile: bool = False

    def __call__(self, data: dict) -> dict:
        wrist_image = _parse_image(data["observation/left_wrist_image"])
        if self.use_front_camera:
            if "observation/base_image" not in data:
                raise KeyError("This AR3 policy was trained with the front camera: send 'observation/base_image'.")
            base_image, base_mask = _parse_image(data["observation/base_image"]), np.True_
        else:
            base_image, base_mask = np.zeros_like(wrist_image), np.False_

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": base_mask,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        for key in ("prev_state", "actions", "prompt", "wrench", "wrench_mask", ADVANTAGE_KEY):
            if key in data:
                inputs[key] = data[key]
        if "wrench" in inputs and self.action_horizon is not None:
            inputs["wrench"] = pad_force_sequence(
                inputs["wrench"], self.action_horizon, history_length=self.history_length
            )
        if "wrench" in inputs and (self.history_length or self.tactile):
            wrench = np.asarray(data["wrench"])
            inputs.update(self._readings(data, 1 if wrench.ndim == 1 else len(wrench)))
        return inputs

    def _readings(self, data: dict, rows: int) -> dict:
        keys = ("sensor_valid", *TACTILE_READING_KEYS) if self.tactile else ("sensor_valid",)
        if self.tactile and (missing := [key for key in TACTILE_READING_KEYS if key not in data]):
            raise KeyError(f"This AR3 policy was trained with the Tac G2 gripper: send {missing}.")
        readings = {key: np.asarray(data[key]) for key in keys if key in data}
        readings.setdefault("sensor_valid", np.ones(rows, dtype=bool))
        for key, value in readings.items():
            if len(value) != rows:
                raise ValueError(f"'{key}' has {len(value)} rows but 'wrench' has {rows}.")
            if self.action_horizon is not None:
                readings[key] = pad_readings(value, self.action_horizon, history_length=self.history_length)
        readings["sensor_valid"] = readings["sensor_valid"].astype(bool).reshape(-1)
        if self.tactile:
            readings["tactile_depth"] = readings["tactile_depth"].astype(np.float32) / TACTILE_DEPTH_SCALE
            readings["tactile_valid"] = readings["tactile_valid"].astype(bool)
            readings["tactile_wrench"] = readings["tactile_wrench"].astype(np.float32)
            aperture = readings["gripper_aperture"].astype(np.float32).reshape(-1, 1) / GRIPPER_OPEN
            readings["gripper_aperture"] = aperture
        return readings


@dataclasses.dataclass(frozen=True)
class AR3LiftOutputs(transforms.DataTransformFn):
    """Returns the 7 single-arm action dimensions (relative pose + absolute gripper, as in LIFT)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


def rpy_to_rotvec(rpy: np.ndarray) -> np.ndarray:
    """AR3 SDK angles [roll, pitch, yaw] with R = Rz(yaw) @ Ry(pitch) @ Rx(roll) -> rotation vector."""
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64)).as_rotvec()


def rotvec_to_rpy(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector -> AR3 SDK angles [roll, pitch, yaw] with R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_euler("xyz")


def relative_to_absolute_actions(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Inverts ``IPhoneZeroStateAndRealRelativeActions`` for one observation.

    Args:
        actions: [..., 7] policy outputs: xyz + xyz Euler of ``inv(T_state) @ T_target``, then the gripper target.
        state: [7] (or [6]) observed state: xyz + rotation vector (+ gripper).

    Returns:
        [..., 7] absolute targets in the robot base frame: xyz [m] + rotation vector [rad] + gripper target.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape[-1] < len(ACTION_NAMES):
        raise ValueError(f"Expected at least {len(ACTION_NAMES)} action dimensions, got {actions.shape}")
    relative = np.zeros((*actions.shape[:-1], 4, 4))
    relative[..., :3, :3] = (
        Rotation.from_euler("xyz", actions[..., 3:6].reshape(-1, 3)).as_matrix().reshape((*actions.shape[:-1], 3, 3))
    )
    relative[..., :3, 3] = actions[..., :3]
    relative[..., 3, 3] = 1.0
    base = pose_utils.pose_to_mat(np.asarray(state, dtype=np.float64)[:6])
    absolute = base @ relative
    return np.concatenate([pose_utils.mat_to_pose(absolute), actions[..., 6:7]], axis=-1)
