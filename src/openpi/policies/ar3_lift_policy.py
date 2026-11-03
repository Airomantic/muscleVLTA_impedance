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
WRIST_IMAGE_KEY = "observation.images.wrist"
FRONT_IMAGE_KEY = "observation.images.front"
ACTIONS_KEY = "actions"

STATE_KEYS = (TCP_POSE_KEY, GRIPPER_POSITION_KEY)
# LIFT force input: the 6D TCP wrench, optionally followed by the 7 measured joint torques.
FORCE_KEYS_WRENCH = (TCP_WRENCH_KEY,)
FORCE_KEYS_WRENCH_TAU_M = (TCP_WRENCH_KEY, TAU_M_KEY)
FORCE_KEY_DIMS = {TCP_WRENCH_KEY: 6, TAU_M_KEY: 7}

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


def make_ar3_lift_example(*, action_horizon: int = 10, wrench_dim: int = 13) -> dict:
    """Random robot-client observation in the format expected by the AR3 LIFT policies."""
    return {
        "observation/state": np.concatenate(
            [np.random.uniform(-0.5, 0.5, size=6), np.random.uniform(GRIPPER_CLOSED, GRIPPER_OPEN, size=1)]
        ).astype(np.float32),
        "observation/left_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/base_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "wrench": np.random.randn(action_horizon, wrench_dim).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AR3LiftRepack(transforms.DataTransformFn):
    """Training only: maps raw LeRobot v3.0 AR3 items to the LIFT client keys listed in the module docstring."""

    force_keys: tuple[str, ...] = FORCE_KEYS_WRENCH_TAU_M
    use_front_camera: bool = True

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([np.asarray(data[key], dtype=np.float32).reshape(-1) for key in STATE_KEYS])
        inputs = {
            "observation/state": state,
            "observation/left_wrist_image": data[WRIST_IMAGE_KEY],
            "actions": np.asarray(data[ACTIONS_KEY], dtype=np.float32),
        }
        if self.use_front_camera:
            inputs["observation/base_image"] = data[FRONT_IMAGE_KEY]
        if self.force_keys:
            inputs["wrench"] = concat_force([data[key] for key in self.force_keys])
        prompt = data.get("prompt", data.get("task"))
        if prompt is not None:
            inputs["prompt"] = prompt
        return inputs


def pad_force_sequence(wrench: np.ndarray, action_horizon: int) -> np.ndarray:
    """Pads the force readings measured so far ([T, d] or [d], T <= action_horizon) to [action_horizon, d].

    Missing future rows repeat the newest reading. Reactive action r only attends to force rows
    <= r - cross_attention_latency (through a causal GRU), so the padding never changes actions whose inputs were
    measured; it only fills the shape the model was trained with.
    """
    wrench = np.asarray(wrench, dtype=np.float32)
    if wrench.ndim == 1:
        wrench = wrench[None]
    if wrench.ndim != 2 or not 1 <= wrench.shape[0] <= action_horizon:
        raise ValueError(f"Expected 1..{action_horizon} force readings of shape [T, d], got {wrench.shape}.")
    if wrench.shape[0] < action_horizon:
        wrench = np.concatenate([wrench, np.repeat(wrench[-1:], action_horizon - wrench.shape[0], axis=0)])
    return wrench


@dataclasses.dataclass(frozen=True)
class AR3LiftInputs(transforms.DataTransformFn):
    """Converts LIFT client keys to model inputs (training and inference): wrist -> left wrist, front -> base.

    With ``action_horizon`` set, a ``wrench`` holding only the readings measured so far in the chunk is padded to the
    action horizon (``pad_force_sequence``); training samples already carry the full sequence.
    """

    model_type: _model.ModelType
    use_front_camera: bool = True
    action_horizon: int | None = None

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
        for key in ("prev_state", "actions", "prompt", "wrench", "wrench_mask"):
            if key in data:
                inputs[key] = data[key]
        if "wrench" in inputs and self.action_horizon is not None:
            inputs["wrench"] = pad_force_sequence(inputs["wrench"], self.action_horizon)
        return inputs


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
