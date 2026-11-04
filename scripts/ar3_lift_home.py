"""Saved AR3 home validation and the recorder's feedback-paced Cartesian return path.

The caller owns the controller, checks tracking/workspace limits, and advances
``path.elapsed`` only after accepting a candidate. This module has no ROS dependency.
"""

import json
import math
from pathlib import Path

import numpy as np

HOME_LINEAR_SPEED = 0.02
HOME_ANGULAR_SPEED = 0.1
HOME_ROTATION_LEAD = 0.1
HOME_POSITION_TOLERANCE = 0.055
HOME_ROTATION_TOLERANCE = 0.06
HOME_SETTLE_TIME = 0.3
HOME_TIMEOUT = 75.0

_MAX_HOME_BYTES = 16 * 1024


def load_home(path: str | Path, base_frame: str) -> np.ndarray:
    """Read a measured TCP home as xyz [m] + SDK roll/pitch/yaw [rad]."""
    path = Path(path).expanduser()
    with path.open("rb") as source:
        payload = source.read(_MAX_HOME_BYTES + 1)
    if len(payload) > _MAX_HOME_BYTES:
        raise ValueError("home file is larger than 16 KiB")
    data = json.loads(payload)
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("home file requires schema_version=1")
    if not isinstance(base_frame, str) or not base_frame or data.get("frame_id") != base_frame:
        raise ValueError("home frame differs from the SDK base frame")
    if data.get("source") != "measured_tcp_pose":
        raise ValueError("home must come from measured_tcp_pose")
    pose = data.get("tcp_pose")
    if (
        not isinstance(pose, list)
        or len(pose) != 6
        or any(isinstance(value, bool) or not isinstance(value, int | float) for value in pose)
    ):
        raise ValueError("home tcp_pose must contain six finite xyz[m]/RPY[rad] values")
    values = np.asarray(pose, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("home tcp_pose must contain six finite xyz[m]/RPY[rad] values")
    return values


def _copy_pose(pose: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    position, rotation = (np.asarray(value, dtype=np.float64).copy() for value in pose)
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("return-home poses require a position [3] and rotation matrix [3, 3]")
    if not np.isfinite(position).all() or not np.isfinite(rotation).all():
        raise ValueError("return-home poses must be finite")
    return position, rotation


class ReturnHomePath:
    """Recorder-compatible quintic path with zero endpoint velocities.

    This follows a straight Cartesian position path and the shortest orientation
    arc. It does not plan around obstacles. Calling ``candidate`` never advances
    the path: the controller must accept a candidate before assigning ``elapsed``.
    """

    def __init__(self, start, goal, linear_speed: float, angular_speed: float) -> None:
        if any(not math.isfinite(speed) or speed <= 0 for speed in (linear_speed, angular_speed)):
            raise ValueError("return-home speeds must be positive and finite")
        self.start, self.goal = _copy_pose(start), _copy_pose(goal)
        self.distance = float(np.linalg.norm(self.goal[0] - self.start[0]))
        delta = self.goal[1] @ self.start[1].T
        self.angle = float(np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))
        if self.angle > np.pi - 1e-5:
            raise ValueError("return-home orientation differs by pi; choose a closer starting pose")
        self._skew = None
        if self.angle >= 1e-10:
            x, y, z = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]]) / (
                2.0 * np.sin(self.angle)
            )
            self._skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        # The peak derivative of 10u^3 - 15u^4 + 6u^5 is 1.875.
        self.duration = max(2.0, 1.875 * self.distance / linear_speed, 1.875 * self.angle / angular_speed)
        self.elapsed = 0.0

    def candidate(self, dt: float) -> tuple[tuple[np.ndarray, np.ndarray], float]:
        """Return the next bounded pose and proposed path time, without committing either."""
        if not math.isfinite(dt):
            raise ValueError("return-home time step must be finite")
        elapsed = min(self.duration, self.elapsed + max(0.0, dt))
        if elapsed >= self.duration:
            return self.goal, elapsed
        u = elapsed / self.duration
        progress = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
        position = self.start[0] + progress * (self.goal[0] - self.start[0])
        if self._skew is None:
            rotation = self.goal[1].copy()
        else:
            angle = self.angle * progress
            increment = np.eye(3) + np.sin(angle) * self._skew + (1.0 - np.cos(angle)) * (self._skew @ self._skew)
            rotation = increment @ self.start[1]
        return (position, rotation), elapsed
