import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts import ar3_lift_home as home

FRAME = "AR3-3_07R-W4C5C5_base"


def _home_data():
    return {
        "schema_version": 1,
        "frame_id": FRAME,
        "source": "measured_tcp_pose",
        "tcp_pose": [0.2, -0.34, 0.1, -2.79, 0.58, -2.99],
    }


def test_load_saved_measured_home(tmp_path):
    path = tmp_path / "cartesian_home.json"
    data = _home_data()
    path.write_text(json.dumps(data))
    pose = home.load_home(path, FRAME)
    np.testing.assert_array_equal(pose, data["tcp_pose"])
    assert pose.dtype == np.float64


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", None),
        ("frame_id", "another_base"),
        ("frame_id", None),
        ("source", "commanded_tcp_pose"),
        ("source", None),
        ("tcp_pose", [0.0] * 5),
        ("tcp_pose", [[0.0] * 6]),
        ("tcp_pose", [0.0] * 5 + [float("nan")]),
        ("tcp_pose", [0.0] * 5 + [float("inf")]),
        ("tcp_pose", [0.0] * 5 + [True]),
        ("tcp_pose", [0.0] * 5 + ["0.0"]),
    ],
)
def test_reject_invalid_home_fields(tmp_path, key, value):
    path = tmp_path / "cartesian_home.json"
    data = _home_data()
    data[key] = value
    path.write_text(json.dumps(data))
    message = {
        "schema_version": "schema_version",
        "frame_id": "home frame",
        "source": "measured_tcp_pose",
        "tcp_pose": "six finite",
    }[key]
    with pytest.raises(ValueError, match=message):
        home.load_home(path, FRAME)


@pytest.mark.parametrize("payload", ["[]", "null", "not JSON", " " * (16 * 1024 + 1)])
def test_reject_invalid_or_oversize_home(tmp_path, payload):
    path = tmp_path / "cartesian_home.json"
    path.write_text(payload)
    with pytest.raises(ValueError, match="schema_version|Expecting value|16 KiB"):
        home.load_home(path, FRAME)


def _poses():
    start = (np.array([0.1, -0.3, 0.2]), Rotation.from_euler("xyz", [0.2, -0.4, 0.3]).as_matrix())
    delta = Rotation.from_rotvec(np.array([0.5, 0.3, -0.2])).as_matrix()
    goal = (np.array([0.22, -0.4, 0.1]), delta @ start[1])
    return start, goal


def test_quintic_endpoints_midpoint_and_speed_bounds():
    start, goal = _poses()
    path = home.ReturnHomePath(start, goal, home.HOME_LINEAR_SPEED, home.HOME_ANGULAR_SPEED)
    first, elapsed = path.candidate(0.0)
    np.testing.assert_allclose(first[0], start[0])
    np.testing.assert_allclose(first[1], start[1])
    assert elapsed == 0.0
    middle, _ = path.candidate(path.duration / 2.0)
    np.testing.assert_allclose(middle[0], (start[0] + goal[0]) / 2.0)
    middle_angle = Rotation.from_matrix(middle[1] @ start[1].T).magnitude()
    assert middle_angle == pytest.approx(path.angle / 2.0)

    dt = path.duration / 1000.0
    previous = start
    first_linear_speed = None
    for _ in range(1000):
        pose, proposed = path.candidate(dt)
        linear_speed = np.linalg.norm(pose[0] - previous[0]) / dt
        angular_speed = Rotation.from_matrix(pose[1] @ previous[1].T).magnitude() / dt
        assert linear_speed <= home.HOME_LINEAR_SPEED + 1e-10
        assert angular_speed <= home.HOME_ANGULAR_SPEED + 1e-10
        if first_linear_speed is None:
            first_linear_speed = linear_speed
        path.elapsed = proposed
        previous = pose
    assert first_linear_speed < home.HOME_LINEAR_SPEED * 1e-4
    assert linear_speed < home.HOME_LINEAR_SPEED * 1e-4
    np.testing.assert_allclose(previous[0], goal[0], atol=1e-12)
    np.testing.assert_allclose(previous[1], goal[1], atol=1e-12)
    final, elapsed = path.candidate(1.0)
    np.testing.assert_array_equal(final[0], goal[0])
    np.testing.assert_array_equal(final[1], goal[1])
    assert elapsed == path.duration


def test_feedback_can_pause_and_resume_path_time():
    path = home.ReturnHomePath(*_poses(), home.HOME_LINEAR_SPEED, home.HOME_ANGULAR_SPEED)
    candidate, proposed = path.candidate(0.1)
    for _ in range(20):
        paused, next_time = path.candidate(0.1)
        np.testing.assert_array_equal(paused[0], candidate[0])
        np.testing.assert_array_equal(paused[1], candidate[1])
        assert next_time == proposed
        assert path.elapsed == 0.0
    path.elapsed = proposed
    _, next_time = path.candidate(0.1)
    assert next_time == pytest.approx(0.2)
    _, unchanged = path.candidate(-1.0)
    assert unchanged == proposed


def test_half_turn_is_rejected_before_motion():
    start = (np.zeros(3), np.eye(3))
    goal = (np.zeros(3), Rotation.from_rotvec([np.pi, 0.0, 0.0]).as_matrix())
    with pytest.raises(ValueError, match="differs by pi"):
        home.ReturnHomePath(start, goal, home.HOME_LINEAR_SPEED, home.HOME_ANGULAR_SPEED)


@pytest.mark.parametrize("speed", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_speed_is_rejected(speed):
    with pytest.raises(ValueError, match="positive and finite"):
        home.ReturnHomePath(*_poses(), speed, home.HOME_ANGULAR_SPEED)


def test_zero_distance_path_keeps_the_measured_pose():
    start, _ = _poses()
    path = home.ReturnHomePath(start, start, home.HOME_LINEAR_SPEED, home.HOME_ANGULAR_SPEED)
    assert path.duration == 2.0
    pose, _ = path.candidate(1.0)
    np.testing.assert_allclose(pose[0], start[0])
    np.testing.assert_allclose(pose[1], start[1])
