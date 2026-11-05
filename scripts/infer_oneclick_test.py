import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "infer_oneclick.sh"
RUNNER = ROOT / "scripts/run_ar3_lift_robot.sh"


def _executable(path, contents="#!/bin/bash\nexit 0\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(0o755)
    return path


@pytest.fixture
def environment(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("INFER_", "OPENPI_"))}
    base = tmp_path / "devices with spaces"
    gripper = base / "V3_Gripper"
    arm = base / "arm"
    tilt = base / "tilt"
    cameras = base / "camera launch"
    data_home = base / "openpi"
    checkpoint = base / "checkpoint"
    for path in [
        arm / "jazzy.sh",
        arm / "install/local_setup.bash",
        arm / "install/ar3_cartesian_impedance/share/ar3_cartesian_impedance/config/rt.yaml",
        arm / "scripts/ar3_wait_cartesian_state.py",
        tilt / "camera_tilt_ros2.py",
        tilt / "usb_servo.py",
        cameras / "gemini_330_series.launch.py",
        cameras / "gemini305.launch.py",
        data_home / "big_vision/paligemma_tokenizer.model",
        checkpoint / "assets/ar3/norm_stats.json",
        base / "setup.bash",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (checkpoint / "params").mkdir()
    _executable(arm / "install/ar3_cartesian_impedance/lib/ar3_cartesian_impedance/cartesian_impedance_node")
    # Import checks (-c) are local and not logged; helper runs are.
    _executable(
        gripper / ".venv-ros/bin/python",
        """#!/bin/bash
[[ "$1" != -c ]] || exit "${INFER_TEST_IMPORT_EXIT:-0}"
printf 'robot-python %s\\n' "$*" >> "$INFER_TEST_CALLS"
if [[ " $* " == *" --connect "* ]]; then
    exit "${INFER_TEST_GRIPPER_PREPARE_EXIT:-0}"
elif [[ " $* " == *" --check "* ]]; then
    exit "${INFER_TEST_GRIPPER_FEEDBACK_EXIT:-0}"
fi
""",
    )
    servo_python = _executable(tilt / "servo-python")
    mock_bin = base / "bin"
    for name in ("ssh", "ping", "curl"):
        _executable(mock_bin / name, '#!/bin/bash\nprintf "%s\\n" "$0 $*" >> "$INFER_TEST_CALLS"\n')
    _executable(
        mock_bin / "ip",
        """#!/bin/bash
printf 'ip %s\\n' "$*" >> "$INFER_TEST_CALLS"
[[ "${1:-}" != -4 ]] || shift
if [[ "$*" == *"addr show"* ]]; then
    if [[ "${INFER_TEST_GRIPPER_ADDRESS_PRESENT:-1}" == 1 || -f "$INFER_TEST_IP_ADDED" ]]; then
        printf '1: %s inet %s scope global\\n' lo 192.168.127.100/24
    fi
elif [[ "$1 $2" == "route get" ]]; then
    printf '%s dev %s src %s\\n' "$3" "${INFER_TEST_ROUTE_INTERFACE:-lo}" 192.168.127.100
elif [[ "$1 $2" == "addr add" ]]; then
    touch "$INFER_TEST_IP_ADDED"
fi
""",
    )
    _executable(
        mock_bin / "sudo",
        """#!/bin/bash
printf 'sudo %s\\n' "$*" >> "$INFER_TEST_CALLS"
[[ "$1 $2 $3" == "ip addr add" ]] || exit 1
exec "$@"
""",
    )
    _executable(
        mock_bin / "timeout",
        """#!/bin/bash
printf '%s\\n' "$*" >> "$INFER_TEST_CALLS"
if [[ "$*" == *"node list"* ]]; then
    printf '%s\\n' "${INFER_TEST_NODES:-}"
elif [[ "$*" == *"/dev/tcp/"* ]]; then
    exit "${INFER_TEST_TCP_EXIT:-0}"
elif [[ "$*" == *"${INFER_TEST_FAIL:-__none__}"* ]]; then
    exit 1
fi
""",
    )
    _executable(
        mock_bin / "gnome-terminal",
        """#!/bin/bash
printf '%s\\n' "GNOME_TERMINAL_SERVICE=:mock" "GNOME_TERMINAL_SCREEN=/mock"
printf 'terminal ' >> "$INFER_TEST_CALLS"
printf '%q ' "$@" >> "$INFER_TEST_CALLS"
printf '\\n' >> "$INFER_TEST_CALLS"
""",
    )
    env.update(
        {
            "INFER_AR3_WS": str(arm),
            "INFER_GRIPPER_DIR": str(gripper),
            "INFER_TILT_DIR": str(tilt),
            "INFER_SERVO_PYTHON": str(servo_python),
            "INFER_CAMERA_LAUNCH_DIR": str(cameras),
            "INFER_ROS_SETUP": str(base / "setup.bash"),
            "INFER_ARM_INTERFACE": "lo",
            "INFER_CHECKPOINT": str(checkpoint),
            "INFER_TEST_CALLS": str(tmp_path / "calls"),
            "INFER_TEST_IP_ADDED": str(tmp_path / "gripper-address-added"),
            "OPENPI_DATA_HOME": str(data_home),
            "OPENPI_PYTHON": sys.executable,
            "PATH": f"{mock_bin}:{env['PATH']}",
            "DISPLAY": ":mock",
            "XDG_RUNTIME_DIR": str(tmp_path),
        }
    )
    return env


def run(env, *args, cwd="/tmp", script=LAUNCHER):
    return subprocess.run(["bash", str(script), *args], cwd=cwd, env=env, text=True, capture_output=True, check=False)


def calls(env):
    path = Path(env["INFER_TEST_CALLS"])
    return path.read_text().splitlines() if path.exists() else []


def test_default_plan_starts_impedance_controller_without_teleoperation_or_policy_actions(environment):
    result = run(environment, "--dry-run")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    arm = next(line for line in lines if "cartesian_impedance_node" in line)
    for parameter in (
        "enable_motion:=true",
        "auto_power_on:=true",
        "workspace_radius_m:=0.45",
        "robot_ip:=192.168.0.160",
    ):
        assert parameter in arm
    front = next(line for line in lines if "camera_name:=front" in line)
    wrist = next(line for line in lines if "camera_name:=wrist" in line)
    assert "color_format:=MJPG" in front
    assert "enable_color_auto_exposure:=false" not in front
    assert "color_format:=YUYV" in wrist
    assert "enable_color_auto_exposure:=false" in wrist
    assert "color_white_balance:=4600" in wrist
    assert "startup_angle_deg:=40.0" in result.stdout
    assert "deploy.sh" in result.stdout
    assert "/ar3/cartesian_impedance/tau_m" in result.stdout
    for absent in ("[policy]", "[sensor]", "[tactile]", "--initialize"):
        assert absent not in result.stdout
    commands = "\n".join(line for line in lines if line.startswith("  "))
    for absent in ("--execute", "ar3_lift_robot_client", "teleop"):
        assert absent not in commands
    assert not calls(environment)


def test_check_is_local_only_and_requires_the_gripper_client(environment):
    result = run(environment, "--check", "--with-policy")
    assert result.returncode == 0, result.stderr
    assert "No hardware, network or ROS commands were run" in result.stdout
    environment["INFER_TEST_IMPORT_EXIT"] = "1"
    result = run(environment, "--check")
    assert result.returncode != 0
    assert "cannot import DM_V3Gripper" in result.stderr
    assert not calls(environment)


@pytest.mark.parametrize("value", ["1", "0.9", "0.01", "0.45;touch /tmp/unwanted", ".45"])
def test_rejects_unsafe_workspace_radius_even_in_dry_run(environment, value):
    environment["INFER_WORKSPACE_RADIUS"] = value
    result = run(environment, "--dry-run")
    assert result.returncode != 0
    assert "INFER_WORKSPACE_RADIUS" in result.stderr
    assert not calls(environment)


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        ("/ar3_cartesian_impedance_teleop", "Teleoperation is running"),
        ("/gripper_teleop", "Teleoperation is running"),
        ("/ar3_lift_robot_client", "robot client is running"),
        ("/cartesian_impedance_node", "arm controller already exists"),
    ],
)
def test_command_sources_or_existing_controller_block_startup_before_any_terminal(environment, nodes, message):
    environment["INFER_TEST_NODES"] = nodes
    result = run(environment)
    assert result.returncode != 0
    assert message in result.stderr
    log = "\n".join(calls(environment))
    assert "node list" in log
    assert "terminal " not in log
    assert "sudo " not in log
    assert "robot-python" not in log


def test_terminal_children_receive_absolute_checkpoint_and_network_is_prepared_first(environment, tmp_path):
    environment["INFER_TEST_TCP_EXIT"] = "1"
    environment["INFER_TEST_GRIPPER_ADDRESS_PRESENT"] = "0"
    result = run(environment, "--with-policy", "--checkpoint", "devices with spaces/checkpoint", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    log = calls(environment)
    terminals = [i for i, line in enumerate(log) if line.startswith("terminal ")]
    tasks = []
    for index in terminals:
        args = shlex.split(log[index].removeprefix("terminal "))
        assert str(ROOT / "scripts/prepare_ar3_inference.sh") in args
        final_checkpoint = max(i for i, arg in enumerate(args) if arg == "--checkpoint")
        assert args[final_checkpoint + 1] == environment["INFER_CHECKPOINT"]
        tasks.append(args[args.index("--task") + 1])
    assert tasks == ["arm", "gripper", "front", "wrist", "tilt", "policy", "ready"]
    add_index = log.index("sudo ip addr add 192.168.127.100/24 dev lo")
    assert add_index < log.index("ip -4 route get 192.168.127.10") < min(terminals)
    assert not any("robot-python" in line for line in log)


def test_readiness_connects_gripper_then_waits_for_every_input_and_ends_with_readonly_check(environment):
    result = run(environment, "--task", "ready")
    assert result.returncode == 0, result.stderr
    assert "PREPARATION READY" in result.stdout
    assert "scripts/run_ar3_lift_robot.sh --check" in result.stdout
    log = calls(environment)

    def index(text):
        return next(i for i, line in enumerate(log) if text in line)

    order = [
        index("node list"),
        index("prepare_ar3_gripper.py --connect"),
        index("impedance-wait-state --timeout 180"),
        index("/ar3/cartesian_impedance/tau_m std_msgs/msg/Float64MultiArray"),
        index("/front/color/image_raw"),
        index("/wrist/color/image_raw"),
        index("--filter abs(m - 40.0) <= 2.0"),
        index("prepare_ar3_gripper.py --check"),
    ]
    assert order == sorted(order)
    prepare = log[order[1]]
    for argument in ("--channel can0", "--motor-id 1", "--timeout 180", "--gripper-address 192.168.127.10:50008"):
        assert argument in prepare
    assert "--initialize" not in "\n".join(log)
    assert "--execute" not in "\n".join(log)


def test_explicit_gripper_initialization_only_reaches_preparation(environment):
    environment["INFER_TEST_NODES"] = "/cartesian_impedance_node\n/front/front\n/wrist/wrist"
    result = run(environment, "--ready-only", "--initialize-gripper")
    assert result.returncode == 0, result.stderr
    log = calls(environment)
    prepare = next(line for line in log if "--connect" in line)
    check = next(line for line in log if "prepare_ar3_gripper.py --check" in line)
    assert "--initialize" in shlex.split(prepare)
    assert "--initialize" not in check
    assert not any(line.startswith("terminal ") for line in log)


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("INFER_TEST_GRIPPER_PREPARE_EXIT", "1", "Gripper preparation failed"),
        ("INFER_TEST_FAIL", "impedance-wait-state", "controller is not ready"),
        ("INFER_TEST_FAIL", "/cartesian_impedance/tau_m", "No qualifying sample"),
        ("INFER_TEST_GRIPPER_FEEDBACK_EXIT", "1", "Gripper feedback is not ready"),
    ],
)
def test_any_failed_readiness_step_never_reports_ready(environment, variable, value, message):
    environment[variable] = value
    result = run(environment, "--ready-only")
    assert result.returncode != 0
    assert message in result.stderr
    assert "PREPARATION READY" not in result.stdout
    if variable == "INFER_TEST_GRIPPER_PREPARE_EXIT":
        assert not any("topic echo" in line for line in calls(environment))


def test_robot_runner_requires_client_dependencies_before_starting_the_client(environment, tmp_path):
    log = tmp_path / "robot-calls"
    environment["INFER_ROBOT_PYTHON"] = str(
        _executable(
            tmp_path / "robot-python",
            f'#!/bin/bash\nprintf "%s\\n" "$*" >> {shlex.quote(str(log))}\n[[ "$1" != -c ]] || exit 1\n',
        )
    )
    result = run(environment, "--execute", script=RUNNER)
    assert result.returncode != 0
    assert "pip install" in result.stderr
    text = log.read_text()
    assert "rclpy" in text
    assert "DM_V3Gripper.gripper_pb2" in text
    assert "pyarrow" not in text
    assert "ar3_lift_robot_client.py" not in text
    result = run(environment, "--execute", "--dagger", "--record-dir=rollouts", script=RUNNER)
    assert result.returncode != 0
    assert "pyarrow.parquet av" in log.read_text().splitlines()[-1]


def test_robot_runner_runs_the_client_in_a_clean_ros_environment(environment, tmp_path):
    log = tmp_path / "robot-calls"
    environment["INFER_ROBOT_PYTHON"] = str(
        _executable(
            tmp_path / "robot-python",
            f'#!/bin/bash\n[[ "$1" == -c ]] || {{ env; printf "ARGS %s\\n" "$*"; }} >> {shlex.quote(str(log))}\n',
        )
    )
    environment.update({"OPENPI_HOST": "10.0.0.5", "OPENPI_PORT": "8100", "PYTHONPATH": "/model/src"})
    result = run(environment, "--check", "--port", "9000", script=RUNNER)
    assert result.returncode == 0, result.stderr
    lines = log.read_text().splitlines()
    assert "ROS_DOMAIN_ID=111" in lines
    assert not any(line.startswith(("PYTHONPATH=", "OPENPI_")) for line in lines)
    client = ROOT / "scripts/ar3_lift_robot_client.py"
    assert lines[-1] == f"ARGS -u {client} --host 10.0.0.5 --port 8100 --check --port 9000"


def test_gripper_gate_runner_checks_the_gripper_modules_and_runs_in_a_clean_ros_environment(environment, tmp_path):
    log = tmp_path / "gate-calls"
    environment["INFER_ROBOT_PYTHON"] = str(
        _executable(
            tmp_path / "robot-python",
            f'#!/bin/bash\n{{ env; printf "ARGS %s\\n" "$*"; }} >> {shlex.quote(str(log))}\n',
        )
    )
    environment.update({"INFER_GRIPPER_HOST": "10.0.0.9", "PYTHONPATH": "/model/src"})
    result = run(environment, "--check", script=ROOT / "scripts/ar3_dagger_gripper_teleop.sh")
    assert result.returncode == 0, result.stderr
    text = log.read_text()
    assert "rclpy numpy grpc DM_V3Gripper.gripper_pb2" in text
    lines = text.splitlines()
    assert "ROS_DOMAIN_ID=111" in lines
    assert not any(line.startswith("PYTHONPATH=") for line in lines)
    gate = ROOT / "scripts/ar3_dagger_gripper_teleop.py"
    assert lines[-1] == f"ARGS -u {gate} --gripper-address 10.0.0.9:50008 --check"
