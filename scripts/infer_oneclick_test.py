import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "infer_oneclick.sh"
RUNNER = ROOT / "scripts/run_ar3_lift_robot.sh"
WRIST_METADATA = {
    "auto_exposure": 0,
    "exposure": 80,
    "gain": 16,
    "auto_white_balance": 0,
    "white_balance": 4600,
}


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
    lerobot = base / "lerobot"
    cameras = base / "camera launch"
    data_home = base / "openpi"
    checkpoint = base / "checkpoint"
    for path in [
        arm / "jazzy.sh",
        arm / "install/local_setup.bash",
        arm / "install/ar3_cartesian_impedance/share/ar3_cartesian_impedance/config/rt.yaml",
        arm / "scripts/ar3_wait_cartesian_state.py",
        arm / "scripts/ar3_cartesian_impedance_teleop.py",
        base / "bridge/config/MessageBridge.yaml",
        tilt / "camera_tilt_ros2.py",
        tilt / "usb_servo.py",
        lerobot / "scripts/ar3_tactile_bridge.py",
        lerobot / "examples/rokae_ar3/tactile_bridge.json",
        cameras / "gemini_330_series.launch.py",
        cameras / "gemini305.launch.py",
        data_home / "big_vision/paligemma_tokenizer.model",
        checkpoint / "assets/ar3/norm_stats.json",
        base / "setup.bash",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (checkpoint / "params").mkdir()
    (arm / ".venv-teleop").mkdir()
    _executable(base / "bridge/MessageBridge")
    _executable(base / "puppetry/Puppetry")
    _executable(arm / "install/ar3_cartesian_impedance/lib/ar3_cartesian_impedance/cartesian_impedance_node")
    _executable(lerobot / ".venv-ar3/bin/python")
    _executable(
        lerobot / "scripts/ar3_python.sh",
        '#!/bin/bash\nprintf "tactile-bridge ROS_DOMAIN_ID=%s AR3_WS=%s %s\\n" "$ROS_DOMAIN_ID" "$AR3_WS" "$*" '
        '>> "$INFER_TEST_CALLS"\n',
    )
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
    if [[ -v INFER_TEST_NODES_SECOND && -f "$INFER_TEST_NODE_QUERY_MARKER" ]]; then
        printf '%s\\n' "$INFER_TEST_NODES_SECOND"
    else
        printf '%s\\n' "${INFER_TEST_NODES:-}"
        [[ ! -v INFER_TEST_NODES_SECOND ]] || touch "$INFER_TEST_NODE_QUERY_MARKER"
    fi
elif [[ "$*" == *"/dev/tcp/"* ]]; then
    [[ " ${INFER_TEST_TCP_OPEN:-} " != *" ${@: -1} "* ]] || exit 0
    exit "${INFER_TEST_TCP_EXIT:-1}"
elif [[ "$*" == *"${INFER_TEST_FAIL:-__none__}"* ]]; then
    exit 1
elif [[ "$*" == *"/wrist/color/metadata"* ]]; then
    printf '%s\\n---\\n' "$INFER_TEST_WRIST_METADATA"
elif [[ "$*" == *"param set"* ]]; then
    printf 'Set parameter successful\n'
elif [[ "$*" == *"param get"* ]]; then
    printf 'Boolean value is: False\n'
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
            "INFER_LEROBOT_DIR": str(lerobot),
            "INFER_SERVO_PYTHON": str(servo_python),
            "INFER_CAMERA_LAUNCH_DIR": str(cameras),
            "INFER_WRIST_DRIVER_OVERLAY": "",
            "INFER_BRIDGE_DIR": str(base / "bridge"),
            "INFER_PUPPETRY_DIR": str(base / "puppetry"),
            "INFER_ROS_SETUP": str(base / "setup.bash"),
            "INFER_ARM_INTERFACE": "lo",
            "INFER_CHECKPOINT": str(checkpoint),
            "INFER_TEST_CALLS": str(tmp_path / "calls"),
            "INFER_TEST_IP_ADDED": str(tmp_path / "gripper-address-added"),
            "INFER_TEST_NODE_QUERY_MARKER": str(tmp_path / "node-query-seen"),
            "INFER_TEST_WRIST_METADATA": json.dumps(WRIST_METADATA),
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


def terminal_commands(env):
    return [shlex.split(line.removeprefix("terminal ")) for line in calls(env) if line.startswith("terminal ")]


def terminal_tasks(env):
    return [command[command.index("--task") + 1] for command in terminal_commands(env)]


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
    assert "enable_ir_auto_exposure:=false" in wrist
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


def _wrist_overlay(path):
    for relative, contents in {
        "lib/liborbbec_camera.so": "test library",
        "share/ament_index/resource_index/packages/orbbec_camera": "",
        "share/ament_index/resource_index/rclcpp_components/orbbec_camera": "orbbec_camera::OBCameraNodeDriver;lib/liborbbec_camera.so\n",
        "wrist_driver.json": '{"g305_v4l2_single_start": true}\n',
    }.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    return path


def test_wrist_overlay_is_applied_after_ros_setup_and_does_not_affect_front(environment, tmp_path):
    overlay = _wrist_overlay(tmp_path / "wrist driver overlay")
    environment["INFER_WRIST_DRIVER_OVERLAY"] = str(overlay)
    mock_bin = tmp_path / "mock ros"
    output = tmp_path / "ros calls"
    _executable(
        mock_bin / "ros2",
        f'#!/bin/bash\nprintf "%s\\n" "$AMENT_PREFIX_PATH" >> {shlex.quote(str(output))}\n',
    )
    Path(environment["INFER_ROS_SETUP"]).write_text(
        f'export AMENT_PREFIX_PATH=/system/ros:/existing/prefix\nexport PATH={shlex.quote(str(mock_bin))}:"$PATH"\n'
    )
    for task in ("wrist", "front"):
        result = run(environment, "--task", task)
        assert result.returncode == 0, result.stderr
    assert output.read_text().splitlines() == [
        f"{overlay}:/system/ros:/existing/prefix",
        "/system/ros:/existing/prefix",
    ]
    result = run(environment, "--dry-run")
    commands = [shlex.split(line.strip()) for line in result.stdout.splitlines() if line.startswith("  env ")]
    wrist = next(command for command in commands if "camera_name:=wrist" in command)
    front = next(command for command in commands if "camera_name:=front" in command)
    assert str(overlay) in wrist
    assert str(overlay) not in front
    assert "Wrist camera driver:" in result.stdout
    assert not calls(environment)


@pytest.mark.parametrize(
    "missing", [None, "lib/liborbbec_camera.so", "share/ament_index/resource_index/rclcpp_components/orbbec_camera"]
)
def test_invalid_wrist_overlay_blocks_startup_before_external_access(environment, tmp_path, missing):
    overlay = tmp_path / "invalid wrist overlay"
    if missing is not None:
        _wrist_overlay(overlay)
        (overlay / missing).unlink()
    environment["INFER_WRIST_DRIVER_OVERLAY"] = str(overlay)
    result = run(environment, "--check")
    assert result.returncode != 0
    assert "wrist driver overlay" in result.stderr
    assert not calls(environment)
    # The direct task entry point must also validate the overlay before executing ros2.
    result = run(environment, "--task", "wrist")
    assert result.returncode != 0
    assert "wrist driver overlay" in result.stderr
    assert not calls(environment)


def test_project_wrist_overlay_is_detected_by_default_but_can_be_disabled(environment, tmp_path):
    project = tmp_path / "project"
    script = project / "scripts/prepare_ar3_inference.sh"
    script.parent.mkdir(parents=True)
    script.write_text((ROOT / "scripts/prepare_ar3_inference.sh").read_text())
    overlay = _wrist_overlay(project / "outputs/orbbec_wrist_driver/overlay")
    environment.pop("INFER_WRIST_DRIVER_OVERLAY")
    result = run(environment, "--dry-run", script=script)
    assert result.returncode == 0, result.stderr
    wrist = next(shlex.split(line.strip()) for line in result.stdout.splitlines() if "camera_name:=wrist" in line)
    assert str(overlay) in wrist
    environment["INFER_WRIST_DRIVER_OVERLAY"] = ""
    result = run(environment, "--dry-run", script=script)
    assert result.returncode == 0, result.stderr
    assert str(overlay) not in result.stdout
    assert "Wrist camera driver: system ROS driver" in result.stdout
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
        ("/controller_manager", "controller"),
        ("/rokae_driver7", "controller"),
    ],
)
def test_conflicting_command_sources_block_startup_before_any_terminal(environment, nodes, message):
    environment["INFER_TEST_NODES"] = nodes
    result = run(environment)
    assert result.returncode != 0
    assert message in result.stderr
    log = "\n".join(calls(environment))
    assert "node list" in log
    assert "terminal " not in log
    assert "sudo " not in log
    assert "robot-python" not in log


@pytest.mark.parametrize(("existing", "missing"), [("front", "wrist"), ("wrist", "front")])
def test_only_missing_camera_is_started_without_replacing_the_existing_owner(environment, existing, missing):
    environment["INFER_TEST_NODES"] = f"/{existing}/{existing}"
    result = run(environment, "--no-arm", "--no-ssh", "--no-tilt", "--no-ready-tab", f"--no-{existing}-camera")
    assert result.returncode == 0, result.stderr
    terminals = [
        shlex.split(line.removeprefix("terminal ")) for line in calls(environment) if line.startswith("terminal ")
    ]
    assert [args[args.index("--task") + 1] for args in terminals] == [missing]


@pytest.mark.parametrize("camera", ["front", "wrist"])
def test_selected_camera_with_an_existing_owner_is_reused_without_manual_skip_flags(environment, camera):
    environment["INFER_TEST_NODES"] = f"/{camera}/{camera}"
    result = run(environment, "--no-arm", "--no-ssh", "--no-tilt", "--no-ready-tab")
    assert result.returncode == 0, result.stderr
    assert terminal_tasks(environment) == ["wrist" if camera == "front" else "front"]
    assert "reus" in result.stdout.lower()


def test_manual_startup_reuses_wrist_and_still_starts_other_devices_and_checks_every_sensor(environment):
    environment["INFER_TEST_NODES"] = "/wrist/wrist"
    result = run(environment)
    assert result.returncode == 0, result.stderr
    assert terminal_tasks(environment) == ["arm", "gripper", "front", "tilt", "ready"]
    ready_command = next(command for command in terminal_commands(environment) if command[-1] == "ready")
    assert "--no-cameras" not in ready_command
    assert "--no-wrist-camera" not in ready_command
    assert "--no-front-camera" not in ready_command
    assert "Wait for /front/color/image_raw" in result.stdout
    assert "Wait for /wrist/color/image_raw" in result.stdout


def test_existing_impedance_controller_is_reused(environment):
    environment["INFER_TEST_NODES"] = "/cartesian_impedance_node"
    result = run(environment)
    assert result.returncode == 0, result.stderr
    assert terminal_tasks(environment) == ["gripper", "front", "wrist", "tilt", "ready"]
    assert "Wait for the controller RT state" in result.stdout


def _all_devices_ready(environment):
    environment["INFER_TEST_NODES"] = "/cartesian_impedance_node\n/front/front\n/wrist/wrist\n/camera_tilt_servo"
    environment["INFER_TEST_TCP_EXIT"] = "0"


def test_manual_startup_reuses_all_devices_and_checks_all_inputs_in_the_same_terminal(environment):
    _all_devices_ready(environment)
    result = run(environment)
    assert result.returncode == 0, result.stderr
    assert not terminal_commands(environment)
    assert "PREPARATION READY" in result.stdout
    log = "\n".join(calls(environment))
    for required in (
        "prepare_ar3_gripper.py --connect",
        "impedance-wait-state --timeout 180",
        "/ar3/cartesian_impedance/tau_m",
        "/front/color/image_raw",
        "/wrist/color/image_raw",
        "/wrist/color/metadata",
        "/camera_tilt/state_deg",
        "prepare_ar3_gripper.py --check",
    ):
        assert required in log
    assert "--initialize" not in log
    assert "Wrist frame metadata verified" in result.stdout


@pytest.mark.parametrize(("key", "value"), [("auto_white_balance", 1), ("white_balance", 4500), ("gain", None)])
def test_reused_wrist_with_wrong_or_missing_control_metadata_cannot_report_ready(environment, key, value):
    _all_devices_ready(environment)
    metadata = dict(WRIST_METADATA)
    if value is None:
        metadata.pop(key)
    else:
        metadata[key] = value
    environment["INFER_TEST_WRIST_METADATA"] = json.dumps(metadata)
    result = run(environment)
    assert result.returncode != 0
    assert f"{key}: expected" in result.stderr
    assert f"got {value!r}" in result.stderr
    assert "PREPARATION READY" not in result.stdout
    assert not terminal_commands(environment)


@pytest.mark.parametrize("mode", ["timeout", "empty", "invalid_json"])
def test_wrist_metadata_without_a_valid_sample_cannot_report_ready(environment, mode):
    _all_devices_ready(environment)
    if mode == "timeout":
        environment["INFER_TEST_FAIL"] = "/wrist/color/metadata"
    else:
        environment["INFER_TEST_WRIST_METADATA"] = "" if mode == "empty" else "not JSON"
    result = run(environment)
    assert result.returncode != 0
    expected = "No wrist frame metadata" if mode == "timeout" else "Invalid wrist frame metadata"
    assert expected in result.stderr
    assert "PREPARATION READY" not in result.stdout
    assert not terminal_commands(environment)


def test_reused_devices_must_still_pass_readiness(environment):
    _all_devices_ready(environment)
    environment["INFER_TEST_FAIL"] = "/wrist/color/image_raw"
    result = run(environment)
    assert result.returncode != 0
    assert "No qualifying sample" in result.stderr
    assert "PREPARATION READY" not in result.stdout
    assert not terminal_commands(environment)


def test_supervisor_reusing_all_devices_can_defer_readiness(environment):
    _all_devices_ready(environment)
    result = run(environment, "--no-ready-tab")
    assert result.returncode == 0, result.stderr
    assert not terminal_commands(environment)
    assert not any("robot-python" in line for line in calls(environment))


def test_two_camera_discovery_snapshots_are_merged_before_reusing_devices(environment):
    _all_devices_ready(environment)
    environment["INFER_TEST_NODES"] = "/cartesian_impedance_node\n/wrist/wrist\n/camera_tilt_servo"
    environment["INFER_TEST_NODES_SECOND"] = "/front/front"
    result = run(environment)
    assert result.returncode == 0, result.stderr
    assert not terminal_commands(environment)
    assert "PREPARATION READY" in result.stdout
    assert sum("node list --no-daemon --spin-time 3" in line for line in calls(environment)) >= 3


def test_conflicting_command_source_discovered_on_second_scan_still_blocks_startup(environment):
    environment["INFER_TEST_NODES"] = "/wrist/wrist"
    environment["INFER_TEST_NODES_SECOND"] = "/ar3_cartesian_impedance_teleop"
    result = run(environment)
    assert result.returncode != 0
    assert "Teleoperation is running" in result.stderr
    assert not terminal_commands(environment)
    assert not any("robot-python" in line for line in calls(environment))


def test_occupied_policy_port_is_not_silently_reused(environment):
    _all_devices_ready(environment)
    result = run(environment, "--with-policy")
    assert result.returncode != 0
    assert "Policy port is occupied" in result.stderr
    assert not terminal_commands(environment)
    assert not any("robot-python" in line for line in calls(environment))


@pytest.mark.parametrize(
    ("skip", "unused_launch"),
    [("front", "gemini_330_series.launch.py"), ("wrist", "gemini305.launch.py")],
)
def test_camera_preflight_only_requires_selected_camera_files(environment, skip, unused_launch, tmp_path):
    (Path(environment["INFER_CAMERA_LAUNCH_DIR"]) / unused_launch).unlink()
    if skip == "wrist":
        environment["INFER_WRIST_DRIVER_OVERLAY"] = str(tmp_path / "unused missing overlay")
    result = run(environment, "--check", f"--no-{skip}-camera")
    assert result.returncode == 0, result.stderr
    assert not calls(environment)


@pytest.mark.parametrize(
    ("flag", "expected_cameras"),
    [("--no-front-camera", ["wrist"]), ("--no-wrist-camera", ["front"]), ("--no-cameras", [])],
)
def test_readiness_only_waits_for_selected_cameras(environment, flag, expected_cameras):
    result = run(environment, "--ready-only", flag)
    assert result.returncode == 0, result.stderr
    assert "PREPARATION READY" in result.stdout
    log = "\n".join(calls(environment))
    for camera in ("front", "wrist"):
        assert (f"/{camera}/color/image_raw" in log) == (camera in expected_cameras)
    assert ("/wrist/color/metadata" in log) == ("wrist" in expected_cameras)


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
        index("/wrist/color/metadata"),
        index("--filter abs(m - 40.0) <= 2.0"),
        index("prepare_ar3_gripper.py --check"),
    ]
    assert order == sorted(order)
    prepare = log[order[1]]
    for argument in ("--channel can0", "--motor-id 1", "--timeout 180", "--gripper-address 192.168.127.10:50008"):
        assert argument in prepare
    assert "--initialize" not in "\n".join(log)
    assert "--execute" not in "\n".join(log)


def test_supervisor_can_start_devices_without_a_competing_readiness_tab(environment):
    result = run(environment, "--no-ready-tab")
    assert result.returncode == 0, result.stderr
    terminals = [
        shlex.split(line.removeprefix("terminal ")) for line in calls(environment) if line.startswith("terminal ")
    ]
    tasks = [args[args.index("--task") + 1] for args in terminals]
    assert tasks == ["arm", "gripper", "front", "wrist", "tilt"]
    assert "Run --ready-only" in result.stdout
    result = run(environment, "--no-ready-tab", "--ready-only")
    assert result.returncode == 0, result.stderr
    assert "PREPARATION READY" in result.stdout


def test_readiness_waits_for_its_existing_owner_before_connecting_gripper(environment, tmp_path):
    state_root = Path(environment["XDG_RUNTIME_DIR"]) / f"ar3-lift-infer-{os.getuid()}-111"
    state_root.mkdir()
    environment["INFER_TIMEOUT"] = "5"
    output_path = tmp_path / "waiting-readiness.log"
    with (state_root / "ready.lock").open("w") as owner, output_path.open("w") as output:
        fcntl.flock(owner, fcntl.LOCK_EX)
        process = subprocess.Popen(
            ["bash", str(LAUNCHER), "--ready-only"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 3
            while "Waiting for the readiness lock" not in output_path.read_text() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert "Waiting for the readiness lock" in output_path.read_text()
            assert process.poll() is None
            assert not any("robot-python" in line for line in calls(environment))
            fcntl.flock(owner, fcntl.LOCK_UN)
            assert process.wait(timeout=5) == 0, output_path.read_text()
        finally:
            fcntl.flock(owner, fcntl.LOCK_UN)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
    log = calls(environment)
    assert sum("prepare_ar3_gripper.py --connect" in line for line in log) == 1
    assert any("node list --no-daemon --spin-time 3" in line for line in log)
    assert "PREPARATION READY" in output_path.read_text()


@pytest.mark.parametrize("task", ["arm", "front", "wrist", "tilt"])
def test_device_task_locks_still_refuse_a_second_owner_immediately(environment, task):
    state_root = Path(environment["XDG_RUNTIME_DIR"]) / f"ar3-lift-infer-{os.getuid()}-111"
    state_root.mkdir()
    with (state_root / f"{task}.lock").open("w") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--task", task],
            env=environment,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    assert result.returncode != 0
    assert f"Task {task} already has an owner" in result.stderr
    assert not calls(environment)


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


def test_dagger_adds_teleoperation_and_gated_gripper_tasks_but_rejects_running_ones(environment):
    result = run(environment, "--dagger")
    assert result.returncode == 0, result.stderr
    log = calls(environment)
    tasks = [shlex.split(line)[-1] for line in log if line.startswith("terminal ")]
    assert tasks == ["arm", "gripper", "front", "wrist", "tilt", "bridge", "puppetry", "teleop", "gate", "ready"]
    assert "./MessageBridge" in result.stdout
    assert "./Puppetry" in result.stdout
    teleop = next(line for line in result.stdout.splitlines() if "impedance-teleop" in line)
    assert "return_home_workspace_radius_m:=0.45" in teleop
    gate = next(line for line in result.stdout.splitlines() if "ar3_dagger_gripper_teleop.sh" in line)
    assert "--wait 360 --gripper-address 192.168.127.10:50008" in gate
    assert "ar3_lift_robot_client" not in result.stdout
    for nodes, message in [
        ("/ar3_cartesian_impedance_teleop", "Teleoperation is running"),
        ("/ar3_dagger_gripper_gate", "gripper gate already runs"),
    ]:
        environment["INFER_TEST_NODES"] = nodes
        result = run(environment, "--dagger")
        assert result.returncode != 0
        assert message in result.stderr


def test_dagger_readiness_enables_the_teleoperation_after_every_input_and_waits_for_the_gate(environment):
    environment["INFER_TEST_NODES"] = "/ar3_cartesian_impedance_teleop\n/ar3_dagger_gripper_gate"
    result = run(environment, "--task", "ready", "--dagger")
    assert result.returncode == 0, result.stderr
    assert "PREPARATION READY" in result.stdout
    assert "--check --dagger" in result.stdout
    log = calls(environment)

    def index(text):
        return next(i for i, line in enumerate(log) if text in line)

    order = [
        index("prepare_ar3_gripper.py --connect"),
        index("prepare_ar3_gripper.py --check"),
        index("/trigger_positions std_msgs/msg/Float64MultiArray"),
        index("param set /ar3_cartesian_impedance_teleop dry_run false"),
        index("param get /ar3_cartesian_impedance_teleop dry_run"),
        max(i for i, line in enumerate(log) if "node list" in line),
    ]
    assert order == sorted(order)
    assert order[-1] > order[-2]


def test_tactile_starts_the_recorded_tac_g2_pipeline_and_reuses_running_parts(environment):
    result = run(environment, "--tactile")
    assert result.returncode == 0, result.stderr
    assert terminal_tasks(environment) == ["arm", "sensor", "gripper", "front", "wrist", "tilt", "tactile", "ready"]
    assert "grpc_server.py" in result.stdout
    assert "ar3_tactile_bridge.py --preview" in result.stdout
    Path(environment["INFER_TEST_CALLS"]).unlink()
    # Listening sensor/gripper ports and a running bridge node are reused, as in the recordings' oneClick.sh.
    environment["INFER_TEST_NODES"] = "/v3_gripper_tactile_recording"
    environment["INFER_TEST_TCP_EXIT"] = "0"
    result = run(environment, "--tactile")
    assert result.returncode == 0, result.stderr
    assert terminal_tasks(environment) == ["arm", "front", "wrist", "tilt", "ready"]
    # The bridge files are required only with --tactile.
    (Path(environment["INFER_LEROBOT_DIR"]) / "scripts/ar3_tactile_bridge.py").unlink()
    assert run(environment, "--check").returncode == 0
    result = run(environment, "--check", "--tactile")
    assert result.returncode != 0
    assert "ar3_tactile_bridge.py" in result.stderr


def test_tac_g2_tasks_start_in_the_recording_order(environment):
    environment["INFER_TIMEOUT"] = "1"
    result = run(environment, "--task", "gripper", "--tactile")
    assert result.returncode != 0
    assert "Nothing listens on 192.168.127.10:50051" in result.stderr
    assert not any("/bin/ssh " in line for line in calls(environment))
    environment["INFER_TEST_TCP_OPEN"] = "50051"
    result = run(environment, "--task", "gripper", "--tactile")
    assert result.returncode == 0, result.stderr
    assert any("/bin/ssh " in line and "deploy.sh" in line for line in calls(environment))
    # The bridge needs both fingers' servers and publishes in the robot's ROS domain.
    result = run(environment, "--task", "tactile", "--tactile")
    assert result.returncode != 0
    assert "Nothing listens on 192.168.127.10:50052" in result.stderr
    environment["INFER_TEST_TCP_OPEN"] = "50051 50052"
    result = run(environment, "--task", "tactile", "--tactile")
    assert result.returncode == 0, result.stderr
    bridge = [line for line in calls(environment) if line.startswith("tactile-bridge ")]
    assert bridge == [
        f"tactile-bridge ROS_DOMAIN_ID=111 AR3_WS={environment['INFER_AR3_WS']} "
        f"{environment['INFER_LEROBOT_DIR']}/scripts/ar3_tactile_bridge.py --preview"
    ]


def test_tactile_readiness_requires_samples_of_both_fingers(environment):
    assert run(environment, "--task", "ready").returncode == 0
    assert not any("/v3_gripper/tactile/" in line for line in calls(environment))
    result = run(environment, "--task", "ready", "--tactile")
    assert result.returncode == 0, result.stderr
    assert "PREPARATION READY" in result.stdout
    log = "\n".join(calls(environment))
    for sensor in ("sensor_0", "sensor_2"):
        assert f"/v3_gripper/tactile/{sensor}/wrench geometry_msgs/msg/WrenchStamped" in log
        assert f"/v3_gripper/tactile/{sensor}/depth sensor_msgs/msg/Image" in log
    environment["INFER_TEST_FAIL"] = "/v3_gripper/tactile/sensor_2/depth"
    result = run(environment, "--task", "ready", "--tactile")
    assert result.returncode != 0
    assert "No qualifying sample on /v3_gripper/tactile/sensor_2/depth" in result.stderr
    assert "PREPARATION READY" not in result.stdout


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
