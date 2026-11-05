import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from scripts import debug_ar3_lift_robot as debug

READY_NODES = {"/cartesian_impedance_node", "/front/camera", "/wrist/camera", "/camera_tilt_servo"}


def test_reuses_existing_hardware_and_rejects_another_command_owner():
    assert debug.hardware_start_options(READY_NODES, gripper_ready=True, skip_tilt=False) is None
    assert debug.hardware_start_options(READY_NODES, gripper_ready=False, skip_tilt=False) == [
        "--no-policy",
        "--no-ready-tab",
        "--no-arm",
        "--no-cameras",
        "--no-tilt",
    ]
    with pytest.raises(RuntimeError, match="conflicting"):
        debug.hardware_start_options(READY_NODES | {"/ar3_lift_robot_client"}, gripper_ready=True, skip_tilt=False)


@pytest.mark.parametrize(("missing", "preserved"), [("wrist", "front"), ("front", "wrist")])
def test_starts_only_the_missing_camera(missing, preserved):
    options = debug.hardware_start_options(READY_NODES - {f"/{missing}/camera"}, gripper_ready=True, skip_tilt=False)
    assert f"--no-{preserved}-camera" in options
    assert f"--no-{missing}-camera" not in options
    assert "--no-cameras" not in options
    assert "--no-arm" in options
    assert "--no-tilt" in options
    assert "--no-ssh" in options


def test_slow_discovery_does_not_start_an_already_running_camera(monkeypatch):
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[0] == "ros2":
            # The first discovery response misses front; the second misses wrist.
            nodes = READY_NODES - ({"/front/camera"} if len(calls) == 1 else {"/wrist/camera"})
            return SimpleNamespace(stdout="\n".join(nodes))
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(debug.subprocess, "run", run)
    monkeypatch.setattr(debug, "port_open", lambda *_: True)
    args, _ = debug.parse_args([])
    debug.prepare_hardware(args)
    assert len(calls) == 3
    assert calls[0] == calls[1] == ["ros2", "node", "list", "--no-daemon", "--spin-time", "3"]
    assert calls[2][2:] == ["--ready-only", "--no-policy"]


def test_preparation_keeps_existing_devices_and_checks_every_required_sensor(monkeypatch):
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="\n".join(READY_NODES - {"/camera_tilt_servo"}))

    monkeypatch.setattr(debug.subprocess, "run", run)
    monkeypatch.setattr(debug, "port_open", lambda *_: True)
    args, _ = debug.parse_args([])
    debug.prepare_hardware(args)
    assert calls[1][2:] == ["--no-policy", "--no-ready-tab", "--no-arm", "--no-cameras", "--no-ssh"]
    assert calls[2][2:] == ["--ready-only", "--no-policy"]
    assert not any("--initialize-gripper" in command for command in calls)


def test_check_configuration_does_not_prepare_hardware(monkeypatch):
    run = mock.Mock()
    monkeypatch.setattr(debug.subprocess, "run", run)
    args, client_args = debug.parse_args(["--hardware", "skip", "--", "--check"])
    debug.prepare_hardware(args)
    run.assert_not_called()
    assert "--check" in client_args


def test_print_plan_never_contacts_devices_or_starts_processes(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("print-plan attempted external access")

    monkeypatch.setattr(debug.subprocess, "run", forbidden)
    monkeypatch.setattr(debug.subprocess, "Popen", forbidden)
    monkeypatch.setattr(debug, "port_open", forbidden)
    monkeypatch.setattr(debug, "policy_healthy", forbidden)
    assert debug.main(["--print-plan", "--", "--execute", "--move-to-start"]) == 0


@pytest.fixture
def fake_client(monkeypatch):
    client = SimpleNamespace(
        parse_args=lambda _: SimpleNamespace(host="127.0.0.1", port=8000), main=mock.Mock(return_value=0)
    )
    monkeypatch.setattr(debug.importlib, "import_module", lambda _: client)
    return client


def test_failed_checkpoint_preflight_never_starts_hardware_or_policy(monkeypatch, fake_client):
    def reject(command, **_kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="Incomplete checkpoint")

    monkeypatch.setattr(debug.subprocess, "run", reject)
    popen = mock.Mock()
    hardware = mock.Mock()
    monkeypatch.setattr(debug.subprocess, "Popen", popen)
    monkeypatch.setattr(debug, "prepare_hardware", hardware)
    assert debug.main(["--", "--execute", "--move-to-start"]) == 1
    popen.assert_not_called()
    hardware.assert_not_called()
    fake_client.main.assert_not_called()


def test_occupied_policy_port_is_preserved_and_never_implicitly_reused(monkeypatch, fake_client):
    monkeypatch.setattr(debug.subprocess, "run", mock.Mock())
    monkeypatch.setattr(debug, "port_open", lambda *_: True)
    popen = mock.Mock()
    kill = mock.Mock()
    monkeypatch.setattr(debug.subprocess, "Popen", popen)
    monkeypatch.setattr(debug.os, "killpg", kill)
    assert debug.main(["--", "--execute"]) == 1
    popen.assert_not_called()
    kill.assert_not_called()
    fake_client.main.assert_not_called()


def test_explicit_policy_reuse_never_stops_the_existing_server(monkeypatch, fake_client):
    monkeypatch.setattr(debug, "policy_healthy", lambda *_: True)
    kill = mock.Mock()
    monkeypatch.setattr(debug.os, "killpg", kill)
    assert debug.main(["--reuse-policy", "--hardware", "skip", "--", "--check"]) == 0
    fake_client.main.assert_called_once()
    kill.assert_not_called()


def test_owned_policy_is_stopped_if_hardware_preparation_fails(monkeypatch, fake_client, tmp_path):
    monkeypatch.setattr(debug, "ROOT", tmp_path)
    monkeypatch.setattr(debug.subprocess, "run", mock.Mock())
    monkeypatch.setattr(debug, "port_open", lambda *_: False)
    monkeypatch.setattr(debug, "policy_healthy", lambda *_: True)
    process = mock.Mock(pid=12345)
    process.poll.return_value = None
    monkeypatch.setattr(debug.subprocess, "Popen", mock.Mock(return_value=process))
    monkeypatch.setattr(debug, "prepare_hardware", mock.Mock(side_effect=RuntimeError("not ready")))
    kill = mock.Mock()
    monkeypatch.setattr(debug.os, "killpg", kill)
    assert debug.main(["--", "--execute"]) == 1
    kill.assert_called_once_with(12345, debug.signal.SIGINT)
    process.wait.assert_called_once()
    fake_client.main.assert_not_called()
    command = debug.subprocess.Popen.call_args.args[0]
    assert command[:5] == [
        debug.sys.executable,
        str(tmp_path / "scripts/ar3_lift_policy_process.py"),
        "--parent-pid",
        str(os.getpid()),
        "--",
    ]
    assert command[5:7] == ["bash", str(tmp_path / "scripts/serve_ar3_lift.sh")]
    assert debug.subprocess.Popen.call_args.kwargs["start_new_session"] is True


@pytest.mark.parametrize("explicit_python", [False, True])
def test_debug_interpreter_isolates_python_while_retaining_preparation_environment(tmp_path, explicit_python):
    setup = tmp_path / "setup.bash"
    setup.write_text("export PYTHONPATH=/clean/ros\n")
    gripper_dir = tmp_path / "custom gripper"
    interpreter = gripper_dir / ".venv-ros/bin/python"
    interpreter.parent.mkdir(parents=True)
    output = tmp_path / "environment"
    interpreter.write_text(f"#!/bin/bash\nenv > {shlex.quote(str(output))}\n")
    interpreter.chmod(0o755)
    environment = {
        **{key: value for key, value in os.environ.items() if key != "INFER_ROBOT_PYTHON"},
        "INFER_ROS_SETUP": str(setup),
        "INFER_GRIPPER_DIR": str(gripper_dir),
        "INFER_ROS_DOMAIN_ID": "111",
        "PYTHONPATH": "/model/python3.11",
        "LD_LIBRARY_PATH": "/unrelated/ros/overlay",
        "SSH_AUTH_SOCK": "/test/ssh-agent",
        "DISPLAY": ":test",
    }
    if explicit_python:
        environment["INFER_ROBOT_PYTHON"] = str(interpreter)
        environment["INFER_GRIPPER_DIR"] = str(tmp_path / "unused")
    wrapper = Path(debug.__file__).with_name("ar3_lift_debug_python.sh")
    subprocess.run(["bash", str(wrapper), "-c", "unused"], env=environment, check=True)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines() if "=" in line)
    assert values["PYTHONPATH"] == "/clean/ros"
    # The execution sandbox may inject its own audit library after env -i.
    assert "/unrelated/ros/overlay" not in values.get("LD_LIBRARY_PATH", "")
    assert values["ROS_DOMAIN_ID"] == "111"
    assert values["PYTHONNOUSERSITE"] == "1"
    assert values["DISPLAY"] == ":test"
    assert values["SSH_AUTH_SOCK"] == "/test/ssh-agent"
