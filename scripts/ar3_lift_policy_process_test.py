"""Lifecycle tests use only short-lived dummy processes, never model/ROS services."""

import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

from scripts import ar3_lift_policy_process as owned

WRAPPER = Path(owned.__file__).resolve()
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death process contract")


def test_wrong_parent_refuses_to_execute_the_command(tmp_path):
    marker = tmp_path / "must-not-run"
    result = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--parent-pid",
            str(os.getpid() + 1),
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(marker),
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 1
    assert "owner process exited before policy startup" in result.stderr
    assert not marker.exists()


def test_parent_loss_during_signal_registration_refuses_to_execute(monkeypatch):
    monkeypatch.setattr(owned.os, "getppid", mock.Mock(side_effect=[1234, 1235]))
    libc = mock.Mock()
    libc.prctl.return_value = 0
    monkeypatch.setattr(owned.ctypes, "CDLL", mock.Mock(return_value=libc))
    execute = mock.Mock()
    monkeypatch.setattr(owned.os, "execvp", execute)
    assert owned.main(["--parent-pid", "1234", "--", "must-not-run"]) == 1
    libc.prctl.assert_called_once()
    execute.assert_not_called()


def test_parent_death_registration_failure_refuses_to_execute(monkeypatch):
    monkeypatch.setattr(owned.os, "getppid", lambda: 1234)
    libc = mock.Mock()
    libc.prctl.return_value = -1
    monkeypatch.setattr(owned.ctypes, "CDLL", mock.Mock(return_value=libc))
    monkeypatch.setattr(owned.ctypes, "get_errno", lambda: 1)
    execute = mock.Mock()
    monkeypatch.setattr(owned.os, "execvp", execute)
    assert owned.main(["--parent-pid", "1234", "--", "must-not-run"]) == 1
    execute.assert_not_called()


# A dedicated supervisor adopts and reaps the orphan, avoiding zombie processes
# in test runners whose PID 1 does not reap children. No changes to pytest's process.
_SUPERVISOR = r'''
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
wrapper, pid_path, ready_path = sys.argv[1:]
parent_code = r"""
import os
from pathlib import Path
import subprocess
import sys
import time
wrapper, pid_path, ready_path = sys.argv[1:]
worker = 'import os, sys, time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)'
child = subprocess.Popen([
    sys.executable, wrapper, '--parent-pid', str(os.getpid()), '--',
    'bash', '-c', 'exec env "$@"', 'owned-policy', sys.executable, '-c', worker, ready_path,
], start_new_session=True)
Path(pid_path).write_text(str(child.pid))
time.sleep(60)
"""
parent = subprocess.Popen([sys.executable, '-c', parent_code, wrapper, pid_path, ready_path])
child_pid = None
reaped = False
try:
    deadline = time.monotonic() + 5
    while not Path(ready_path).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    child_pid = int(Path(pid_path).read_text())
    assert Path(ready_path).exists(), 'owned command did not become ready'
    assert int(Path(ready_path).read_text()) == child_pid, 'exec must preserve the owned PID'
    assert os.getpgid(child_pid) == child_pid, 'policy must retain its separate process group'
    parent.kill()  # Equivalent to the debugger bypassing Python cleanup with SIGKILL.
    parent.wait(timeout=2)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        pid, status = os.waitpid(child_pid, os.WNOHANG)
        if pid:
            reaped = True
            assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
            break
        time.sleep(0.01)
    assert reaped, 'owned command survived its debug parent'
finally:
    if parent.poll() is None:
        parent.kill()
        parent.wait(timeout=2)
    if child_pid is None and Path(pid_path).exists():
        child_pid = int(Path(pid_path).read_text())
    if child_pid is not None and not reaped:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(child_pid, 0)
'''


def test_sigkill_of_debug_parent_kills_the_separate_policy_group_after_exec(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", _SUPERVISOR, str(WRAPPER), str(tmp_path / "pid"), str(tmp_path / "ready")],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
