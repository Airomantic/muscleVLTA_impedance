from collections import deque
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import ar3_lift_robot_client as client
from scripts import prepare_ar3_gripper as helper


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class RpcError(Exception):
    pass


def state(**values):
    return SimpleNamespace(
        **{
            "connected": True,
            "valid": True,
            "stale": False,
            "init_state": 1,
            "grip_state": 1,
            "position": 432,
            "error": 0,
            "update_age_ms": 0.0,
            "message": "",
            **values,
        }
    )


class ConnectConfig:
    def __init__(self, **values):
        self.values = values

    def SerializeToString(self):  # noqa: N802
        return json.dumps(self.values).encode()


@pytest.fixture
def rig(monkeypatch):
    clock = Clock()
    calls, bindings, opened = [], [], []
    states = deque([state()])
    responses = {}
    fields = {
        name: SimpleNamespace(number=number, type=kind) for name, (number, kind) in client.GRIPPER_STATE_FIELDS.items()
    }
    protobuf = SimpleNamespace(
        StateResponse=SimpleNamespace(
            DESCRIPTOR=SimpleNamespace(fields_by_name=fields), FromString=lambda value: value
        ),
        Response=SimpleNamespace(FromString=lambda value: value),
        ConnectConfig=ConnectConfig,
    )

    class Channel:
        closed = False

        def unary_unary(self, path, **kwargs):
            name = path.rsplit("/", 1)[-1]
            bindings.append(name)

            def rpc(request, *, timeout):
                assert timeout > 0
                calls.append((name, request, timeout))
                if name == "GetAllState":
                    result = states.popleft() if len(states) > 1 else states[0]
                else:
                    result = responses.get(name, SimpleNamespace(success=True, message="ok"))
                if isinstance(result, BaseException):
                    raise result
                return result

            return rpc

        def close(self):
            self.closed = True

    channel = Channel()

    def open_channel(address, *, options):
        opened.append(address)
        return channel

    monkeypatch.setitem(sys.modules, "grpc", SimpleNamespace(insecure_channel=open_channel, RpcError=RpcError))
    monkeypatch.setitem(sys.modules, "DM_V3Gripper", SimpleNamespace(gripper_pb2=protobuf))
    return SimpleNamespace(
        clock=clock,
        channel=channel,
        calls=calls,
        bindings=bindings,
        states=states,
        responses=responses,
        protobuf=protobuf,
        opened=opened,
    )


def run(rig, *argv):
    return helper.prepare(helper.parse_args(["--connect", *argv]), clock=rig.clock, sleep=rig.clock.sleep)


def writes(rig):
    return [name for name, _, _ in rig.calls if name != "GetAllState"]


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--initialize"],
        ["--check", "--initialize"],
        ["--connect", "--check"],
        ["--connect", "--timeout", "nan"],
        ["--connect", "--motor-id", "0"],
    ],
)
def test_cli_requires_one_explicit_mode_and_valid_parameters_before_opening_hardware(rig, argv):
    with pytest.raises(SystemExit) as exc:
        helper.main(argv)
    assert exc.value.code == 2
    assert not rig.opened


def test_disconnected_gripper_is_connected_once_then_needs_fresh_feedback(rig):
    rig.states.extendleft(
        reversed([state(connected=False, valid=False, stale=True, init_state=-1), state(valid=False, stale=True)])
    )
    run(rig)
    assert writes(rig) == ["Connect"]
    _, payload, timeout = next(call for call in rig.calls if call[0] == "Connect")
    assert json.loads(payload) == {
        "channel": "can0",
        "motor_id": 1,
        "read_interval_s": 0.05,
        "read_timeout_s": 0.1,
        "retry_count": 2,
        "stale_timeout_s": 0.5,
        "command_min_interval_s": 0.02,
    }
    assert timeout == 5.0
    assert [name for name, _, _ in rig.calls].count("GetAllState") == 3
    assert rig.channel.closed
    assert "Disconnect" not in rig.bindings


def test_ready_gripper_needs_no_write_rpc_and_keeps_the_server_connection(rig):
    run(rig, "--initialize")
    assert rig.bindings == ["GetAllState"]
    assert rig.channel.closed


def test_uninitialized_gripper_is_never_calibrated_without_the_explicit_flag(rig):
    rig.states.clear()
    rig.states.extend([state(connected=False), state(init_state=0)])
    with pytest.raises(RuntimeError, match="infer_oneclick.sh --ready-only --initialize-gripper"):
        run(rig)
    assert writes(rig) == ["Connect"]
    assert rig.channel.closed


def test_explicit_initialization_waits_for_completion_without_target_commands(rig):
    rig.states.clear()
    rig.states.extend([state(init_state=0), state(init_state=2, position=-12), state()])
    run(rig, "--initialize")
    assert writes(rig) == ["Enable", "Initialization"]
    assert rig.channel.closed


@pytest.mark.parametrize("name", ["Enable", "Initialization"])
def test_failed_initialization_attempts_a_stop(rig, name, capsys):
    rig.states[0] = state(init_state=0)
    rig.responses[name] = RpcError("timeout")
    with pytest.raises(RuntimeError, match=name):
        run(rig, "--initialize")
    assert writes(rig)[-1] == "Stop"
    assert "physical stop is unverified" in capsys.readouterr().err
    assert rig.channel.closed


@pytest.mark.parametrize("overrides", [{"valid": False}, {"stale": True}, {"update_age_ms": 501}])
def test_stale_feedback_times_out_with_the_actual_state_and_no_writes(rig, overrides):
    rig.states[0] = state(**overrides)
    with pytest.raises(TimeoutError, match="CAN feedback timed out.*connected=True"):
        run(rig, "--timeout", "0.2")
    assert not writes(rig)
    assert rig.channel.closed


def test_incompatible_feedback_schema_is_rejected_before_opening_a_channel(rig):
    rig.protobuf.StateResponse.DESCRIPTOR.fields_by_name.pop("connected")
    with pytest.raises(client.GripperError, match="schema is incompatible"):
        run(rig)
    assert not rig.opened


@pytest.mark.parametrize(
    ("feedback", "ready"),
    [
        (state(), True),
        (state(connected=False, valid=False, stale=True, init_state=-1), False),
        (state(init_state=0), False),
        (state(update_age_ms=600), False),
    ],
)
def test_check_reads_once_and_never_commands(rig, feedback, ready, capsys):
    rig.states[0] = feedback
    args = helper.parse_args(["--check"])
    if ready:
        assert helper.check(args, clock=rig.clock) == 432
        assert "No gripper command was sent" in capsys.readouterr().out
    else:
        with pytest.raises(client.GripperError):
            helper.check(args, clock=rig.clock)
    assert rig.bindings == ["GetAllState"]
    assert len(rig.calls) == 1
    assert rig.channel.closed


def test_cli_reports_failures_without_a_traceback(monkeypatch, capsys):
    def timeout(args):
        raise TimeoutError("CAN feedback timed out")

    monkeypatch.setattr(helper, "prepare", timeout)
    assert helper.main(["--connect"]) == 1
    assert "Gripper preparation failed: CAN feedback timed out" in capsys.readouterr().err
