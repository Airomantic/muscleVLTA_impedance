import pytest

from scripts import ar3_dagger_gripper_teleop as gate_module
from scripts import ar3_lift_robot_client as client


class FakeGripper:
    def __init__(self, position):
        self.position = position
        self.error = None
        self.sent = []
        self.target = None
        self.forgotten = 0

    def read(self):
        return self.position

    def command(self, target):
        if self.error is not None:
            raise self.error
        self.target = float(target)
        self.sent.append(target)

    def forget_target(self):
        self.forgotten += 1
        self.target = None


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_gate(position=300.0):
    gripper, published, clock = FakeGripper(position), [], Clock()
    gate = gate_module.GripperGate(gripper, published.append, match_tolerance=50, trigger_timeout=0.5, clock=clock)
    return gate, gripper, published, clock


def test_trigger_maps_like_teleop_gripper():
    assert gate_module.trigger_to_position([0.2, 0.75]) == 250
    assert gate_module.trigger_to_position([0.2, 0.75], hand="left") == 800
    assert gate_module.trigger_to_position([0.2, 0.75], invert=True) == 750
    assert gate_module.trigger_to_position([9.0, 0.2, 0.75], offset=1) == 250
    for data in ([0.2, 0.75, 0.1], [0.2, float("nan")], [0.2, 1.2]):
        with pytest.raises(ValueError, match="trigger"):
            gate_module.trigger_to_position(data)


def test_trigger_drives_gripper_only_during_live_takeover_after_meeting_the_fingers():
    gate, gripper, published, clock = make_gate(position=300.0)
    gate.on_trigger(900)  # trigger released (open) while the policy holds a grasp
    assert gate.step(client_alive=True) is None  # no takeover
    gate.on_human_control(active=True)
    assert gate.step(client_alive=False) is None  # the client's latched True outlived it
    assert gate.step(client_alive=True) is None  # would open the grasp
    gate.on_trigger(380)
    assert gate.step(client_alive=True) is None  # same side, outside the tolerance
    assert gripper.sent == []
    gate.on_trigger(330)
    assert gate.step(client_alive=True) == 330
    gate.on_trigger(600)
    assert gate.step(client_alive=True) == 600  # engaged: follows without matching
    assert gripper.sent == [330, 600]
    assert published == [330.0, 600.0]

    clock.now += 1.0  # stale trigger: let go; it has to meet the fingers again
    assert gate.step(client_alive=True) is None
    assert not gate.engaged
    gate.on_trigger(900)
    gripper.position = 600.0
    assert gate.step(client_alive=True) is None
    assert gripper.sent == [330, 600]


def test_crossing_the_fingers_engages_and_takeover_end_lets_go():
    gate, gripper, _, _ = make_gate(position=300.0)
    gate.on_human_control(active=True)
    gate.on_trigger(500)
    assert gate.step(client_alive=True) is None
    gate.on_trigger(100)  # swept through the fingers' opening between two ticks
    assert gate.step(client_alive=True) == 100
    gate.on_human_control(active=False)
    assert gate.step(client_alive=True) is None
    assert gripper.forgotten == 1
    assert not gate.engaged
    gate.on_human_control(active=True)
    gate.on_trigger(800)
    assert gate.step(client_alive=True) is None
    assert gripper.sent == [100]


def test_loop_lets_go_on_gripper_errors_and_keeps_running(monkeypatch):
    gate, gripper, _, _ = make_gate(position=300.0)
    gate.on_human_control(active=True)
    gate.on_trigger(310)
    ticks = []

    def sleep(_):
        ticks.append(gate.engaged)
        gripper.error = client.GripperError("gripper SetTargetPosition failed") if len(ticks) == 1 else None
        if len(ticks) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(gate_module.time, "sleep", sleep)
    gate_module.loop(gate_module.parse_args([]), gate, lambda: True)
    assert ticks == [True, False, True]  # engaged, let go after the error, engaged again
    assert gripper.forgotten == 1
