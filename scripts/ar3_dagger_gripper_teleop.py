#!/usr/bin/env python3
"""Gated gripper teleoperation for DAgger takeovers of ``scripts/ar3_lift_robot_client.py --dagger``.

The lerobot recordings drove the DM gripper with V3_Gripper's ``teleop_gripper.py``: the Puppetry trigger
(``/trigger_positions`` from MessageBridge, [left, right] in 0..1) became the SDK target ``1000 * (1 - trigger)``
(1000 = open). During DAgger rollouts the policy owns the gripper, so this node follows the trigger only

* while the robot client hands the robot to the operator (latched ``/ar3_lift/human_control`` = True after a clutch
  press; nothing while the client is not running), and
* once the trigger has met the fingers' measured opening (within ``--match-tolerance`` or by crossing it), so taking
  over never opens a grasp or snaps the fingers.

Each target is published on ``/v3_gripper/action`` (JointState, SDK position / 1000) where the client records it as
the gripper action of the operator's frames. Unlike ``teleop_gripper.py`` this node never connects, enables,
configures, stops or disconnects the gripper: it shares the gRPC connection prepared by ``./infer_oneclick.sh`` and
the speed / force set by the robot client, and only sends ``SetTargetPosition``. A stale trigger or a gripper error
only lets go of the gripper (it keeps its target); the node keeps running.

Run it with ``scripts/ar3_dagger_gripper_teleop.sh`` (ROS 2 Jazzy + the gripper's Python, like the client).
"""

import argparse
from collections.abc import Callable, Sequence
import logging
import math
import threading
import time
from typing import Any

if __package__:
    from . import ar3_lift_robot_client as client
else:
    import ar3_lift_robot_client as client

logger = logging.getLogger("ar3_dagger_gripper_gate")


def trigger_to_position(data: Sequence[float], *, hand: str = "right", invert: bool = False, offset: int = 0) -> int:
    """``teleop_gripper.trigger_to_position``: MessageBridge sends [left, right] in 0..1; 1000 = fully open."""
    if offset < 0 or len(data) - offset != 2:
        raise ValueError(f"the trigger message must hold [left, right], got {len(data) - offset} values")
    values = tuple(float(value) for value in data[offset:])
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"trigger values must be finite and within 0..1, got {values}")
    trigger = values[0 if hand == "left" else 1]
    return round(client.GRIPPER_OPEN * (trigger if invert else 1.0 - trigger))


class GripperGate:
    """Decides each control tick whether the operator's trigger drives the gripper (no ROS, testable).

    ``gripper`` is a ``client.DMGripper``-like object (``read``, ``command``, ``forget_target``, ``target``);
    ``publish(target)`` reports a commanded target (``/v3_gripper/action``).
    """

    def __init__(
        self,
        gripper: Any,
        publish: Callable[[float], None],
        *,
        match_tolerance: float,
        trigger_timeout: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._gripper = gripper
        self._publish = publish
        self._tolerance = match_tolerance
        self._timeout = trigger_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._trigger: tuple[int, float] | None = None  # newest (target, received_at)
        self._human_control = False
        self.active = False  # the operator has control
        self.engaged = False  # the trigger drives the gripper
        self._offset: float | None = None  # trigger - fingers of the previous tick before engaging
        self._fresh = True

    # ROS callbacks (executor thread).
    def on_trigger(self, target: int) -> None:
        with self._lock:
            self._trigger = (target, self._clock())

    def on_human_control(self, *, active: bool) -> None:
        with self._lock:
            self._human_control = active

    def latest_trigger(self) -> int | None:
        with self._lock:
            return None if self._trigger is None else self._trigger[0]

    def step(self, *, client_alive: bool) -> int | None:
        """One control tick; returns the target sent to the gripper (None: the gate does not drive it)."""
        now = self._clock()
        with self._lock:
            active, trigger = self._human_control and client_alive, self._trigger
        if not active:
            if self.active:
                logger.info("takeover ended: the policy owns the gripper again")
            self.release()
            self.active = False
            return None
        if not self.active:
            self.active = True
            logger.warning("takeover: move the trigger to the fingers' opening to take the gripper")
        if trigger is None or now - trigger[1] > self._timeout:
            if self._fresh:
                logger.warning("no fresh trigger: the gate lets go of the gripper (it keeps its target)")
            self.release()
            self._fresh = False
            return None
        if not self._fresh:
            logger.info("the trigger is fresh again")
            self._fresh = True
        target = trigger[0]
        if not self.engaged:
            fingers = self._gripper.read()
            offset = target - fingers
            crossed = self._offset is not None and offset * self._offset <= 0
            if abs(offset) > self._tolerance and not crossed:
                self._offset = offset
                return None
            self.engaged = True
            logger.warning("the gripper follows the trigger (trigger %d, fingers %.0f)", target, fingers)
        self._gripper.command(target)
        self._publish(self._gripper.target)
        return target

    def release(self) -> None:
        """Lets go of the gripper (it keeps its target); the trigger has to meet the fingers again."""
        if self.engaged:
            self._gripper.forget_target()
        self.engaged = False
        self._offset = None
        self._fresh = True


def wait_for_gripper(gripper: Any, timeout: float) -> float:
    """Reads the gripper until its feedback is valid (``./infer_oneclick.sh`` connects it) or ``timeout`` passes."""
    deadline = time.monotonic() + timeout
    reported = None
    while True:
        try:
            return gripper.read()
        except client.GripperError as error:
            if time.monotonic() >= deadline:
                raise
            if str(error) != reported:
                logger.info("waiting for the prepared gripper: %s", error)
                reported = str(error)
        time.sleep(1.0)


def run(args: argparse.Namespace) -> int:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool
    from std_msgs.msg import Float64MultiArray

    gripper = client.DMGripper.connect(
        args.gripper_address,
        writable=not args.check,
        deadband=args.deadband,
        max_age=args.gripper_max_age,
    )
    try:
        fingers = wait_for_gripper(gripper, args.wait)
        rclpy.init(args=[])
        node = Node(client.GRIPPER_GATE_NODE)
        action = node.create_publisher(JointState, client.GRIPPER_ACTION_TOPIC, 10)

        def publish(target: float) -> None:
            message = JointState()
            message.header.stamp = node.get_clock().now().to_msg()
            message.name = ["gripper"]
            message.position = [float(target) / client.GRIPPER_OPEN]
            action.publish(message)

        gate = GripperGate(
            gripper,
            publish,
            match_tolerance=args.match_tolerance,
            trigger_timeout=args.trigger_timeout,
        )
        invalid = threading.Event()

        def on_trigger(message: Any) -> None:
            try:
                target = trigger_to_position(
                    message.data, hand=args.hand, invert=args.invert, offset=message.layout.data_offset
                )
            except (TypeError, ValueError, OverflowError) as error:
                # Ignored like a missing sample (the gate lets go once the trigger is stale); logged once.
                if not invalid.is_set():
                    logger.error("invalid trigger message on %s: %s", args.trigger_topic, error)
                    invalid.set()
                return
            invalid.clear()
            gate.on_trigger(target)

        # The client's latched publisher, and MessageBridge's QoS (as teleop_gripper.py subscribes).
        latched = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        node.create_subscription(
            Bool, client.HUMAN_CONTROL_TOPIC, lambda m: gate.on_human_control(active=m.data), latched
        )
        trigger_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE
        )
        node.create_subscription(Float64MultiArray, args.trigger_topic, on_trigger, trigger_qos)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        threading.Thread(target=executor.spin, name="ros", daemon=True).start()
        try:
            if args.check:
                return check(args, gate, fingers)
            logger.info(
                "gate ready (fingers %.0f): follows %s (%s hand) only during DAgger takeovers of the robot client",
                fingers,
                args.trigger_topic,
                args.hand,
            )
            loop(args, gate, lambda: node.count_publishers(client.HUMAN_CONTROL_TOPIC) > 0)
        finally:
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
            rclpy.try_shutdown()
    finally:
        gripper.close()
    logger.info("gate stopped: the gripper keeps its last target (no Stop / disconnect)")
    return 0


def loop(args: argparse.Namespace, gate: GripperGate, client_alive: Callable[[], bool]) -> None:
    period = 1.0 / args.rate
    reported = None
    deadline = time.monotonic()
    try:
        while True:
            try:
                gate.step(client_alive=client_alive())
                reported = None
            except client.GripperError as error:
                if str(error) != reported:
                    logger.error("%s: the gate lets go of the gripper", error)
                    reported = str(error)
                gate.release()
            deadline = max(deadline + period, time.monotonic())
            time.sleep(max(deadline - time.monotonic(), 0.0))
    except KeyboardInterrupt:
        pass


def check(args: argparse.Namespace, gate: GripperGate, fingers: float) -> int:
    """Prints the trigger -> gripper target mapping for a few seconds; sends nothing."""
    logger.info("check: fingers at %.0f (0 = closed .. 1000 = open); squeeze and release the trigger", fingers)
    deadline = time.monotonic() + args.check_seconds
    seen = None
    while time.monotonic() < deadline:
        target = gate.latest_trigger()
        if target is not None and target != seen:
            seen = target
            logger.info("trigger -> gripper target %d", seen)
        time.sleep(0.1)
    if seen is None:
        logger.error(
            "no trigger message on %s: start MessageBridge and Puppetry (./infer_oneclick.sh --dagger) with the same "
            "ROS_DOMAIN_ID",
            args.trigger_topic,
        )
        return 1
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true", help="read the gripper and print the trigger mapping, then exit (no writes)"
    )
    parser.add_argument("--check-seconds", type=float, default=10.0)
    parser.add_argument("--gripper-address", default="192.168.127.10:50008")
    parser.add_argument("--trigger-topic", default="/trigger_positions", help="MessageBridge [left, right] 0..1")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--invert", action="store_true", help="trigger 1 = open (teleop_gripper.py --invert)")
    parser.add_argument("--rate", type=float, default=20.0, help="Hz (the recording teleoperation's)")
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=50.0,
        help="the trigger takes the gripper within this distance of the fingers (0..1000) or by crossing them",
    )
    parser.add_argument("--trigger-timeout", type=float, default=0.5, help="older triggers let go of the gripper (s)")
    parser.add_argument("--deadband", type=int, default=5, help="as teleop_gripper.py (SDK units)")
    parser.add_argument("--gripper-max-age", type=float, default=0.5)
    parser.add_argument("--wait", type=float, default=0.0, help="wait this long for valid gripper feedback (s)")
    args = parser.parse_args(argv)
    for name in ("check_seconds", "rate", "trigger_timeout", "gripper_max_age"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.match_tolerance <= client.GRIPPER_OPEN or not 0 <= args.deadband <= 100:
        parser.error("--match-tolerance must be within 0..1000 and --deadband within 0..100")
    if not math.isfinite(args.wait) or args.wait < 0:
        parser.error("--wait must not be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    try:
        return run(args)
    except client.GripperError as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
