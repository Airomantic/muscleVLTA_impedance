#!/usr/bin/env python3
"""Prepares (``--connect``) or checks (``--check``) the DM tac G2/V3 gripper's CAN connection for AR3 LIFT inference.

--connect sends Connect only when the gripper server reports disconnected (Connect replaces an existing connection).
--initialize additionally allows Enable and the full calibration, which MOVES the fingers. No target, open, close or
Disconnect RPC is ever sent; closing this client's gRPC channel keeps the server's CAN connection for inference.
--check reads GetAllState once and requires connected, initialized, error-free and fresh feedback. It sends nothing.
"""

import argparse
from collections.abc import Callable, Sequence
import math
import sys
import time
from typing import Any

if __package__:
    from scripts import ar3_lift_robot_client as client
else:
    import ar3_lift_robot_client as client


MAX_AGE = 0.5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--connect", action="store_true", help="allow the server's CAN connection")
    mode.add_argument("--check", action="store_true", help="read-only feedback check")
    parser.add_argument("--gripper-address", default="192.168.127.10:50008")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--motor-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--initialize", action="store_true", help="allow calibration if needed; moves the gripper")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if not 1 <= args.motor_id <= 255:
        parser.error("--motor-id must be 1..255")
    if not args.gripper_address.strip() or not args.channel.strip():
        parser.error("--gripper-address and --channel must be nonempty")
    if args.initialize and not args.connect:
        parser.error("--initialize requires --connect")
    return args


def describe_state(state: Any) -> str:
    fields = ("connected", "valid", "stale", "init_state", "error", "position", "update_age_ms", "message")
    return ", ".join(f"{name}={getattr(state, name, '<missing>')}" for name in fields)


def _open(address: str) -> tuple[Any, Any, Any, Any]:
    from DM_V3Gripper import gripper_pb2
    import grpc

    # Reject an incompatible feedback schema before opening a channel or binding any RPC.
    client.check_gripper_schema(gripper_pb2.StateResponse)
    channel = grpc.insecure_channel(address, options=(("grpc.enable_http_proxy", 0),))
    reader = channel.unary_unary(
        client.GRIPPER_RPC_PREFIX + "GetAllState", response_deserializer=gripper_pb2.StateResponse.FromString
    )
    return gripper_pb2, grpc, channel, reader


def check(args: argparse.Namespace, *, clock: Callable[[], float] = time.monotonic) -> float:
    _, grpc, channel, reader = _open(args.gripper_address)
    try:
        started = clock()
        try:
            state = reader(b"", timeout=min(args.timeout, 1.0))
        except grpc.RpcError as exc:
            raise client.GripperError(f"GetAllState failed: {exc}") from exc
        position = client.gripper_position(state, max_age=MAX_AGE, elapsed=clock() - started)
    finally:
        channel.close()
    print(
        f"Read-only gripper feedback ready at {args.gripper_address}: position={position:g} (0 = closed ... "
        "1000 = open), connected, initialized and fresh. No gripper command was sent.",
        flush=True,
    )
    return position


def prepare(
    args: argparse.Namespace,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if not args.connect:
        raise ValueError("CAN preparation requires explicit --connect")
    gripper_pb2, grpc, channel, reader = _open(args.gripper_address)
    deadline = clock() + args.timeout
    attempted_initialization = False
    operations = {}

    def call(name: str, *, until: float, request: bytes = b"", limit: float = 5.0) -> None:
        remaining = until - clock()
        if remaining <= 0:
            raise TimeoutError(f"Deadline expired before gripper {name}")
        if name not in operations:
            operations[name] = channel.unary_unary(
                client.GRIPPER_RPC_PREFIX + name, response_deserializer=gripper_pb2.Response.FromString
            )
        try:
            response = operations[name](request, timeout=min(limit, remaining))
        except grpc.RpcError as exc:
            raise RuntimeError(f"Gripper {name} RPC failed; delivery is uncertain: {exc}") from exc
        if not response.success:
            raise RuntimeError(f"Gripper {name} rejected: {response.message}")

    try:
        inspected_connection = False
        initialization_deadline = None
        detail = "No GetAllState response received"
        while True:
            until = min(deadline, initialization_deadline) if initialization_deadline is not None else deadline
            remaining = until - clock()
            if remaining <= 0:
                phase = "initialization" if initialization_deadline is not None else "CAN feedback"
                raise TimeoutError(f"Gripper {phase} timed out: {detail}")
            started = clock()
            try:
                state = reader(b"", timeout=min(1.0, remaining))
            except grpc.RpcError as exc:
                detail = f"GetAllState failed: {exc}"
            else:
                elapsed = clock() - started
                detail = describe_state(state)
                if clock() >= until:
                    continue
                if not inspected_connection:
                    inspected_connection = True
                    if not state.connected:
                        config = gripper_pb2.ConnectConfig(
                            channel=args.channel,
                            motor_id=args.motor_id,
                            read_interval_s=0.05,
                            read_timeout_s=0.1,
                            retry_count=2,
                            stale_timeout_s=MAX_AGE,
                            command_min_interval_s=0.02,
                        )
                        print(f"Connecting gripper CAN {args.channel}, motor {args.motor_id}...", flush=True)
                        call("Connect", until=deadline, request=config.SerializeToString())
                        continue
                age = float(state.update_age_ms) / 1000.0 + elapsed
                fresh = math.isfinite(age) and state.update_age_ms >= 0 and elapsed >= 0 and age <= MAX_AGE
                if state.connected and state.valid and not state.stale and fresh:
                    if state.error != 0:
                        raise RuntimeError(f"Gripper reports a motor error: {detail}")
                    if state.init_state == 1:
                        client.gripper_position(state, max_age=MAX_AGE, elapsed=elapsed)
                        print(
                            f"Gripper CAN ready at {args.gripper_address}: position={state.position}. "
                            "Closing only the local gRPC channel; the server keeps the CAN connection for inference. "
                            "No target command was sent.",
                            flush=True,
                        )
                        return
                    if state.init_state not in (0, 2):
                        raise RuntimeError(f"Unknown gripper initialization state: {detail}")
                    if state.init_state == 0 and not args.initialize:
                        raise RuntimeError(
                            "Gripper CAN is connected but not initialized. Clear the fingers' workspace and run "
                            "./infer_oneclick.sh --ready-only --initialize-gripper to allow the calibration motion."
                        )
                    if initialization_deadline is None:
                        initialization_deadline = min(deadline, clock() + 15.0)
                        if state.init_state == 0:
                            print("Initializing gripper; calibration moves the fingers...", flush=True)
                            # Even a rejected or timed-out Enable may have reached the motor.
                            attempted_initialization = True
                            call("Enable", until=initialization_deadline)
                            call("Initialization", until=initialization_deadline)
                        else:
                            print("Waiting for the gripper initialization already in progress...", flush=True)
            sleep(max(0.0, min(0.05, until - clock())))
    except BaseException:
        if attempted_initialization:
            try:
                call("Stop", until=clock() + 2.0, limit=2.0)
            except BaseException as exc:
                print(f"Gripper cleanup Stop failed; physical stop is unverified: {exc}", file=sys.stderr, flush=True)
            else:
                print(
                    "Gripper cleanup Stop RPC acknowledged; physical stop is unverified.", file=sys.stderr, flush=True
                )
        raise
    finally:
        channel.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.check:
            check(args)
        else:
            prepare(args)
    except KeyboardInterrupt:
        print("Gripper preparation interrupted.", file=sys.stderr)
        return 130
    except (ImportError, RuntimeError, TimeoutError, ValueError, AttributeError) as exc:
        print(f"Gripper {'check' if args.check else 'preparation'} failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
