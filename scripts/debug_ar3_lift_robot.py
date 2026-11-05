#!/usr/bin/env python3
"""VS Code entry point: start a local policy, prepare devices, then debug the robot client in this process.

Use scripts/ar3_lift_debug_python.sh as the interpreter, as configured in .vscode/launch.json.
Arguments after -- go to ar3_lift_robot_client.py. Hardware is prepared but gripper calibration is never implicit.
Only the policy server started by this invocation is stopped on exit; hardware terminal tabs remain available.
"""

import argparse
import importlib
import logging
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("ar3_lift_debug")


def model_environment() -> dict[str, str]:
    """The local model uses Python 3.11 and must not load Jazzy's Python 3.12 modules/libraries."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ROS_", "AMENT_", "COLCON_", "RMW_"))
        and key not in {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "CMAKE_PREFIX_PATH", "OPENPI_DRY_RUN"}
    }


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def policy_healthy(host: str, port: int) -> bool:
    # Never route localhost through a shell's HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://{host}:{port}/healthz", timeout=2.0) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def hardware_start_options(nodes: set[str], *, gripper_ready: bool, skip_tilt: bool) -> list[str] | None:
    """Start only missing devices, retaining each existing camera's owner."""
    forbidden = ("teleop", "cartesian_retargeting", "rokae_driver", "controller_manager", "ar3_lift_robot_client")
    conflicts = sorted(node for node in nodes if any(name in node for name in forbidden))
    if conflicts:
        raise RuntimeError(f"Stop conflicting robot command sources before debugging: {', '.join(conflicts)}")
    arm = "/cartesian_impedance_node" in nodes
    front = any(node.startswith("/front/") for node in nodes)
    wrist = any(node.startswith("/wrist/") for node in nodes)
    tilt = skip_tilt or any(node.rsplit("/", 1)[-1] == "camera_tilt_servo" for node in nodes)
    if arm and front and wrist and tilt and gripper_ready:
        return None
    options = ["--no-policy", "--no-ready-tab"]
    for present, flag in ((arm, "--no-arm"), (front and wrist, "--no-cameras"), (tilt, "--no-tilt")):
        if present:
            options.append(flag)
    if front != wrist:
        options.append("--no-front-camera" if front else "--no-wrist-camera")
    if gripper_ready:
        options.append("--no-ssh")
    return options


def discover_hardware_nodes() -> set[str]:
    """Give DDS time to discover camera containers before deciding which devices to start."""
    nodes: set[str] = set()
    for attempt in range(2):
        result = subprocess.run(
            ["ros2", "node", "list", "--no-daemon", "--spin-time", "3"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        nodes.update(line.strip() for line in result.stdout.splitlines() if line.startswith("/"))
        if all(any(node.startswith(f"/{camera}/") for node in nodes) for camera in ("front", "wrist")):
            break
        if attempt == 0:
            LOGGER.info("Waiting for camera discovery before starting missing devices")
    LOGGER.info("Discovered ROS nodes: %s", ", ".join(sorted(nodes)) or "none")
    return nodes


def prepare_hardware(args: argparse.Namespace) -> None:
    if args.hardware == "skip":
        return
    launcher = ["bash", str(ROOT / "infer_oneclick.sh")]
    if args.hardware == "auto":
        nodes = discover_hardware_nodes()
        options = hardware_start_options(
            nodes,
            gripper_ready=port_open(os.environ.get("INFER_GRIPPER_HOST", "192.168.127.10"), 50008),
            skip_tilt=args.skip_tilt,
        )
        if options is not None:
            LOGGER.info("Starting missing devices: %s", shlex.join(launcher + options))
            subprocess.run(launcher + options, cwd=ROOT, check=True)
    ready = [*launcher, "--ready-only", "--no-policy"]
    if args.skip_tilt:
        ready.append("--no-tilt")
    if args.initialize_gripper:
        ready.append("--initialize-gripper")
    # --no-ready-tab above ensures there is exactly one owner of the readiness lock.
    LOGGER.info("Checking all required devices; gripper calibration is %s", args.initialize_gripper)
    subprocess.run(ready, cwd=ROOT, stdin=subprocess.DEVNULL, check=True)


def stop_policy(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    LOGGER.info("Stopping this debug session's policy server (PID %s)", process.pid)
    for sig, timeout in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    arguments = list(sys.argv[1:] if argv is None else argv)
    split = arguments.index("--") if "--" in arguments else len(arguments)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "infer_checkpoint/20000")
    parser.add_argument("--host", default=os.environ.get("OPENPI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OPENPI_PORT", "8000")))
    parser.add_argument("--hardware", choices=("auto", "ready", "skip"), default="auto")
    parser.add_argument("--skip-tilt", action="store_true", help="do not start or verify the camera tilt servo")
    parser.add_argument("--initialize-gripper", action="store_true", help="explicitly allow finger calibration motion")
    parser.add_argument(
        "--reuse-policy", action="store_true", help="use an existing server; verify its checkpoint yourself"
    )
    parser.add_argument("--policy-start-timeout", type=float, default=600)
    parser.add_argument("--print-plan", action="store_true", help="print commands only; no device or network access")
    args = parser.parse_args(arguments[:split])
    if not 1 <= args.port <= 65535 or args.policy_start_timeout <= 0:
        parser.error("port must be 1..65535 and policy-start-timeout must be positive")
    if not args.reuse_policy and args.host not in {"127.0.0.1", "localhost"}:
        parser.error("local policy startup requires localhost; use --reuse-policy for a remote service")
    if args.hardware == "skip" and args.initialize_gripper:
        parser.error("--initialize-gripper requires --hardware auto or ready")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    client_args = ["--host", args.host, "--port", str(args.port)]
    if "INFER_GRIPPER_HOST" in os.environ:
        client_args += ["--gripper-address", f"{os.environ['INFER_GRIPPER_HOST']}:50008"]
    client_args += arguments[split + 1 :]
    return args, client_args


def main(argv: list[str] | None = None) -> int:
    args, client_args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info(
        "Robot Python: %s (%s); ROS domain: %s",
        sys.executable,
        sys.version.split()[0],
        os.environ.get("ROS_DOMAIN_ID", "unset"),
    )
    serve = ["bash", str(ROOT / "scripts/serve_ar3_lift.sh"), str(args.checkpoint)]
    if args.print_plan:
        print("Policy:", "reuse existing" if args.reuse_policy else shlex.join(serve))
        print(
            f"Hardware: {args.hardware}; initialize gripper: {args.initialize_gripper}; ROS domain: "
            f"{os.environ.get('ROS_DOMAIN_ID', '111')}"
        )
        print(
            "Client (same debugger process):",
            shlex.join([str(ROOT / "scripts/ar3_lift_robot_client.py"), *client_args]),
        )
        return 0

    process = None
    log_file = None
    try:
        for name in (
            "rclpy",
            "numpy",
            "msgpack",
            "websockets.sync.client",
            "typing_extensions",
            "PIL",
            "grpc",
            "DM_V3Gripper.gripper_pb2",
        ):
            importlib.import_module(name)
        # Import in the debuggee rather than spawning a child: breakpoints in the client work normally.
        sys.path.insert(0, str(ROOT / "scripts"))
        client = importlib.import_module("ar3_lift_robot_client")
        client_options = client.parse_args(client_args)
        if (client_options.host, client_options.port) != (args.host, args.port):
            raise RuntimeError("Pass policy --host/--port before -- so startup and client use the same endpoint.")
        if client_options.dagger and args.hardware != "skip":
            # auto/ready treat the DAgger teleoperation as a conflicting command source.
            raise RuntimeError("--dagger: run ./infer_oneclick.sh --dagger first, then pass --hardware skip before --.")
        if client_options.record_dir is not None:
            try:
                importlib.import_module("pyarrow")
                importlib.import_module("av")
            except ImportError as error:
                raise RuntimeError(
                    f"--record-dir needs pyarrow and PyAV ({error}): "
                    f"PYTHONNOUSERSITE=1 {sys.executable} -m pip install pyarrow av"
                ) from error
        environment = model_environment()
        environment.update(OPENPI_HOST=args.host, OPENPI_PORT=str(args.port))
        if args.reuse_policy:
            if not policy_healthy(args.host, args.port):
                raise RuntimeError(f"No healthy policy at {args.host}:{args.port}; the existing service is unchanged.")
            LOGGER.info("Reusing policy at %s:%s (checkpoint selection remains with its owner)", args.host, args.port)
        else:
            # Reuse the shell's checkpoint/tokenizer/interpreter checks before any hardware preparation.
            subprocess.run(
                serve, cwd=ROOT, env={**environment, "OPENPI_DRY_RUN": "1"}, check=True, capture_output=True, text=True
            )
            if port_open(args.host, args.port):
                raise RuntimeError(
                    f"Policy port {args.host}:{args.port} is occupied. Keep it and use --reuse-policy only if its "
                    "checkpoint is correct, or choose --port 8001."
                )
            log_dir = ROOT / "logs"
            log_dir.mkdir(exist_ok=True)
            log_path = log_dir / f"ar3_lift_policy_debug_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
            log_file = log_path.open("w")
            LOGGER.info("Loading %s; policy output: %s", args.checkpoint, log_path)
            # VS Code can SIGKILL the debuggee, bypassing finally. Bind the model
            # to this owner before exec while retaining its separate process group.
            owned_serve = [
                sys.executable,
                str(ROOT / "scripts/ar3_lift_policy_process.py"),
                "--parent-pid",
                str(os.getpid()),
                "--",
                *serve,
            ]
            process = subprocess.Popen(
                owned_serve,
                cwd=ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + args.policy_start_timeout
            next_status = time.monotonic() + 15
            while not policy_healthy(args.host, args.port):
                if process.poll() is not None:
                    raise RuntimeError(f"Policy server exited ({process.returncode}). See {log_path}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Policy startup timed out. See {log_path}")
                if time.monotonic() >= next_status:
                    LOGGER.info("Still loading the local policy; see %s", log_path)
                    next_status = time.monotonic() + 15
                time.sleep(1)
            LOGGER.info("Local policy is ready")
        prepare_hardware(args)
        LOGGER.info("Running robot client: %s", shlex.join(client_args))
        return client.main(client_args)
    except KeyboardInterrupt:
        LOGGER.info("Debug preparation interrupted")
        return 130
    except (ImportError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        LOGGER.error("%s", error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            LOGGER.error("%s", error.stderr.strip())
        return 1
    finally:
        stop_policy(process)
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    # During preparation VS Code's stop/Ctrl+C should clean up the owned model service too.
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    sys.exit(main())
