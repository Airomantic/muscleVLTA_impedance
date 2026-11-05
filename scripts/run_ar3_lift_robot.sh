#!/usr/bin/env bash
# Run the AR3 LIFT policy on the real robot (scripts/ar3_lift_robot_client.py) with ROS 2 Jazzy and the DM gripper's
# Python 3.12 environment. Prepare the devices first (./infer_oneclick.sh, wait for PREPARATION READY) and serve the
# policy (scripts/serve_ar3_lift.sh; from the GPU server: ssh -N -L 8000:127.0.0.1:8000 <user>@<server>).
#
# Usage (all arguments go to the client; it moves the robot only with --execute):
#   scripts/run_ar3_lift_robot.sh --check                       # controller, cameras, gripper; no motion
#   scripts/run_ar3_lift_robot.sh --max-chunks 3                # dry run: targets only on preview_command
#   scripts/run_ar3_lift_robot.sh --execute --move-to-start     # MOVES the arm and gripper
#   scripts/run_ar3_lift_robot.sh --execute --record-dir DIR    # MOVES; records each episode (LeRobot v3.0)
#   scripts/run_ar3_lift_robot.sh --execute --dagger --record-dir DIR   # DAgger (./infer_oneclick.sh --dagger)
#   scripts/run_ar3_lift_robot.sh --help                        # all client options
#
# Environment:
#   INFER_ROBOT_PYTHON    Python 3.12 with grpc, DM_V3Gripper, numpy, msgpack, websockets>=11,<18, typing_extensions
#                         and pillow; pyarrow and av (PyAV) for --record-dir
#                         (default: <repo>/../V3_Gripper/.venv-ros/bin/python)
#   INFER_GRIPPER_DIR     default prefix of INFER_ROBOT_PYTHON (default: <repo>/../V3_Gripper)
#   INFER_ROS_SETUP       ROS setup script (default: /opt/ros/jazzy/setup.bash)
#   INFER_ROS_DOMAIN_ID   ROS domain of the prepared nodes (default: 111)
#   OPENPI_HOST, OPENPI_PORT   policy server (default: the client's 127.0.0.1:8000)
#   INFER_GRIPPER_HOST    gripper gRPC host, port 50008 (default: the client's 192.168.127.10)
#   OPENPI_DRY_RUN=1      print the command without running it
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRIPPER_DIR="${INFER_GRIPPER_DIR:-$(dirname -- "$ROOT")/V3_Gripper}"
PYTHON="${INFER_ROBOT_PYTHON:-$GRIPPER_DIR/.venv-ros/bin/python}"
ROS_SETUP="${INFER_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
DOMAIN="${INFER_ROS_DOMAIN_ID:-111}"
CLIENT="$ROOT/scripts/ar3_lift_robot_client.py"

die() { printf '[run_ar3_lift_robot] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$DOMAIN" =~ ^[0-9]+$ && ${#DOMAIN} -le 3 ]] && ((10#$DOMAIN <= 232)) || die "INFER_ROS_DOMAIN_ID must be 0..232."
[[ -f "$ROS_SETUP" ]] || die "Missing ROS setup: $ROS_SETUP (set INFER_ROS_SETUP)."
[[ -x "$PYTHON" ]] || die "Missing robot Python: $PYTHON (set INFER_ROBOT_PYTHON)."

defaults=()
[[ -z "${OPENPI_HOST:-}" ]] || defaults+=(--host "$OPENPI_HOST")
[[ -z "${OPENPI_PORT:-}" ]] || defaults+=(--port "$OPENPI_PORT")
[[ -z "${INFER_GRIPPER_HOST:-}" ]] || defaults+=(--gripper-address "$INFER_GRIPPER_HOST:50008")

# ROS Jazzy runs on Python 3.12: never inherit the model's Python 3.11 environment or other overlays.
ros=(env -i "HOME=$HOME" "PATH=/usr/bin:/bin:/usr/local/bin" "LANG=${LANG:-C.UTF-8}" "TERM=${TERM:-xterm-256color}"
    "ROS_DOMAIN_ID=$((10#$DOMAIN))" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
    bash --noprofile --norc -c 'set -e; source "$1"; shift; exec "$@"' _ "$ROS_SETUP")
# Explicit arguments come last and override the environment defaults.
command=("${ros[@]}" "$PYTHON" -u "$CLIENT" "${defaults[@]}" "$@")

if [[ "${OPENPI_DRY_RUN:-0}" == "1" ]]; then
    printf 'Would run:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

help=0 gripper=1 record=0
for arg in "$@"; do
    case "$arg" in
        -h|--help) help=1 ;;
        --no-gripper) gripper=0 ;;
        --record-dir|--record-dir=*) record=1 ;;
    esac
done
if ((help)); then
    sed -n '2,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
    echo
else
    modules=(rclpy numpy msgpack websockets.sync.client typing_extensions PIL.Image)
    ((gripper)) && modules+=(grpc DM_V3Gripper.gripper_pb2)
    ((record)) && modules+=(pyarrow.parquet av)
    "${ros[@]}" "$PYTHON" -c '
import importlib
import sys

missing = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception as error:  # noqa: BLE001
        missing.append(f"{name} ({type(error).__name__}: {error})")
if missing:
    sys.exit("missing modules: " + "; ".join(missing))
' "${modules[@]}" || die "The robot Python lacks the modules above: rclpy comes from $ROS_SETUP (Python 3.12), grpc and DM_V3Gripper from the gripper's .venv-ros; install into this environment (user-site packages are disabled): PYTHONNOUSERSITE=1 $PYTHON -m pip install 'websockets>=11,<18' msgpack typing_extensions pillow pyarrow av"
fi

exec "${command[@]}"
