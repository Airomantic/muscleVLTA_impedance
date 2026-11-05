#!/usr/bin/env bash
# Gated gripper teleoperation for DAgger (scripts/ar3_dagger_gripper_teleop.py): the Puppetry trigger drives the DM
# gripper only while the robot client (--dagger) hands the robot to the operator, and only after the trigger met the
# fingers' opening. It shares the gripper connection prepared by ./infer_oneclick.sh and never stops or disconnects it.
# ./infer_oneclick.sh --dagger starts it in its own terminal.
#
# Usage (all arguments go to the gate):
#   scripts/ar3_dagger_gripper_teleop.sh --check     # read the gripper, print trigger -> target for 10 s; no writes
#   scripts/ar3_dagger_gripper_teleop.sh             # run the gate (Ctrl+C stops it; the gripper keeps its target)
#   scripts/ar3_dagger_gripper_teleop.sh --help      # all options
#
# Environment (as scripts/run_ar3_lift_robot.sh):
#   INFER_ROBOT_PYTHON    Python 3.12 with grpc, DM_V3Gripper and numpy
#                         (default: <repo>/../V3_Gripper/.venv-ros/bin/python)
#   INFER_GRIPPER_DIR     default prefix of INFER_ROBOT_PYTHON (default: <repo>/../V3_Gripper)
#   INFER_ROS_SETUP       ROS setup script (default: /opt/ros/jazzy/setup.bash)
#   INFER_ROS_DOMAIN_ID   ROS domain of the robot client and MessageBridge (default: 111)
#   INFER_GRIPPER_HOST    gripper gRPC host, port 50008 (default: the gate's 192.168.127.10)
#   OPENPI_DRY_RUN=1      print the command without running it
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRIPPER_DIR="${INFER_GRIPPER_DIR:-$(dirname -- "$ROOT")/V3_Gripper}"
PYTHON="${INFER_ROBOT_PYTHON:-$GRIPPER_DIR/.venv-ros/bin/python}"
ROS_SETUP="${INFER_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
DOMAIN="${INFER_ROS_DOMAIN_ID:-111}"
GATE="$ROOT/scripts/ar3_dagger_gripper_teleop.py"

die() { printf '[ar3_dagger_gripper_teleop] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$DOMAIN" =~ ^[0-9]+$ && ${#DOMAIN} -le 3 ]] && ((10#$DOMAIN <= 232)) || die "INFER_ROS_DOMAIN_ID must be 0..232."
[[ -f "$ROS_SETUP" ]] || die "Missing ROS setup: $ROS_SETUP (set INFER_ROS_SETUP)."
[[ -x "$PYTHON" ]] || die "Missing robot Python: $PYTHON (set INFER_ROBOT_PYTHON)."

defaults=()
[[ -z "${INFER_GRIPPER_HOST:-}" ]] || defaults+=(--gripper-address "$INFER_GRIPPER_HOST:50008")

# ROS Jazzy runs on Python 3.12: never inherit the model's Python 3.11 environment or other overlays.
ros=(env -i "HOME=$HOME" "PATH=/usr/bin:/bin:/usr/local/bin" "LANG=${LANG:-C.UTF-8}" "TERM=${TERM:-xterm-256color}"
    "ROS_DOMAIN_ID=$((10#$DOMAIN))" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
    bash --noprofile --norc -c 'set -e; source "$1"; shift; exec "$@"' _ "$ROS_SETUP")
command=("${ros[@]}" "$PYTHON" -u "$GATE" "${defaults[@]}" "$@")

if [[ "${OPENPI_DRY_RUN:-0}" == "1" ]]; then
    printf 'Would run:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

help=0
for arg in "$@"; do
    [[ "$arg" != -h && "$arg" != --help ]] || help=1
done
if ((!help)); then
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
' rclpy numpy grpc DM_V3Gripper.gripper_pb2 \
        || die "The robot Python lacks the modules above: rclpy comes from $ROS_SETUP (Python 3.12), grpc and DM_V3Gripper from the gripper's .venv-ros."
fi

exec "${command[@]}"
