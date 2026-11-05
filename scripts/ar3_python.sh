#!/usr/bin/env bash
# Robot-side Python: the DM gripper's Python 3.12 (ROS 2 Jazzy's rclpy) in the clean ROS environment of
# scripts/run_ar3_lift_robot.sh. .vscode/launch.json uses it as the debugpy "python"; every argument goes unchanged to
# the interpreter, e.g. scripts/ar3_python.sh scripts/ar3_lift_robot_client.py --check
#
# Environment (as scripts/run_ar3_lift_robot.sh):
#   INFER_ROBOT_PYTHON    (default: <repo>/../V3_Gripper/.venv-ros/bin/python)
#   INFER_GRIPPER_DIR     default prefix of INFER_ROBOT_PYTHON (default: <repo>/../V3_Gripper)
#   INFER_ROS_SETUP       ROS setup script (default: /opt/ros/jazzy/setup.bash)
#   INFER_ROS_DOMAIN_ID   ROS domain of the prepared nodes (default: 111)
# Only the debugger (DEBUGPY_*, PYDEVD_*), ROS_LOG_DIR and display variables are inherited.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRIPPER_DIR="${INFER_GRIPPER_DIR:-$(dirname -- "$ROOT")/V3_Gripper}"
PYTHON="${INFER_ROBOT_PYTHON:-$GRIPPER_DIR/.venv-ros/bin/python}"
ROS_SETUP="${INFER_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
DOMAIN="${INFER_ROS_DOMAIN_ID:-111}"

die() { printf '[ar3_python] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$DOMAIN" =~ ^[0-9]+$ && ${#DOMAIN} -le 3 ]] && ((10#$DOMAIN <= 232)) || die "INFER_ROS_DOMAIN_ID must be 0..232."
[[ -f "$ROS_SETUP" ]] || die "Missing ROS setup: $ROS_SETUP (set INFER_ROS_SETUP)."
[[ -x "$PYTHON" ]] || die "Missing robot Python: $PYTHON (set INFER_ROBOT_PYTHON)."

keep=()
while IFS= read -r -d '' entry; do
    case "${entry%%=*}" in
        DEBUGPY_* | PYDEVD_* | PYTHONFAULTHANDLER | ROS_LOG_DIR | DISPLAY | XAUTHORITY | WAYLAND_DISPLAY | XDG_RUNTIME_DIR)
            keep+=("$entry")
            ;;
    esac
done < <(env -0)

# ROS Jazzy runs on Python 3.12: never inherit the model's Python 3.11 environment or other overlays.
exec env -i "HOME=$HOME" "PATH=/usr/bin:/bin:/usr/local/bin" "LANG=${LANG:-C.UTF-8}" "TERM=${TERM:-xterm-256color}" \
    "ROS_DOMAIN_ID=$((10#$DOMAIN))" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 "${keep[@]}" \
    bash --noprofile --norc -c 'set -e; source "$1"; shift; exec "$@"' _ "$ROS_SETUP" "$PYTHON" "$@"
