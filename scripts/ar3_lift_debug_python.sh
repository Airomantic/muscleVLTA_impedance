#!/usr/bin/env bash
# Python executable for VS Code: ROS Jazzy + the gripper's Python 3.12, independent of the activated model venv.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRIPPER_DIR="${INFER_GRIPPER_DIR:-$(dirname -- "$ROOT")/V3_Gripper}"
ROBOT_PYTHON="${INFER_ROBOT_PYTHON:-$GRIPPER_DIR/.venv-ros/bin/python}"
ROS_SETUP="${INFER_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
DOMAIN="${INFER_ROS_DOMAIN_ID:-111}"
[[ "$DOMAIN" =~ ^[0-9]+$ && ${#DOMAIN} -le 3 ]] && ((10#$DOMAIN <= 232)) \
    || { echo 'INFER_ROS_DOMAIN_ID must be 0..232.' >&2; exit 1; }
[[ -x "$ROBOT_PYTHON" && -f "$ROS_SETUP" ]] \
    || { echo "Missing robot Python ($ROBOT_PYTHON) or ROS setup ($ROS_SETUP)." >&2; exit 1; }

clean=(env -i "HOME=$HOME" "PATH=/usr/bin:/bin:/usr/local/bin" "LANG=${LANG:-C.UTF-8}"
    "TERM=${TERM:-xterm-256color}" "ROS_DOMAIN_ID=$((10#$DOMAIN))" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1)
# The preparation script opens terminal tabs and may use the user's SSH agent. Never inherit Python/ROS overlays.
for name in DISPLAY WAYLAND_DISPLAY XAUTHORITY DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR XDG_DATA_DIRS SSH_AUTH_SOCK; do
    [[ ! -v "$name" ]] || clean+=("$name=${!name}")
done
while IFS= read -r name; do
    case "$name" in INFER_*|OPENPI_*) clean+=("$name=${!name}") ;; esac
done < <(compgen -e)
exec "${clean[@]}" bash --noprofile --norc -c 'set -e; source "$1"; shift; exec "$@"' \
    _ "$ROS_SETUP" "$ROBOT_PYTHON" "$@"
