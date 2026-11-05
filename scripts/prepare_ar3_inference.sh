#!/usr/bin/env bash
# Prepare the AR3 LIFT inference devices; never records or executes policy actions. Only --dagger starts the
# operator's teleoperation (MessageBridge, Puppetry, AR3 Cartesian teleoperation) and the gated gripper teleoperation.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/prepare_ar3_inference.sh"
UPPERLIMB="$(dirname -- "$ROOT")"
AR3_WS="${INFER_AR3_WS:-$UPPERLIMB/ar3_CartesianImpedanceControl}"
GRIPPER_DIR="${INFER_GRIPPER_DIR:-$UPPERLIMB/V3_Gripper}"
ROBOT_PYTHON="${INFER_ROBOT_PYTHON:-$GRIPPER_DIR/.venv-ros/bin/python}"
TILT_DIR="${INFER_TILT_DIR:-$UPPERLIMB/../../huanEr}"
SERVO_PYTHON="${INFER_SERVO_PYTHON:-$HOME/anaconda3/envs/qwen3VL/bin/python}"
ROS_SETUP="${INFER_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
CAMERA_LAUNCH_DIR="${INFER_CAMERA_LAUNCH_DIR:-/opt/ros/jazzy/share/orbbec_camera/launch}"
# --dagger: the recordings' teleoperation (lerobot oneClick.sh) takes over from the policy.
REMOTE_DIR="${INFER_REMOTE_DIR:-$UPPERLIMB/remote/pp_ubuntu24.04_V1.1.2.38_202606301404_release_ros2_x86_jazzy}"
BRIDGE_DIR="${INFER_BRIDGE_DIR:-$REMOTE_DIR/x86/standalone/bin/ROS_2}"
PUPPETRY_DIR="${INFER_PUPPETRY_DIR:-$REMOTE_DIR/x86/standalone/bin/Puppetry}"
TELEOP_NODE=/ar3_cartesian_impedance_teleop
GATE_NODE=/ar3_dagger_gripper_gate
TRIGGER_TOPIC=/trigger_positions
ARM_INTERFACE="${INFER_ARM_INTERFACE:-enp132s0}"
ARM_CIDR="${INFER_ARM_CIDR:-192.168.0.100/24}"
ARM_IP="${INFER_ARM_IP:-192.168.0.160}"
ARM_LOCAL_IP="${ARM_CIDR%/*}"
# The robot client centres the controller's workspace sphere at the recorded start pose, and the recorded motions
# reach 0.43 m from it (the recording's teleoperation used 0.35 m around its own start).
WORKSPACE_RADIUS="${INFER_WORKSPACE_RADIUS:-0.45}"
# The gripper has its own wired subnet.
GRIPPER_INTERFACE="${INFER_GRIPPER_INTERFACE:-$ARM_INTERFACE}"
GRIPPER_CIDR="${INFER_GRIPPER_CIDR:-192.168.127.100/24}"
GRIPPER_LOCAL_IP="${GRIPPER_CIDR%/*}"
GRIPPER_HOST="${INFER_GRIPPER_HOST:-192.168.127.10}"
GRIPPER_PORT=50008
GRIPPER_USER="${INFER_GRIPPER_USER:-cftc}"
GRIPPER_CHANNEL="${INFER_GRIPPER_CHANNEL:-can0}"
GRIPPER_MOTOR_ID="${INFER_GRIPPER_MOTOR_ID:-1}"
FRONT_SERIAL="${INFER_FRONT_SERIAL:-CPC8563000HX}"
WRIST_SERIAL="${INFER_WRIST_SERIAL:-CV2856D000BV}"
TILT_ANGLE="${INFER_TILT_ANGLE:-40.0}"
WAIT_TIMEOUT="${INFER_TIMEOUT:-180}"
CHECKPOINT="${INFER_CHECKPOINT:-$ROOT/infer_checkpoint/29999}"
export ROS_DOMAIN_ID="${INFER_ROS_DOMAIN_ID:-111}"
export OPENPI_HOST="${OPENPI_HOST:-127.0.0.1}"
export OPENPI_PORT="${OPENPI_PORT:-8000}"
CONTROLLER=/ar3/cartesian_impedance
RT_PARAMS=install/ar3_cartesian_impedance/share/ar3_cartesian_impedance/config/rt.yaml
# The recordings read the wrist camera with fixed exposure, gain and white balance (the front one with auto settings).
WRIST_COLOR=(enable_color_auto_exposure:=false color_exposure:=80 color_gain:=16
    enable_color_auto_white_balance:=false color_white_balance:=4600)

WITH_ARM=1 WITH_SSH=1 WITH_CAMERAS=1 WITH_TILT=1 WITH_POLICY=0 WITH_DAGGER=0
WINDOWS=0 MODE=start TASK=""
INITIALIZE_GRIPPER=0
ORIGINAL_ARGS=("$@")

info() { printf '[infer_oneclick] %s\n' "$*"; }
die() { printf '[infer_oneclick] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: ./infer_oneclick.sh [OPTIONS]
  --dry-run          Print commands/dependencies; no device or network access.
  --check            Check local files/interpreters only; no device/network access.
  --checkpoint DIR   Checkpoint step directory for --with-policy (default: infer_checkpoint/29999).
  --ready-only       Connect the gripper and recheck running services; no new tabs.
  --initialize-gripper
                     Allow gripper calibration if needed (MOVES the gripper).
  --no-ssh           Reuse the running remote gripper service (gRPC 192.168.127.10:50008).
  --no-arm           Do not launch or check the arm's Cartesian impedance controller.
  --no-cameras       Do not open/check the front/wrist RGB cameras.
  --no-tilt          Do not start or move the camera tilt servo.
  --with-policy      Also serve the policy on this PC (scripts/serve_ar3_lift.sh; needs a GPU).
  --no-policy        Do not launch/check the policy server (default; serve it on the GPU server).
  --dagger           Also start the teleoperation for DAgger takeovers: MessageBridge, Puppetry,
                     the AR3 Cartesian teleoperation (the ready task sets dry_run=false: it
                     MOVES the arm while the clutch is pressed) and the gated gripper
                     teleoperation (scripts/ar3_dagger_gripper_teleop.sh).
  --windows          Separate GNOME windows instead of tabs in one new window.
  -h, --help         Show this help.

Default: the AR3 Cartesian impedance controller of the recordings (xCoreSDK RT,
enable_motion, workspace_radius_m 0.45), the remote DM gripper service, front/wrist
Orbbec RGB, 40-degree camera tilt and a readiness tab, ROS domain 111. The controller
only monitors until a client enables it; it then switches to automatic mode and
POWERS the arm. The tilt servo MOVES. The readiness task connects the gripper's CAN
interface if disconnected, preserving it for the robot client. Calibration requires
--initialize-gripper. Without --dagger no teleoperation, recording or policy action is run.
After PREPARATION READY run the robot client (it MOVES the arm only with --execute):
  scripts/run_ar3_lift_robot.sh --check
  scripts/run_ar3_lift_robot.sh --max-chunks 3                 (dry run)
  scripts/run_ar3_lift_robot.sh --execute --move-to-start
With --dagger (each episode: the policy until the clutch is pressed, then the operator
finishes the task; releasing the clutch ends the episode):
  scripts/run_ar3_lift_robot.sh --check --dagger
  scripts/run_ar3_lift_robot.sh --execute --dagger --record-dir DIR
Stop teleoperation, recording and other arm controllers first. One owner per camera.

Paths: INFER_AR3_WS, INFER_GRIPPER_DIR, INFER_ROBOT_PYTHON, INFER_TILT_DIR,
INFER_SERVO_PYTHON, INFER_ROS_SETUP, INFER_CAMERA_LAUNCH_DIR, INFER_REMOTE_DIR,
INFER_BRIDGE_DIR (MessageBridge), INFER_PUPPETRY_DIR (Puppetry).
Devices: INFER_ARM_INTERFACE, INFER_ARM_CIDR, INFER_ARM_IP, INFER_WORKSPACE_RADIUS,
INFER_GRIPPER_INTERFACE, INFER_GRIPPER_CIDR, INFER_GRIPPER_HOST, INFER_GRIPPER_USER,
INFER_GRIPPER_CHANNEL, INFER_GRIPPER_MOTOR_ID, INFER_FRONT_SERIAL, INFER_WRIST_SERIAL,
INFER_TILT_ANGLE.
Runtime: INFER_ROS_DOMAIN_ID, INFER_TIMEOUT, INFER_CHECKPOINT, OPENPI_SERVE_GPU,
OPENPI_CONFIG_NAME, OPENPI_PYTHON, OPENPI_DATA_HOME, OPENPI_HOST, OPENPI_PORT.

Before opening tabs, add the gripper's local wired address if missing (sudo may
prompt): 192.168.127.100/24 on INFER_GRIPPER_INTERFACE (default: arm interface),
and verify that 192.168.127.10 routes through this interface/source address. The
arm tab likewise adds INFER_ARM_CIDR and pings the arm before starting the
controller. Existing addresses are preserved. --check and --dry-run never change
networking.

SSH uses keys/agent or interactive authentication and VERIFIED known_hosts entries.
If a host is new, verify its fingerprint with a separate ssh login first.
Ctrl+C stops a tab's foreground process; press Enter after exit to close the tab.
The remote deploy.sh leaves the gripper service running. Closing tabs is NOT an
emergency stop and does not guarantee arm power-off or a physical stop.

Readiness requires fresh, connected, initialized gripper feedback, not just a
responsive gRPC server. If the motor is uninitialized, clear the gripper workspace
and rerun with --ready-only --initialize-gripper.
EOF
}

while (($#)); do
    case "$1" in
        --dry-run|--check)
            [[ "$MODE" == start ]] || die "Choose only one of --check and --dry-run."
            MODE="${1#--}" ;;
        --checkpoint)
            [[ $# -ge 2 && -n "$2" ]] || die "--checkpoint requires a step directory."
            CHECKPOINT="$2"; shift ;;
        --no-ssh) WITH_SSH=0 ;;
        --ready-only) TASK=ready ;;
        --initialize-gripper) INITIALIZE_GRIPPER=1 ;;
        --no-arm) WITH_ARM=0 ;;
        --no-cameras) WITH_CAMERAS=0 ;;
        --no-tilt) WITH_TILT=0 ;;
        --with-policy) WITH_POLICY=1 ;;
        --no-policy) WITH_POLICY=0 ;;
        --dagger) WITH_DAGGER=1 ;;
        --windows) WINDOWS=1 ;;
        --task)
            [[ $# -ge 2 ]] || die "--task requires a name."
            TASK="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (see --help)." ;;
    esac
    shift
done
[[ "$CHECKPOINT" == /* ]] || CHECKPOINT="$PWD/$CHECKPOINT"
[[ "$WAIT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "INFER_TIMEOUT must be a positive integer."
[[ "$GRIPPER_MOTOR_ID" =~ ^[0-9]+$ && ${#GRIPPER_MOTOR_ID} -le 3 ]] \
    && ((10#$GRIPPER_MOTOR_ID >= 1 && 10#$GRIPPER_MOTOR_ID <= 255)) \
    || die "INFER_GRIPPER_MOTOR_ID must be 1..255."
[[ "$ROS_DOMAIN_ID" =~ ^[0-9]+$ && ${#ROS_DOMAIN_ID} -le 3 ]] \
    && ((10#$ROS_DOMAIN_ID <= 232)) || die "INFER_ROS_DOMAIN_ID must be 0..232."
[[ "$OPENPI_PORT" =~ ^[0-9]+$ && ${#OPENPI_PORT} -le 5 ]] \
    && ((10#$OPENPI_PORT >= 1 && 10#$OPENPI_PORT <= 65535)) || die "OPENPI_PORT must be 1..65535."
[[ "$TILT_ANGLE" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    && awk -v angle="$TILT_ANGLE" 'BEGIN {exit !(angle >= 0 && angle <= 60)}' \
    || die "INFER_TILT_ANGLE must be 0..60 degrees."
# A decimal point keeps the ROS parameter a double.
[[ "$WORKSPACE_RADIUS" =~ ^[0-9]+[.][0-9]+$ ]] \
    && awk -v radius="$WORKSPACE_RADIUS" 'BEGIN {exit !(radius >= 0.05 && radius <= 0.8)}' \
    || die "INFER_WORKSPACE_RADIUS must be 0.05..0.8 m with a decimal point (e.g. 0.45)."

# ROS Jazzy uses system Python 3.12, never the model's Python 3.11 or inherited overlays.
ROS_CMD=(env -i "HOME=$HOME" "PATH=/usr/bin:/bin:/usr/local/bin"
    "LANG=${LANG:-C.UTF-8}" "TERM=${TERM:-xterm-256color}" "DISPLAY=${DISPLAY:-}"
    "XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}" "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
    bash --noprofile --norc -c 'set -e; source "$1"; shift; exec "$@"' _ "$ROS_SETUP")
TASKS=()
((WITH_ARM)) && TASKS+=(arm)
((WITH_SSH)) && TASKS+=(gripper)
((WITH_CAMERAS)) && TASKS+=(front wrist)
((WITH_TILT)) && TASKS+=(tilt)
((WITH_DAGGER)) && TASKS+=(bridge puppetry teleop gate)
((WITH_POLICY)) && TASKS+=(policy)
TASKS+=(ready)

quote_remote() {
    local arg
    for arg in "$@"; do printf "'%s' " "${arg//\'/\'\\\'\'}"; done
}

command_for() {
    TASK_CWD="$ROOT"
    case "$1" in
        arm)
            TASK_CWD="$AR3_WS"
            # The recordings' controller: xCoreSDK RT Cartesian impedance, as started by their oneClick.sh.
            COMMAND=(bash "$AR3_WS/jazzy.sh" ros2 run ar3_cartesian_impedance cartesian_impedance_node --ros-args
                --params-file "$AR3_WS/$RT_PARAMS" -p "robot_ip:=$ARM_IP" -p "local_ip:=$ARM_LOCAL_IP"
                -p enable_motion:=true -p auto_power_on:=true -p "workspace_radius_m:=$WORKSPACE_RADIUS") ;;
        gripper)
            local remote='cd "$HOME/gripper_sdk" && exec bash deploy.sh'
            COMMAND=(ssh -tt -o StrictHostKeyChecking=yes -o ConnectTimeout=10
                -o ServerAliveInterval=15 -o ServerAliveCountMax=3
                "$GRIPPER_USER@$GRIPPER_HOST" "$(quote_remote bash -lic "$remote")") ;;
        front|wrist)
            local launch=gemini_330_series.launch.py serial="$FRONT_SERIAL" format=MJPG
            # Gemini 305 exposes YUYV, not the front camera's MJPG profile.
            [[ "$1" == front ]] || { launch=gemini305.launch.py; serial="$WRIST_SERIAL"; format=YUYV; }
            COMMAND=("${ROS_CMD[@]}" ros2 launch "$CAMERA_LAUNCH_DIR/$launch"
                "camera_name:=$1" "serial_number:=$serial" device_num:=2 enable_color:=true
                color_width:=640 color_height:=480 color_fps:=30 "color_format:=$format"
                enable_depth:=false enable_point_cloud:=false enable_colored_point_cloud:=false
                enable_frame_sync:=false)
            [[ "$1" == front ]] \
                || COMMAND+=(uvc_backend:=v4l2 enable_left_color:=false enable_right_color:=false "${WRIST_COLOR[@]}") ;;
        tilt)
            TASK_CWD="$TILT_DIR"
            COMMAND=("${ROS_CMD[@]}" /usr/bin/python3 "$TILT_DIR/camera_tilt_ros2.py" --ros-args
                -p "startup_angle_deg:=$TILT_ANGLE" -p "servo_python:=$SERVO_PYTHON") ;;
        policy)
            COMMAND=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH
                -u COLCON_PREFIX_PATH -u CMAKE_PREFIX_PATH -u ROS_PACKAGE_PATH
                bash "$ROOT/scripts/serve_ar3_lift.sh" "$CHECKPOINT") ;;
        bridge|puppetry)
            # As in the recordings' oneClick.sh: the vendor binary from its directory with ROS sourced, keeping the
            # desktop session (Puppetry opens a window) but not the model's Python or other ROS overlays.
            local binary=./MessageBridge
            TASK_CWD="$BRIDGE_DIR"
            [[ "$1" == bridge ]] || { binary=./Puppetry; TASK_CWD="$PUPPETRY_DIR"; }
            COMMAND=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH
                -u CMAKE_PREFIX_PATH -u ROS_PACKAGE_PATH "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
                bash --noprofile --norc -c 'set -e; source "$1"; shift; exec "$@"' _ "$ROS_SETUP" "$binary") ;;
        teleop)
            TASK_CWD="$AR3_WS"
            # The recordings' teleoperation; its return-home range matches the controller's workspace radius. It
            # starts with dry_run=true and the ready task enables it.
            COMMAND=(bash "$AR3_WS/jazzy.sh" impedance-teleop -p return_home_position_tolerance_m:=0.055
                -p return_home_rotation_tolerance_rad:=0.06 -p return_home_timeout_s:=75.0
                -p return_home_max_rotation_lead_rad:=0.10 -p "return_home_workspace_radius_m:=$WORKSPACE_RADIUS") ;;
        gate)
            # It waits for the gripper the ready task connects (after its own checks).
            COMMAND=(env "INFER_ROBOT_PYTHON=$ROBOT_PYTHON" "INFER_ROS_SETUP=$ROS_SETUP"
                "INFER_ROS_DOMAIN_ID=$ROS_DOMAIN_ID" bash "$ROOT/scripts/ar3_dagger_gripper_teleop.sh"
                --wait "$((2 * WAIT_TIMEOUT))" --gripper-address "$GRIPPER_HOST:$GRIPPER_PORT") ;;
        ready) COMMAND=() ;;
        *) die "Unknown task: $1" ;;
    esac
}

gripper_helper() {
    env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
        "$ROBOT_PYTHON" "$ROOT/scripts/prepare_ar3_gripper.py" "$@" --gripper-address "$GRIPPER_HOST:$GRIPPER_PORT"
}

prepare_gripper() {
    local args=(--connect --channel "$GRIPPER_CHANNEL" --motor-id "$GRIPPER_MOTOR_ID" --timeout "$WAIT_TIMEOUT")
    ((INITIALIZE_GRIPPER)) && args+=(--initialize)
    gripper_helper "${args[@]}" \
        || die "Gripper preparation failed; inspect its error above. If calibration is required, clear the fingers and use --ready-only --initialize-gripper (moves the gripper)."
}

check_gripper() {
    gripper_helper --check \
        || die "Gripper feedback is not ready. Check the preparation output and whether another client disconnected the motor."
}

preflight() {
    local cmd path
    for cmd in bash env timeout awk ssh flock ip sudo; do
        command -v "$cmd" >/dev/null || die "Missing command: $cmd"
    done
    [[ -d "/sys/class/net/$GRIPPER_INTERFACE" ]] || die "Missing gripper interface: $GRIPPER_INTERFACE"
    local files=("$ROS_SETUP" "$ROOT/scripts/prepare_ar3_gripper.py" "$ROOT/scripts/ar3_lift_robot_client.py")
    local executables=("$ROBOT_PYTHON")
    if ((WITH_ARM)); then
        for cmd in ping pgrep; do command -v "$cmd" >/dev/null || die "Missing command: $cmd"; done
        files+=("$AR3_WS/jazzy.sh" "$AR3_WS/install/local_setup.bash" "$AR3_WS/$RT_PARAMS"
            "$AR3_WS/scripts/ar3_wait_cartesian_state.py")
        executables+=("$AR3_WS/install/ar3_cartesian_impedance/lib/ar3_cartesian_impedance/cartesian_impedance_node")
        [[ -d "/sys/class/net/$ARM_INTERFACE" ]] || die "Missing arm interface: $ARM_INTERFACE"
    fi
    if ((WITH_CAMERAS)); then
        files+=("$CAMERA_LAUNCH_DIR/gemini_330_series.launch.py" "$CAMERA_LAUNCH_DIR/gemini305.launch.py")
        [[ -n "$FRONT_SERIAL" && -n "$WRIST_SERIAL" && "$FRONT_SERIAL" != "$WRIST_SERIAL" ]] \
            || die "Set distinct nonempty INFER_FRONT_SERIAL and INFER_WRIST_SERIAL."
    fi
    if ((WITH_TILT)); then
        files+=("$TILT_DIR/camera_tilt_ros2.py" "$TILT_DIR/usb_servo.py")
        executables+=("$SERVO_PYTHON")
    fi
    if ((WITH_DAGGER)); then
        command -v pgrep >/dev/null || die "Missing command: pgrep"
        files+=("$BRIDGE_DIR/config/MessageBridge.yaml" "$AR3_WS/jazzy.sh"
            "$AR3_WS/scripts/ar3_cartesian_impedance_teleop.py" "$ROOT/scripts/ar3_dagger_gripper_teleop.py"
            "$ROOT/scripts/ar3_dagger_gripper_teleop.sh")
        executables+=("$BRIDGE_DIR/MessageBridge" "$PUPPETRY_DIR/Puppetry")
        # jazzy.sh impedance-teleop keeps its per-domain lock there.
        [[ -d "$AR3_WS/.venv-teleop" ]] || die "Missing $AR3_WS/.venv-teleop (the AR3 teleoperation environment)."
    fi
    if ((WITH_POLICY)); then
        command -v curl >/dev/null || die "Missing command: curl"
        files+=("${OPENPI_DATA_HOME:-$HOME/.cache/openpi}/big_vision/paligemma_tokenizer.model")
        executables+=("${OPENPI_PYTHON:-$ROOT/.venv/bin/python}")
        [[ -d "$CHECKPOINT/params" && -d "$CHECKPOINT/assets" ]] \
            || die "Checkpoint needs params/ and assets/: $CHECKPOINT"
        local stats=("$CHECKPOINT"/assets/*/norm_stats.json)
        if [[ -n "${OPENPI_AR3_ASSET_ID:-}" ]]; then
            files+=("$CHECKPOINT/assets/$OPENPI_AR3_ASSET_ID/norm_stats.json")
        else
            [[ ${#stats[@]} -eq 1 && -f "${stats[0]}" ]] \
                || die "Expected one normalization asset; set OPENPI_AR3_ASSET_ID."
        fi
    fi
    for path in "${files[@]}"; do [[ -f "$path" ]] || die "Missing file: $path"; done
    for path in "${executables[@]}"; do [[ -x "$path" ]] || die "Missing executable: $path"; done
    env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$ROBOT_PYTHON" -c 'import DM_V3Gripper.gripper_pb2, grpc, numpy' \
        || die "$ROBOT_PYTHON cannot import DM_V3Gripper, grpc and numpy; set INFER_ROBOT_PYTHON to the gripper's ROS venv."
    info "Local prerequisites exist. No hardware, network or ROS commands were run."
}

print_plan() {
    info "ROS domain $ROS_DOMAIN_ID; arm $ARM_IP (workspace radius $WORKSPACE_RADIUS m); gripper $GRIPPER_HOST:$GRIPPER_PORT"
    ((WITH_POLICY)) && info "Policy: $CHECKPOINT on GPU ${OPENPI_SERVE_GPU:-0}, ws://$OPENPI_HOST:$OPENPI_PORT"
    ((WITH_DAGGER)) && info "DAgger: MessageBridge, Puppetry, the AR3 teleoperation (enabled by the ready task: it MOVES the arm while the clutch is pressed) and the gated gripper teleoperation."
    if [[ "$TASK" == ready ]]; then
        info "Recheck existing services; require gripper $GRIPPER_HOST via $GRIPPER_INTERFACE src $GRIPPER_LOCAL_IP. No new tabs."
    else
        info "The controller monitors only until a client enables it (then automatic mode + POWER ON); the tilt servo moves to $TILT_ANGLE degrees. NO policy action is executed."
        info "Before tabs: add $GRIPPER_CIDR to $GRIPPER_INTERFACE only if absent; require gripper $GRIPPER_HOST via this interface with source $GRIPPER_LOCAL_IP."
    fi
    ((INITIALIZE_GRIPPER)) && info "Gripper calibration is enabled if needed; this MOVES the gripper."
    local task
    local planned_tasks=("${TASKS[@]}")
    [[ -z "$TASK" ]] || planned_tasks=("$TASK")
    for task in "${planned_tasks[@]}"; do
        command_for "$task"
        printf '\n[%s] cwd=%s\n' "$task" "$TASK_CWD"
        if [[ "$task" == ready ]]; then
            printf '  Connect CAN %s motor %s at %s:%s if disconnected; retain the connection for inference.\n' \
                "$GRIPPER_CHANNEL" "$GRIPPER_MOTOR_ID" "$GRIPPER_HOST" "$GRIPPER_PORT"
            printf '  Require fresh, connected, initialized motor feedback; calibration only with the explicit flag.\n'
            ((WITH_ARM)) && printf '  Wait for the controller RT state, metadata and status, then %s/tau_m.\n' "$CONTROLLER"
            ((WITH_CAMERAS)) && printf '  Wait for /front/color/image_raw and /wrist/color/image_raw samples.\n'
            ((WITH_TILT)) && printf '  Wait for measured /camera_tilt/state_deg within %s +/- 2 degrees.\n' "$TILT_ANGLE"
            ((WITH_DAGGER)) && printf '  Wait for %s, set %s dry_run=false and verify it, wait for %s.\n' \
                "$TRIGGER_TOPIC" "$TELEOP_NODE" "$GATE_NODE"
            ((WITH_POLICY)) && printf '  Wait for http://%s:%s/healthz.\n' "$OPENPI_HOST" "$OPENPI_PORT"
        else
            printf '  '; printf '%q ' "${COMMAND[@]}"; printf '\n'
        fi
        case "$task" in
            arm) printf '  Add %s to %s only if absent; verify route and ping; refuse old joint drivers.\n' "$ARM_CIDR" "$ARM_INTERFACE" ;;
            gripper) printf '  Reuse the service if %s:%s already listens.\n' "$GRIPPER_HOST" "$GRIPPER_PORT" ;;
            bridge|puppetry) printf '  Reuse it if it already runs.\n' ;;
            teleop) printf '  Start after the controller is ready, with dry_run=true; the ready task enables it.\n' ;;
            gate) printf '  Drives the gripper only during DAgger takeovers; never stops or disconnects it.\n' ;;
        esac
    done
    ((WITH_SSH)) || info "Remote startup skipped; the gripper gRPC service must already run."
    info "Preparation never executes policy actions: scripts/run_ar3_lift_robot.sh does, only with --execute."
}

tcp_ready() {
    timeout 2 bash --noprofile --norc -c 'exec 3<>/dev/tcp/"$1"/"$2"' _ "$1" "$2" >/dev/null 2>&1
}

wait_topic() {
    local topic="$1" type="$2"; shift 2
    info "Waiting for data on $topic"
    timeout --signal=TERM --kill-after=3 "$WAIT_TIMEOUT" "${ROS_CMD[@]}" ros2 topic echo \
        "$topic" "$type" --once --no-daemon --qos-reliability best_effort "$@" \
        || die "No qualifying sample on $topic; inspect its task tab and ROS domain."
}

wait_node() {
    local deadline=$((SECONDS + WAIT_TIMEOUT)) nodes
    info "Waiting for the ROS node $1"
    until nodes=$(timeout --kill-after=3 15 "${ROS_CMD[@]}" ros2 node list --no-daemon) \
        && grep -qxF -- "$1" <<<"$nodes"; do
        ((SECONDS < deadline)) || die "No $1 node; inspect its task tab."
        sleep 2
    done
}

enable_teleop() {
    local deadline=$((SECONDS + WAIT_TIMEOUT)) result=""
    info "Waiting for $TELEOP_NODE, then setting dry_run=false"
    while ((SECONDS < deadline)); do
        # Jazzy's param set may exit 0 when the parameter is rejected: check its message.
        if result=$(timeout --kill-after=3 10 bash "$AR3_WS/jazzy.sh" ros2 param set "$TELEOP_NODE" dry_run false \
            --no-daemon --timeout 2 2>&1) && [[ "$result" == *"Set parameter successful"* ]]; then
            result=$(timeout --kill-after=3 10 bash "$AR3_WS/jazzy.sh" ros2 param get "$TELEOP_NODE" dry_run \
                --no-daemon 2>&1) || true
            [[ "$result" == *"Boolean value is: False"* ]] || die "Could not verify dry_run=false: $result"
            info "The AR3 teleoperation is enabled (dry_run=false): the clutch takes over the arm."
            return
        fi
        sleep 2
    done
    printf '%s\n' "$result" >&2
    die "The AR3 teleoperation did not accept dry_run=false; inspect its tab."
}

wait_controller() {
    info "Waiting for the controller's RT state, metadata and a non-fault status"
    timeout --signal=TERM --kill-after=3 "$((WAIT_TIMEOUT + 3))" bash "$AR3_WS/jazzy.sh" impedance-wait-state \
        --timeout "$WAIT_TIMEOUT" || die "The Cartesian impedance controller is not ready; inspect the arm tab."
}

check_conflicts() {
    local nodes
    nodes=$(timeout --kill-after=3 15 "${ROS_CMD[@]}" ros2 node list --no-daemon) \
        || die "Cannot inspect ROS nodes; no tasks started."
    reject_command_sources "$nodes"
    if ((WITH_ARM)) && [[ "$nodes" == *cartesian_impedance_node* || "$nodes" == *controller_manager* ]]; then
        die "An arm controller already exists; inspect it and use --no-arm to keep a running impedance controller."
    fi
    if ((WITH_CAMERAS)) && [[ "$nodes" == *"/front/"* || "$nodes" == *"/wrist/"* ]]; then
        die "Camera nodes already exist; use --no-cameras to preserve their ownership."
    fi
    if ((WITH_TILT)) && [[ "$nodes" == *camera_tilt_servo* ]]; then
        die "The tilt node already exists; use --no-tilt (no servo command will be sent)."
    fi
    if ((WITH_POLICY)) && tcp_ready "$OPENPI_HOST" "$OPENPI_PORT"; then
        die "Policy port is occupied; use --no-policy or choose another OPENPI_PORT."
    fi
    if ((WITH_DAGGER)) && [[ "$nodes" == *ar3_dagger_gripper_gate* ]]; then
        die "The DAgger gripper gate already runs; stop it first."
    fi
}

reject_command_sources() {
    # With --dagger the ready task expects the AR3 teleoperation started by this script (not a gripper teleoperation).
    if [[ "$TASK" != ready || $WITH_DAGGER == 0 ]] && [[ "$1" == *ar3_cartesian_impedance_teleop* ]]; then
        die "Teleoperation is running. Stop it before inference preparation."
    fi
    [[ "$1" != *ar3r_cartesian_retargeting* && "$1" != *gripper_teleop* ]] \
        || die "Teleoperation is running. Stop it before inference preparation."
    [[ "$1" != *ar3_lift_robot_client* ]] \
        || die "The AR3 LIFT robot client is running. Stop it before gripper preparation."
}

check_gripper_route() {
    local route
    route=$(ip -4 route get "$GRIPPER_HOST") || die "Cannot resolve route to gripper $GRIPPER_HOST."
    [[ " $route " == *" dev $GRIPPER_INTERFACE "* && " $route " == *" src $GRIPPER_LOCAL_IP "* ]] \
        || die "Unexpected gripper route: $route. Expected dev $GRIPPER_INTERFACE src $GRIPPER_LOCAL_IP; run ./infer_oneclick.sh to prepare the wired network."
}

prepare_gripper_network() {
    local addresses
    addresses=$(ip -o -4 addr show dev "$GRIPPER_INTERFACE")
    if ! awk -v cidr="$GRIPPER_CIDR" '$4 == cidr {found=1} END {exit !found}' <<<"$addresses"; then
        info "Adding gripper wired address $GRIPPER_CIDR to $GRIPPER_INTERFACE; enter the local sudo password if prompted."
        sudo ip addr add "$GRIPPER_CIDR" dev "$GRIPPER_INTERFACE" \
            || die "Could not add gripper wired address $GRIPPER_CIDR; no tasks started."
    fi
    check_gripper_route
    info "Gripper route ready: $GRIPPER_HOST via $GRIPPER_INTERFACE src $GRIPPER_LOCAL_IP."
}

run_task() {
    local addresses route nodes deadline binary
    case "$TASK" in
        gripper|ready) check_gripper_route ;;
    esac
    case "$TASK" in
        arm)
            addresses=$(ip -o -4 addr show dev "$ARM_INTERFACE")
            if ! awk -v cidr="$ARM_CIDR" '$4 == cidr {found=1} END {exit !found}' <<<"$addresses"; then
                info "Adding arm wired address $ARM_CIDR to $ARM_INTERFACE; enter the local sudo password if prompted."
                sudo ip addr add "$ARM_CIDR" dev "$ARM_INTERFACE"
            fi
            route=$(ip route get "$ARM_IP")
            [[ " $route " == *" dev $ARM_INTERFACE "* && " $route " == *" src $ARM_LOCAL_IP "* ]] \
                || die "Unexpected arm route: $route. Expected dev $ARM_INTERFACE src $ARM_LOCAL_IP."
            ping -I "$ARM_LOCAL_IP" -c 4 -W 2 "$ARM_IP" || die "The arm $ARM_IP does not answer; the controller was not started."
            # The old joint driver / MoveIt and this SDK controller must never command the same robot.
            if pgrep -f '[r]os2_control_node|[r]okae_driver7|[r]okae_driver($| )' >/dev/null; then
                die "An old joint-control process (MoveIt / rokae_driver) is running; stop it first."
            fi ;;
        gripper)
            if tcp_ready "$GRIPPER_HOST" "$GRIPPER_PORT"; then
                info "Reusing the gripper service listening on $GRIPPER_HOST:$GRIPPER_PORT; the ready tab checks it."
                return
            fi ;;
        bridge|puppetry)
            binary=MessageBridge
            [[ "$TASK" == bridge ]] || binary=Puppetry
            if pgrep -x "$binary" >/dev/null; then
                info "Reusing the running $binary."
                return
            fi ;;
        teleop) wait_controller ;;
        ready)
            nodes=$(timeout --kill-after=3 15 "${ROS_CMD[@]}" ros2 node list --no-daemon) \
                || die "Cannot inspect ROS nodes; gripper preparation was not attempted."
            reject_command_sources "$nodes"
            prepare_gripper
            if ((WITH_ARM)); then
                wait_controller
                wait_topic "$CONTROLLER/tau_m" std_msgs/msg/Float64MultiArray --field data
            fi
            if ((WITH_CAMERAS)); then
                wait_topic /front/color/image_raw sensor_msgs/msg/Image --field header
                wait_topic /wrist/color/image_raw sensor_msgs/msg/Image --field header
            fi
            if ((WITH_TILT)); then
                wait_topic /camera_tilt/state_deg std_msgs/msg/Float64 --field data \
                    --qos-reliability reliable --qos-durability transient_local \
                    --filter "abs(m - $TILT_ANGLE) <= 2.0"
            fi
            check_gripper
            if ((WITH_DAGGER)); then
                wait_topic "$TRIGGER_TOPIC" std_msgs/msg/Float64MultiArray --field data
                enable_teleop
                wait_node "$GATE_NODE"
            fi
            if ((WITH_POLICY)); then
                deadline=$((SECONDS + WAIT_TIMEOUT))
                until curl --fail --silent --show-error --noproxy '*' --max-time 2 \
                    "http://$OPENPI_HOST:$OPENPI_PORT/healthz"; do
                    ((SECONDS < deadline)) || die "Policy readiness timed out; inspect its tab."
                    sleep 2
                done
            fi
            info "PREPARATION READY for the selected services. No policy action was executed."
            info "Gripper feedback is valid and fresh; its CAN connection is retained for inference."
            if ((WITH_DAGGER)); then
                info "DAgger teleoperation is live: the clutch takes over the arm; the trigger takes the gripper once it meets the fingers."
                info "Next: scripts/run_ar3_lift_robot.sh --check --dagger, then --execute --dagger --record-dir DIR."
            else
                info "Next: scripts/run_ar3_lift_robot.sh --check, a dry run (--max-chunks 3), then --execute --move-to-start."
            fi
            return ;;
    esac
    command_for "$TASK"
    cd -- "$TASK_CWD"
    "${COMMAND[@]}"
}

open_terminal() {
    local option="$1" task="$2" output key value service="" screen=""
    local terminal_env=(env -u GNOME_TERMINAL_SERVICE -u GNOME_TERMINAL_SCREEN)
    if [[ "$option" == --tab ]]; then
        terminal_env+=("GNOME_TERMINAL_SERVICE=$TERMINAL_SERVICE" "GNOME_TERMINAL_SCREEN=$TERMINAL_SCREEN")
    fi
    output=$("${terminal_env[@]}" gnome-terminal "$option" --print-environment \
        "--title=AR3 LIFT: $task" "--working-directory=$ROOT" -- \
        bash --noprofile --norc "$SELF" "${ORIGINAL_ARGS[@]}" --checkpoint "$CHECKPOINT" --task "$task") \
        || die "Could not open task $task; inspect already opened tabs before retrying."
    while IFS='=' read -r key value; do
        case "$key" in
            GNOME_TERMINAL_SERVICE) service="$value" ;;
            GNOME_TERMINAL_SCREEN) screen="$value" ;;
        esac
    done <<<"$output"
    [[ -n "$service" && -n "$screen" ]] || die "GNOME did not return identifiers for $task."
    TERMINAL_SERVICE="$service" TERMINAL_SCREEN="$screen"
}

if [[ "$MODE" == dry-run ]]; then print_plan; exit 0; fi
if [[ "$MODE" == check ]]; then preflight; exit 0; fi

if [[ -n "$TASK" ]]; then
    [[ "$TASK" != ready ]] || preflight
    command_for "$TASK"
    state_root="${XDG_RUNTIME_DIR:-/tmp}/ar3-lift-infer-$UID-$ROS_DOMAIN_ID"
    umask 077
    mkdir -p -- "$state_root"
    [[ -O "$state_root" && ! -L "$state_root" ]] || die "Unsafe runtime directory: $state_root"
    exec 9>"$state_root/$TASK.lock"
    flock -n 9 || die "Task $TASK already has an owner in ROS domain $ROS_DOMAIN_ID."
    set +e
    (set -e; run_task)
    status=$?
    set -e
    flock -u 9
    info "Task $TASK exited with status $status. Closing a tab is not a physical stop."
    if [[ -t 0 ]]; then read -r -p "Press Enter to close this tab. " _; fi
    exit "$status"
fi

preflight
command -v gnome-terminal >/dev/null || die "Missing gnome-terminal; use --dry-run to launch tasks manually."
[[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] || die "No graphical session; use --dry-run."
check_conflicts
prepare_gripper_network
print_plan
option=--window
for task in "${TASKS[@]}"; do
    open_terminal "$option" "$task"
    ((WINDOWS)) || option=--tab
done
info "Preparation tasks opened. Wait for PREPARATION READY in the ready tab."
