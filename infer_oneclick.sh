#!/usr/bin/env bash
# Public entrypoint: prepare the AR3 arm controller, DM gripper, cameras and camera tilt for AR3 LIFT inference.
# Then run the policy on the robot with scripts/run_ar3_lift_robot.sh (see --help).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/scripts/prepare_ar3_inference.sh" "$@"
