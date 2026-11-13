#!/usr/bin/env bash
# LIFT stage two (DAgger post-training) on the AR3 with H100 GPUs 0, 1 and 2: continue an offline checkpoint of
# scripts/train_ar3_lift_gpu012.sh on batches with a fixed 1:1 ratio of offline demonstrations and human corrections
# recorded by `scripts/ar3_lift_robot_client.py --dagger --record-dir DIR`. As in LIFT (--online-intervention-only),
# only action chunks recorded entirely under human control (control_flag == -1) are used; the policy-controlled
# frames before a takeover are not training targets.
#
# Usage (from the repository root):
#   bash scripts/train_ar3_lift_dagger_gpu012.sh <offline checkpoint step dir> <DAgger dataset dir> [<DAgger dir> ...]
# e.g.
#   bash scripts/train_ar3_lift_dagger_gpu012.sh \
#       /data/jiang/train_checkpoints/pi05_ar3_lift_reactive/ar3_lift_30k_gpu012/29999 /data/jiang/ar3_dagger/round1
# Next round: start from this run's last step dir and pass every correction dataset collected so far.
# Serve the result like an offline checkpoint: bash scripts/serve_ar3_lift.sh <checkpoint>/<config>/<exp>/<step>
#
# Overrides (environment variables), otherwise as in scripts/train_ar3_lift_gpu012.sh:
#   OPENPI_CONFIG_NAME       pi05_ar3_lift_reactive_dagger (default, 13D force) | pi05_ar3_lift_reactive_wrench6_dagger;
#                            must match the offline checkpoint's model
#   OPENPI_EXP_NAME          experiment name (default: ar3_lift_dagger_<date>)
#   OPENPI_AR3_DAGGER_RATIO  share of DAgger samples in every batch (default 0.5 = LIFT 1:1 offline:online)
#   OPENPI_AR3_REPO_ID       offline LeRobot v3.0 dataset (default /data/jiang/ar3_eef_forceVisuoTactaile_front_wrist);
#                            several ':'-separated ones as in scripts/train_ar3_lift_gpu012.sh
#   OPENPI_AR3_ASSET_ID      norm-stats id under <step>/assets (default: the only one there)
#   OPENPI_CUDA_VISIBLE_DEVICES (default 0,1,2), OPENPI_BATCH_SIZE (default 96), OPENPI_NUM_WORKERS (default 16)
#   OPENPI_NUM_TRAIN_STEPS (default 10000), OPENPI_SAVE_INTERVAL (default 1000), OPENPI_KEEP_PERIOD (default 2000)
#   OPENPI_RESUME=1 / OPENPI_OVERWRITE=1, WANDB_MODE (default offline), OPENPI_DRY_RUN=1 prints the commands
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if (($# < 2)); then
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//' >&2
    exit 2
fi
STEP_DIR="$(realpath -m "$1")"
shift
DAGGER_DIRS=()
for directory in "$@"; do
    DAGGER_DIRS+=("$(realpath -m "${directory}")")
done

CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_ar3_lift_reactive_dagger}"
EXP_NAME="${OPENPI_EXP_NAME:-ar3_lift_dagger_$(date +%Y%m%d_%H%M)}"
GPUS="${OPENPI_CUDA_VISIBLE_DEVICES:-0,1,2}"
NUM_GPUS="$(awk -F, '{print NF}' <<<"${GPUS}")"
BATCH_SIZE="${OPENPI_BATCH_SIZE:-96}"
NUM_TRAIN_STEPS="${OPENPI_NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${OPENPI_SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${OPENPI_LOG_INTERVAL:-100}"
KEEP_PERIOD="${OPENPI_KEEP_PERIOD:-2000}"
CHECKPOINT_BASE_DIR="${OPENPI_CHECKPOINT_BASE_DIR:-/data/jiang/train_checkpoints}"
NUM_WORKERS="${OPENPI_NUM_WORKERS:-16}"
PYTHON="${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ "${CONFIG_NAME}" != *_dagger ]]; then
    echo "OPENPI_CONFIG_NAME=${CONFIG_NAME} is not a DAgger preset (pi05_ar3_lift_reactive_dagger, ..._wrench6_dagger)" >&2
    exit 1
fi
if ((BATCH_SIZE % NUM_GPUS != 0)); then
    echo "OPENPI_BATCH_SIZE=${BATCH_SIZE} must be divisible by the number of GPUs (${NUM_GPUS})" >&2
    exit 1
fi
if [[ ! -d "${STEP_DIR}/params" ]]; then
    echo "${STEP_DIR} is not a checkpoint step directory (missing params/)" >&2
    exit 1
fi
if [[ -z "${OPENPI_AR3_ASSET_ID:-}" ]]; then
    stats=("${STEP_DIR}"/assets/*/norm_stats.json)
    if ((${#stats[@]} != 1)) || [[ ! -f "${stats[0]}" ]]; then
        echo "Expected exactly one ${STEP_DIR}/assets/<asset_id>/norm_stats.json; set OPENPI_AR3_ASSET_ID" >&2
        exit 1
    fi
    OPENPI_AR3_ASSET_ID="$(basename "$(dirname "${stats[0]}")")"
fi
for directory in "${DAGGER_DIRS[@]}"; do
    if [[ ! -f "${directory}/meta/info.json" ]]; then
        echo "${directory} is not a LeRobot v3.0 recording (missing meta/info.json)" >&2
        exit 1
    fi
    if [[ "${directory}" == *:* ]]; then
        echo "DAgger dataset paths must not contain ':' (${directory})" >&2
        exit 1
    fi
done

export OPENPI_AR3_REPO_ID="${OPENPI_AR3_REPO_ID:-/data/jiang/ar3_eef_forceVisuoTactaile_front_wrist}"
export OPENPI_AR3_ASSET_ID
export OPENPI_AR3_DAGGER_INIT_PARAMS="${STEP_DIR}/params"
export OPENPI_AR3_DAGGER_ASSETS_DIR="${STEP_DIR}/assets"
OPENPI_AR3_DAGGER_REPO_IDS="$(IFS=:; echo "${DAGGER_DIRS[*]}")"
export OPENPI_AR3_DAGGER_REPO_IDS
export OPENPI_AR3_DAGGER_RATIO="${OPENPI_AR3_DAGGER_RATIO:-0.5}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
# As in scripts/train_ar3_lift_gpu012.sh: with more than 4 GPUs NCCL needs more of the memory outside JAX's pool.
if ((NUM_GPUS > 4)); then
    DEFAULT_MEM_FRACTION=0.90
else
    DEFAULT_MEM_FRACTION=0.95
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-${DEFAULT_MEM_FRACTION}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

run() {
    echo "+ $*"
    if [[ "${OPENPI_DRY_RUN:-0}" != "1" ]]; then
        "$@"
    fi
}

IFS=: read -r -a OFFLINE_DIRS <<<"${OPENPI_AR3_REPO_ID}"
for directory in "${OFFLINE_DIRS[@]}"; do
    if [[ -n "${directory}" && ! -e "${directory}/meta/info.json" ]]; then
        echo "Missing ${directory}/meta/info.json" >&2
        exit 1
    fi
done
for path in "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model" "${PYTHON}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${path}" >&2
        exit 1
    fi
done

echo "Initialization: ${OPENPI_AR3_DAGGER_INIT_PARAMS} (norm stats: ${OPENPI_AR3_DAGGER_ASSETS_DIR}/${OPENPI_AR3_ASSET_ID})"
echo "Offline data:   ${OPENPI_AR3_REPO_ID}"
echo "DAgger data:    ${DAGGER_DIRS[*]} (ratio ${OPENPI_AR3_DAGGER_RATIO} per batch)"

# Usable correction chunks per dataset (fails early if none of them has a chunk entirely under human control).
run env JAX_PLATFORMS=cpu "${PYTHON}" -c '
import sys
from openpi.data import lerobot_v3
from openpi.training import config as _config
model = _config.get_config(sys.argv[1]).model
horizon = model.async_action_horizon if getattr(model, "async_action_horizon", -1) > 0 else model.action_horizon
total = 0
for root in sys.argv[2:]:
    options = lerobot_v3.LeRobotV3Options(decode_videos=False, intervention_only=True)
    try:
        dataset = lerobot_v3.LeRobotV3Dataset(root, sequence_keys=("actions",), sequence_length=horizon, options=options)
        usable = len(dataset)
    except ValueError as error:
        print(f"{root}: {error}")
        usable = 0
    total += usable
    print(f"{root}: {usable} human-correction chunks (horizon {horizon})")
sys.exit(0 if total else "No human-correction chunks (control_flag == -1) in the DAgger datasets")
' "${CONFIG_NAME}" "${DAGGER_DIRS[@]}"

TRAIN_ARGS=(
    "${CONFIG_NAME}"
    --exp-name "${EXP_NAME}"
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
    --fsdp-devices "${NUM_GPUS}"
    --batch-size "${BATCH_SIZE}"
    --num-train-steps "${NUM_TRAIN_STEPS}"
    --save-interval "${SAVE_INTERVAL}"
    --log-interval "${LOG_INTERVAL}"
    --keep-period "${KEEP_PERIOD}"
    --num-workers "${NUM_WORKERS}"
)
if [[ "${OPENPI_RESUME:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--resume)
elif [[ "${OPENPI_OVERWRITE:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--overwrite)
fi
if [[ "${WANDB_MODE}" == "disabled" ]]; then
    TRAIN_ARGS+=(--no-wandb-enabled)
fi

mkdir -p "${REPO_ROOT}/logs" "${CHECKPOINT_BASE_DIR}"
LOG_FILE="${REPO_ROOT}/logs/${CONFIG_NAME}_${EXP_NAME}.log"
echo "JAX preallocates ${XLA_PYTHON_CLIENT_MEM_FRACTION} of each of the ${NUM_GPUS} GPUs"
echo "Logging to ${LOG_FILE}"
if [[ "${OPENPI_DRY_RUN:-0}" == "1" ]]; then
    run "${PYTHON}" scripts/train.py "${TRAIN_ARGS[@]}"
else
    run "${PYTHON}" scripts/train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
fi
