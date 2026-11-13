#!/usr/bin/env bash
# Train the LIFT reactive-force policy (pi05 + reactive force expert) on the AR3 Cartesian impedance + DM V3 gripper
# LeRobot v3.0 dataset with H100 GPUs 0, 1 and 2.
#
# Inputs: front camera (base image), wrist camera (left wrist image), TCP pose + gripper opening (state, zeroed by the
# relative-action transform as in LIFT) and the measured force sequence over the action horizon:
# observation.tcp_wrench (6D) + observation.tau_m (7 joint torques) = 13D. Actions: relative TCP pose + gripper target.
#
# Quick start (from the repository root):
#   bash scripts/train_ar3_lift_gpu012.sh
#
# Overrides (environment variables):
#   OPENPI_CONFIG_NAME     pi05_ar3_lift_reactive (default: 13D force = tcp_wrench + tau_m) |
#                          pi05_ar3_lift_reactive_wrench6 (tcp_wrench only) | pi05_ar3_lift (vision only, no force)
#   OPENPI_EXP_NAME        experiment name (default: ar3_lift_<date>), <checkpoint root>/<config>/<exp>/
#   OPENPI_AR3_REPO_ID     LeRobot v3.0 dataset dir (default: /data/jiang/ar3_eef_forceVisuoTactaile_front_wrist), or
#                          several ':'-separated dirs recorded alike (same fps and sensors) to train as one, e.g.
#                          /data/jiang/ar3_eef_forceVisuoTactaile_plugIn:/data/jiang/ar3_eef_forceVisuoTactaile_plugIn_35
#   OPENPI_AR3_ASSET_ID    norm stats id, assets/<config>/<asset_id>/ (default: the dataset dir names joined by '+')
#   OPENPI_AR3_WEIGHT_PATH initialization params (default: /data/jiang/checkpoints/openpi-assets/checkpoints/pi05_base/params)
#   OPENPI_CUDA_VISIBLE_DEVICES  GPUs (default 0,1,2); fsdp devices = number of GPUs
#   OPENPI_BATCH_SIZE      global batch, divisible by the GPU count (default 96 = 32 per H100)
#   OPENPI_NUM_TRAIN_STEPS (default 30000, as in LIFT), OPENPI_NUM_WORKERS (default 16)
#   OPENPI_LR_DECAY_STEPS  cosine decay length (default: the preset's LIFT schedule)
#   OPENPI_SAVE_INTERVAL   checkpoint interval (default 1000), OPENPI_LOG_INTERVAL (default 100)
#   OPENPI_KEEP_PERIOD     keep every N-th checkpoint permanently (default 5000); the latest one is always kept
#   OPENPI_CHECKPOINT_BASE_DIR  checkpoint root (default /data/jiang/train_checkpoints)
#   XLA_PYTHON_CLIENT_MEM_FRACTION  fraction of each GPU preallocated by JAX (default 0.95; 0.90 with more than 4 GPUs)
#   NCCL_SOCKET_IFNAME / NCCL_IB_DISABLE  NCCL bootstrap interface / IB switch (default lo / 1: single node, NVLink only)
#   OPENPI_RESUME=1 resumes the experiment, OPENPI_OVERWRITE=1 overwrites it
#   OPENPI_RECOMPUTE_NORM_STATS=1 recomputes assets/<config>/<asset_id>/norm_stats.json
#   WANDB_MODE  defaults to offline; use online / disabled as needed
#   OPENPI_DRY_RUN=1 prints the commands without running them
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_ar3_lift_reactive}"
EXP_NAME="${OPENPI_EXP_NAME:-ar3_lift_$(date +%Y%m%d_%H%M)}"
GPUS="${OPENPI_CUDA_VISIBLE_DEVICES:-0,1,2}"
NUM_GPUS="$(awk -F, '{print NF}' <<<"${GPUS}")"
BATCH_SIZE="${OPENPI_BATCH_SIZE:-96}"
NUM_TRAIN_STEPS="${OPENPI_NUM_TRAIN_STEPS:-600000}"
SAVE_INTERVAL="${OPENPI_SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${OPENPI_LOG_INTERVAL:-100}"
KEEP_PERIOD="${OPENPI_KEEP_PERIOD:-5000}"
CHECKPOINT_BASE_DIR="${OPENPI_CHECKPOINT_BASE_DIR:-/data/jiang/train_checkpoints}"
NUM_WORKERS="${OPENPI_NUM_WORKERS:-16}"
PYTHON="${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if ((BATCH_SIZE % NUM_GPUS != 0)); then
    echo "OPENPI_BATCH_SIZE=${BATCH_SIZE} must be divisible by the number of GPUs (${NUM_GPUS})" >&2
    exit 1
fi

export OPENPI_AR3_REPO_ID="${OPENPI_AR3_REPO_ID:-/data/jiang/ar3_eef_forceVisuoTactaile_front_wrist}"
# ':'-separated datasets are read as one (src/openpi/training/config.py) with statistics named after all of them.
IFS=: read -r -a DATASET_ENTRIES <<<"${OPENPI_AR3_REPO_ID}"
DATASET_NAMES=()
DATASET_INFOS=()
for dataset in "${DATASET_ENTRIES[@]}"; do
    if [[ -n "${dataset}" ]]; then
        DATASET_NAMES+=("$(basename "${dataset}")")
        DATASET_INFOS+=("${dataset}/meta/info.json")
    fi
done
if ((${#DATASET_NAMES[@]} == 0)); then
    echo "OPENPI_AR3_REPO_ID=${OPENPI_AR3_REPO_ID} names no dataset" >&2
    exit 1
fi
export OPENPI_AR3_ASSET_ID="${OPENPI_AR3_ASSET_ID:-$(IFS=+; echo "${DATASET_NAMES[*]}")}"
export OPENPI_AR3_WEIGHT_PATH="${OPENPI_AR3_WEIGHT_PATH:-/data/jiang/checkpoints/openpi-assets/checkpoints/pi05_base/params}"
# paligemma_tokenizer.model is read from ${OPENPI_DATA_HOME}/big_vision/.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
# Always import this repository's openpi.
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
# JAX preallocates this fraction of every GPU; NCCL allocates outside of it. On the 81.5 GB H100s the rest was
# measured at 1.5-2.4 GB per GPU with up to 4 GPUs and 4.2 GB with 8: 0.95 leaves 4.0 GB, and 8-GPU training aborts in
# its first step with NCCL "Cuda failure 2 'out of memory'".
if ((NUM_GPUS > 4)); then
    DEFAULT_MEM_FRACTION=0.90
else
    DEFAULT_MEM_FRACTION=0.95
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-${DEFAULT_MEM_FRACTION}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# Step / loss lines go through stdout: unbuffered, so they reach the log while training runs.
export PYTHONUNBUFFERED=1
# Single-node training only needs NVLink / shared memory: keep NCCL's bootstrap on loopback and off the IB / RoCE NICs
# (their addresses are periodically re-applied on this host, which breaks the first collective).
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

run() {
    echo "+ $*"
    if [[ "${OPENPI_DRY_RUN:-0}" != "1" ]]; then
        "$@"
    fi
}

for path in "${DATASET_INFOS[@]}" "${OPENPI_AR3_WEIGHT_PATH}" "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model" "${PYTHON}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${path}" >&2
        exit 1
    fi
done

if command -v nvidia-smi >/dev/null; then
    echo "GPU memory in use before training (GPUs ${GPUS}):"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "${GPUS}" || true
fi
echo "JAX preallocates ${XLA_PYTHON_CLIENT_MEM_FRACTION} of each of the ${NUM_GPUS} GPUs"

# Each checkpoint (params + EMA + AdamW state) needs several tens of GB.
mkdir -p "${CHECKPOINT_BASE_DIR}"
FREE_GB="$(df -BG --output=avail "${CHECKPOINT_BASE_DIR}" | tail -1 | tr -dc 0-9)"
echo "Free space for checkpoints in ${CHECKPOINT_BASE_DIR}: ${FREE_GB} GB"
if ((FREE_GB < 100)); then
    echo "WARNING: less than 100 GB free; checkpoint saving may fail. Set OPENPI_CHECKPOINT_BASE_DIR to a larger disk." >&2
fi

# 1) Normalization statistics (state / actions / measured force), computed once per dataset (combination) on the CPU.
NORM_STATS="${REPO_ROOT}/assets/${CONFIG_NAME}/${OPENPI_AR3_ASSET_ID}/norm_stats.json"
if [[ ! -f "${NORM_STATS}" || "${OPENPI_RECOMPUTE_NORM_STATS:-0}" == "1" ]]; then
    echo "Computing ${NORM_STATS}"
    run env JAX_PLATFORMS=cpu "${PYTHON}" scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
else
    echo "Using existing ${NORM_STATS}"
fi

# 2) Training.
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
if [[ -n "${OPENPI_LR_DECAY_STEPS:-}" ]]; then
    TRAIN_ARGS+=(--lr-schedule.decay-steps "${OPENPI_LR_DECAY_STEPS}")
fi
if [[ "${OPENPI_RESUME:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--resume)
elif [[ "${OPENPI_OVERWRITE:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--overwrite)
fi
if [[ "${WANDB_MODE}" == "disabled" ]]; then
    TRAIN_ARGS+=(--no-wandb-enabled)
fi

mkdir -p "${REPO_ROOT}/logs"
LOG_FILE="${REPO_ROOT}/logs/${CONFIG_NAME}_${EXP_NAME}.log"
echo "Logging to ${LOG_FILE}"
if [[ "${OPENPI_DRY_RUN:-0}" == "1" ]]; then
    run "${PYTHON}" scripts/train.py "${TRAIN_ARGS[@]}"
else
    run "${PYTHON}" scripts/train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
fi
