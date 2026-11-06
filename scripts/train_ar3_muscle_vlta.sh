#!/usr/bin/env bash
# Train muscleVLTA (pi05_ar3_muscle_vlta) on the AR3 plug-insertion dataset: the LIFT reactive force expert also reads
# 1 s of force history and the Tac G2 hand stream (finger indentation maps + finger 6D wrenches + gripper opening).
#
# It builds the Tac G2 indentation-map cache of the dataset (skipped when up to date), then runs
# scripts/train_ar3_lift_gpu012.sh, which computes the normalization statistics (with the finger wrenches) and trains.
# All overrides of that launcher apply, e.g. on GPUs 4-7:
#   OPENPI_CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train_ar3_muscle_vlta.sh
#
# Defaults set here: OPENPI_CONFIG_NAME=pi05_ar3_muscle_vlta, OPENPI_EXP_NAME=muscle_vlta_<date>,
# OPENPI_AR3_REPO_ID=/data/jiang/ar3_eef_forceVisuoTactaile_plugIn. OPENPI_TACTILE_CACHE_WORKERS (default 16).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_ar3_muscle_vlta}"
export OPENPI_EXP_NAME="${OPENPI_EXP_NAME:-muscle_vlta_$(date +%Y%m%d_%H%M)}"
export OPENPI_AR3_REPO_ID="${OPENPI_AR3_REPO_ID:-/data/jiang/ar3_eef_forceVisuoTactaile_plugIn}"
PYTHON="${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

CACHE_COMMAND=(
    "${PYTHON}" scripts/build_ar3_tactile_cache.py
    --dataset "${OPENPI_AR3_REPO_ID}"
    --workers "${OPENPI_TACTILE_CACHE_WORKERS:-16}"
)
echo "+ ${CACHE_COMMAND[*]}"
if [[ "${OPENPI_DRY_RUN:-0}" != "1" ]]; then
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${CACHE_COMMAND[@]}"
fi

exec bash "${REPO_ROOT}/scripts/train_ar3_lift_gpu012.sh"
