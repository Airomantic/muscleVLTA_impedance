#!/usr/bin/env bash
# One offline RL round for a trained muscleVLTA policy on H100 GPUs 4-7: advantage-conditioned fine-tuning with the
# force-touch critic (src/openpi/rl, README "Offline RL"). The model and its losses stay those of the policy. Action
# chunks with a negative advantage are trained with "Advantage: negative" appended to the prompt, the others with the
# prompt as it is, which is also how the result is served.
#
# Usage (from the repository root):
#   OPENPI_AR3_REPO_ID=<demonstrations>:<rollouts>[:...] \
#       bash scripts/train_ar3_muscle_vlta_rl.sh <policy checkpoint step dir>
# e.g. the first round on the demonstrations alone:
#   OPENPI_AR3_REPO_ID=/data/jiang/ar3_eef_forceVisuoTactaile_plugIn:/data/jiang/ar3_eef_forceVisuoTactaile_plugIn_35 \
#       bash scripts/train_ar3_muscle_vlta_rl.sh \
#       /data/jiang/train_checkpoints/pi05_ar3_muscle_vlta/muscle_vlta_30k_plugin_30s35s/141000
# Next round: record rollouts of the new policy (scripts/run_ar3_lift_robot.sh --execute --record-dir DIR, with
# --dagger for corrections) and answer y / n at its save prompt (success / failure), then run this script again from
# the new checkpoint with every dataset collected so far.
#
# Every dataset needs its episode outcomes (<dataset>/rl/outcomes.json), e.g. for demonstrations
#   .venv/bin/python scripts/ar3_rl.py label --dataset <demonstrations> --default success
# and to correct or complete the labels of rollouts
#   .venv/bin/python scripts/ar3_rl.py label --dataset <rollouts> --success 0-4 --failure 5 7
#
# Stages (OPENPI_RL_STAGES, default "cache features critic policy"; finished stages are skipped):
#   cache     Tac G2 map cache of every dataset (scripts/build_ar3_tactile_cache.py)
#   features  vision features of every frame, <dataset>/rl/vision_siglip.npy             (first GPU)
#   critic    cross-fitted critic and advantage labels, <dataset>/rl/advantage_<tag>.npz  (first GPU)
#   policy    fine-tuning of the policy from the checkpoint                               (all GPUs)
#
# Overrides (environment variables), otherwise as in scripts/train_ar3_lift_gpu012.sh:
#   OPENPI_CONFIG_NAME       pi05_ar3_muscle_vlta_rl | pi05_ar3_muscle_vlta_v2_rl (default: from the checkpoint path),
#                            or with _recap appended for RECAP's two indicators: "Advantage: positive" /
#                            "Advantage: negative" in training, served with "Advantage: positive"
#   OPENPI_EXP_NAME          experiment name (default: muscle_vlta_rl_<date>)
#   OPENPI_AR3_RL_TAG        name of the advantage labels (default: the experiment name)
#   OPENPI_RL_FIT_ARGS       further arguments of `scripts/ar3_rl.py fit`, e.g. "--n-step 30 --demo-fraction 0.3"
#   OPENPI_RL_REFIT=1        train the critic again although the labels exist
#   OPENPI_AR3_RL_DROPOUT    *_recap presets: share of training samples without an indicator (default 0.3)
#   OPENPI_AR3_ASSET_ID      norm-stats id under <step>/assets (default: the only one there)
#   OPENPI_CUDA_VISIBLE_DEVICES (default 4,5,6,7), OPENPI_BATCH_SIZE (default 128), OPENPI_NUM_WORKERS (default 16)
#   OPENPI_NUM_TRAIN_STEPS (default 10000), OPENPI_SAVE_INTERVAL (default 1000), OPENPI_KEEP_PERIOD (default 2000)
#   OPENPI_RESUME=1 / OPENPI_OVERWRITE=1 (force: also when the experiment holds saved checkpoints)
#   WANDB_MODE (default offline), OPENPI_DRY_RUN=1 prints the commands
#
# Serve the result with its preset name:
#   OPENPI_CONFIG_NAME=<config> bash scripts/serve_ar3_lift.sh <checkpoint root>/<config>/<exp>/<step>
# (OPENPI_AR3_RL_ADVANTAGE=negative serves the same checkpoint conditioned on a negative advantage, to compare.)
set -euo pipefail

run() {
    echo "+ $*"
    if [[ "${OPENPI_DRY_RUN:-0}" != "1" ]]; then
        "$@"
    fi
}

# Bash reads a script while it runs it: after an edit it continues at the old offset of the new text, and a round
# that had finished could start its training again and, with OPENPI_OVERWRITE=1, delete its own checkpoints. The round
# is therefore a function, which is parsed completely before it starts, and nothing is read after it.
main() {
    local repo_root step_dir source_config default_config config_name exp_name tag stages gpus num_gpus first_gpu
    local batch_size checkpoint_base_dir python log_file dataset path labelled mem_fraction exp_dir saved
    local -a fit_args stats dataset_entries datasets train_args

    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "${repo_root}"

    if (($# != 1)); then
        sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//' >&2
        return 2
    fi
    step_dir="$(realpath -m "$1")"

    # <checkpoint root>/<config>/<exp>/<step>: an offline preset gets its RL preset, an RL round continues with its
    # own.
    source_config="$(basename "$(dirname "$(dirname "${step_dir}")")")"
    case "${source_config}" in
        *_rl | *_rl_recap) default_config="${source_config}" ;;
        *) default_config="${source_config}_rl" ;;
    esac
    config_name="${OPENPI_CONFIG_NAME:-${default_config}}"
    exp_name="${OPENPI_EXP_NAME:-muscle_vlta_rl_$(date +%Y%m%d_%H%M)}"
    tag="${OPENPI_AR3_RL_TAG:-${exp_name}}"
    stages=" ${OPENPI_RL_STAGES:-cache features critic policy} "
    gpus="${OPENPI_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
    num_gpus="$(awk -F, '{print NF}' <<<"${gpus}")"
    first_gpu="${gpus%%,*}"
    batch_size="${OPENPI_BATCH_SIZE:-128}"
    checkpoint_base_dir="${OPENPI_CHECKPOINT_BASE_DIR:-/data/jiang/train_checkpoints}"
    python="${OPENPI_PYTHON:-${repo_root}/.venv/bin/python}"
    read -r -a fit_args <<<"${OPENPI_RL_FIT_ARGS:-}"

    case "${config_name}" in
        pi05_ar3_muscle_vlta_rl | pi05_ar3_muscle_vlta_v2_rl) ;;
        pi05_ar3_muscle_vlta_rl_recap | pi05_ar3_muscle_vlta_v2_rl_recap) ;;
        *)
            echo "OPENPI_CONFIG_NAME=${config_name} is not an offline RL preset" \
                "(pi05_ar3_muscle_vlta[_v2]_rl[_recap])" >&2
            return 1
            ;;
    esac
    if ((batch_size % num_gpus != 0)); then
        echo "OPENPI_BATCH_SIZE=${batch_size} must be divisible by the number of GPUs (${num_gpus})" >&2
        return 1
    fi
    # As in scripts/train_ar3_lift_gpu012.sh: OPENPI_OVERWRITE=1 does not delete an experiment that saved checkpoints.
    exp_dir="${checkpoint_base_dir}/${config_name}/${exp_name}"
    if [[ "${stages}" == *" policy "* && "${OPENPI_RESUME:-0}" != "1" && "${OPENPI_OVERWRITE:-0}" == "1" &&
        -d "${exp_dir}" ]]; then
        saved="$(find "${exp_dir}" -mindepth 1 -maxdepth 1 ! -name wandb_id.txt -printf '%f\n' | sort -V |
            paste -sd ' ' -)"
        if [[ -n "${saved}" ]]; then
            echo "${exp_dir} holds: ${saved}" >&2
            echo "OPENPI_OVERWRITE=1 does not delete saved checkpoints: use another OPENPI_EXP_NAME, OPENPI_RESUME=1" \
                "to continue this run, or OPENPI_OVERWRITE=force to delete them." >&2
            return 1
        fi
    fi
    if [[ ! -d "${step_dir}/params" ]]; then
        echo "${step_dir} is not a checkpoint step directory (missing params/)" >&2
        return 1
    fi
    if [[ -z "${OPENPI_AR3_ASSET_ID:-}" ]]; then
        stats=("${step_dir}"/assets/*/norm_stats.json)
        if ((${#stats[@]} != 1)) || [[ ! -f "${stats[0]}" ]]; then
            echo "Expected exactly one ${step_dir}/assets/<asset_id>/norm_stats.json; set OPENPI_AR3_ASSET_ID" >&2
            return 1
        fi
        OPENPI_AR3_ASSET_ID="$(basename "$(dirname "${stats[0]}")")"
    fi
    if [[ -z "${OPENPI_AR3_REPO_ID:-}" ]]; then
        echo "Set OPENPI_AR3_REPO_ID to the ':'-separated datasets of this round (demonstrations and rollouts)" >&2
        return 1
    fi
    IFS=: read -r -a dataset_entries <<<"${OPENPI_AR3_REPO_ID}"
    datasets=()
    for dataset in "${dataset_entries[@]}"; do
        if [[ -z "${dataset}" ]]; then
            continue
        fi
        if [[ ! -f "${dataset}/meta/info.json" ]]; then
            echo "${dataset} is not a LeRobot v3.0 dataset (missing meta/info.json)" >&2
            return 1
        fi
        if [[ ! -f "${dataset}/rl/outcomes.json" ]]; then
            echo "${dataset} has no episode outcomes: ${python} scripts/ar3_rl.py label --dataset ${dataset}" \
                "--default success [--failure EPISODES]" >&2
            return 1
        fi
        datasets+=("${dataset}")
    done

    export OPENPI_AR3_REPO_ID
    export OPENPI_AR3_ASSET_ID
    export OPENPI_AR3_RL_TAG="${tag}"
    export OPENPI_AR3_RL_INIT_PARAMS="${step_dir}/params"
    export OPENPI_AR3_RL_ASSETS_DIR="${step_dir}/assets"
    export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
    export PYTHONPATH="${repo_root}/src:${repo_root}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export PYTHONUNBUFFERED=1
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

    for path in "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model" "${python}"; do
        if [[ ! -e "${path}" ]]; then
            echo "Missing ${path}" >&2
            return 1
        fi
    done

    echo "Policy:   ${step_dir} -> ${config_name} / ${exp_name} (norm stats: ${OPENPI_AR3_ASSET_ID})"
    echo "Datasets: ${datasets[*]}"
    echo "Labels:   <dataset>/rl/advantage_${tag}.npz"

    if [[ "${stages}" == *" cache "* ]]; then
        run "${python}" scripts/build_ar3_tactile_cache.py --dataset "${datasets[@]}" \
            --workers "${OPENPI_TACTILE_CACHE_WORKERS:-16}"
    fi
    if [[ "${stages}" == *" features "* ]]; then
        run env CUDA_VISIBLE_DEVICES="${first_gpu}" "${python}" scripts/ar3_rl.py features --dataset "${datasets[@]}"
    fi
    if [[ "${stages}" == *" critic "* ]]; then
        labelled=1
        for dataset in "${datasets[@]}"; do
            if [[ ! -f "${dataset}/rl/advantage_${tag}.npz" ]]; then
                labelled=0
            fi
        done
        if ((labelled)) && [[ "${OPENPI_RL_REFIT:-0}" != "1" ]]; then
            echo "Using the existing advantage labels '${tag}' (OPENPI_RL_REFIT=1 trains the critic again)"
        else
            run env CUDA_VISIBLE_DEVICES="${first_gpu}" "${python}" scripts/ar3_rl.py fit \
                --dataset "${datasets[@]}" --tag "${tag}" --out "${checkpoint_base_dir}/ar3_rl_critic/${tag}" \
                "${fit_args[@]}"
        fi
    fi
    if [[ "${stages}" != *" policy "* ]]; then
        return 0
    fi

    train_args=(
        "${config_name}"
        --exp-name "${exp_name}"
        --checkpoint-base-dir "${checkpoint_base_dir}"
        --fsdp-devices "${num_gpus}"
        --batch-size "${batch_size}"
        --num-train-steps "${OPENPI_NUM_TRAIN_STEPS:-10000}"
        --save-interval "${OPENPI_SAVE_INTERVAL:-1000}"
        --log-interval "${OPENPI_LOG_INTERVAL:-100}"
        --keep-period "${OPENPI_KEEP_PERIOD:-2000}"
        --num-workers "${OPENPI_NUM_WORKERS:-16}"
    )
    if [[ "${OPENPI_RESUME:-0}" == "1" ]]; then
        train_args+=(--resume)
    elif [[ "${OPENPI_OVERWRITE:-0}" == "1" || "${OPENPI_OVERWRITE:-0}" == "force" ]]; then
        train_args+=(--overwrite)
    fi
    if [[ "${WANDB_MODE}" == "disabled" ]]; then
        train_args+=(--no-wandb-enabled)
    fi

    export CUDA_VISIBLE_DEVICES="${gpus}"
    # As in scripts/train_ar3_lift_gpu012.sh: with more than 4 GPUs NCCL needs more of the memory outside JAX's pool.
    if ((num_gpus > 4)); then
        mem_fraction=0.90
    else
        mem_fraction=0.95
    fi
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-${mem_fraction}}"
    mkdir -p "${repo_root}/logs" "${checkpoint_base_dir}"
    log_file="${repo_root}/logs/${config_name}_${exp_name}.log"
    echo "JAX preallocates ${XLA_PYTHON_CLIENT_MEM_FRACTION} of each of the ${num_gpus} GPUs"
    echo "Logging to ${log_file}"
    if [[ "${OPENPI_DRY_RUN:-0}" == "1" ]]; then
        run "${python}" scripts/train.py "${train_args[@]}"
    else
        run "${python}" scripts/train.py "${train_args[@]}" 2>&1 | tee "${log_file}"
    fi
}

main "$@"; exit
