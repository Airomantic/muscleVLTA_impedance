import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import pytest

from scripts import ar3_dagger_gripper_teleop as gate
from scripts import ar3_lift_robot_client as client
from scripts import ar3_rl
from scripts import debug_ar3_lift_robot as debug

SCRIPTS_DIR = Path(__file__).resolve().parent


@pytest.fixture
def launcher_env(tmp_path):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OPENPI_", "REMOTE_", "SSH_"))
        and key not in {"CHECKPOINT_DIR", "SOURCE_DIR", "OUTPUT_PATH", "CONFIG_PATH"}
    }
    checkpoint = tmp_path / "checkpoint with spaces" / "params"
    checkpoint.mkdir(parents=True)
    environment.update(
        {
            "OPENPI_PYTHON": sys.executable,
            "OPENPI_DRY_RUN": "1",
            "OPENPI_INIT_CHECKPOINT": str(checkpoint),
            "OPENPI_BASE_INIT_CHECKPOINT": str(checkpoint),
        }
    )
    return environment


def run_launcher(name, environment, *arguments):
    return subprocess.run(
        ["bash", str(SCRIPTS_DIR / name), *arguments],
        env=environment,
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )


def command_args(result):
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Would run:")
    return shlex.split(result.stdout.removeprefix("Would run:"))


def test_converter_requires_explicit_input_and_output(launcher_env):
    result = run_launcher("nedf2_to_lerobot_incremental_flexiv_tdk.sh", launcher_env)
    assert result.returncode != 0
    assert "SOURCE_DIR" in result.stderr


def test_converter_uses_public_towel_preset(launcher_env, tmp_path):
    launcher_env.update({"SOURCE_DIR": str(tmp_path / "raw data"), "OUTPUT_PATH": str(tmp_path / "output")})
    arguments = command_args(run_launcher("nedf2_to_lerobot_incremental_flexiv_tdk.sh", launcher_env))
    assert str(SCRIPTS_DIR.parent / "preprocess_data/configs/towelv3_online.yaml") in arguments
    assert arguments[arguments.index("--source-dir") + 1] == launcher_env["SOURCE_DIR"]
    assert not Path(launcher_env["OUTPUT_PATH"]).exists()


def test_converter_rejects_same_source_and_output(launcher_env, tmp_path):
    launcher_env.update({"SOURCE_DIR": str(tmp_path), "OUTPUT_PATH": str(tmp_path / "nested" / "..")})
    result = run_launcher("nedf2_to_lerobot_incremental_flexiv_tdk.sh", launcher_env)
    assert result.returncode != 0
    assert "different directories" in result.stderr


@pytest.mark.parametrize("name", ["nedf2_to_lerobot_incremental_flexiv_tdk.sh"])
def test_utility_help_needs_no_private_configuration(name, launcher_env):
    result = run_launcher(name, launcher_env, "--help")
    assert result.returncode == 0
    assert "Usage:" in result.stdout


@pytest.mark.parametrize("suffix", ["", "_reactive", "_residual"])
def test_training_does_not_overwrite_by_default(suffix, launcher_env):
    arguments = command_args(run_launcher(f"train_online_dagger_lerobot{suffix}.sh", launcher_env))
    assert "--overwrite" not in arguments
    assert "--resume" not in arguments
    assert arguments[arguments.index("--config-name") + 1] == f"pi05_iPhoneSingle_book_insertion_v3_100{suffix}"


@pytest.mark.parametrize(("ratio", "fraction"), [("0to1", "1.0"), ("1to1", "0.5"), ("1to2", "0.6666666667")])
def test_ratio_wrapper_preserves_sampling_setting(ratio, fraction, launcher_env):
    arguments = command_args(run_launcher(f"train_online_dagger_lerobot_reactive_ratio_{ratio}.sh", launcher_env))
    assert arguments[arguments.index("--min-online-ratio") + 1] == fraction
    assert arguments[arguments.index("--max-online-ratio") + 1] == fraction


def test_force_ablation_still_reaches_python(launcher_env):
    arguments = command_args(run_launcher("train_online_dagger_lerobot_reactive_no_force_history.sh", launcher_env))
    assert "--disable-force-history" in arguments


def test_general_entrypoint_defaults_to_help(launcher_env):
    arguments = command_args(run_launcher("train_online_dagger.sh", launcher_env))
    assert arguments[-1] == "--help"


@pytest.mark.parametrize("suffix", ["", "_reactive", "_residual"])
def test_presets_preserve_gpu_visibility_and_forward_arguments(suffix, launcher_env):
    launcher_env["CUDA_VISIBLE_DEVICES"] = "5"
    arguments = command_args(run_launcher(f"train_online_dagger_lerobot{suffix}.sh", launcher_env, "--batch-size", "7"))
    assert "CUDA_VISIBLE_DEVICES=5" in arguments
    assert arguments[-2:] == ["--batch-size", "7"]
    launcher_env["OPENPI_CUDA_VISIBLE_DEVICES"] = "3"
    arguments = command_args(run_launcher(f"train_online_dagger_lerobot{suffix}.sh", launcher_env))
    assert "CUDA_VISIBLE_DEVICES=3" in arguments


def test_ratio_ablation_is_strict_and_uses_selected_task_name(launcher_env):
    launcher_env["OPENPI_CONFIG_NAME"] = "custom_task"
    arguments = command_args(run_launcher("train_online_dagger_lerobot_reactive_ratio_1to2.sh", launcher_env))
    assert "--no-allow-offline-warm-start" in arguments
    assert arguments[arguments.index("--exp-name") + 1].startswith("custom_task_ratio_1to2_")
    assert arguments[arguments.index("--task-description") + 1] == ""


def test_ar3_dagger_launcher_continues_offline_checkpoint(launcher_env, tmp_path):
    step = tmp_path / "ckpt" / "29999"
    (step / "params").mkdir(parents=True)
    (step / "assets" / "ar3_offline").mkdir(parents=True)
    (step / "assets" / "ar3_offline" / "norm_stats.json").write_text("{}")
    rounds = [tmp_path / "round 1", tmp_path / "round2"]
    for directory in [tmp_path / "offline", *rounds]:
        (directory / "meta").mkdir(parents=True)
        (directory / "meta" / "info.json").write_text("{}")
    (tmp_path / "home" / "big_vision").mkdir(parents=True)
    (tmp_path / "home" / "big_vision" / "paligemma_tokenizer.model").write_text("")
    launcher_env.update({"OPENPI_AR3_REPO_ID": str(tmp_path / "offline"), "OPENPI_DATA_HOME": str(tmp_path / "home")})

    result = run_launcher("train_ar3_lift_dagger_gpu012.sh", launcher_env, str(step), *map(str, rounds))
    assert result.returncode == 0, result.stderr
    assert f"Initialization: {step}/params (norm stats: {step}/assets/ar3_offline)" in result.stdout
    assert f"DAgger data:    {rounds[0]} {rounds[1]} (ratio 0.5 per batch)" in result.stdout
    train = shlex.split(next(line for line in result.stdout.splitlines() if "scripts/train.py" in line)[2:])
    assert train[2] == "pi05_ar3_lift_reactive_dagger"
    assert train[train.index("--num-train-steps") + 1] == "10000"
    assert train[train.index("--fsdp-devices") + 1] == "3"

    result = run_launcher("train_ar3_lift_dagger_gpu012.sh", launcher_env, str(step))
    assert result.returncode == 2
    assert "Usage" in result.stderr
    result = run_launcher("train_ar3_lift_dagger_gpu012.sh", launcher_env, str(step), str(tmp_path / "missing"))
    assert result.returncode == 1
    assert "not a LeRobot v3.0 recording" in result.stderr

    # Several offline datasets (':'-separated) stay in the 1:1 mixture as one.
    (tmp_path / "offline_35" / "meta").mkdir(parents=True)
    (tmp_path / "offline_35" / "meta" / "info.json").write_text("{}")
    launcher_env["OPENPI_AR3_REPO_ID"] += f":{tmp_path / 'offline_35'}"
    result = run_launcher("train_ar3_lift_dagger_gpu012.sh", launcher_env, str(step), str(rounds[0]))
    assert result.returncode == 0, result.stderr
    assert f"Offline data:   {tmp_path / 'offline'}:{tmp_path / 'offline_35'}" in result.stdout


def test_muscle_vlta_launcher_builds_the_tactile_cache_before_training(launcher_env, tmp_path):
    dataset = tmp_path / "plug_in"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}")
    (tmp_path / "home" / "big_vision").mkdir(parents=True)
    (tmp_path / "home" / "big_vision" / "paligemma_tokenizer.model").write_text("")
    launcher_env.update(
        {
            "OPENPI_AR3_REPO_ID": str(dataset),
            "OPENPI_AR3_WEIGHT_PATH": launcher_env["OPENPI_INIT_CHECKPOINT"],
            "OPENPI_DATA_HOME": str(tmp_path / "home"),
            "OPENPI_CHECKPOINT_BASE_DIR": str(tmp_path / "checkpoints"),
            "OPENPI_CUDA_VISIBLE_DEVICES": "4,5,6,7",
        }
    )
    result = run_launcher("train_ar3_muscle_vlta.sh", launcher_env)
    assert result.returncode == 0, result.stderr
    cache, norm_stats, train = (shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ "))
    assert cache[1:] == ["scripts/build_ar3_tactile_cache.py", "--dataset", str(dataset), "--workers", "16"]
    assert norm_stats[-2:] == ["--config-name", "pi05_ar3_muscle_vlta"]
    assert train[2] == "pi05_ar3_muscle_vlta"
    assert train[train.index("--exp-name") + 1].startswith("muscle_vlta_")
    assert train[train.index("--fsdp-devices") + 1] == "4"
    assert not (dataset / "tactile_cache").exists()


def ar3_training_setup(launcher_env, tmp_path):
    """Prepares what the AR3 training launchers check before a dry run; returns their arguments by launcher name."""
    dataset = tmp_path / "plug_in"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}")
    (dataset / "rl").mkdir()
    (dataset / "rl" / "outcomes.json").write_text("{}")
    step = tmp_path / "ckpt" / "pi05_ar3_muscle_vlta" / "run" / "32000"
    (step / "params").mkdir(parents=True)
    (step / "assets" / "plug_in").mkdir(parents=True)
    (step / "assets" / "plug_in" / "norm_stats.json").write_text("{}")
    (tmp_path / "home" / "big_vision").mkdir(parents=True)
    (tmp_path / "home" / "big_vision" / "paligemma_tokenizer.model").write_text("")
    launcher_env.update(
        {
            "OPENPI_AR3_REPO_ID": str(dataset),
            "OPENPI_AR3_WEIGHT_PATH": launcher_env["OPENPI_INIT_CHECKPOINT"],
            "OPENPI_DATA_HOME": str(tmp_path / "home"),
            "OPENPI_CHECKPOINT_BASE_DIR": str(tmp_path / "checkpoints"),
        }
    )
    return {
        "train_ar3_muscle_vlta_rl.sh": [str(step)],
        "train_ar3_lift_dagger_gpu012.sh": [str(step), str(dataset)],
    }


@pytest.mark.parametrize(
    ("launcher", "gpus", "override", "fraction"),
    [
        ("train_ar3_muscle_vlta.sh", "4,5,6,7", None, "0.95"),
        ("train_ar3_muscle_vlta.sh", "0,1,2,3,4,5,6,7", None, "0.90"),
        ("train_ar3_muscle_vlta.sh", "0,1,2,3,4,5,6,7", "0.8", "0.8"),
        ("train_ar3_muscle_vlta_rl.sh", "0,1,2,3,4,5,6,7", None, "0.90"),
        ("train_ar3_lift_dagger_gpu012.sh", "0,1,2,3,4,5,6,7", None, "0.90"),
    ],
)
def test_training_on_many_gpus_leaves_memory_for_nccl(launcher, gpus, override, fraction, launcher_env, tmp_path):
    """With 8 GPUs NCCL needs about 4.2 GB per GPU outside JAX's pool: preallocating 0.95 of the H100s aborted the
    first step ("Cuda failure 2 'out of memory'")."""
    arguments = ar3_training_setup(launcher_env, tmp_path)
    launcher_env.update({"OPENPI_CUDA_VISIBLE_DEVICES": gpus, "OPENPI_BATCH_SIZE": "256"})
    launcher_env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    if override is not None:
        launcher_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = override
    result = run_launcher(launcher, launcher_env, *arguments.get(launcher, []))
    assert result.returncode == 0, result.stderr
    assert f"JAX preallocates {fraction} of each of the {len(gpus.split(','))} GPUs" in result.stdout


@pytest.mark.parametrize(
    ("launcher", "config"),
    [
        ("train_ar3_muscle_vlta.sh", "pi05_ar3_muscle_vlta"),
        ("train_ar3_muscle_vlta_rl.sh", "pi05_ar3_muscle_vlta_rl"),
        ("train_ar3_lift_dagger_gpu012.sh", "pi05_ar3_lift_reactive_dagger"),
    ],
)
def test_overwrite_does_not_delete_saved_checkpoints(launcher, config, launcher_env, tmp_path):
    """A command whose first line lacked its continuation lost OPENPI_CONFIG_NAME, ran with the default preset under the
    same experiment name and, with OPENPI_OVERWRITE=1, wiped the checkpoints of the run that had trained there."""
    arguments = ar3_training_setup(launcher_env, tmp_path).get(launcher, [])
    experiment = tmp_path / "checkpoints" / config / "new_visuotactile"
    experiment.mkdir(parents=True)
    (experiment / "wandb_id.txt").write_text("v1it8tr3")
    launcher_env.update({"OPENPI_EXP_NAME": "new_visuotactile", "OPENPI_OVERWRITE": "1"})

    def train_command():
        result = run_launcher(launcher, launcher_env, *arguments)
        assert result.returncode == 0, result.stderr
        (line,) = [line for line in result.stdout.splitlines() if line.startswith("+ ") and "scripts/train.py" in line]
        return shlex.split(line[2:])

    # A run that stopped before its first checkpoint is simply started again.
    assert "--overwrite" in train_command()

    (experiment / "25000" / "params").mkdir(parents=True)
    (experiment / "24000.tar").write_text("")
    result = run_launcher(launcher, launcher_env, *arguments)
    assert result.returncode == 1
    assert f"{experiment} holds: 24000.tar 25000" in result.stderr
    assert "OPENPI_OVERWRITE=force" in result.stderr
    assert "scripts/train.py" not in result.stdout

    launcher_env["OPENPI_RESUME"] = "1"
    assert "--resume" in train_command()
    del launcher_env["OPENPI_RESUME"]
    launcher_env["OPENPI_OVERWRITE"] = "force"
    assert "--overwrite" in train_command()


def test_muscle_vlta_launcher_trains_several_sessions_as_one(launcher_env, tmp_path):
    sessions = [tmp_path / "plug_in", tmp_path / "plug in_35"]
    for session in sessions:
        (session / "meta").mkdir(parents=True)
        (session / "meta" / "info.json").write_text("{}")
    (tmp_path / "home" / "big_vision").mkdir(parents=True)
    (tmp_path / "home" / "big_vision" / "paligemma_tokenizer.model").write_text("")
    launcher_env.update(
        {
            "OPENPI_AR3_REPO_ID": f"{sessions[0]}::{sessions[1]}:",
            "OPENPI_AR3_WEIGHT_PATH": launcher_env["OPENPI_INIT_CHECKPOINT"],
            "OPENPI_DATA_HOME": str(tmp_path / "home"),
            "OPENPI_CHECKPOINT_BASE_DIR": str(tmp_path / "checkpoints"),
        }
    )
    result = run_launcher("train_ar3_muscle_vlta.sh", launcher_env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0].endswith(f"--dataset {sessions[0]} {sessions[1]} --workers 16")
    assert "assets/pi05_ar3_muscle_vlta/plug_in+plug in_35/norm_stats.json" in result.stdout

    (sessions[1] / "meta" / "info.json").unlink()
    result = run_launcher("train_ar3_muscle_vlta.sh", launcher_env)
    assert result.returncode == 1
    assert f"Missing {sessions[1]}/meta/info.json" in result.stderr


def test_ar3_configs_train_every_listed_dataset_as_one(launcher_env):
    root = SCRIPTS_DIR.parent
    launcher_env.update(
        {
            "OPENPI_AR3_REPO_ID": "/data/plug_in:/data/plug_in_35:",
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": f"{root / 'src'}:{root / 'packages/openpi-client/src'}",
        }
    )
    script = (
        "import json\nfrom openpi.training import config\n"
        "configs = map(config.get_config, ['pi05_ar3_muscle_vlta', 'pi05_ar3_lift_reactive_dagger'])\n"
        "print(json.dumps([[c.data.repo_id, c.data.base_config.extra_repo_ids, c.data.assets.asset_id] for c in configs]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], env=launcher_env, cwd="/tmp", capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    expected = ["/data/plug_in", ["/data/plug_in_35"], "plug_in+plug_in_35"]
    assert json.loads(result.stdout.splitlines()[-1]) == [expected, expected]


def test_rl_presets_continue_a_policy_checkpoint_with_the_advantage_in_the_prompt(launcher_env):
    root = SCRIPTS_DIR.parent
    launcher_env.update(
        {
            "OPENPI_AR3_REPO_ID": "/data/demos:/data/rollouts",
            "OPENPI_AR3_ASSET_ID": "plug_in",
            "OPENPI_AR3_RL_TAG": "round1",
            "OPENPI_AR3_RL_INIT_PARAMS": "/ckpt/32000/params",
            "OPENPI_AR3_RL_ASSETS_DIR": "/ckpt/32000/assets",
            "OPENPI_AR3_RL_ADVANTAGE": "none",
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": f"{root / 'src'}:{root / 'packages/openpi-client/src'}",
        }
    )
    script = (
        "import json\nfrom openpi.training import config\n"
        "def transforms(name):\n"
        "    c = config.get_config(name)\n"
        "    return c, c.data.create(c.assets_dirs, c.model).data_transforms.inputs\n"
        "rl, inputs = transforms('pi05_ar3_muscle_vlta_v2_rl')\n"
        "recap, recap_inputs = transforms('pi05_ar3_muscle_vlta_v2_rl_recap')\n"
        "base, base_inputs = transforms('pi05_ar3_muscle_vlta_v2')\n"
        "print(json.dumps([\n"
        "    [rl.data.repo_id, rl.data.base_config.extra_repo_ids, rl.data.base_config.advantage_tag],\n"
        "    [rl.weight_loader.params_path, rl.data.assets.assets_dir, rl.data.assets.asset_id],\n"
        "    [type(t).__name__ for t in inputs], inputs[1].inference, rl.model == base.model,\n"
        "    rl.copy_reactive_from_action_expert, base.data.base_config.advantage_tag,\n"
        "    [type(t).__name__ for t in base_inputs],\n"
        "    [[c.data.base_config.advantage_dropout, t[1].marking] for c, t in ((rl, inputs), (recap, recap_inputs))],\n"
        "]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], env=launcher_env, cwd="/tmp", capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    data, init, inputs, indicator, same_model, copy_expert, base_tag, base_inputs, markings = json.loads(
        result.stdout.splitlines()[-1]
    )
    assert data == ["/data/demos", ["/data/rollouts"], "round1"]
    assert init == ["/ckpt/32000/params", "/ckpt/32000/assets", "plug_in"]
    # The same model as the offline preset; only the prompt carries the advantage label.
    assert inputs == ["AR3LiftInputs", "AdvantagePrompt", "IPhoneZeroStateAndRealRelativeActions"]
    assert (indicator, same_model, copy_expert) == ("none", True, False)
    assert base_tag is None
    assert base_inputs == ["AR3LiftInputs", "IPhoneZeroStateAndRealRelativeActions"]
    # Labels are only dropped where a positive indicator tells the positive chunks from the unlabelled ones.
    assert markings == [[0.0, "negative"], [0.3, "both"]]


def test_rl_launcher_runs_the_stages_of_a_round_from_a_policy_checkpoint(launcher_env, tmp_path):
    step = tmp_path / "ckpt" / "pi05_ar3_muscle_vlta_v2" / "run" / "32000"
    (step / "params").mkdir(parents=True)
    (step / "assets" / "plug_in").mkdir(parents=True)
    (step / "assets" / "plug_in" / "norm_stats.json").write_text("{}")
    datasets = [tmp_path / "demos", tmp_path / "rollouts"]
    for directory in datasets:
        (directory / "meta").mkdir(parents=True)
        (directory / "meta" / "info.json").write_text("{}")
        (directory / "rl").mkdir()
        (directory / "rl" / "outcomes.json").write_text("{}")
    (tmp_path / "home" / "big_vision").mkdir(parents=True)
    (tmp_path / "home" / "big_vision" / "paligemma_tokenizer.model").write_text("")
    launcher_env.update(
        {
            "OPENPI_AR3_REPO_ID": f"{datasets[0]}:{datasets[1]}",
            "OPENPI_DATA_HOME": str(tmp_path / "home"),
            "OPENPI_EXP_NAME": "round1",
        }
    )

    def stages(result):
        assert result.returncode == 0, result.stderr
        return [line[2:] for line in result.stdout.splitlines() if line.startswith("+ ")]

    roots = f"{datasets[0]} {datasets[1]}"
    cache, features, fit, train = stages(run_launcher("train_ar3_muscle_vlta_rl.sh", launcher_env, str(step)))
    assert cache.endswith(f"scripts/build_ar3_tactile_cache.py --dataset {roots} --workers 16")
    assert features.startswith("env CUDA_VISIBLE_DEVICES=4 ")
    assert features.endswith(f"scripts/ar3_rl.py features --dataset {roots}")
    assert fit.endswith(
        f"scripts/ar3_rl.py fit --dataset {roots} --tag round1 --out /data/jiang/train_checkpoints/ar3_rl_critic/round1"
    )
    train = shlex.split(train)
    assert train[2] == "pi05_ar3_muscle_vlta_v2_rl"
    assert train[train.index("--fsdp-devices") + 1] == "4"
    assert train[train.index("--batch-size") + 1] == "128"

    # Labels of this tag are reused unless the critic is to be trained again.
    for directory in datasets:
        (directory / "rl" / "advantage_round1.npz").write_text("")
    result = run_launcher("train_ar3_muscle_vlta_rl.sh", launcher_env, str(step))
    assert len(stages(result)) == 3
    assert "Using the existing advantage labels 'round1'" in result.stdout
    launcher_env.update({"OPENPI_RL_REFIT": "1", "OPENPI_RL_FIT_ARGS": "--n-step 30", "OPENPI_RL_STAGES": "critic"})
    (fit,) = stages(run_launcher("train_ar3_muscle_vlta_rl.sh", launcher_env, str(step)))
    assert fit.endswith("ar3_rl_critic/round1 --n-step 30")

    # The next round continues an RL checkpoint with the same preset, also that of RECAP's two indicators.
    later = tmp_path / "ckpt" / "pi05_ar3_muscle_vlta_v2_rl" / "round1" / "9999"
    later.parent.mkdir(parents=True)
    step.rename(later)
    launcher_env["OPENPI_RL_STAGES"] = "policy"
    (train,) = stages(run_launcher("train_ar3_muscle_vlta_rl.sh", launcher_env, str(later)))
    assert shlex.split(train)[2] == "pi05_ar3_muscle_vlta_v2_rl"
    recap = tmp_path / "ckpt" / "pi05_ar3_muscle_vlta_v2_rl_recap" / "round1" / "9999"
    recap.parent.mkdir(parents=True)
    later.rename(recap)
    (train,) = stages(run_launcher("train_ar3_muscle_vlta_rl.sh", launcher_env, str(recap)))
    assert shlex.split(train)[2] == "pi05_ar3_muscle_vlta_v2_rl_recap"

    lift = tmp_path / "ckpt" / "pi05_ar3_lift_reactive" / "run" / "29999"
    lift.parent.mkdir(parents=True)
    recap.rename(lift)
    result = run_launcher("train_ar3_muscle_vlta_rl.sh", launcher_env, str(lift))
    assert result.returncode == 1
    assert "pi05_ar3_lift_reactive_rl is not an offline RL preset" in result.stderr

    (datasets[1] / "rl" / "outcomes.json").unlink()
    launcher_env["OPENPI_CONFIG_NAME"] = "pi05_ar3_muscle_vlta_rl"
    result = run_launcher("train_ar3_muscle_vlta_rl.sh", launcher_env, str(lift))
    assert result.returncode == 1
    assert f"{datasets[1]} has no episode outcomes" in result.stderr


def test_rl_launcher_finishes_its_round_once_although_the_script_is_edited_meanwhile(launcher_env, tmp_path):
    """Bash reads a script while running it: an edited launcher once started the finished training again, which
    deleted its checkpoints (OPENPI_OVERWRITE=1)."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    launcher = repo / "scripts" / "train_ar3_muscle_vlta_rl.sh"
    launcher.write_text((SCRIPTS_DIR / "train_ar3_muscle_vlta_rl.sh").read_text())
    step = tmp_path / "ckpt" / "pi05_ar3_muscle_vlta" / "run" / "32000"
    (step / "params").mkdir(parents=True)
    (step / "assets" / "plug_in").mkdir(parents=True)
    (step / "assets" / "plug_in" / "norm_stats.json").write_text("{}")
    dataset = tmp_path / "demos"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}")
    (dataset / "rl").mkdir()
    (dataset / "rl" / "outcomes.json").write_text("{}")
    (tmp_path / "home" / "big_vision").mkdir(parents=True)
    (tmp_path / "home" / "big_vision" / "paligemma_tokenizer.model").write_text("")
    # Stands in for the interpreter: notes every command of the round and takes a moment.
    calls, python = tmp_path / "calls", tmp_path / "python"
    python.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{calls}"\nsleep 1\n')
    python.chmod(0o755)
    launcher_env.update(
        {
            "OPENPI_PYTHON": str(python),
            "OPENPI_DRY_RUN": "0",
            "OPENPI_OVERWRITE": "1",
            "OPENPI_AR3_REPO_ID": str(dataset),
            "OPENPI_DATA_HOME": str(tmp_path / "home"),
            "OPENPI_EXP_NAME": "round1",
            "OPENPI_CHECKPOINT_BASE_DIR": str(tmp_path / "out"),
        }
    )
    process = subprocess.Popen(
        ["bash", str(launcher), str(step)], env=launcher_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        deadline = time.monotonic() + 20
        while not calls.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        # The first stage runs: every later offset of the file moves, as when a comment is added to the header.
        launcher.write_text(launcher.read_text().replace("\n", "\n# edited while the round runs\n", 30))
        output, _ = process.communicate(timeout=60)
    finally:
        process.kill()
    assert process.returncode == 0, output.decode()
    commands = [line.split()[0] for line in calls.read_text().splitlines()]
    assert commands == [
        "scripts/build_ar3_tactile_cache.py",
        "scripts/ar3_rl.py",
        "scripts/ar3_rl.py",
        "scripts/train.py",
    ]


def parse_jsonc(text):
    """VS Code's launch.json is JSONC: drop comments and trailing commas outside strings."""
    pattern = r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/|,(?=\s*[}\]])'
    return json.loads(re.sub(pattern, lambda match: match[1] or "", text, flags=re.DOTALL))


def check_debug_training(configuration, root, inputs):
    """train.py's CLI accepts the arguments in the configuration's environment, which config.py reads on import."""

    def resolve(value):
        return inputs.get(value, value).replace("${workspaceFolder}", str(root))

    environment = {key: value for key, value in os.environ.items() if not key.startswith("OPENPI_")}
    environment.update({name: resolve(value) for name, value in configuration["env"].items()}, JAX_PLATFORMS="cpu")
    script = (
        "import json\nfrom openpi.training import config\n"
        "c = config.cli()\nprint(json.dumps([str(c.checkpoint_dir), c.batch_size, c.fsdp_devices]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *map(resolve, configuration["args"])],
        env=environment,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    checkpoint_dir, batch_size, fsdp_devices = json.loads(result.stdout.splitlines()[-1])
    gpus = len(environment["CUDA_VISIBLE_DEVICES"].split(","))
    assert batch_size % gpus == 0, configuration["name"]
    assert gpus % fsdp_devices == 0, configuration["name"]
    # --overwrite deletes the experiment directory: debugging must never point at the batch runs.
    assert not Path(checkpoint_dir).is_relative_to("/data/jiang/train_checkpoints"), configuration["name"]


def test_vscode_configurations_pass_arguments_the_launcher_and_scripts_accept(tmp_path, monkeypatch):
    root = SCRIPTS_DIR.parent
    text = (root / ".vscode" / "launch.json").read_text()
    launch = parse_jsonc(text)
    # The configurations name the saved home of the adjacent arm workspace.
    workspace = tmp_path / "muscleVLTA_impedance"
    saved_home = tmp_path / "ar3_CartesianImpedanceControl" / "config" / "cartesian_home.json"
    saved_home.parent.mkdir(parents=True)
    saved_home.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame_id": client.BASE_FRAME,
                "source": "measured_tcp_pose",
                "tcp_pose": [0.4, -0.3, 0.8, 0.2, -0.1, 0.3],
            }
        )
    )
    for name in list(os.environ):
        if name.startswith(("INFER_", "OPENPI_")):
            monkeypatch.delenv(name)
    parsers = {"ar3_dagger_gripper_teleop.py": gate.parse_args, "ar3_rl.py": ar3_rl.parse_args}
    inputs = {f"${{input:{item['id']}}}": item["default"] for item in launch.get("inputs", [])}
    assert launch["configurations"]
    for config in launch["configurations"]:
        program = Path(config["program"].replace("${workspaceFolder}", str(root)))
        assert program.is_file(), config["name"]
        if program.name == "train.py":
            check_debug_training(config, root, inputs)
            continue
        for name, value in config.get("env", {}).items():
            monkeypatch.setenv(name, value.replace("${workspaceFolder}", str(workspace)))
        args = [inputs.get(arg, arg).replace("${workspaceFolder}", str(workspace)) for arg in config["args"]]
        if program.name == "debug_ar3_lift_robot.py":
            launcher, client_args = debug.parse_args(args)
            options = client.parse_args(client_args)
            # DAgger reuses the teleoperation of ./infer_oneclick.sh --dagger (auto/ready refuse it).
            assert not options.dagger or launcher.hardware == "skip", config["name"]
        else:
            parsers[program.name](args)
    assert set(re.findall(r"\$\{input:(\w+)\}", text)) == {item["id"] for item in launch.get("inputs", [])}
