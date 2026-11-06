"""Task, data, and model presets used by training and inference.

Start with _CONFIGS. Public task datasets are flexiv/book_insertion_v3_100,
flexiv/hanoi_v1_100, and flexiv/towelv3_100; the suffix counts demonstrations.
For another task, change repo_id, weight_loader, and AssetsConfig together.
For another observation/action space, also adapt LeRobotSingleiPhoneFlexivDataConfig
and src/openpi/policies/single_iphone_flexiv_policy.py rather than only renaming
the dataset. Zarr conversion presets live in preprocess_data/configs/ and are
consumed by preprocess_data/iphone_zarr_to_lerobot.py.

HF_LEROBOT_HOME is the dataset parent, with <repo_id>/meta/info.json and
<repo_id>/data/ below it. AssetsConfig loads <assets_dir>/<asset_id>/norm_stats.json;
weight_loader points to a checkpoint's params/ directory. The checkpoint root
environment variables below are read when Python imports this module.

To inspect available local data without training, use scripts/revalidate_lift.py
with --tasks book, or --tasks book hanoi towel. Its optional --online-dataset
accepts a complete dataset, not a parent of per-episode exports. This validator
checks paths and schema and lists normalization candidates; it does not restore
weights or verify checkpoint tensor contents.
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.ar3_lift_policy as ar3_lift_policy
from openpi.policies.flexiv_transforms import ACTION_NAMES
import openpi.policies.single_iphone_flexiv_policy as single_iphone_flexiv_policy
import openpi.shared.download as _download
from openpi.shared.episode_schema import DEFAULT_INTERVENTION_VALUE
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter
_DEFAULT_CHECKPOINT_ROOT = pathlib.Path(os.environ.get("OPENPI_CHECKPOINT_ROOT", "./checkpoints")).expanduser()
# Towel initialization uses <checkpoint-dir>/params and normalization uses
# <checkpoint-dir>/assets/<asset-id>/norm_stats.json. Override all three Towel
# variables below to reuse data/assets stored under different names; no files
# are renamed. For online training, additionally point OPENPI_INIT_CHECKPOINT
# (LIFT) or OPENPI_BASE_INIT_CHECKPOINT (residual) to that params directory.
_DEFAULT_TOWEL_CHECKPOINT_DIR = pathlib.Path(
    os.environ.get("OPENPI_TOWEL_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_ROOT / "towelv3"))
).expanduser()
_DEFAULT_TOWEL_REPO_ID = os.environ.get("OPENPI_TOWEL_REPO_ID", "flexiv/towelv3_100")
_DEFAULT_TOWEL_ASSET_ID = os.environ.get("OPENPI_TOWEL_ASSET_ID", "flexiv/towelv3_100")
# AR3 Cartesian impedance + DM V3 gripper presets (pi05_ar3_lift*): a local LeRobot v3.0 dataset directory, the
# normalization asset id (assets/<config>/<asset_id>/norm_stats.json; default: the dataset directory name) and the
# initialization params (default: pi05_base).
_DEFAULT_AR3_REPO_ID = os.environ.get("OPENPI_AR3_REPO_ID", "/data/jiang/ar3_eef_forceVisuoTactaile_front_wrist")
_DEFAULT_AR3_ASSET_ID = os.environ.get("OPENPI_AR3_ASSET_ID") or pathlib.Path(_DEFAULT_AR3_REPO_ID).name
_DEFAULT_AR3_WEIGHT_PATH = os.environ.get(
    "OPENPI_AR3_WEIGHT_PATH", "/data/jiang/checkpoints/openpi-assets/checkpoints/pi05_base/params"
)
# AR3 DAgger post-training presets (pi05_ar3_lift*_dagger, scripts/train_ar3_lift_dagger_gpu012.sh): ':'-separated
# LeRobot v3.0 recordings of scripts/ar3_lift_robot_client.py --dagger, the per-batch share of their human-correction
# chunks (LIFT 1:1 offline:online = 0.5), the offline checkpoint params to continue from and its assets directory
# (<step>/assets, so the offline normalization statistics are reused).
_DEFAULT_AR3_DAGGER_REPO_IDS = tuple(
    path for path in os.environ.get("OPENPI_AR3_DAGGER_REPO_IDS", "").split(":") if path
)
_DEFAULT_AR3_DAGGER_RATIO = float(os.environ.get("OPENPI_AR3_DAGGER_RATIO", "0.5"))
_DEFAULT_AR3_DAGGER_INIT_PARAMS = os.environ.get(
    "OPENPI_AR3_DAGGER_INIT_PARAMS", "<set OPENPI_AR3_DAGGER_INIT_PARAMS to the offline checkpoint <step>/params>"
)
_DEFAULT_AR3_DAGGER_ASSETS_DIR = os.environ.get("OPENPI_AR3_DAGGER_ASSETS_DIR") or None


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    action_names: tuple[str, ...] | None = None
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # DAgger post-training (LIFT stage two, LeRobot v3.0 only): robot rollout recordings with human corrections. Only
    # chunks recorded entirely under human control (control_flag == -1, LIFT --online-intervention-only) are used, and
    # every batch holds exactly round(batch_size * dagger_ratio) of them (LIFT: 1:1 offline:online, i.e. 0.5).
    dagger_repo_ids: tuple[str, ...] = ()
    dagger_ratio: float = 0.5

    # LeRobot v3.0 only: action sequence keys that also hold the ``history_length`` frames before the current one
    # (sensor history of the muscleVLTA reactive memory), and the Tac G2 indentation map cache of the datasets
    # (a directory or "auto", see ``LeRobotV3Options.tactile_cache``) loaded over the same frames.
    history_keys: Sequence[str] = ()
    history_length: int = 0
    tactile_cache: str | None = None
    # Keys that need normalization stats: training fails if the loaded stats lack one (stats computed before the key
    # was added to scripts/compute_norm_stats.py would leave it unnormalized).
    required_norm_stats: Sequence[str] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotSingleiPhoneFlexivDataConfig(DataConfigFactory):
    """Map single-arm LeRobot samples to policy inputs.

    Bundled datasets use 10 Hz RGB wrist images (left_wrist_img), 7D state and
    actions, and task descriptions. The extra_delta_transform path converts
    these poses to the model's relative-action representation; preserve the
    coordinate and gripper conventions when adapting a dataset.

    Force-aware models additionally map left_wrench to wrench. Offline hybrid
    samples omit force; online LIFT samples require real 6D wrench sequences.
    Human-intervention control flags are handled by the online data loader,
    separately from these policy transforms.
    """

    extra_delta_transform: bool = True
    use_wrench: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig, *args, **kwargs) -> DataConfig:
        include_wrench = (
            self.use_wrench
            or getattr(model_config, "reactive_in_use", False)
            or getattr(model_config, "residual_policy_in_use", False)
        )
        # include_wrench = self.use_wrench

        repack_structure = {
            "observation/state": "state",
            "observation/left_wrist_image": "left_wrist_img",
            "actions": "actions",
            "prompt": "task",
        }
        if include_wrench:
            repack_structure["wrench"] = "left_wrench"

        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])
        data_transforms = _transforms.Group(
            inputs=[single_iphone_flexiv_policy.SingleiPhoneFlexivInputs(model_type=model_config.model_type)],
            outputs=[single_iphone_flexiv_policy.SingleiPhoneFlexivOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        if self.extra_delta_transform:
            data_transforms = data_transforms.push(
                inputs=[_transforms.IPhoneZeroStateAndRealRelativeActions(bimanual=False)],
                outputs=[_transforms.IPhoneIdentityActions(bimanual=False)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            action_names=ACTION_NAMES,
            action_sequence_keys=("actions", "left_wrench") if include_wrench else ("actions",),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAR3LiftDataConfig(DataConfigFactory):
    """Map AR3 Cartesian impedance LeRobot v3.0 samples (recorder training_schema "lift") to LIFT policy inputs.

    State is the measured TCP pose (xyz + rotation vector, robot base frame) and gripper opening (0 closed ...
    1000 open); actions are the accepted absolute TCP targets and gripper targets. As in the LIFT presets,
    extra_delta_transform makes the actions relative to the observed TCP pose. The wrist camera is the LIFT
    left wrist image, the front camera fills the base image slot (use_front_camera), and force-aware models
    receive the measured force_keys sequences (concatenated per step) over the action horizon as wrench.
    See src/openpi/policies/ar3_lift_policy.py for the robot-client keys used at inference.
    """

    extra_delta_transform: bool = True
    use_front_camera: bool = True
    # Force signals concatenated per step; their total width must equal Pi0Config.wrench_dim.
    force_keys: tuple[str, ...] = ar3_lift_policy.FORCE_KEYS_WRENCH_TAU_M
    use_wrench: bool = False
    # Tac G2 indentation map cache of the datasets of tactile models (Pi0Config.tactile_in_use): a directory, or
    # "auto" for <dataset>/tactile_cache/depth_48x64 (built by scripts/build_ar3_tactile_cache.py).
    tactile_cache: str = "auto"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig, *args, **kwargs) -> DataConfig:
        include_wrench = (
            self.use_wrench
            or getattr(model_config, "reactive_in_use", False)
            or getattr(model_config, "residual_policy_in_use", False)
        )
        force_keys = tuple(self.force_keys) if include_wrench else ()
        if include_wrench:
            wrench_dim = getattr(model_config, "wrench_dim", 6)
            if ar3_lift_policy.force_dim(force_keys) != wrench_dim:
                raise ValueError(
                    f"force_keys {force_keys} give {ar3_lift_policy.force_dim(force_keys)}D force, "
                    f"but the model expects wrench_dim={wrench_dim}."
                )
        # muscleVLTA: the force (and hand) readings of the sensor_history steps before the chunk start, and the Tac G2
        # hand stream (finger wrenches, indentation maps, gripper opening).
        history_length = getattr(model_config, "sensor_history", 0)
        tactile = getattr(model_config, "tactile_in_use", False)
        if (history_length or tactile) and not force_keys:
            raise ValueError("sensor_history and tactile_in_use need the force sequence (reactive_in_use).")
        sensor_keys = (*force_keys, *(ar3_lift_policy.HAND_KEYS if tactile else ()))

        repack_transform = _transforms.Group(
            inputs=[
                ar3_lift_policy.AR3LiftRepack(
                    force_keys=force_keys,
                    use_front_camera=self.use_front_camera,
                    history_length=history_length,
                    tactile=tactile,
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                ar3_lift_policy.AR3LiftInputs(
                    model_type=model_config.model_type,
                    use_front_camera=self.use_front_camera,
                    action_horizon=model_config.action_horizon,
                    history_length=history_length,
                    tactile=tactile,
                )
            ],
            outputs=[ar3_lift_policy.AR3LiftOutputs()],
        )
        if self.extra_delta_transform:
            data_transforms = data_transforms.push(
                inputs=[_transforms.IPhoneZeroStateAndRealRelativeActions(bimanual=False)],
                outputs=[_transforms.IPhoneIdentityActions(bimanual=False)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            action_names=ar3_lift_policy.ACTION_NAMES,
            action_sequence_keys=("actions", *sensor_keys),
            history_keys=sensor_keys if history_length else (),
            history_length=history_length,
            tactile_cache=self.tactile_cache if tactile else None,
            required_norm_stats=("tactile_wrench",) if tactile else (),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # If true, initialize reactive action expert weights from action expert weights.
    copy_reactive_from_action_expert: bool = False

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # How often (in steps) to plot trajectory.
    plot_interval: int = 5000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If true, will use wrench in online dataset
    online_use_wrench: bool = False

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if (
            getattr(self.model, "residual_policy_in_use", False)
            and isinstance(self.freeze_filter, nnx.Nothing)
            and hasattr(self.model, "get_freeze_filter")
        ):
            object.__setattr__(self, "freeze_filter", self.model.get_freeze_filter())


@dataclasses.dataclass(frozen=True)
class OnlineDaggerTrainConfig:
    """Configuration for Online DAgger training with adaptive sampling."""

    # Base training config name
    config_name: str = tyro.MISSING
    # Experiment name
    exp_name: str | None = None
    # Root directory for generated checkpoints and run metadata.
    checkpoint_base_dir: str | None = None
    # If true, will overwrite the checkpoint directory if it already exists
    overwrite: bool = False
    # If true, will resume training from the last checkpoint
    resume: bool = False

    # Path to a checkpoint to initialize model weights from (e.g., "./checkpoints/<config>/<exp>/<step>/params")
    # If not provided, uses the weight_loader defined in the base config
    init_checkpoint: str | None = None
    # Set only when init_checkpoint contains residual parameters without the frozen base policy.
    init_checkpoint_is_residual: bool = False
    # Residual-policy base checkpoint. The path points to a checkpoint `params` directory.
    base_init_checkpoint: str | None = None
    # Residual-policy checkpoint containing the trainable residual parameters.
    residual_init_checkpoint: str | None = None
    # Override checkpoint save frequency (in training steps). If None, use base config value.
    save_interval: int | None = None
    # Override metric logging frequency. If None, use the base config value.
    log_interval: int | None = None
    # Override the total number of steps. Useful for smoke tests and bounded resume checks.
    num_train_steps: int | None = None
    # Override global batch size. Increase this when GPU memory allows to raise per-device work.
    batch_size: int | None = None
    fsdp_devices: int | None = None
    # Optional directory for residual target sidecar caches. Defaults to a hidden directory next to the offline dataset.
    residual_target_cache_dir: str | None = None
    # Number of base-policy diffusion steps used when generating residual targets.
    residual_target_num_steps: int = 10
    # Batch size used by the one-time/online base-policy rollout cache builder.
    residual_target_batch_size: int | None = None

    # Data cloud endpoint for fetching new episodes
    datacloud_endpoint: str = ""
    # Identifier for the data cloud recordings
    identifier: str = ""
    # Query filter for data cloud API
    query_filter: dict = dataclasses.field(default_factory=dict)
    # Fetch new data every N training steps
    fetch_interval: int = 1
    # Prepare this many batches in a background thread while the current step runs.
    prefetch_batches: int = 2
    # Update adaptive-sampler losses at this interval to avoid a device-to-host sync every step.
    sampler_update_interval: int = 10
    # Each immediate child of the local root is treated as one dataset source.
    local_lerobot_data_root: str = ""

    # Online LeRobot dataset repo id
    online_repo_id: str = ""
    # Robot type for online dataset creation
    robot_type: str = "single_iphone_flexiv"
    # FPS for online dataset
    fps: int = 10
    # Task description for online data
    task_description: str = ""
    # Features configuration (if None, will be inferred from offline dataset)
    features: dict | None = None
    # If true, will use wrench in online dataset
    online_use_wrench: bool = False
    online_intervention_only: bool = False
    intervention_value: float = DEFAULT_INTERVENTION_VALUE
    allow_offline_warm_start: bool = True
    online_require_control_flag_one: bool = False
    # Disable the historical force-memory ablation and expose only the first admissible force token.
    disable_force_history: bool = False
    # zarr configs
    use_absolute_action: bool = True
    action_type: str = "left_arm_6DOF_gripper_width"
    temporal_downsample_ratio: int = 0
    use_dino: bool = False
    episode_clip_head_seconds: float = 0.0
    episode_clip_tail_seconds: float = 0.0
    gripper_width_bias: float = 0.0
    gripper_width_scale: float = 1.0

    # Adaptive sampling parameters (SOP-style)
    window_size: int = 200  # Sliding window for loss estimation
    boost_factor: float = 1.5  # alpha > 1 to prioritize online data
    min_online_ratio: float = 0.2  # Minimum online sampling weight
    max_online_ratio: float = 0.8  # Maximum online sampling weight
    initial_online_weight: float = 0.5  # Initial online weight

    def __post_init__(self):
        if self.online_require_control_flag_one:
            logging.warning("--online-require-control-flag-one is deprecated; use --online-intervention-only.")
            object.__setattr__(self, "online_intervention_only", True)


pi05_residual = pi0_config.Pi0Config(
    pi05=True,
    action_horizon=10,
    discrete_state_input=False,
    reactive_in_use=False,
    residual_policy_in_use=True,
    use_state=False,
    original_head=False,
    cross_attention_latency=3,
)

pi05_residual_debug = dataclasses.replace(
    pi05_residual,
    paligemma_variant="dummy",
    action_expert_variant="dummy",
    cross_attention_config="dummy",
    action_horizon=4,
    residual_width=64,
    residual_mlp_dim=128,
    residual_num_layers=2,
    residual_num_heads=4,
    residual_dropout_rate=0.0,
    residual_wrench_hidden_dim=32,
)

# AR3 LIFT presets: the pi05_iPhoneSingle_*_reactive recipe on the AR3 Cartesian impedance dataset. The reactive
# expert reads the measured TCP wrench and, by default, the 7 measured joint torques tau_m (13D force).
pi05_ar3_lift_reactive = pi0_config.Pi0Config(
    pi05=True,
    action_horizon=10,
    discrete_state_input=False,
    reactive_in_use=True,
    use_state=False,
    original_head=False,
    cross_attention_latency=3,
    wrench_dim=ar3_lift_policy.force_dim(ar3_lift_policy.FORCE_KEYS_WRENCH_TAU_M),
)

# muscleVLTA: the LIFT preset whose reactive expert also reads 1 s of force history and the Tac G2 hand stream
# (finger indentation maps, finger wrenches, gripper opening), with the training-only contact and forward-model
# supervision of the sensor encoders. Needs the tactile cache (scripts/build_ar3_tactile_cache.py).
pi05_ar3_muscle_vlta = dataclasses.replace(pi05_ar3_lift_reactive, sensor_history=10, tactile_in_use=True)


def _ar3_lift_train_config(
    name: str, model: pi0_config.Pi0Config, force_keys: tuple[str, ...] = ar3_lift_policy.FORCE_KEYS_WRENCH_TAU_M
) -> TrainConfig:
    """Batch 96 = 32 per GPU with FSDP over 3 GPUs; override --batch-size / --fsdp-devices for other setups."""
    return TrainConfig(
        name=name,
        model=model,
        data=LeRobotAR3LiftDataConfig(
            repo_id=_DEFAULT_AR3_REPO_ID,
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            force_keys=force_keys,
            assets=AssetsConfig(asset_id=_DEFAULT_AR3_ASSET_ID),
        ),
        batch_size=96,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(_DEFAULT_AR3_WEIGHT_PATH),
        copy_reactive_from_action_expert=model.reactive_in_use,
        num_train_steps=30_000,
        keep_period=5000,
        num_workers=16,
        fsdp_devices=3,
    )


def _ar3_lift_dagger_train_config(
    name: str, model: pi0_config.Pi0Config, force_keys: tuple[str, ...] = ar3_lift_policy.FORCE_KEYS_WRENCH_TAU_M
) -> TrainConfig:
    """LIFT stage two on the AR3: continue an offline checkpoint on 1:1 offline + DAgger human-correction batches.

    Like LIFT's online trainer, it starts from the complete offline params (its trained reactive expert is kept, not
    copied from the action expert again), reuses the offline normalization statistics and keeps the constant 5e-5
    learning rate (after a short warmup for the fresh optimizer state).
    """
    config = _ar3_lift_train_config(name, model, force_keys)
    return dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            base_config=DataConfig(
                prompt_from_task=True,
                dagger_repo_ids=_DEFAULT_AR3_DAGGER_REPO_IDS,
                dagger_ratio=_DEFAULT_AR3_DAGGER_RATIO,
            ),
            assets=AssetsConfig(assets_dir=_DEFAULT_AR3_DAGGER_ASSETS_DIR, asset_id=_DEFAULT_AR3_ASSET_ID),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_DEFAULT_AR3_DAGGER_INIT_PARAMS),
        copy_reactive_from_action_expert=False,
        num_train_steps=10_000,
        keep_period=2_000,
    )


# Use `get_config` if you need to get a config by name in your
_CONFIGS = [
    TrainConfig(
        name="pi05_iPhoneSingle_book_insertion_v3_100",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            reactive_in_use=False,
            use_state=False,
            original_head=False,
            cross_attention_latency=3,
        ),
        data=LeRobotSingleiPhoneFlexivDataConfig(
            repo_id="flexiv/book_insertion_v3_100",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            assets=AssetsConfig(
                assets_dir=str(_DEFAULT_CHECKPOINT_ROOT / "30000/assets"),
                asset_id="flexiv/book_insertion_v3_100",
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_DEFAULT_CHECKPOINT_ROOT / "30000/params")),
        num_train_steps=30_000,
        keep_period=500,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_iPhoneSingle_book_insertion_v3_100_reactive",  # change config name here
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            reactive_in_use=True,
            use_state=False,
            original_head=False,
            cross_attention_latency=3,
        ),  # change chunk size here
        data=LeRobotSingleiPhoneFlexivDataConfig(
            repo_id="flexiv/book_insertion_v3_100",  # change dataset name here
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            assets=AssetsConfig(
                assets_dir=str(_DEFAULT_CHECKPOINT_ROOT / "30000/assets"),
                asset_id="flexiv/book_insertion_v3_100",
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_DEFAULT_CHECKPOINT_ROOT / "30000/params")),
        copy_reactive_from_action_expert=True,
        num_train_steps=30_000,
        keep_period=500,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_iPhoneSingle_hanoi_v1_100_reactive",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            reactive_in_use=True,
            use_state=False,
            original_head=False,
            cross_attention_latency=3,
        ),
        data=LeRobotSingleiPhoneFlexivDataConfig(
            repo_id="flexiv/hanoi_v1_100",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        copy_reactive_from_action_expert=True,
        num_train_steps=30_000,
        keep_period=500,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_iPhoneSingle_towelv3_100_reactive",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            reactive_in_use=True,
            use_state=False,
            original_head=False,
            cross_attention_latency=3,
        ),  # change chunk size here
        data=LeRobotSingleiPhoneFlexivDataConfig(
            repo_id=_DEFAULT_TOWEL_REPO_ID,
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            assets=AssetsConfig(
                assets_dir=str(_DEFAULT_TOWEL_CHECKPOINT_DIR / "assets"),
                asset_id=_DEFAULT_TOWEL_ASSET_ID,
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_DEFAULT_TOWEL_CHECKPOINT_DIR / "params")),
        copy_reactive_from_action_expert=True,
        num_train_steps=30_000,
        keep_period=1000,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_iPhoneSingle_hanoi_v1_100_residual",
        model=pi05_residual,
        data=LeRobotSingleiPhoneFlexivDataConfig(
            repo_id="flexiv/hanoi_v1_100",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            assets=AssetsConfig(
                assets_dir=str(_DEFAULT_CHECKPOINT_ROOT / "25000/assets"),
                asset_id="flexiv/hanoi_v1_100",
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_DEFAULT_CHECKPOINT_ROOT / "25000/params")),
        copy_reactive_from_action_expert=False,
        freeze_filter=pi05_residual.get_freeze_filter(),
        num_train_steps=30_000,
        keep_period=500,
        # Residual training freezes the large base model. Replicate it across
        # data-parallel devices to avoid FSDP all-gathers for every train step.
        fsdp_devices=1,
        online_use_wrench=True,
    ),
    TrainConfig(
        name="pi05_iPhoneSingle_towelv3_100_residual",
        model=pi05_residual,
        data=LeRobotSingleiPhoneFlexivDataConfig(
            repo_id=_DEFAULT_TOWEL_REPO_ID,
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            assets=AssetsConfig(
                assets_dir=str(_DEFAULT_TOWEL_CHECKPOINT_DIR / "assets"),
                asset_id=_DEFAULT_TOWEL_ASSET_ID,
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_DEFAULT_TOWEL_CHECKPOINT_DIR / "params")),
        copy_reactive_from_action_expert=False,
        freeze_filter=pi05_residual.get_freeze_filter(),
        num_train_steps=30_000,
        keep_period=500,
        # Residual training freezes the large base model. Replicate it across
        # data-parallel devices to avoid FSDP all-gathers for every train step.
        fsdp_devices=1,
        online_use_wrench=True,
    ),
    TrainConfig(
        name="pi05_iPhoneSingle_book_insertion_v3_100_residual",
        model=pi05_residual,
        data=LeRobotSingleiPhoneFlexivDataConfig(
            repo_id="flexiv/book_insertion_v3_100",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            assets=AssetsConfig(
                assets_dir=str(_DEFAULT_CHECKPOINT_ROOT / "30000/assets"),
                asset_id="flexiv/book_insertion_v3_100",
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_DEFAULT_CHECKPOINT_ROOT / "30000/params")),
        copy_reactive_from_action_expert=False,
        freeze_filter=pi05_residual.get_freeze_filter(),
        num_train_steps=30_000,
        keep_period=500,
        # Residual training freezes the large base model. Replicate it across
        # data-parallel devices to avoid FSDP all-gathers for every train step.
        fsdp_devices=1,
        online_use_wrench=True,
    ),
    _ar3_lift_train_config("pi05_ar3_lift_reactive", pi05_ar3_lift_reactive),
    _ar3_lift_train_config(
        "pi05_ar3_lift_reactive_wrench6",
        dataclasses.replace(pi05_ar3_lift_reactive, wrench_dim=6),
        force_keys=ar3_lift_policy.FORCE_KEYS_WRENCH,
    ),
    # Vision-only pi05 (LIFT's offline stage); its params can initialize the reactive presets via
    # OPENPI_AR3_WEIGHT_PATH, like the Book preset initializes from an offline checkpoint.
    _ar3_lift_train_config("pi05_ar3_lift", dataclasses.replace(pi05_ar3_lift_reactive, reactive_in_use=False)),
    # scripts/train_ar3_muscle_vlta.sh (plug insertion dataset by default).
    _ar3_lift_train_config("pi05_ar3_muscle_vlta", pi05_ar3_muscle_vlta),
    # DAgger post-training of the presets above (scripts/train_ar3_lift_dagger_gpu012.sh). Serve the resulting
    # checkpoints with the matching offline preset name (same model; normalization statistics in the checkpoint).
    _ar3_lift_dagger_train_config("pi05_ar3_lift_reactive_dagger", pi05_ar3_lift_reactive),
    _ar3_lift_dagger_train_config(
        "pi05_ar3_lift_reactive_wrench6_dagger",
        dataclasses.replace(pi05_ar3_lift_reactive, wrench_dim=6),
        force_keys=ar3_lift_policy.FORCE_KEYS_WRENCH,
    ),
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_iPhoneSingle_book_insertion_v3_100_residual_debug",
        model=pi05_residual_debug,
        data=LeRobotSingleiPhoneFlexivDataConfig(
            repo_id="flexiv/book_insertion_v3_100",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            assets=AssetsConfig(
                assets_dir="assets/pi05_iPhoneSingle_book_insertion_v3_100_reactive",
                asset_id="flexiv/book_insertion_v3_100",
            ),
        ),
        batch_size=1,
        num_workers=0,
        lr_schedule=_optimizer.RsqrtDecaySchedule(warmup_steps=1, peak_lr=5e-5, timescale=10),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.NoOpWeightLoader(),
        copy_reactive_from_action_expert=False,
        overwrite=True,
        save_interval=1,
        log_interval=1,
        keep_period=1,
        num_train_steps=2,
        fsdp_devices=1,
        wandb_enabled=False,
        online_use_wrench=True,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
