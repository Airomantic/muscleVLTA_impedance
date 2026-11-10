import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
from openpi.shared.episode_schema import DEFAULT_INTERVENTION_VALUE
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    reactive_in_use: bool = False
    residual_policy_in_use: bool = False
    reactive_action_expert_variant: _gemma.Variant = "gemma_300m"
    cross_attention_config: _gemma.Variant = "gemma_300m"
    original_head: bool = False
    # False restricts the reactive cross-attention mask to force-memory token 0
    # while preserving its latency constraint; it does not remove force input.
    # Use the same value in the inference config as in the training ablation.
    use_force_history: bool = True
    use_state: bool = True
    # Per-step width of the force sequence fed to the wrench GRU (6 = TCP wrench only, as in LIFT).
    wrench_dim: int = 6

    # muscleVLTA (reactive policies only; 0 / False keeps LIFT unchanged). The sensor readings are time-aligned rows:
    # ``sensor_history`` steps before the chunk start followed by the action horizon.
    sensor_history: int = 0
    # Adds the Tac G2 hand stream (finger indentation maps + finger wrenches + gripper opening) to the force memory.
    tactile_in_use: bool = False
    tactile_depth_shape: tuple[int, int] = (48, 64)
    tactile_num_fingers: int = 2
    # Model-scale indentation above which a pixel counts as contact (0.2 = 0.05 raw Tac G2 depth; open-gripper
    # noise peaks stay below 0.04 raw on the plug-insertion data).
    tactile_contact_depth: float = 0.2
    tactile_feature_dim: int = 256
    # Training-time sensor dropout, constant over a sample: the whole hand, or each finger independently.
    hand_dropout: float = 0.1
    finger_dropout: float = 0.1
    # Training-only supervision of the sensor encoders (0 skips a term; its parameters stay): the finger contact gates
    # against the indentation-map labels, and an action-conditioned forward model of the readings (force memory of
    # chunk step p + commands a_{p-1}, a_p -> arm wrench, finger wrenches, contact patches and contact of step p + 1).
    contact_loss_weight: float = 0.05
    forward_loss_weight: float = 0.1
    # Balanced joint force fusion (muscleVLTA v2; False / 0 keeps the fusion above). A trained policy reads little
    # more from its sensors than the level of the arm force (its sustained load: posture, payload, bias), and ignores
    # the hand stream (scripts/ar3_sensor_attribution.py). ``force_reference`` + ``load_token`` decompose the arm
    # force of the sensor window: the oldest valid row becomes one load token, and the wrench GRU reads the rows
    # relative to it, i.e. the interaction force. ``stream_attention``: the reactive cross-attention normalises its
    # weights within each stream (interaction, hand, load) and sums the streams, so none is crowded out of a shared
    # softmax. ``arm_dropout``: training-time dropout of the whole interaction stream of a sample.
    stream_attention: bool = False
    force_reference: bool = False
    load_token: bool = False
    arm_dropout: float = 0.0

    # The residual branch reuses the frozen base-policy SigLIP. These settings
    # control only the trainable correction decoder and wrench encoder.
    residual_image_keys: tuple[str, ...] = ("left_wrist_0_rgb",)
    residual_width: int = 512
    residual_mlp_dim: int = 2048
    residual_num_layers: int = 6
    residual_num_heads: int = 4
    residual_dropout_rate: float = 0.1
    residual_wrench_hidden_dim: int = 256
    residual_action_scale: tuple[float, ...] = (0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 1.0)
    residual_intervention_value: float = DEFAULT_INTERVENTION_VALUE

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    cross_attention_latency: int = 0
    max_token_len: int = None  # type: ignore
    async_action_horizon: int = -1
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.reactive_in_use and self.residual_policy_in_use:
            raise ValueError("reactive_in_use and residual_policy_in_use cannot both be enabled.")
        if self.sensor_history < 0:
            raise ValueError("sensor_history must be non-negative.")
        if not 0.0 <= self.arm_dropout < 1.0:
            raise ValueError("arm_dropout must be in [0, 1).")
        if (self.stream_attention or self.force_reference or self.arm_dropout) and not (
            self.sensor_history or self.tactile_in_use
        ):
            raise ValueError("stream_attention, force_reference and arm_dropout need sensor_history or tactile_in_use.")
        if self.load_token and not self.force_reference:
            raise ValueError("load_token holds the reference of force_reference: enable both.")
        if self.sensor_history or self.tactile_in_use:
            if not self.reactive_in_use:
                raise ValueError("sensor_history and tactile_in_use require reactive_in_use=True.")
            if not self.pi05 and self.use_state:
                raise ValueError("sensor_history and tactile_in_use require pi05=True or use_state=False.")
            if self.sensor_history and not self.use_force_history:
                raise ValueError("use_force_history=False keeps only the chunk-start reading; drop sensor_history.")
            if not (0.0 <= self.hand_dropout < 1.0 and 0.0 <= self.finger_dropout < 1.0):
                raise ValueError("hand_dropout and finger_dropout must be in [0, 1).")
            if self.contact_loss_weight < 0 or self.forward_loss_weight < 0:
                raise ValueError("contact_loss_weight and forward_loss_weight must be non-negative.")
            if self.forward_loss_weight and self.action_horizon < 3:
                raise ValueError("forward_loss_weight > 0 needs action_horizon >= 3 (it predicts chunk steps 2..H-1).")
        if self.residual_policy_in_use:
            if not self.residual_image_keys:
                raise ValueError("residual_image_keys must contain at least one image key.")
            if len(set(self.residual_image_keys)) != len(self.residual_image_keys):
                raise ValueError("residual_image_keys must not contain duplicates.")
            if self.residual_width % self.residual_num_heads != 0:
                raise ValueError("residual_width must be divisible by residual_num_heads.")
            if self.residual_num_layers <= 0:
                raise ValueError("residual_num_layers must be positive.")
            if not 0.0 <= self.residual_dropout_rate < 1.0:
                raise ValueError("residual_dropout_rate must be in [0, 1).")
            from openpi.models import residual_policy as _residual_policy

            _residual_policy.expand_action_scale(self.residual_action_scale, self.action_dim)
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            sensor_rows = self.sensor_history + self.action_horizon
            wrench_spec = (
                jax.ShapeDtypeStruct([batch_size, sensor_rows, self.wrench_dim], jnp.float32)
                if self.reactive_in_use or self.residual_policy_in_use
                else None
            )
            sensor_specs = {}
            if self.sensor_history or self.tactile_in_use:
                sensor_specs["sensor_valid"] = jax.ShapeDtypeStruct([batch_size, sensor_rows], jnp.bool_)
            if self.tactile_in_use:
                fingers = self.tactile_num_fingers
                sensor_specs.update(
                    tactile_depth=jax.ShapeDtypeStruct(
                        [batch_size, sensor_rows, fingers, *self.tactile_depth_shape], jnp.float32
                    ),
                    tactile_valid=jax.ShapeDtypeStruct([batch_size, sensor_rows, fingers], jnp.bool_),
                    tactile_wrench=jax.ShapeDtypeStruct([batch_size, sensor_rows, 6 * fingers], jnp.float32),
                    gripper_aperture=jax.ShapeDtypeStruct([batch_size, sensor_rows, 1], jnp.float32),
                )
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                wrench=wrench_spec,
                **sensor_specs,
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if self.residual_policy_in_use:
            residual_filter = nnx_utils.PathRegex(".*residual_policy.*")
            return nnx.Not(residual_filter)

        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        reactive_action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_2.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            if self.reactive_in_use and "lora" not in self.reactive_action_expert_variant:
                filters.append(
                    nnx.Not(reactive_action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True
        elif self.reactive_in_use and "lora" in self.reactive_action_expert_variant:
            filters.append(
                reactive_action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
