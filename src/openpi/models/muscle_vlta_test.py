import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gru as _gru
from openpi.models import hand_encoder as _hand_encoder
from openpi.models import pi0 as _pi0
import openpi.models.pi0_config as _pi0_config
import openpi.shared.nnx_utils as nnx_utils

HISTORY = 2
HORIZON = 4
LATENCY = 1


def _config(**kwargs) -> _pi0_config.Pi0Config:
    defaults = {
        "pi05": True,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "reactive_action_expert_variant": "dummy",
        "cross_attention_config": "dummy",
        "action_horizon": HORIZON,
        "action_dim": 8,
        "reactive_in_use": True,
        "cross_attention_latency": LATENCY,
        "wrench_dim": 13,
        "sensor_history": HISTORY,
        "tactile_in_use": True,
        "tactile_depth_shape": (8, 8),
        "tactile_feature_dim": 16,
    }
    return _pi0_config.Pi0Config(**{**defaults, **kwargs})


@pytest.mark.parametrize("use_force_history", [True, False])
def test_sensor_mask_without_history_reproduces_lift_mask(use_force_history):
    lift = _pi0.make_cross_attn_mask((2, 5, 5, 2), use_force_history=use_force_history)
    mask = _pi0.make_sensor_cross_attn_mask(
        jnp.ones((2, 5), dtype=bool), history_length=0, latency=2, use_force_history=use_force_history
    )
    np.testing.assert_array_equal(mask, lift)


def test_sensor_mask_history_validity_and_streams():
    valid = jnp.array([[False, True, True, True, True, True]])
    mask = _pi0.make_sensor_cross_attn_mask(valid, history_length=2, latency=1, num_streams=2)

    assert mask.shape == (1, 1, 4, 12)
    # Rows are chunk steps -2..3; action i sees steps <= i - 1 of valid rows, in both streams.
    expected = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask[0, 0], np.concatenate([expected, expected], axis=1))


def test_masked_gru_skips_invalid_rows():
    network = _gru.create_gru_network(hidden_size=8, output_size=4, action_horizon=5, input_size=3)
    network.lazy_init(rngs=nnx.Rngs(0), method="init")
    x = jax.random.normal(jax.random.key(1), (2, 5, 3))
    valid = jnp.array([[False, False, True, True, True], [True] * 5])

    np.testing.assert_array_equal(network(x), network(x, valid=jnp.ones((2, 5), dtype=bool)))
    out = network(x, valid=valid)
    changed = network(x.at[0, :2].add(100.0), valid=valid)
    np.testing.assert_array_equal(out, changed)
    # The skipped rows keep the initial state, so the valid rows match a run without them.
    np.testing.assert_array_equal(out[0, 2:], network(x[:1, 2:])[0])


def test_contact_geometry_locates_the_patch_and_ignores_noise():
    depth = jnp.full((3, 8, 16), 0.1).at[0].set(0.0).at[1, 2:4, 12:14].set(1.0)
    geometry = _hand_encoder.contact_geometry(depth, contact_depth=0.2)

    np.testing.assert_array_equal(geometry[0], np.zeros(8))
    # Sub-threshold noise: intensity only, no contact patch.
    np.testing.assert_allclose(geometry[2, :3], [0.1, 0.1, 0.0], rtol=1e-6)
    np.testing.assert_array_equal(geometry[2, 3:], np.zeros(5))
    peak, _, area, cx, cy = geometry[1, :5]
    assert peak == 1.0
    assert area == pytest.approx(4 / 128)
    assert 0.5 < cx < 0.8
    assert -0.5 < cy < 0.0
    np.testing.assert_array_equal(_hand_encoder.contact_labels(depth, 0.2), [False, True, False])


def _hand_inputs(rows: int, batch: int = 2):
    keys = jax.random.split(jax.random.key(3), 4)
    return {
        "depth": jax.random.uniform(keys[0], (batch, rows, 2, 8, 8)),
        "finger_valid": jnp.ones((batch, rows, 2), dtype=bool).at[0, :, 1].set(False),
        "finger_wrench": jax.random.normal(keys[1], (batch, rows, 12)),
        "aperture": jax.random.uniform(keys[2], (batch, rows, 1)),
        "valid": jnp.ones((batch, rows), dtype=bool),
    }


def test_hand_encoder_ignores_missing_fingers_and_is_causal():
    encoder = _hand_encoder.create_hand_encoder(
        width=8, hidden_size=8, features=4, contact_depth=0.2, depth_shape=(8, 8), rows=5
    )
    encoder.lazy_init(rngs=nnx.Rngs(0), method="init")
    inputs = _hand_inputs(rows=5)

    def tokens_of(**kwargs):
        return np.asarray(encoder(**kwargs)[0], dtype=np.float32)

    tokens, logits = tokens_of(**inputs), encoder(**inputs)[1]
    assert tokens.shape == (2, 5, 8)
    assert logits.shape == (2, 5, 2)

    missing_changed = dict(
        inputs,
        depth=inputs["depth"].at[0, :, 1].add(5.0),
        finger_wrench=inputs["finger_wrench"].at[0, :, 6:].add(5.0),
    )
    np.testing.assert_array_equal(tokens_of(**missing_changed)[0], tokens[0])

    future_changed = dict(inputs, depth=inputs["depth"].at[:, 3].add(5.0))
    changed_tokens = tokens_of(**future_changed)
    np.testing.assert_array_equal(changed_tokens[:, :3], tokens[:, :3])
    assert not np.allclose(changed_tokens[:, 3:], tokens[:, 3:])


def _copy_collisions(tree, path=()):
    if not isinstance(tree, dict):
        return []
    found = [(*path, key) for key in tree if key.endswith("_2") and f"{key[:-2]}_1" in tree]
    for key, value in tree.items():
        found += _copy_collisions(value, (*path, key))
    return found


def test_new_params_are_safe_from_reactive_expert_copy():
    # maybe_copy_reactive_expert_params overwrites every "*_2" entry that has a "*_1" sibling; only the experts may.
    params = nnx.state(nnx.eval_shape(_config().create, jax.random.key(0))).to_pure_dict()
    del params["PaliGemma"]["llm"]
    assert params["PaliGemma"]["hand_encoder"]
    assert _copy_collisions(params) == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"reactive_in_use": False}, "require reactive_in_use"),
        ({"pi05": False, "use_state": True}, "require pi05"),
        ({"use_force_history": False}, "drop sensor_history"),
        ({"finger_dropout": 1.0}, "dropout"),
    ],
)
def test_config_rejects_unsupported_muscle_vlta_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _config(**kwargs)


@pytest.fixture(scope="module")
def reactive_model():
    config = _config()
    model = config.create(jax.random.key(0))
    # The cross-attention output projection starts at zero; randomize it so the force memory reaches the actions.
    state = nnx.state(model)
    params = state.to_pure_dict()
    cross = params["PaliGemma"]["llm"]["layers"]["cross_attn"]["cross_attn_vec_einsum"]
    cross["w"] = 0.5 * jax.random.normal(jax.random.key(1), cross["w"].shape, cross["w"].dtype)
    state.replace_by_pure_dict(params)
    nnx.update(model, state)

    rows = HISTORY + HORIZON
    keys = jax.random.split(jax.random.key(2), 5)
    observation = dataclasses.replace(
        config.fake_obs(batch_size=1),
        wrench=jax.random.normal(keys[0], (1, rows, 13)),
        sensor_valid=jnp.array([[False] + [True] * (rows - 1)]),
        tactile_depth=jax.random.uniform(keys[1], (1, rows, 2, 8, 8)),
        tactile_valid=jnp.ones((1, rows, 2), dtype=bool),
        tactile_wrench=jax.random.normal(keys[2], (1, rows, 12)),
        gripper_aperture=jax.random.uniform(keys[3], (1, rows, 1)),
    )
    _, prefix_tokens, prefix_mask, prefix_ar_mask, dt, kv_cache, _ = model.reactive_replan(observation)
    noise = jax.random.normal(keys[4], (1, HORIZON, 8))
    get_actions = nnx_utils.module_jit(model.reactive_get_k_actions)

    def actions(obs):
        return np.asarray(
            get_actions(obs, noise, -noise, prefix_tokens, prefix_mask, prefix_ar_mask, dt, kv_cache)[0],
            dtype=np.float32,
        )

    return config, model, observation, actions


def _perturb(observation, row: int, *, arm: bool = True):
    fields = {
        "tactile_depth": observation.tactile_depth.at[:, row].add(3.0),
        "tactile_wrench": observation.tactile_wrench.at[:, row].add(3.0),
        "gripper_aperture": observation.gripper_aperture.at[:, row].add(1.0),
    }
    if arm:
        fields["wrench"] = observation.wrench.at[:, row].add(3.0)
    return dataclasses.replace(observation, **fields)


def test_reactive_actions_only_see_readings_up_to_the_latency(reactive_model):
    _, _, observation, actions = reactive_model
    base = actions(observation)

    # Row 0 precedes the episode start.
    np.testing.assert_array_equal(actions(_perturb(observation, 0)), base)
    # Row 1 (chunk step -1) is history, visible to every action.
    assert not np.allclose(actions(_perturb(observation, 1))[0], base[0])
    # Chunk step 1 reaches actions i >= 1 + latency, through both the arm and the hand streams.
    for arm in (True, False):
        changed = actions(_perturb(observation, HISTORY + 1, arm=arm))
        np.testing.assert_array_equal(changed[: 1 + LATENCY], base[: 1 + LATENCY])
        assert not np.allclose(changed[1 + LATENCY :], base[1 + LATENCY :])


def test_reactive_context_drops_fingers_only_in_training(reactive_model):
    config, model, _, _ = reactive_model
    keep = model._drop_fingers(jax.random.key(4), jnp.ones((64, HISTORY + HORIZON, 2), dtype=bool))
    assert not keep.all()
    assert (keep == keep[:, :1]).all()
    assert 0.6 < keep.mean() < 0.95
    # Evaluation keeps the readings: the context ignores the rng without train=True.
    observation = config.fake_obs(batch_size=1)
    context = model._reactive_context(observation, jax.random.key(4))
    np.testing.assert_array_equal(context.memory, model._reactive_context(observation).memory)
    assert context.memory.shape == (1, 2 * (HISTORY + HORIZON), 64)
    assert context.cross_attn_mask.shape == (1, 1, HORIZON, 2 * (HISTORY + HORIZON))
    assert context.contact_logits.shape == (1, HISTORY + HORIZON, 2)


def test_muscle_vlta_training_loss(reactive_model):
    config, model, _, _ = reactive_model
    observation, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)
    loss = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))(
        jax.random.key(5), observation, actions, train=True
    )
    assert loss.shape == (2, HORIZON)
    assert np.isfinite(np.asarray(loss)).all()
