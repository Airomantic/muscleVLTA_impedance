import os
import subprocess
import sys
import textwrap

from flax import nnx
import jax
import optax

from openpi.models import pi0_config
from openpi.training import checkpoints
from openpi.training import utils as training_utils


def test_residual_checkpoint_filter_keeps_only_residual_params():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        cross_attention_config="dummy",
        action_horizon=4,
        use_state=False,
        residual_policy_in_use=True,
        residual_width=32,
        residual_mlp_dim=64,
        residual_num_layers=2,
        residual_num_heads=4,
        residual_dropout_rate=0.0,
        residual_wrench_hidden_dim=16,
    )
    model = nnx.eval_shape(config.create, jax.random.key(0))
    params = nnx.state(model)
    residual_filter = nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))
    tx = optax.identity()
    state = training_utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(residual_filter)),
        ema_decay=0.99,
        ema_params=params,
    )

    filtered = checkpoints._filter_checkpoint_params(state, residual_filter)
    params_paths = ["/".join(str(part) for part in path) for path in filtered.params.flat_state()]
    ema_paths = ["/".join(str(part) for part in path) for path in filtered.ema_params.flat_state()]

    assert params_paths
    assert all("residual_policy" in path for path in params_paths)
    assert any("residual_policy/core" in path for path in params_paths)
    assert any("residual_policy/wrench_gru" in path for path in params_paths)
    assert all("image_encoder" not in path for path in params_paths)
    assert ema_paths == params_paths


_RESHARD_SCRIPT = textwrap.dedent(
    """
    import sys
    import types

    from flax import nnx
    import jax
    import numpy as np
    import optax

    from openpi.training import checkpoints
    from openpi.training import utils as training_utils

    P = jax.sharding.PartitionSpec


    def make_state(devices, spec):
        mesh = jax.sharding.Mesh(np.array(devices).reshape(1, -1), ("batch", "fsdp"))
        sharding = jax.sharding.NamedSharding(mesh, spec)
        replicated = jax.sharding.NamedSharding(mesh, P())
        model = nnx.Linear(8, 16, rngs=nnx.Rngs(0))
        params = jax.tree.map(lambda x: jax.device_put(x, sharding if x.ndim == 2 else replicated), nnx.state(model))
        tx = optax.identity()
        state = training_utils.TrainState(
            step=jax.device_put(np.int32(0), replicated),
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=(),
            ema_decay=None,
        )
        return state, sharding

    loader = types.SimpleNamespace(data_config=lambda: types.SimpleNamespace(norm_stats=None, asset_id=None))
    saved, _ = make_state(jax.devices()[:2], P("fsdp", None))
    manager, _ = checkpoints.initialize_checkpoint_dir(sys.argv[1], keep_period=None, overwrite=False, resume=False)
    checkpoints.save_state(manager, saved, loader, 1)
    manager.wait_until_finished()

    target, sharding = make_state(jax.devices()[:4], P(None, "fsdp"))
    target = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), target)
    manager, resuming = checkpoints.initialize_checkpoint_dir(sys.argv[1], keep_period=None, overwrite=False, resume=True)
    assert resuming
    restored = checkpoints.restore_state(manager, target, loader)
    kernel = restored.params["kernel"].value
    assert kernel.sharding == sharding, kernel.sharding
    np.testing.assert_array_equal(np.asarray(kernel), np.asarray(saved.params["kernel"].value))
    """
)


def test_restore_state_reshards_to_a_different_fsdp_mesh(tmp_path):
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    result = subprocess.run(
        [sys.executable, "-c", _RESHARD_SCRIPT, str(tmp_path / "run")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
