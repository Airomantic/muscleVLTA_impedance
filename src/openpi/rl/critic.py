"""Force-touch critic: a distributional value function over the frames of AR3 recordings.

The critic predicts the value target of a frame (``openpi.rl.outcomes``: minus the normalized steps left until the
task is complete, -1 for episodes that fail) from what the robot observes at that frame:

* ``vision``: cached features of the front and wrist images (``scripts/ar3_rl.py features``),
* ``pose``: the TCP pose and the gripper opening,
* ``arm``: the last ``history`` readings of the TCP wrench and the joint torques (a GRU, as in the policy),
* ``touch``: the last ``history`` readings of the Tac G2 fingers (the contact-gated hand encoder of the policy).

``CriticConfig.streams`` selects the inputs, so that critics without touch, force or vision measure what each sensor
adds to predicting the outcome. As in RECAP the value is a categorical distribution over bins of [-1, 0], trained with
the cross-entropy to the two-hot return. With few episodes a critic memorizes the returns of the episodes it was
trained on, which would make every advantage zero, so ``cross_fit`` scores every episode with a critic trained on
the other folds.
"""

from collections.abc import Callable, Sequence
import dataclasses
import json
import pathlib

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import gru as _gru
from openpi.models import hand_encoder as _hand_encoder
from openpi.policies import ar3_lift_policy as _ar3
from openpi.rl import frames as _frames
from openpi.rl import outcomes as _outcomes

STREAMS = ("vision", "pose", "arm", "touch")
# Indentation (map / TACTILE_DEPTH_SCALE) above which a map counts as contact: the policy's tactile_contact_depth.
CONTACT_DEPTH = 0.2


@dataclasses.dataclass(frozen=True)
class CriticConfig:
    streams: tuple[str, ...] = STREAMS
    num_tasks: int = 1
    # Sensor rows read by the arm and touch streams: the frame and the history - 1 frames before it.
    history: int = 10
    bins: int = 201
    width: int = 256
    hidden: int = 512
    tactile_features: int = 64
    dropout: float = 0.1
    # Weight of the contact-gate supervision of the hand encoder (as in the policy).
    gate_loss_weight: float = 0.05

    def __post_init__(self):
        if unknown := sorted(set(self.streams) - set(STREAMS)):
            raise ValueError(f"Unknown critic streams {unknown}: choose from {STREAMS}")


class Critic(nn.Module):
    """Batch of ``gather`` -> (value logits [B, bins], finger contact logits [B, K, fingers] or None)."""

    config: CriticConfig

    @nn.compact
    def __call__(self, batch: dict, *, train: bool = False):
        config = self.config
        features = [nn.Embed(config.num_tasks, config.width, name="task")(batch["task"])]
        gate_logits = None
        if "vision" in config.streams:
            vision = nn.gelu(nn.Dense(config.width, name="vision_in")(batch["vision"]))
            features.append(vision.reshape(vision.shape[0], -1))
        if "pose" in config.streams:
            features.append(nn.gelu(nn.Dense(config.width, name="pose_in")(batch["pose"])))
        if "arm" in config.streams:
            arm = _gru._LinenGRUNetwork(  # noqa: SLF001
                hidden_size=config.width,
                output_size=config.width,
                action_horizon=config.history,
                input_size=batch["arm"].shape[-1],
                name="arm",
            )(batch["arm"], valid=batch["valid"])
            features.append(arm[:, -1].astype(jnp.float32))
        if "touch" in config.streams:
            hand, gate_logits = _hand_encoder.HandEncoder(
                width=config.width,
                hidden_size=config.width,
                features=config.tactile_features,
                contact_depth=CONTACT_DEPTH,
                depth_shape=tuple(batch["depth"].shape[-2:]),
                num_fingers=batch["depth"].shape[2],
                rows=config.history,
                name="hand",
            )(batch["depth"], batch["finger_valid"], batch["finger_wrench"], batch["aperture"], batch["valid"])
            features.append(hand[:, -1].astype(jnp.float32))
        x = jnp.concatenate(features, axis=-1)
        for index in range(2):
            x = nn.gelu(nn.Dense(config.hidden, name=f"trunk{index}")(x))
            x = nn.Dropout(config.dropout, deterministic=not train)(x)
        return nn.Dense(config.bins, name="value")(x), gate_logits


def two_hot(targets: jnp.ndarray, bins: int) -> jnp.ndarray:
    """Values in [-1, 0] -> distributions [..., bins] over the two neighbouring bins with the same mean."""
    position = (jnp.clip(targets, -1.0, 0.0) + 1.0) * (bins - 1)
    low = jnp.floor(position)
    upper = position - low
    low = low.astype(jnp.int32)
    high = jnp.minimum(low + 1, bins - 1)
    return jax.nn.one_hot(low, bins) * (1.0 - upper)[..., None] + jax.nn.one_hot(high, bins) * upper[..., None]


def expected_value(logits: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softmax(logits, axis=-1) @ jnp.linspace(-1.0, 0.0, logits.shape[-1])


@dataclasses.dataclass
class CriticData:
    """The frames of several datasets as one table (global rows), with their value targets."""

    tables: list[_frames.FrameTable]
    task_names: list[str]
    # Per-frame critic inputs, see ``gather``.
    arrays: dict[str, np.ndarray]
    targets: np.ndarray  # [F] value targets in [-1, 0]
    costs: np.ndarray  # [F] normalized reward of the step
    done: np.ndarray  # [F] bool: the task of this (successful) episode is complete
    success: np.ndarray  # [F] bool: frame of a successful episode
    human: np.ndarray  # [F] bool: teleoperated frame
    demonstration: np.ndarray  # [F] bool: frame of an episode that is teleoperated throughout
    episode: np.ndarray  # [F] episode index within its dataset
    last: np.ndarray  # [F] global row of the last frame of the episode
    norm: np.ndarray  # [F] normalizer of the task, in steps

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def offsets(self) -> list[int]:
        """Global row of the first frame of every table."""
        return np.concatenate(([0], np.cumsum([len(table) for table in self.tables])))[:-1].tolist()


def build_data(
    tables: Sequence[_frames.FrameTable], outcomes: Sequence[_outcomes.Outcomes], *, fail_penalty: float = 1.0
) -> CriticData:
    """Value targets of every frame, and the datasets concatenated. Tasks are matched by their prompt."""
    task_names = sorted({name for table in tables for name in table.tasks.values()})
    # Per table: episode index -> (success, completion frame, time to completion or failure).
    episodes: list[dict[int, tuple[bool, int, int]]] = []
    times: dict[str, list[int]] = {name: [] for name in task_names}
    for table, labels in zip(tables, outcomes, strict=True):
        info = {}
        for index, (start, end) in table.bounds().items():
            try:
                success = labels.success(index)
            except KeyError:
                # Rollouts saved without an outcome (Enter at the save prompt of the robot client).
                raise ValueError(
                    f"{table.root}: episode {index} has no outcome label. Label it with `python scripts/ar3_rl.py "
                    f"label --dataset {table.root} --success ... --failure ...` (or --default for all unlisted ones)"
                ) from None
            completion = labels.completion_frames.get(
                index, _outcomes.completion_frame(table.gripper_target[start:end], labels.completion)
            )
            if not 0 <= completion < end - start:
                raise ValueError(f"{table.root}: completion frame {completion} is outside episode {index}")
            info[index] = (success, completion, completion if success else end - start - 1)
            times[table.tasks[int(table.task[start])]].append(info[index][2])
        episodes.append(info)
    norms = {name: _outcomes.task_norm(values) for name, values in times.items()}

    per_table: dict[str, list[np.ndarray]] = {
        key: [] for key in ("targets", "costs", "done", "success", "demonstration", "last", "norm", "task", "first")
    }
    offset = 0
    for table, labels_info in zip(tables, episodes, strict=True):
        for index, (start, end) in table.bounds().items():
            success, completion, _ = labels_info[index]
            length = end - start
            name = table.tasks[int(table.task[start])]
            kwargs = {"success": success, "completion": completion, "norm": norms[name]}
            per_table["targets"].append(_outcomes.value_targets(length, **kwargs, fail_penalty=fail_penalty))
            per_table["costs"].append(_outcomes.step_costs(length, **kwargs).astype(np.float32))
            per_table["done"].append(success & (np.arange(length) >= completion))
            per_table["success"].append(np.full(length, success))
            per_table["demonstration"].append(np.full(length, bool(table.human[start:end].all())))
            per_table["last"].append(np.full(length, offset + end - 1))
            per_table["norm"].append(np.full(length, norms[name], dtype=np.float32))
            per_table["task"].append(np.full(length, task_names.index(name), dtype=np.int32))
            per_table["first"].append(np.full(length, offset + start, dtype=np.int32))
        offset += len(table)
    merged = {key: np.concatenate(values) for key, values in per_table.items()}

    arrays = {
        "task": merged["task"],
        "first": merged["first"],
        "pose": np.concatenate(
            [np.concatenate([table.pose, table.gripper[:, None] / _ar3.GRIPPER_OPEN], axis=1) for table in tables]
        ),
        "arm": np.concatenate([table.arm for table in tables]),
        "finger_wrench": np.concatenate([table.finger_wrench for table in tables]),
        "finger_valid": np.concatenate([table.finger_valid for table in tables]),
        "aperture": np.concatenate([table.gripper / _ar3.GRIPPER_OPEN for table in tables]).astype(np.float32),
    }
    shapes = {table.depth.shape[1:] for table in tables if table.depth is not None}
    if len(shapes) > 1:
        raise ValueError(f"The datasets have tactile maps of different shapes: {sorted(shapes)}")
    if shapes:
        shape = next(iter(shapes))
        arrays["depth"] = np.concatenate(
            [
                table.depth if table.depth is not None else np.zeros((len(table), *shape), dtype=np.float16)
                for table in tables
            ]
        )
    if all(table.vision is not None for table in tables):
        arrays["vision"] = np.concatenate([np.asarray(table.vision) for table in tables])
    return CriticData(
        tables=list(tables),
        task_names=task_names,
        arrays=arrays,
        targets=merged["targets"],
        costs=merged["costs"],
        done=merged["done"],
        success=merged["success"],
        human=np.concatenate([table.human for table in tables]),
        demonstration=merged["demonstration"],
        episode=np.concatenate([table.episode for table in tables]),
        last=merged["last"],
        norm=merged["norm"],
    )


def check_streams(config: CriticConfig, data: CriticData) -> None:
    if "vision" in config.streams and "vision" not in data.arrays:
        raise ValueError("The vision stream needs the vision features of every dataset (scripts/ar3_rl.py features).")
    if "touch" in config.streams and "depth" not in data.arrays:
        raise ValueError("The touch stream needs a dataset with Tac G2 data (scripts/build_ar3_tactile_cache.py).")


def input_stats(data: CriticData) -> dict[str, np.ndarray]:
    """Mean and standard deviation of the unbounded inputs over all frames."""
    stats = {}
    for key in ("pose", "arm", "finger_wrench"):
        values = data.arrays[key].astype(np.float64)
        stats[f"{key}_mean"] = values.mean(axis=0).astype(np.float32)
        stats[f"{key}_std"] = (values.std(axis=0) + 1e-6).astype(np.float32)
    if "vision" in data.arrays:
        # Mean-pooled image tokens are dominated by a component shared by all frames: standardize every feature.
        values = data.arrays["vision"].astype(np.float32)
        stats["vision_mean"] = values.mean(axis=0)
        stats["vision_std"] = values.std(axis=0) + 1e-3
    return stats


def gather(tables: dict, rows: jnp.ndarray, history: int) -> dict:
    """Critic inputs of the frames ``rows`` [B] from the per-frame arrays and statistics ``tables``. Sensor windows
    hold the frame and the ``history - 1`` frames before it; rows before the episode start are flagged invalid."""
    window = rows[:, None] + jnp.arange(-history + 1, 1)
    first = tables["first"][rows][:, None]
    valid = window >= first
    window = jnp.maximum(window, first)

    def normalized(key, index):
        return (tables[key][index] - tables[f"{key}_mean"]) / tables[f"{key}_std"]

    batch = {
        "task": tables["task"][rows],
        "pose": normalized("pose", rows),
        "arm": normalized("arm", window),
        "valid": valid,
    }
    if "vision" in tables:
        batch["vision"] = (tables["vision"][rows].astype(jnp.float32) - tables["vision_mean"]) / tables["vision_std"]
    if "depth" in tables:
        batch["depth"] = tables["depth"][window].astype(jnp.float32) / _ar3.TACTILE_DEPTH_SCALE
        batch["finger_valid"] = tables["finger_valid"][window] & valid[..., None]
        batch["finger_wrench"] = normalized("finger_wrench", window)
        batch["aperture"] = tables["aperture"][window][..., None]
    return batch


def device_tables(config: CriticConfig, data: CriticData, stats: dict[str, np.ndarray]) -> dict:
    """The arrays that the streams of ``config`` read, on the default device."""
    keys = ["task", "first", "pose", "arm"]
    if "vision" in config.streams:
        keys.append("vision")
    if "touch" in config.streams:
        keys += ["depth", "finger_valid", "finger_wrench", "aperture"]
    wanted = {key: data.arrays[key] for key in keys}
    wanted.update({key: value for key, value in stats.items() if key.rsplit("_", 1)[0] in keys})
    return {key: jnp.asarray(value) for key, value in wanted.items()}


def _losses(module: Critic, params, tables: dict, rows, targets, key):
    config = module.config
    batch = gather(tables, rows, config.history)
    logits, gate_logits = module.apply({"params": params}, batch, train=True, rngs={"dropout": key})
    loss = optax.softmax_cross_entropy(logits, two_hot(targets, config.bins)).mean()
    if gate_logits is not None and config.gate_loss_weight > 0:
        contact = batch["depth"].max(axis=(-2, -1)) > CONTACT_DEPTH
        gate = optax.sigmoid_binary_cross_entropy(gate_logits, contact.astype(jnp.float32))
        measured = batch["finger_valid"].astype(jnp.float32)
        loss = loss + config.gate_loss_weight * (gate * measured).sum() / jnp.maximum(measured.sum(), 1.0)
    return loss


def fit(
    config: CriticConfig,
    tables: dict,
    targets: np.ndarray,
    rows: np.ndarray,
    *,
    steps: int,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    seed: int = 0,
    log: Callable[[int, float], None] | None = None,
    log_every: int = 500,
):
    """Trains a critic on the frames ``rows`` and returns its parameters."""
    module = Critic(config)
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    params = module.init(init_key, gather(tables, jnp.asarray(rows[:2]), config.history))["params"]
    schedule = optax.warmup_cosine_decay_schedule(0.0, learning_rate, min(200, steps // 10 + 1), steps)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=weight_decay))
    state = optimizer.init(params)
    device_targets = jnp.asarray(targets)

    @jax.jit
    def step(params, state, tables, batch_rows, key):
        loss, grads = jax.value_and_grad(
            lambda p: _losses(module, p, tables, batch_rows, device_targets[batch_rows], key)
        )(params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    rng = np.random.default_rng(seed)
    running = []
    for index in range(steps):
        key, step_key = jax.random.split(key)
        batch_rows = jnp.asarray(rng.choice(rows, size=batch_size))
        params, state, loss = step(params, state, tables, batch_rows, step_key)
        running.append(loss)
        if log is not None and ((index + 1) % log_every == 0 or index + 1 == steps):
            log(index + 1, float(jnp.mean(jnp.stack(running))))
            running = []
    return params


def predict(config: CriticConfig, params, tables: dict, rows: np.ndarray, *, batch_size: int = 2048) -> np.ndarray:
    """Values [len(rows)] of the frames ``rows``."""
    module = Critic(config)

    @jax.jit
    def values(params, tables, batch_rows):
        logits, _ = module.apply({"params": params}, gather(tables, batch_rows, config.history), train=False)
        return expected_value(logits)

    rows = np.asarray(rows)
    out = np.zeros(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        # A fixed batch shape, so that the last batch does not compile again.
        chunk = rows[start : start + batch_size]
        padded = np.concatenate([chunk, np.repeat(chunk[-1:], batch_size - len(chunk))])
        out[start : start + len(chunk)] = np.asarray(values(params, tables, jnp.asarray(padded)))[: len(chunk)]
    return out


def cross_fit(
    config: CriticConfig,
    data: CriticData,
    *,
    folds: int = 4,
    steps: int = 4000,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
):
    """Trains one critic per fold on the other folds' episodes (episode index modulo ``folds``, per dataset).

    Returns the values of all frames, each from the critic that did not see its episode, the parameters of the
    critics and the input statistics. ``folds=1`` trains one critic on everything and scores its own training data.
    """
    check_streams(config, data)
    stats = input_stats(data)
    tables = device_tables(config, data, stats)
    fold = data.episode % folds
    values = np.zeros(len(data), dtype=np.float32)
    params_per_fold = []
    for index in range(folds):
        held_out = np.flatnonzero(fold == index)
        train_rows = np.flatnonzero(fold != index) if folds > 1 else held_out
        if not len(train_rows) or not len(held_out):
            raise ValueError(f"Fold {index} of {folds} is empty: use fewer folds than episodes per dataset")

        def report(step, loss, index=index):
            if log is not None:
                log(f"fold {index + 1}/{folds}: step {step}/{steps}, loss {loss:.4f}")

        params = fit(
            config,
            tables,
            data.targets,
            train_rows,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed + index,
            log=report,
        )
        values[held_out] = predict(config, params, tables, held_out)
        params_per_fold.append(params)
    return values, params_per_fold, stats


def save(
    directory: str | pathlib.Path, config: CriticConfig, params_per_fold: Sequence, stats: dict, task_names: list[str]
) -> None:
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    info = {
        "config": dataclasses.asdict(config),
        "task_names": task_names,
        "stats": {key: np.asarray(value).tolist() for key, value in stats.items()},
        "folds": len(params_per_fold),
    }
    (directory / "critic.json").write_text(json.dumps(info, indent=2) + "\n")
    for index, params in enumerate(params_per_fold):
        (directory / f"fold{index}.msgpack").write_bytes(flax.serialization.to_bytes(params))


def load(directory: str | pathlib.Path) -> tuple[CriticConfig, list, dict[str, np.ndarray], list[str]]:
    """The critics saved by ``save``: (config, parameters per fold, input statistics, task names)."""
    directory = pathlib.Path(directory)
    info = json.loads((directory / "critic.json").read_text())
    config = CriticConfig(**{**info["config"], "streams": tuple(info["config"]["streams"])})
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in info["stats"].items()}
    params = [
        flax.serialization.msgpack_restore((directory / f"fold{index}.msgpack").read_bytes())
        for index in range(info["folds"])
    ]
    return config, params, stats, info["task_names"]
