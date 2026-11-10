"""Measures which sensor streams a trained AR3 reactive policy uses (LIFT and muscleVLTA), on recorded frames.

The policy samples every action chunk twice with the same noise: from the recorded readings and from counterfactual
ones, in which one stream is removed or taken from another sample. It reports, separately for chunks that start in
free space and with a finger in contact,

* the change of the chunk against the recorded readings, in standard deviations of the recorded actions, and
* the mean squared error against the recorded actions (normalized actions).

A stream the policy ignores changes nothing; a stream it relies on raises the error once it is replaced. "arm level
swapped" keeps the force changes of the window but gives them the level of another sample (pose-dependent gravity and
payload load, sensor bias); "arm changes removed" keeps the level and freezes the readings at the chunk start.

    OPENPI_AR3_REPO_ID=/data/jiang/ar3_eef_forceVisuoTactaile_plugIn:/data/jiang/ar3_eef_forceVisuoTactaile_plugIn_35 \
    .venv/bin/python scripts/ar3_sensor_attribution.py --config-name pi05_ar3_muscle_vlta_v2 \
        --checkpoint /data/jiang/train_checkpoints/pi05_ar3_muscle_vlta_v2/<run>/<step> --holdout-every 8

--holdout-every N reads only every N-th episode: the episodes a run trained with OPENPI_AR3_HOLDOUT_EVERY=N never saw.
"""

import argparse
from collections.abc import Sequence
import dataclasses
import json
import logging
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import hand_encoder
from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

RECORDED = "recorded readings"
ACTION_GROUPS = {"xyz": slice(0, 3), "rotation": slice(3, 6), "gripper": slice(6, 7)}
ACTION_DIM = 7


def counterfactuals(observation: _model.Observation, history: int) -> dict[str, _model.Observation]:
    """The observation with one sensor stream removed, or replaced by that of the next sample of the batch."""

    def other(value):
        return jnp.roll(value, 1, axis=0)

    wrench = observation.wrench
    start = wrench[:, history : history + 1]
    variants = {
        "arm swapped": dataclasses.replace(observation, wrench=other(wrench)),
        "arm level swapped": dataclasses.replace(observation, wrench=wrench - start + other(start)),
        "arm changes removed": dataclasses.replace(observation, wrench=jnp.broadcast_to(start, wrench.shape)),
    }
    if observation.tactile_depth is not None:
        variants["hand removed"] = dataclasses.replace(
            observation, tactile_valid=jnp.zeros_like(observation.tactile_valid)
        )
        variants["hand swapped"] = dataclasses.replace(
            observation,
            tactile_depth=other(observation.tactile_depth),
            tactile_valid=other(observation.tactile_valid),
            tactile_wrench=other(observation.tactile_wrench),
            gripper_aperture=other(observation.gripper_aperture),
        )
    if history and observation.sensor_valid is not None:
        variants["history removed"] = dataclasses.replace(
            observation, sensor_valid=observation.sensor_valid.at[:, :history].set(False)
        )
    variants["all sensors removed"] = dataclasses.replace(
        observation, wrench_mask=jnp.zeros(wrench.shape[0], dtype=jnp.bool_)
    )
    return variants


def summarize(sampled: dict[str, np.ndarray], recorded: np.ndarray, contact: np.ndarray) -> dict[str, dict]:
    """Per variant and chunk type: the chunk change against ``sampled[RECORDED]`` per action group, in standard
    deviations of the recorded actions, and the squared error against the recorded actions."""
    scale = recorded.reshape(-1, recorded.shape[-1]).std(axis=0) + 1e-8
    summary = {}
    for name, actions in sampled.items():
        change = np.abs(actions - sampled[RECORDED]) / scale
        error = np.square(actions - recorded).mean(axis=(1, 2))
        summary[name] = {
            phase: {
                **{group: float(change[chunks][..., dims].mean()) for group, dims in ACTION_GROUPS.items()},
                "error": float(error[chunks].mean()),
                "chunks": int(chunks.sum()),
            }
            for phase, chunks in (("free", ~contact), ("contact", contact))
            if chunks.any()
        }
    return summary


def format_summary(summary: dict[str, dict]) -> str:
    phases = [phase for phase in ("free", "contact") if phase in summary[RECORDED]]
    columns = [*ACTION_GROUPS, "error"]
    lines = [
        " " * 22
        + "".join(f"| {phase + ' (' + str(summary[RECORDED][phase]['chunks']) + ' chunks)':<43}" for phase in phases),
        f"{'readings':<22}" + "".join("| " + "".join(f"{column:>10}" for column in columns) + "   " for _ in phases),
    ]
    for name, row in summary.items():
        cells = ("".join(f"{row[phase][column]:10.4f}" for column in columns) for phase in phases)
        lines.append(f"{name:<22}" + "".join(f"| {cell}   " for cell in cells))
    return "\n".join(lines)


def _train_config(args: argparse.Namespace) -> _config.TrainConfig:
    config = _config.get_config(args.config_name)
    assets_dir = args.checkpoint / "assets"
    asset_ids = [path.name for path in sorted(assets_dir.iterdir()) if (path / "norm_stats.json").exists()]
    asset_id = args.asset_id or (asset_ids[0] if len(asset_ids) == 1 else None)
    if asset_id is None:
        raise SystemExit(f"{assets_dir} holds the normalization statistics {asset_ids}: choose one with --asset-id")
    base = config.data.base_config or _config.DataConfig()
    data = dataclasses.replace(config.data, assets=_config.AssetsConfig(assets_dir=str(assets_dir), asset_id=asset_id))
    if args.dataset:
        paths = tuple(path for path in args.dataset.split(":") if path)
        data = dataclasses.replace(data, repo_id=paths[0])
        base = dataclasses.replace(base, extra_repo_ids=paths[1:])
    base = dataclasses.replace(base, holdout_every=args.holdout_every, holdout=args.holdout_every > 0)
    return dataclasses.replace(
        config,
        data=dataclasses.replace(data, base_config=base),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="pi05_ar3_muscle_vlta", help="Training config of the checkpoint.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Step directory (params/, assets/).")
    parser.add_argument("--dataset", default="", help="':'-separated datasets (default: OPENPI_AR3_REPO_ID).")
    parser.add_argument("--asset-id", default=None, help="Normalization statistics, if the checkpoint has several.")
    parser.add_argument("--holdout-every", type=int, default=0, help="Read only every N-th episode (0: all episodes).")
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=pathlib.Path, default=None, help="Also write the summary to this file.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)
    config = _train_config(args)
    if not getattr(config.model, "reactive_in_use", False):
        raise SystemExit(f"{args.config_name} has no force input: there is no sensor stream to attribute.")
    loader = _data_loader.create_data_loader(config, shuffle=True, num_batches=args.num_batches)
    model = config.model.load(_model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16))
    sample = nnx_utils.module_jit(model.sample_actions)
    history = config.model.sensor_history

    sampled: dict[str, list[np.ndarray]] = {}
    recorded, contact = [], []
    rng = jax.random.key(args.seed)
    for index, (observation, actions) in enumerate(loader):
        rng, batch_rng = jax.random.split(rng)
        for name, variant in {RECORDED: observation, **counterfactuals(observation, history)}.items():
            chunk = np.asarray(sample(batch_rng, variant)[..., :ACTION_DIM], dtype=np.float32)
            sampled.setdefault(name, []).append(chunk)
        recorded.append(np.asarray(actions[..., :ACTION_DIM], dtype=np.float32))
        touching = np.zeros(actions.shape[0], dtype=bool)
        if observation.tactile_depth is not None:
            labels = hand_encoder.contact_labels(
                observation.tactile_depth[:, history], config.model.tactile_contact_depth
            )
            touching = np.asarray((labels & observation.tactile_valid[:, history]).any(axis=-1))
        contact.append(touching)
        logging.info("batch %d/%d", index + 1, args.num_batches)

    summary = summarize(
        {name: np.concatenate(chunks) for name, chunks in sampled.items()},
        np.concatenate(recorded),
        np.concatenate(contact),
    )
    print(format_summary(summary))
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
