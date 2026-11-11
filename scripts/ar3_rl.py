"""Offline RL data preparation for the AR3 muscleVLTA policies (``openpi.rl``): episode outcomes, vision features,
the force-touch critic and the advantage labels that ``pi05_ar3_muscle_vlta_rl`` trains with.
``scripts/train_ar3_muscle_vlta_rl.sh`` runs the steps of one round; by hand:

    # 1. How did the episodes end? Demonstrations are successes; rollouts name their failures.
    .venv/bin/python scripts/ar3_rl.py label --dataset /data/jiang/ar3_eef_forceVisuoTactaile_plugIn --default success
    .venv/bin/python scripts/ar3_rl.py label --dataset /data/jiang/ar3_rollouts/round1 --default success --failure 3 7-9

    # 2. Vision features of the frames (once per dataset; one GPU).
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/ar3_rl.py features --dataset DATASET [DATASET ...]

    # 3. Cross-fitted critic and advantage labels (<dataset>/rl/advantage_<tag>.npz).
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/ar3_rl.py fit --dataset DATASET [DATASET ...] --tag round1

``fit --streams`` trains the critic on a subset of its inputs (vision, pose, arm, touch), e.g. to measure what touch
adds to predicting the outcome; ``relabel`` relabels with another lookahead or positive fraction without training.
Datasets may also be given as one ':'-separated list, as in OPENPI_AR3_REPO_ID.
"""

import argparse
from collections.abc import Sequence
import json
import logging
import os
import pathlib

# The critic is small: do not reserve the visible GPUs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from openpi.rl import advantage as _advantage
from openpi.rl import critic as _critic
from openpi.rl import frames as _frames
from openpi.rl import outcomes as _outcomes

logger = logging.getLogger("ar3_rl")

DEFAULT_PARAMS = os.environ.get(
    "OPENPI_AR3_WEIGHT_PATH", "/data/jiang/checkpoints/openpi-assets/checkpoints/pi05_base/params"
)
DEFAULT_CRITIC_ROOT = "/data/jiang/train_checkpoints/ar3_rl_critic"
VISION_NAME = "siglip"
# Raw Tac G2 indentation above which a finger is in contact (openpi_client.tactile.CONTACT_THRESHOLD).
RAW_CONTACT_DEPTH = 0.05


def dataset_roots(values: Sequence[str]) -> list[str]:
    return [path for value in values for path in value.split(":") if path]


def label(args: argparse.Namespace) -> None:
    for root in dataset_roots(args.dataset):
        path = pathlib.Path(root) / _outcomes.OUTCOMES_PATH
        current = _outcomes.load(root) if path.is_file() else _outcomes.Outcomes()
        table = _frames.load(root)
        episodes = dict(current.episodes)
        for outcome, specs in ((_outcomes.SUCCESS, args.success), (_outcomes.FAILURE, args.failure)):
            for index in _outcomes.parse_episodes(specs):
                if index not in table.lengths:
                    raise SystemExit(f"{root} has no episode {index}")
                episodes[index] = outcome
        frames = dict(current.completion_frames)
        for spec in args.completion_frame:
            index, _, frame = spec.partition("=")
            frames[int(index)] = int(frame)
        updated = _outcomes.Outcomes(
            default=args.default or current.default,
            episodes=episodes,
            completion=args.completion or current.completion,
            completion_frames=frames,
        )
        failures, completions = [], []
        for index, (start, end) in table.bounds().items():
            try:
                success = updated.success(index)
            except KeyError:
                raise SystemExit(f"{root}: episode {index} has no label; pass --default or list it") from None
            if success:
                frame = updated.completion_frames.get(
                    index, _outcomes.completion_frame(table.gripper_target[start:end], updated.completion)
                )
                completions.append(frame / table.fps)
            else:
                failures.append(index)
        _outcomes.save(root, updated)
        times = (
            f"; completed after {np.median(completions):.1f} s (median), {min(completions):.1f}-{max(completions):.1f} s"
            if completions
            else ""
        )
        print(f"{path}: {len(completions)} successes, {len(failures)} failures {failures}{times}")


class _ImageFrames:
    """The resized camera images uint8 [cameras, size, size, 3] of every frame of a dataset, in dataset order."""

    def __init__(self, root: str, cameras: Sequence[str], size: int):
        from openpi.data import lerobot_v3

        options = lerobot_v3.LeRobotV3Options(video_keys=tuple(cameras), require_valid=False)
        self._dataset = lerobot_v3.LeRobotV3Dataset(root, options=options)
        self._cameras = tuple(cameras)
        self._size = size

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> np.ndarray:
        from openpi_client import image_tools

        item = self._dataset[index]
        return np.stack([image_tools.resize_with_pad(item[key], self._size, self._size) for key in self._cameras])


def features(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    import torch

    from openpi.models import model as _model
    from openpi.models import siglip as _siglip
    from openpi.policies import ar3_lift_policy as _ar3

    cameras = (_ar3.FRONT_IMAGE_KEY, _ar3.WRIST_IMAGE_KEY)
    pending = []
    for root in dataset_roots(args.dataset):
        path = _frames.vision_path(root, args.name)
        if _frames.vision_is_current(root, args.name) and not args.overwrite:
            print(f"{path} is up to date (pass --overwrite to recompute)")
        else:
            pending.append((root, path))
    if not pending:
        return

    # The decoding workers are forked here, before JAX starts its threads (forking afterwards can deadlock).
    loaders = []
    for root, path in pending:
        table = _frames.load(root)
        dataset = _ImageFrames(root, cameras, args.image_size)
        if len(dataset) != len(table):
            raise SystemExit(f"{root}: {len(dataset)} decodable frames for {len(table)} table rows")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=np.stack
        )
        loaders.append((root, path, table, iter(loader)))

    logger.info("Loading the image encoder of %s", args.params)
    params = _model.restore_params(args.params, restore_type=np.ndarray)["PaliGemma"]["img"]
    module = _siglip.Module(
        num_classes=params["head"]["kernel"].shape[-1],
        variant=args.variant,
        pool_type="none",
        scan=True,
        dtype_mm="bfloat16",
    )
    params = jax.tree.map(jnp.asarray, params)

    @jax.jit
    def encode(params, images):
        flat = images.reshape(-1, *images.shape[2:]).astype(jnp.float32) / 255.0 * 2.0 - 1.0
        tokens, _ = module.apply({"params": params}, flat, train=False)
        pooled = tokens.astype(jnp.float32).mean(axis=1)
        return pooled.reshape(*images.shape[:2], -1).astype(jnp.float16)

    for root, path, table, batches in loaders:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        out = None
        done = 0
        for batch in batches:
            count = len(batch)
            # A fixed batch shape, so that the last batch does not compile again.
            images = np.concatenate([batch, np.repeat(batch[-1:], args.batch_size - count, axis=0)])
            encoded = np.asarray(encode(params, jnp.asarray(images)))[:count]
            if out is None:
                out = np.lib.format.open_memmap(
                    temporary, mode="w+", dtype=np.float16, shape=(len(table), *encoded.shape[1:])
                )
            out[done : done + count] = encoded
            done += count
            if (done // args.batch_size) % 20 == 0 or done == len(table):
                logger.info("%s: %d / %d frames", root, done, len(table))
        out.flush()
        del out
        os.replace(temporary, path)
        info = {
            "episodes": {str(index): length for index, length in table.lengths.items()},
            "cameras": list(cameras),
            "params": str(args.params),
            "variant": args.variant,
            "image_size": args.image_size,
            "pooling": "mean of the image tokens",
        }
        path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"wrote {path}")


def load_data(args: argparse.Namespace, *, vision: str | None) -> _critic.CriticData:
    roots = dataset_roots(args.dataset)
    tables = [_frames.load(root, vision=vision) for root in roots]
    return _critic.build_data(
        tables, [_outcomes.load(root) for root in roots], fail_penalty=getattr(args, "fail_penalty", 1.0)
    )


def contact_frames(data: _critic.CriticData) -> np.ndarray:
    """Frames [F] at which a finger is in contact."""
    if "depth" not in data.arrays:
        return np.zeros(len(data), dtype=bool)
    peak = data.arrays["depth"].reshape(*data.arrays["depth"].shape[:2], -1).max(axis=-1)
    return ((peak > RAW_CONTACT_DEPTH) & data.arrays["finger_valid"]).any(axis=1)


def evaluate(data: _critic.CriticData, values: np.ndarray) -> dict:
    """Error of the values against the value targets, per task: over the frames before the task is complete, in
    normalized units and (successful episodes) in seconds, for all frames and split by finger contact."""
    fps = data.tables[0].fps
    contact = contact_frames(data)
    error = np.abs(values - data.targets)
    report = {}
    for index, name in enumerate(data.task_names):
        task = (data.arrays["task"] == index) & ~data.done
        if not task.any():
            continue
        entry = {
            "frames": int(task.sum()),
            "episodes": len(np.unique(data.last[task])),
            "failed_episodes": len(np.unique(data.last[task & ~data.success])),
            "norm_seconds": float(data.norm[task][0] / fps),
            "mae": float(error[task].mean()),
        }
        for label_, rows in (("free", task & ~contact), ("contact", task & contact)):
            if rows.any():
                entry[f"mae_{label_}"] = float(error[rows].mean())
        succeeded = task & data.success
        if succeeded.any():
            entry["mae_seconds"] = float((error[succeeded] * data.norm[succeeded]).mean() / fps)
        if (task & ~data.success).any() and succeeded.any():
            # How well the value separates frames of failed from frames of successful episodes (rank statistic).
            order = np.argsort(np.argsort(values[task]))
            failed = ~data.success[task]
            auc = (order[~failed].sum() - (~failed).sum() * ((~failed).sum() - 1) / 2) / (
                (~failed).sum() * failed.sum()
            )
            entry["outcome_auc"] = float(auc)
        report[name] = entry
    return report


def write_labels(data: _critic.CriticData, values: np.ndarray, args: argparse.Namespace, info: dict) -> dict:
    """Labels the frames and writes the advantage file of every dataset. Returns the label summary."""
    advantage = _advantage.n_step_advantage(values, data.targets, data.costs, data.last, args.n_step)
    forced = data.done | (data.human & ~data.demonstration)
    labels, thresholds = _advantage.label_frames(
        advantage,
        task=data.arrays["task"],
        rollout=~data.human,
        forced=forced,
        demo_fraction=args.demo_fraction,
        rollout_fraction=args.rollout_fraction,
    )
    summary = {
        "n_step": args.n_step,
        "demo_fraction": args.demo_fraction,
        "rollout_fraction": args.rollout_fraction,
        "thresholds": {
            f"{data.task_names[int(key.split('/')[0])]} / {key.split('/')[1]}": value
            for key, value in thresholds.items()
        },
        "positive": _advantage.positive_fractions(
            labels,
            {
                "demonstration frames": data.demonstration & ~forced,
                "policy frames": ~data.human,
                "policy frames of failed episodes": ~data.human & ~data.success,
                "human corrections": data.human & ~data.demonstration,
                "all frames": np.ones(len(data), dtype=bool),
            },
        ),
    }
    for table, offset in zip(data.tables, data.offsets, strict=True):
        rows = slice(offset, offset + len(table))
        path = _advantage.save_labels(
            table.root,
            args.tag,
            lengths=table.lengths,
            arrays={
                "label": labels[rows],
                "advantage": advantage[rows],
                "value": values[rows],
                "target": data.targets[rows],
            },
            info={**info, **summary},
        )
        print(f"wrote {path}")
    return summary


def fit(args: argparse.Namespace) -> None:
    vision = None if args.vision == "none" else args.vision
    streams = tuple(args.streams)
    if vision is None:
        streams = tuple(stream for stream in streams if stream != "vision")
    data = load_data(args, vision=vision if "vision" in streams else None)
    if "depth" not in data.arrays and "touch" in streams:
        logger.warning("No dataset has Tac G2 data: the critic is trained without the touch stream")
        streams = tuple(stream for stream in streams if stream != "touch")
    config = _critic.CriticConfig(streams=streams, num_tasks=len(data.task_names), history=args.history)
    logger.info("%d frames of %d datasets, tasks %s, streams %s", len(data), len(data.tables), data.task_names, streams)
    values, params, stats = _critic.cross_fit(
        config,
        data,
        folds=args.folds,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log=logger.info,
    )
    report = {"streams": list(streams), "folds": args.folds, "steps": args.steps, "critic": evaluate(data, values)}
    out = pathlib.Path(args.out or pathlib.Path(DEFAULT_CRITIC_ROOT) / args.tag)
    _critic.save(out, config, params, stats, data.task_names)
    if not args.no_labels:
        info = {"critic": str(out), "streams": list(streams), "folds": args.folds}
        report["labels"] = write_labels(data, values, args, info)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"critic and report: {out}")


def relabel(args: argparse.Namespace) -> None:
    data = load_data(args, vision=None)
    values, info = [], {}
    for table in data.tables:
        source = _advantage.AdvantageLabels.load(table.root, args.from_tag)
        source.check_episodes(table.lengths)
        values.append(source.arrays["value"])
        info = {key: source.info[key] for key in ("critic", "streams", "folds") if key in source.info}
    print(json.dumps(write_labels(data, np.concatenate(values), args, info), indent=2))


def _add_label_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", required=True, help="Name of the labels: <dataset>/rl/advantage_<tag>.npz")
    # One action chunk (10 frames) advances the value by less than the critic's error on a few hundred episodes, so
    # its advantage is mostly noise; over 5 s two critics agree on most labels (README "Offline RL").
    parser.add_argument("--n-step", type=int, default=50, help="Lookahead of the advantage in frames")
    parser.add_argument(
        "--demo-fraction", type=float, default=0.5, help="Share of demonstration frames labelled positive"
    )
    parser.add_argument("--rollout-fraction", type=float, default=0.4, help="Share of policy frames labelled positive")
    parser.add_argument("--fail-penalty", type=float, default=1.0, help="Value penalty of a failed episode")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, function, description: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=description, description=description)
        sub.add_argument("--dataset", nargs="+", required=True, help="LeRobot v3.0 dataset directories")
        sub.set_defaults(function=function)
        return sub

    sub = command("label", label, "Record how the episodes ended (<dataset>/rl/outcomes.json).")
    sub.add_argument("--default", choices=(_outcomes.SUCCESS, _outcomes.FAILURE), help="Outcome of unlisted episodes")
    sub.add_argument("--success", nargs="*", default=[], help='Successful episodes, e.g. "0-4" 7')
    sub.add_argument("--failure", nargs="*", default=[], help='Failed episodes, e.g. 3 "8-10"')
    sub.add_argument("--completion", choices=_outcomes.COMPLETION_MODES, help="Completion frame of successes")
    sub.add_argument("--completion-frame", nargs="*", default=[], metavar="EPISODE=FRAME", help="Explicit frames")

    sub = command("features", features, "Cache the vision features of every frame (<dataset>/rl/vision_<name>.npy).")
    sub.add_argument("--params", default=DEFAULT_PARAMS, help="Checkpoint params with the image encoder")
    sub.add_argument("--variant", default="So400m/14")
    sub.add_argument("--name", default=VISION_NAME)
    sub.add_argument("--image-size", type=int, default=224)
    sub.add_argument("--batch-size", type=int, default=128)
    sub.add_argument("--workers", type=int, default=16)
    sub.add_argument("--overwrite", action="store_true")

    sub = command("fit", fit, "Cross-fit the critic and label the frames with their advantage.")
    _add_label_arguments(sub)
    sub.add_argument("--out", help=f"Critic directory (default: {DEFAULT_CRITIC_ROOT}/<tag>)")
    sub.add_argument("--streams", nargs="+", default=list(_critic.STREAMS), choices=_critic.STREAMS)
    sub.add_argument("--vision", default=VISION_NAME, help='Name of the cached vision features, or "none"')
    sub.add_argument("--history", type=int, default=10, help="Sensor rows read by the arm and touch streams")
    sub.add_argument("--folds", type=int, default=4, help="Cross-fitting folds (1: no cross-fitting)")
    sub.add_argument("--steps", type=int, default=4000, help="Training steps per fold")
    sub.add_argument("--batch-size", type=int, default=512)
    sub.add_argument("--learning-rate", type=float, default=3e-4)
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--no-labels", action="store_true", help="Only train and evaluate the critic")

    sub = command("relabel", relabel, "Relabel with the values of existing labels (no critic training).")
    _add_label_arguments(sub)
    sub.add_argument("--from-tag", required=True, help="Labels whose critic values are reused")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", force=True)
    args = parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
