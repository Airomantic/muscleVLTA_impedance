"""Replays recorded AR3 frames through a running AR3 LIFT policy server, exactly as the robot client queries it.

Use it to validate a checkpoint before running it on the robot (start the server with scripts/serve_ar3_lift.sh):

    .venv/bin/python scripts/ar3_lift_replay_check.py --port 8000 --episode 0 --num-chunks 5 --k 1

For every chunk starting at frame t, the chunk-start observation (front + wrist images, TCP pose + gripper state, the
force reading of frame t) is sent with ``reactive_replan``; then, as on the robot, step i of the chunk sends the force
readings of frames t..t+i with ``reactive_step`` (LIFT Algorithm 1). The predicted relative actions are converted to
absolute targets and compared with the recorded command targets. With ``--check-causality`` the step-wise actions are
compared with one call that sees the whole recorded force sequence: reactive action r only attends to force rows
<= r - cross_attention_latency, so both must agree (up to numerical noise) for k <= latency + 1.

muscleVLTA servers (metadata ``sensor_history`` / ``tactile_in_use``) get every per-step reading key, built from the
dataset exactly as for training (``AR3LiftRepack``): the history rows of the frames before the chunk start and the
Tac G2 hand stream of the dataset's tactile cache (scripts/build_ar3_tactile_cache.py), e.g.

    .venv/bin/python scripts/ar3_lift_replay_check.py --dataset /data/jiang/ar3_eef_forceVisuoTactaile_plugIn
"""

import dataclasses
import logging
import pathlib

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation
import tyro

from openpi.data import lerobot_v3
from openpi.policies import ar3_lift_policy as ar3


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    dataset: str = "/data/jiang/ar3_eef_forceVisuoTactaile_front_wrist"
    episode: int = 0
    start_frame: int = 0
    num_chunks: int = 3
    # Actions requested per reactive call (1 = re-plan every step; <= latency + 1 is exact).
    k: int = 1
    # "reactive": replan + step (LIFT Algorithm 1); "infer": one call per chunk with the chunk-start force reading.
    mode: str = "reactive"
    prompt: str | None = None
    check_causality: bool = True
    # Client-side resize (as the robot client does): identical model inputs, less bandwidth and server CPU per call.
    # 0 sends the raw 640x480 frames.
    resize: int = 224


def _force_keys(wrench_dim: int) -> tuple[str, ...]:
    for keys in (ar3.FORCE_KEYS_WRENCH_TAU_M, ar3.FORCE_KEYS_WRENCH):
        if ar3.force_dim(keys) == wrench_dim:
            return keys
    raise ValueError(f"No AR3 force layout with wrench_dim={wrench_dim}")


def _rows(readings: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    """The first ``count`` rows of every per-step reading key (what the robot client has measured so far)."""
    return {key: value[:count] for key, value in readings.items()}


def _errors(predicted: np.ndarray, recorded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position [mm], rotation [deg] and gripper errors between absolute [H, 7] targets."""
    position = 1000 * np.linalg.norm(predicted[:, :3] - recorded[:, :3], axis=-1)
    rotation = np.degrees(
        (Rotation.from_rotvec(recorded[:, 3:6]).inv() * Rotation.from_rotvec(predicted[:, 3:6])).magnitude()
    )
    return position, rotation, np.abs(predicted[:, 6] - recorded[:, 6])


def main(args: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    horizon = int(metadata.get("action_horizon", 10))
    reactive = bool(metadata.get("reactive", False))
    force_keys = _force_keys(int(metadata.get("wrench_dim", 13))) if reactive else ()
    # muscleVLTA: readings of the steps before the chunk, and the Tac G2 hand stream.
    history = int(metadata.get("sensor_history", 0)) if reactive else 0
    tactile = reactive and bool(metadata.get("tactile_in_use", False))
    logging.info("Server metadata: %s", metadata)
    if args.mode == "reactive" and not reactive:
        raise ValueError("The served model is not force-reactive; use --mode infer.")
    if args.mode == "reactive" and args.k > int(metadata.get("cross_attention_latency", 0)) + 1:
        logging.warning("k=%d exceeds latency + 1: the last actions of each call use padded force rows.", args.k)

    # The sensor keys, history window and tactile cache of the training data (config._ar3_lift_train_config).
    sensor_keys = (*force_keys, *(ar3.HAND_KEYS if tactile else ()))
    dataset = lerobot_v3.LeRobotV3Dataset(
        pathlib.Path(args.dataset),
        sequence_keys=(ar3.ACTIONS_KEY, *sensor_keys),
        sequence_length=horizon,
        history_keys=sensor_keys if history else (),
        history_length=history,
        options=lerobot_v3.LeRobotV3Options(
            video_keys=(ar3.WRIST_IMAGE_KEY, ar3.FRONT_IMAGE_KEY),
            require_valid=False,
            episodes=(args.episode,),
            tactile_cache="auto" if tactile else None,
        ),
    )
    repack = ar3.AR3LiftRepack(force_keys=force_keys, history_length=history, tactile=tactile)

    all_errors = []
    for chunk in range(args.num_chunks):
        frame = args.start_frame + chunk * horizon
        if frame >= len(dataset):
            break
        item = dataset[frame]
        inputs = repack(item)
        state = inputs["observation/state"]
        # [history + horizon, ...] per key: history rows, then the readings of the chunk's steps.
        readings = {key: inputs[key] for key in ar3.READING_KEYS if key in inputs}
        images = {key: np.asarray(item[key]) for key in (ar3.WRIST_IMAGE_KEY, ar3.FRONT_IMAGE_KEY)}
        if args.resize:
            images = {
                key: image_tools.resize_with_pad(image, args.resize, args.resize) for key, image in images.items()
            }
        obs = {
            "observation/state": state,
            "observation/left_wrist_image": images[ar3.WRIST_IMAGE_KEY],
            "observation/base_image": images[ar3.FRONT_IMAGE_KEY],
            "prompt": args.prompt or item["task"],
        }
        if readings:
            obs.update(_rows(readings, history + 1))

        timings = []
        if args.mode == "reactive":
            out = client.reactive_replan(obs, k=args.k)
            parts, timings = [out["actions"]], [out["policy_timing"]["infer_ms"]]
            while out["next_step"] < horizon:
                out = client.reactive_step(_rows(readings, history + out["next_step"] + 1), k=args.k)
                parts.append(out["actions"])
                timings.append(out["policy_timing"]["infer_ms"])
            relative = np.concatenate(parts)
        else:
            out = client.infer(obs)
            relative, timings = out["actions"], [out["policy_timing"]["infer_ms"]]

        details = ""
        if args.mode == "reactive" and args.check_causality:
            full = client.reactive_step(readings, k=horizon, current_step=0)["actions"]
            details = f" | step-wise vs full-force max |diff| {np.max(np.abs(full - relative)):.2e}"
        if tactile:
            details += (
                f" | Tac G2 peak {readings['tactile_depth'][history:].max():.2f}, "
                f"fingers valid {100 * readings['tactile_valid'][history:].mean():.0f}%, "
                f"gripper {readings['gripper_aperture'][history]:.0f}"
            )

        predicted = ar3.relative_to_absolute_actions(relative, state)
        position, rotation, gripper = _errors(predicted, np.asarray(item[ar3.ACTIONS_KEY], np.float64))
        all_errors.append((position, rotation, gripper))
        logging.info(
            "chunk %d (frame %d): %d calls, first %.0f ms, rest mean %.0f ms | pos err mean %.1f / max %.1f mm, "
            "rot err mean %.2f deg, gripper err mean %.0f%s",
            chunk,
            frame,
            len(timings),
            timings[0],
            np.mean(timings[1:]) if len(timings) > 1 else float("nan"),
            position.mean(),
            position.max(),
            rotation.mean(),
            gripper.mean(),
            details,
        )

    if all_errors:
        position, rotation, gripper = (np.concatenate(values) for values in zip(*all_errors, strict=True))
        logging.info(
            "Overall: pos err mean %.1f mm, rot err mean %.2f deg, gripper err mean %.0f (0..1000 units)",
            position.mean(),
            rotation.mean(),
            gripper.mean(),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
