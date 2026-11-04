"""Replays recorded AR3 frames through a running AR3 LIFT policy server, exactly as the robot client queries it.

Use it to validate a checkpoint before running it on the robot (start the server with scripts/serve_ar3_lift.sh):

    .venv/bin/python scripts/ar3_lift_replay_check.py --port 8000 --episode 0 --num-chunks 5 --k 1

For every chunk starting at frame t, the chunk-start observation (front + wrist images, TCP pose + gripper state, the
force reading of frame t) is sent with ``reactive_replan``; then, as on the robot, step i of the chunk sends the force
readings of frames t..t+i with ``reactive_step`` (LIFT Algorithm 1). The predicted relative actions are converted to
absolute targets and compared with the recorded command targets. With ``--check-causality`` the step-wise actions are
compared with one call that sees the whole recorded force sequence: reactive action r only attends to force rows
<= r - cross_attention_latency, so both must agree (up to numerical noise) for k <= latency + 1.
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
    logging.info("Server metadata: %s", metadata)
    if args.mode == "reactive" and not reactive:
        raise ValueError("The served model is not force-reactive; use --mode infer.")
    if args.mode == "reactive" and args.k > int(metadata.get("cross_attention_latency", 0)) + 1:
        logging.warning("k=%d exceeds latency + 1: the last actions of each call use padded force rows.", args.k)

    dataset = lerobot_v3.LeRobotV3Dataset(
        pathlib.Path(args.dataset),
        sequence_keys=(ar3.ACTIONS_KEY, *force_keys),
        sequence_length=horizon,
        options=lerobot_v3.LeRobotV3Options(
            video_keys=(ar3.WRIST_IMAGE_KEY, ar3.FRONT_IMAGE_KEY), require_valid=False, episodes=(args.episode,)
        ),
    )

    all_errors = []
    for chunk in range(args.num_chunks):
        frame = args.start_frame + chunk * horizon
        if frame >= len(dataset):
            break
        item = dataset[frame]
        state = np.concatenate([np.asarray(item[key], np.float32).reshape(-1) for key in ar3.STATE_KEYS])
        force = ar3.concat_force([item[key] for key in force_keys]) if force_keys else None
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
        if force is not None:
            obs["wrench"] = force[:1]

        timings = []
        if args.mode == "reactive":
            out = client.reactive_replan(obs, k=args.k)
            parts, timings = [out["actions"]], [out["policy_timing"]["infer_ms"]]
            while out["next_step"] < horizon:
                out = client.reactive_step({"wrench": force[: out["next_step"] + 1]}, k=args.k)
                parts.append(out["actions"])
                timings.append(out["policy_timing"]["infer_ms"])
            relative = np.concatenate(parts)
        else:
            out = client.infer(obs)
            relative, timings = out["actions"], [out["policy_timing"]["infer_ms"]]

        causality = ""
        if args.mode == "reactive" and args.check_causality:
            full = client.reactive_step({"wrench": force}, k=horizon, current_step=0)["actions"]
            causality = f" | step-wise vs full-force max |diff| {np.max(np.abs(full - relative)):.2e}"

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
            causality,
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
