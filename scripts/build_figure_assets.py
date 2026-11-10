"""Extracts the recorded data shown in the architecture figures (picture/src/frames): the front and wrist camera
frames at a chunk start, both Tac G2 indentation maps with their change since the previous step, and (printed) the
SVG paths of the TCP force and joint torque over the 1 s history + action chunk of that frame.

The frames of picture/src/musclevlta*_architecture.svg are frame 240 of episode 3 (plug held above the socket):

    .venv/bin/python scripts/build_figure_assets.py /data/jiang/ar3_eef_forceVisuoTactaile_plugIn 3 240 \
        picture/src/frames
"""

import pathlib
import sys

from matplotlib import colormaps
import numpy as np
from PIL import Image
from PIL import ImageDraw

from openpi.data import lerobot_v3

HISTORY, HORIZON = 10, 10
FRONT_CROP = (240, 150, 560, 390)  # left, top, right, bottom of the 640x480 front frame
VIEW_SIZE = (256, 192)


def rounded(image: Image.Image, radius: int) -> Image.Image:
    """The image with rounded corners in its alpha channel, so the figure needs no clip-path (Office ignores it)."""
    scale = 4
    mask = Image.new("L", (image.width * scale, image.height * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, mask.width - 1, mask.height - 1), radius=radius * scale, fill=255)
    image = image.convert("RGB")
    image.putalpha(mask.resize(image.size, Image.LANCZOS))
    return image


def sparkline(values: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> str:
    low, high = values.min(), values.max()
    xs = np.linspace(x0, x1, len(values))
    ys = y1 - (values - low) / max(high - low, 1e-9) * (y1 - y0)
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in zip(xs, ys, strict=True))


def main() -> None:
    root, episode, start, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), pathlib.Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    dataset = lerobot_v3.LeRobotV3Dataset(
        root,
        sequence_keys=("actions", "observation.tcp_wrench", "observation.tau_m"),
        sequence_length=1,
        options=lerobot_v3.LeRobotV3Options(episodes=(episode,), tactile_cache="auto"),
    )
    item = dataset[start]
    front = Image.fromarray(np.asarray(item["observation.images.front"])).crop(FRONT_CROP)
    wrist = Image.fromarray(np.asarray(item["observation.images.wrist"]))
    for name, view in (("front", front), ("wrist", wrist)):
        rounded(view.resize(VIEW_SIZE, Image.LANCZOS), radius=16).save(out / f"view_{name}.png", optimize=True)

    depth = np.asarray(item[lerobot_v3.TACTILE_DEPTH_KEY])[-1]
    previous = np.asarray(dataset[start - 1][lerobot_v3.TACTILE_DEPTH_KEY])[-1]
    change = depth - previous
    print("depth peak per finger", depth.max(axis=(1, 2)).round(3))
    print("largest change per finger", np.abs(change).max(axis=(1, 2)).round(3))
    for finger in range(depth.shape[0]):
        rgb = colormaps["viridis"](np.clip(depth[finger] / 0.3, 0, 1))[..., :3]
        Image.fromarray((rgb * 255).astype(np.uint8)).resize((192, 144), Image.BICUBIC).save(
            out / f"tactile_depth_f{finger}.png"
        )
        rgb = colormaps["RdBu_r"](np.clip(0.5 + change[finger] / (2 * 0.06), 0, 1))[..., :3]
        Image.fromarray((rgb * 255).astype(np.uint8)).resize((192, 144), Image.BICUBIC).save(
            out / f"tactile_change_f{finger}.png"
        )

    rows = range(start - HISTORY, start + HORIZON)
    wrench = np.stack([np.ravel(dataset[i]["observation.tcp_wrench"]) for i in rows])
    torque = np.stack([np.ravel(dataset[i]["observation.tau_m"]) for i in rows])
    axis = int(np.argmax(np.ptp(wrench[:, :3], axis=0)))
    joint = int(np.argmax(np.ptp(torque, axis=0)))
    print(f"TCP force axis {axis}: {wrench[:, axis].round(1).tolist()}")
    print(f"joint torque {joint + 1}: {torque[:, joint].round(2).tolist()}")
    print("force path :", sparkline(wrench[:, axis], 165, 279, 228.5, 245.5))
    print("torque path:", sparkline(torque[:, joint], 165, 279, 228.5, 245.5))


if __name__ == "__main__":
    main()
