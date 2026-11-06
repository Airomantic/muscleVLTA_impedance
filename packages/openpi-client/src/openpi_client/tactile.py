"""DM Tac G2 visuotactile preprocessing shared by training (tactile cache, transforms) and the AR3 robot client.

The two tactile fingers ``sensor_0`` and ``sensor_2`` of the gripper publish a 6D wrench (Fx Fy Fz Mx My Mz in SDK
units, each sensor's own frame) and a 288x384 ``32FC1`` indentation ("depth") map in raw SDK units. The map is ~0
without contact (noise max < 0.035) and 0.08..0.35 where an object presses into the gel; the shape and position of the
contact patch encode how the object sits in the hand. Training and inference reduce every map to ``DEPTH_SHAPE`` by
area pooling (6x6 for the native resolution), which keeps the patch geometry at 1/36 of the size.
"""

from collections.abc import Sequence

import numpy as np

SENSORS = ("sensor_0", "sensor_2")
RAW_DEPTH_SHAPE = (288, 384)
DEPTH_SHAPE = (48, 64)
WRENCH_DIM = 6
# A finger touches something when its pooled indentation exceeds this value (raw SDK depth units).
CONTACT_THRESHOLD = 0.05


def pool_depth(depth: np.ndarray, shape: Sequence[int] = DEPTH_SHAPE) -> np.ndarray:
    """Area-pools depth maps [..., H, W] to float32 [..., h, w]; H and W must be multiples of ``shape``."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim < 2:
        raise ValueError(f"Expected depth maps of shape [..., H, W], got {depth.shape}")
    height, width = depth.shape[-2:]
    out_height, out_width = (int(size) for size in shape)
    if out_height <= 0 or out_width <= 0 or height % out_height or width % out_width:
        raise ValueError(f"Cannot area-pool {height}x{width} depth maps to {out_height}x{out_width}")
    blocks = depth.reshape(*depth.shape[:-2], out_height, height // out_height, out_width, width // out_width)
    return blocks.mean(axis=(-3, -1))


def contact(depth: np.ndarray, threshold: float = CONTACT_THRESHOLD) -> np.ndarray:
    """Per-map contact flags (bool [...]) of pooled depth maps [..., h, w]."""
    return np.asarray(depth).max(axis=(-2, -1)) > threshold
