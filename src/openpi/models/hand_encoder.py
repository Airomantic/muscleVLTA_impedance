"""Tac G2 hand stream of muscleVLTA.

Finger contact is sparse: the indentation maps and finger wrenches carry signal only while an object is held, and the
noise of a free finger is not a weak version of that signal. Each finger reading therefore becomes a token through a
contact gate (supervised with ``contact_labels``): without contact the token collapses to a learned no-contact
embedding, and an unavailable (or dropped) finger uses a separate missing-sensor embedding. A GRU fuses the finger
tokens, their contact probabilities and the gripper opening over time, skipping the rows before the episode start.

Layer names avoid ``*_1``/``*_2`` pairs: ``maybe_copy_reactive_expert_params`` copies every ``*_1`` entry to ``*_2``.
"""

import flax.linen as nn
import flax.nnx.bridge as nnx_bridge
import jax.numpy as jnp

from openpi.models import gru as _gru

GEOMETRY_DIM = 8
FINGER_WRENCH_DIM = 6


def contact_geometry(depth: jnp.ndarray, contact_depth: float) -> jnp.ndarray:
    """Indentation maps [..., h, w] -> contact-patch features [..., 8].

    Peak and mean indentation, the contact area fraction, then the centroid (x, y in [-1, 1]) and the central second
    moments (xx, yy, xy) of the indentation above ``contact_depth``. The shape features fade in with the patch mass
    and are zero without contact.
    """
    height, width = depth.shape[-2:]
    depth = jnp.maximum(depth.astype(jnp.float32), 0.0)
    excess = jnp.maximum(depth - contact_depth, 0.0)
    mass = excess.sum(axis=(-2, -1))
    weights = excess / jnp.maximum(mass, 1e-6)[..., None, None]
    ys = jnp.linspace(-1.0, 1.0, height)[:, None]
    xs = jnp.linspace(-1.0, 1.0, width)[None, :]

    def moment(values):
        return (weights * values).sum(axis=(-2, -1))

    cx, cy = moment(xs), moment(ys)
    dx = xs - cx[..., None, None]
    dy = ys - cy[..., None, None]
    shape = jnp.stack([cx, cy, moment(dx**2), moment(dy**2), moment(dx * dy)], axis=-1)
    presence = mass / (mass + 1.0)
    intensity = jnp.stack(
        [depth.max(axis=(-2, -1)), depth.mean(axis=(-2, -1)), (depth > contact_depth).mean(axis=(-2, -1))], axis=-1
    )
    return jnp.concatenate([intensity, shape * presence[..., None]], axis=-1)


def contact_labels(depth: jnp.ndarray, contact_depth: float) -> jnp.ndarray:
    """Per-map contact: peak indentation above ``contact_depth``."""
    return depth.max(axis=(-2, -1)) > contact_depth


class _FingerEncoder(nn.Module):
    """One finger reading (indentation map [N, h, w], wrench [N, 6]) -> (features [N, F], contact logit [N])."""

    features: int
    contact_depth: float

    @nn.compact
    def __call__(self, depth, wrench):
        geometry = contact_geometry(depth, self.contact_depth)
        x = depth.astype(jnp.float32)[..., None]
        for channels in (16, 32, 64):
            x = nn.gelu(nn.Conv(channels, (3, 3), strides=(2, 2), name=f"conv{channels}")(x))
        x = x.reshape(*x.shape[:-3], -1)
        wrench_features = nn.Dense(self.features, name="wrench_out")(nn.gelu(nn.Dense(128, name="wrench_in")(wrench)))
        features = (
            nn.Dense(self.features, name="map_proj")(x)
            + nn.Dense(self.features, name="geometry_proj")(geometry)
            + wrench_features
        )
        features = nn.LayerNorm(name="norm")(features)
        gate_hidden = nn.gelu(nn.Dense(64, name="gate_in")(jnp.concatenate([geometry, wrench], axis=-1)))
        return features, nn.Dense(1, name="gate_out")(gate_hidden)[..., 0]


class HandEncoder(nn.Module):
    """Tac G2 readings -> (hand tokens [B, T, width] bf16, finger contact logits [B, T, f]).

    Inputs: indentation maps [B, T, f, h, w], finger validity [B, T, f], finger wrenches [B, T, 6 f], gripper opening
    [B, T, 1] and row validity [B, T]. The finger encoder is shared by the fingers; their tokens are concatenated, so
    each finger keeps its own slot in the GRU input.
    """

    width: int
    hidden_size: int
    features: int
    contact_depth: float
    depth_shape: tuple[int, int]
    num_fingers: int = 2
    rows: int = 1

    @nn.compact
    def __call__(self, depth, finger_valid, finger_wrench, aperture, valid):
        batch, rows, fingers = depth.shape[:3]
        wrench = finger_wrench.reshape(batch, rows, fingers, FINGER_WRENCH_DIM)
        features, logits = _FingerEncoder(self.features, self.contact_depth, name="finger")(
            depth.reshape(-1, *depth.shape[-2:]), wrench.reshape(-1, FINGER_WRENCH_DIM).astype(jnp.float32)
        )
        features = features.reshape(batch, rows, fingers, self.features)
        logits = logits.reshape(batch, rows, fingers)

        no_contact = self.param("no_contact_embedding", nn.initializers.normal(1.0), (self.features,))
        missing = self.param("missing_embedding", nn.initializers.normal(1.0), (self.features,))
        gate = nn.sigmoid(logits)
        tokens = gate[..., None] * features + (1.0 - gate[..., None]) * no_contact
        tokens = jnp.where(finger_valid[..., None], tokens, missing)
        gate = jnp.where(finger_valid, gate, 0.0)
        x = jnp.concatenate(
            [tokens.reshape(batch, rows, -1), gate, aperture.astype(jnp.float32)],
            axis=-1,
        )
        gru = _gru._LinenGRUNetwork(  # noqa: SLF001
            hidden_size=self.hidden_size,
            output_size=self.width,
            action_horizon=rows,
            input_size=x.shape[-1],
            name="gru",
        )
        return gru(x, valid=valid), logits

    def init(self):
        rows, fingers = self.rows, self.num_fingers
        self(
            jnp.zeros((1, rows, fingers, *self.depth_shape)),
            jnp.ones((1, rows, fingers), dtype=bool),
            jnp.zeros((1, rows, FINGER_WRENCH_DIM * fingers)),
            jnp.zeros((1, rows, 1)),
            jnp.ones((1, rows), dtype=bool),
        )


def create_hand_encoder(**kwargs) -> nnx_bridge.ToNNX:
    return nnx_bridge.ToNNX(HandEncoder(**kwargs))
