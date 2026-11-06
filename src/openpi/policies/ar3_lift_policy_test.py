import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import ar3_lift_policy as ar3

L, H = 3, 4


def _history_item(*, finger_wrench: bool = True) -> dict:
    """Dataset item with L history frames, the first before the episode start and the last two past its end."""
    frames = np.arange(L + H, dtype=np.float32)
    item = {
        ar3.TCP_POSE_KEY: np.arange(6, dtype=np.float32),
        ar3.GRIPPER_POSITION_KEY: 100 * frames,
        ar3.TCP_WRENCH_KEY: np.repeat(frames[:, None], 6, axis=1),
        ar3.TAU_M_KEY: np.repeat(-frames[:, None], 7, axis=1),
        ar3.TACTILE_DEPTH_KEY: np.full((L + H, 2, 48, 64), 0.5, np.float16),
        ar3.TACTILE_VALID_KEY: np.ones((L + H, 2), bool),
        ar3.WRIST_IMAGE_KEY: np.zeros((8, 8, 3), np.uint8),
        ar3.FRONT_IMAGE_KEY: np.zeros((8, 8, 3), np.uint8),
        ar3.ACTIONS_KEY: np.zeros((H, 7), np.float32),
        "task": "Insert the plug",
    }
    if finger_wrench:
        for sign, key in zip((1, -1), ar3.GRIPPER_WRENCH_KEYS, strict=True):
            item[key] = sign * np.repeat(frames[:, None], 6, axis=1)
    for key in (ar3.GRIPPER_POSITION_KEY, *ar3.FORCE_KEYS_WRENCH_TAU_M, *ar3.GRIPPER_WRENCH_KEYS):
        if key in item:
            item[f"{key}_is_pad"] = np.array([True, False, False, False, False, True, True])
    return item


def test_repack_splits_the_history_window_into_state_and_readings():
    inputs = ar3.AR3LiftRepack(history_length=L, tactile=True)(_history_item())
    # The state is the current frame (row L), not the oldest history frame.
    np.testing.assert_array_equal(inputs["observation/state"], [0, 1, 2, 3, 4, 5, 100 * L])
    assert inputs["wrench"].shape == (L + H, 13)
    np.testing.assert_array_equal(inputs["wrench"][:, 6], -np.arange(L + H))
    # Only history rows before the episode start lack a reading; rows past its end repeat the last one (LIFT).
    np.testing.assert_array_equal(inputs["sensor_valid"], [False] + [True] * (L + H - 1))
    np.testing.assert_array_equal(inputs["tactile_wrench"][:, 6], -np.arange(L + H))
    np.testing.assert_array_equal(inputs["gripper_aperture"], 100 * np.arange(L + H))
    assert inputs["tactile_valid"].all()
    assert inputs["prompt"] == "Insert the plug"

    # Recordings without the Tac G2 wrenches (LIFT DAgger rounds): both fingers are unknown.
    inputs = ar3.AR3LiftRepack(history_length=L, tactile=True)(_history_item(finger_wrench=False))
    assert inputs["tactile_wrench"].shape == (L + H, ar3.TACTILE_WRENCH_DIM)
    assert not inputs["tactile_valid"].any()


def test_repack_without_history_keeps_the_lift_keys():
    history_item = _history_item()
    item = {key: history_item[key] for key in (ar3.TCP_POSE_KEY, ar3.WRIST_IMAGE_KEY, ar3.FRONT_IMAGE_KEY, "task")}
    item.update({key: history_item[key][L:] for key in (ar3.ACTIONS_KEY, *ar3.FORCE_KEYS_WRENCH_TAU_M)})
    item[ar3.GRIPPER_POSITION_KEY] = np.array([700.0])
    inputs = ar3.AR3LiftRepack()(item)
    assert inputs.keys() == {
        "observation/state",
        "observation/left_wrist_image",
        "observation/base_image",
        "actions",
        "wrench",
        "prompt",
    }
    np.testing.assert_array_equal(inputs["observation/state"], [0, 1, 2, 3, 4, 5, 700])
    np.testing.assert_array_equal(inputs["wrench"][:, 0], np.arange(L, L + H))


@pytest.mark.parametrize("measured", [1, H])
def test_inputs_pad_the_client_readings_to_history_and_horizon(measured):
    example = ar3.make_ar3_lift_example(action_horizon=H, history_length=L, tactile=True)
    rows = L + measured
    for key in ar3.READING_KEYS:
        example[key] = example[key][:rows]
    inputs = ar3.AR3LiftInputs(_model.ModelType.PI05, action_horizon=H, history_length=L, tactile=True)(example)
    for key in ar3.READING_KEYS:
        assert len(inputs[key]) == L + H
        # Rows not measured yet repeat the newest reading.
        np.testing.assert_array_equal(inputs[key][rows:], np.repeat(inputs[key][rows - 1 : rows], L + H - rows, 0))
    np.testing.assert_array_equal(inputs["sensor_valid"][:rows], example["sensor_valid"])
    assert inputs["tactile_depth"].dtype == np.float32
    np.testing.assert_allclose(
        inputs["tactile_depth"][:rows], example["tactile_depth"].astype(np.float32) / ar3.TACTILE_DEPTH_SCALE
    )
    assert inputs["gripper_aperture"].shape == (L + H, 1)
    np.testing.assert_allclose(inputs["gripper_aperture"][:rows, 0], example["gripper_aperture"] / ar3.GRIPPER_OPEN)


def test_inputs_reject_incomplete_readings():
    policy_inputs = ar3.AR3LiftInputs(_model.ModelType.PI05, action_horizon=H, history_length=L, tactile=True)
    example = ar3.make_ar3_lift_example(action_horizon=H, history_length=L, tactile=True)
    with pytest.raises(KeyError, match="tactile_depth"):
        policy_inputs({key: value for key, value in example.items() if key != "tactile_depth"})
    with pytest.raises(ValueError, match="'gripper_aperture' has"):
        policy_inputs({**example, "gripper_aperture": example["gripper_aperture"][:-1]})
    with pytest.raises(ValueError, match="history readings"):
        policy_inputs({**example, **{key: example[key][:L] for key in ar3.READING_KEYS}})


def test_inputs_without_history_keep_the_lift_protocol():
    example = ar3.make_ar3_lift_example(action_horizon=H)
    example["wrench"] = example["wrench"][0]
    inputs = ar3.AR3LiftInputs(_model.ModelType.PI05, action_horizon=H)(example)
    np.testing.assert_array_equal(inputs["wrench"], np.repeat(example["wrench"][None], H, axis=0))
    assert not inputs.keys() & set(ar3.READING_KEYS[1:])
