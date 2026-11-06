import numpy as np
import pytest

from openpi_client import tactile


def test_pool_depth_area_averages_blocks():
    depth = np.arange(2 * 4 * 6, dtype=np.float64).reshape(2, 4, 6)
    pooled = tactile.pool_depth(depth, (2, 3))
    assert pooled.dtype == np.float32
    assert pooled.shape == (2, 2, 3)
    expected = depth.reshape(2, 2, 2, 3, 2).mean(axis=(2, 4))
    np.testing.assert_allclose(pooled, expected)

    native = np.zeros((2, *tactile.RAW_DEPTH_SHAPE), dtype=np.float32)
    native[0, 6:12, 12:18] = 0.36
    pooled = tactile.pool_depth(native)
    assert pooled.shape == (2, *tactile.DEPTH_SHAPE)
    assert pooled[0, 1, 2] == pytest.approx(0.36)
    assert pooled[0].sum() == pytest.approx(0.36)


@pytest.mark.parametrize(("shape", "pooled"), [((4, 6), (3, 3)), ((4, 6), (0, 3)), ((6,), (2, 3))])
def test_pool_depth_rejects_incompatible_shapes(shape, pooled):
    with pytest.raises(ValueError, match="pool|depth maps"):
        tactile.pool_depth(np.zeros(shape), pooled)


def test_contact_thresholds_the_peak_indentation():
    depth = np.zeros((3, 2, 4, 4), dtype=np.float32)
    depth[0, 1, 2, 3] = 0.2
    depth[1, 0] = 0.03
    np.testing.assert_array_equal(tactile.contact(depth), [[False, True], [False, False], [False, False]])
    np.testing.assert_array_equal(
        tactile.contact(depth, threshold=0.01), [[False, True], [True, False], [False, False]]
    )
