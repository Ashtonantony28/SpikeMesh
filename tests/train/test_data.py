import numpy as np
import pytest

from train.data import DATA_DIR, N_IN, images_to_events, saccade_path
from train.data_real import events_to_raster


def _digit_like(n=4, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((n, 28, 28), np.uint8)
    for i in range(n):
        r0, c0 = rng.integers(4, 12, 2)
        img[i, r0:r0 + 12, c0:c0 + 3] = 255     # a vertical bar
        img[i, r0:r0 + 3, c0:c0 + 10] = 200     # and a horizontal one
    return img


def test_saccade_path_is_closed_triangle():
    p = saccade_path(24, 2.0)
    assert p.shape == (25, 2)
    np.testing.assert_allclose(p[0], p[-1])
    # three straight segments of 8 steps each, with distinct directions
    dirs = {tuple(np.round(np.sign(p[k + 1] - p[k]), 3)) for k in range(24)}
    assert len(dirs) == 3


def test_events_shape_dtype_deterministic():
    img = _digit_like()
    a = images_to_events(img, T=24)
    b = images_to_events(img, T=24)
    assert a.shape == (4, 24, N_IN) and a.dtype == bool
    assert np.array_equal(a, b)
    assert a.sum() > 0
    assert a.sum((1, 2)).min() > 20             # every sample produces events
    assert not images_to_events(np.zeros((1, 28, 28), np.uint8)).any()  # blank -> no events


def test_batching_does_not_change_result():
    img = _digit_like(7, seed=3)
    np.testing.assert_array_equal(images_to_events(img, batch=2), images_to_events(img, batch=1000))


@pytest.mark.skipif(not (DATA_DIR / "mnist" / "t10k-images-idx3-ubyte.gz").exists(),
                    reason="MNIST not downloaded")
def test_real_mnist_event_rate():
    from train.data import load_mnist
    x, y = load_mnist("test")
    ev = images_to_events(x[:200])
    per_step = ev.sum(2).mean()
    assert 10 < per_step < 120, per_step       # sparse but informative


def test_real_dataset_binning_on_synthetic_events():
    """data_real is untestable without the dataset hosts; test its binning only."""
    dt = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)])
    ev = np.array([(0, 0, 0, 1), (33, 33, 999, 0), (17, 2, 500, 1), (18, 2, 501, 0)], dt)
    r = events_to_raster(ev, (34, 34), T=10)
    assert r.shape == (10, 256) and r.sum() == 3   # two events merge into one axon/bin
    assert r[0, 0] and r[9, 15 * 16 + 15] and r[5, 0 * 16 + 8]
    assert not events_to_raster(ev[:0], (34, 34), 10).any()
