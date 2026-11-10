import numpy as np
import pytest

from eeg_golf.data import apply_car, load_splits


def save_npz(path, data, labels, channels):
    np.savez(path, X=data, resp_chars=np.asarray(labels), chan_names=np.asarray(channels))


def test_channel_order_and_train_only_stats(tmp_path):
    names = ['Cz', 'EXTRA', 'Pz', 'Fz']
    train = np.ones((4, 4, 64), dtype=np.float32)
    train[:, 0, :] *= 4
    train[:, 2, :] *= 6
    train[:, 3, :] *= 2
    # Different distribution for validation must not affect training mean and std.
    val = np.full((4, 4, 64), 99.0, dtype=np.float32)
    save_npz(tmp_path / 'train.npz', train, ['S', 'U', 'S', 'U'], names)
    save_npz(tmp_path / 'val.npz', val, ['U', 'S', 'U', 'S'], names)
    Xtr, ytr, Xva, yva, mean, std = load_splits(tmp_path / 'train.npz', tmp_path / 'val.npz')
    assert Xtr.shape == (4, 3, 64)
    assert Xva.shape == (4, 3, 64)
    np.testing.assert_allclose(mean.flatten(), [2., 4., 6.])
    assert (Xva > 1.0).all()
    assert ytr.tolist() == [1, 0, 1, 0]
    assert yva.tolist() == [0, 1, 0, 1]


def test_validation_name_reordering(tmp_path):
    names_train = ['Fz', 'Cz', 'Pz', 'OTHER']
    names_val = ['Pz', 'Fz', 'Cz', 'OTHER']
    Xtr = np.random.default_rng(1).standard_normal((4, 4, 64)).astype(np.float32)
    Xva = np.ones((4, 4, 64), dtype=np.float32)
    Xva[:, 0, :] *= 3
    Xva[:, 1, :] *= 1
    Xva[:, 2, :] *= 2
    save_npz(tmp_path / 'train.npz', Xtr, ['S', 'U', 'U', 'S'], names_train)
    save_npz(tmp_path / 'val.npz', Xva, ['S', 'U', 'U', 'S'], names_val)
    *_, mean, std = load_splits(tmp_path / 'train.npz', tmp_path / 'val.npz')
    assert mean.shape == (1, 3, 1)
    Xtr_n, _, Xva_n, *_ = load_splits(tmp_path / 'train.npz', tmp_path / 'val.npz')
    np.testing.assert_allclose(Xva_n[0, :, 0], (np.asarray([1, 2, 3]) - mean.reshape(3)) / std.reshape(3))


def test_missing_channel_raises(tmp_path):
    X = np.zeros((4, 3, 64), dtype=np.float32)
    for name in ('train', 'val'):
        save_npz(tmp_path / (name + '.npz'), X, ['S', 'U', 'S', 'U'], ['Fz', 'Cz', 'Oz'])
    with pytest.raises(ValueError, match='PZ'):
        load_splits(tmp_path / 'train.npz', tmp_path / 'val.npz')


def test_car_removes_channel_mean_at_each_timepoint():
    X = np.random.default_rng(42).standard_normal((3, 3, 64))
    np.testing.assert_allclose(apply_car(X).mean(axis=1), 0., atol=1e-12)
