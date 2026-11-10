import numpy as np
from eeg_golf.data import _channel_indices


def test_bytes_channel_names():
    idx = _channel_indices(np.asarray([b'Cz', b'Fz', b'Pz']), 3,
                           ('FZ', 'CZ', 'PZ'))
    assert idx.tolist() == [1, 0, 2]
