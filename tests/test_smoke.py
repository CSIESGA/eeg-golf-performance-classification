import json
import numpy as np
from eeg_golf.train import train_resnet_tiny


def test_synthetic_end_to_end(tmp_path):
    rng = np.random.default_rng(9)
    names = np.asarray(['Fz', 'Cz', 'Pz'] + [f'CH{i}' for i in range(4, 31)])
    for tag, count in [('train', 12), ('val', 8)]:
        data = rng.standard_normal((count, 30, 256)).astype(np.float32)
        labels = np.asarray(['S', 'U'] * (count // 2))
        np.savez(tmp_path / (tag + '.npz'), X=data, resp_chars=labels, chan_names=names)
    out = tmp_path / 'out'
    metrics = train_resnet_tiny(str(tmp_path / 'train.npz'), str(tmp_path / 'val.npz'), str(out),
                                epochs=1, batch_size=4, base_channels=8, num_workers=0,
                                device='cpu')
    assert 0 <= metrics['auroc'] <= 1
    assert metrics['selected_epoch'] == 1
    assert (out / 'best.ckpt').is_file()
    assert json.loads((out / 'metrics_val.json').read_text()) == metrics
    assert 'train_path' not in (out / 'run_config.json').read_text()
