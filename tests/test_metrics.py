import numpy as np
from eeg_golf.metrics import metrics_from_logits


def test_binary_metrics():
    r = metrics_from_logits(np.asarray([-2.0, 2.0, 4.0, -4.0]), np.asarray([0, 1, 1, 0]))
    assert r['acc'] == 1.0
    assert r['f1_macro'] == 1.0
    assert r['auroc'] == 1.0


def test_single_class_auroc_is_nan():
    r = metrics_from_logits(np.asarray([1., -1.]), np.asarray([1, 1]))
    assert np.isnan(r['auroc'])
