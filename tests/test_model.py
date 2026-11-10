import torch
from eeg_golf.model import ResNetTiny1D


def test_output_is_one_logit_per_sample():
    model = ResNetTiny1D(in_channels=3)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 256))
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()
