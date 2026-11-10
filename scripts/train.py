"""CLI wrapper around the training pipeline. Run from repository root."""

import argparse
import json
from pathlib import Path

from eeg_golf.train import train_resnet_tiny


def main():
    parser = argparse.ArgumentParser(description="Train 1D ResNet-Tiny on research NPZ EEG splits")
    parser.add_argument("--train-npz", required=True, help="Private training .npz path")
    parser.add_argument("--val-npz", required=True, help="Private validation .npz path")
    parser.add_argument("--out-dir", default="outputs/eeg_run")
    parser.add_argument("--config", default="configs/three_channel_resnet.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print("Best validation checkpoint:", train_resnet_tiny(
        args.train_npz, args.val_npz, args.out_dir, **config))


if __name__ == "__main__":
    main()
