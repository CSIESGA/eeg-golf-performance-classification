"""Create purely synthetic inputs to exercise the pipeline; NOT real study data."""

import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/synthetic")
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--val-count", type=int, default=16)
    parser.add_argument("--timepoints", type=int, default=256)
    args = parser.parse_args()
    if min(args.train_count, args.val_count) < 4 or args.timepoints < 40:
        raise ValueError("At least 4 examples per split and 40 timepoints are required")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    names = np.asarray(["Fz", "Cz", "Pz"] + [f"CH{i}" for i in range(4, 31)])
    for file_name, size in (("train.npz", args.train_count), ("val.npz", args.val_count)):
        X = rng.standard_normal((size, 30, args.timepoints)).astype(np.float32)
        labels = np.asarray(["S", "U"] * ((size + 1) // 2))[:size]
        rng.shuffle(labels)
        np.savez(out / file_name, X=X, resp_chars=labels, chan_names=names)
        print(f"Wrote synthetic-only fixture: {out / file_name}")


if __name__ == "__main__":
    main()
