# EEG-Based Golf Performance Classification (1D ResNet)

A research-code portfolio showing an end-to-end **Python/PyTorch binary classification pipeline for multichannel EEG time series**, developed during undergraduate research at **National Taiwan University** (Sep 2025–Mar 2026). This repository packages the researcher's own model-training scripts into a more readable, reusable structure for code review.

**Scope:** A small 1D residual CNN, selected EEG channel processing, training-only normalization, weighted classification loss, AUROC-based checkpoint selection, and reproducible training configuration. This is an **offline experimental pipeline**, not a deployed application or clinically validated model.

## Technical overview

| Component | Implementation |
|---|---|
| Input | EEG tensor `(N, channels, time)`; study documentation records `(N, 30, 2000)` for prepared windows |
| Channel selection | Select **Fz, Cz, Pz**, reducing input channel count from 30 to 3 (90% fewer input channels; not a measured speedup) |
| Preprocessing | Optional CAR (off in recorded runs), per-channel z-score calculated **only on training samples** |
| Architecture | Compact 1D ResNet with convolutional stem, three residual stages and global average pooling |
| Training | PyTorch DataLoader, class-weighted `BCEWithLogitsLoss`, AdamW, seeded training |
| Evaluation | Accuracy, macro-F1, AUROC; best checkpoint chosen by **validation AUROC** |
| Outputs | Local checkpoint and configuration/metric JSON files (ignored by Git) |

The binary label comes from the archived script's `resp_chars == "S"` rule. The exact semantics of the study's label codes are not asserted here.

## Repository layout

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── configs/
│   └── three_channel_resnet.json
├── src/eeg_golf/
│   ├── __init__.py
│   ├── model.py           # BasicBlock1D, ResNetTiny1D
│   ├── data.py            # NPZ, channel selection, normalization
│   ├── metrics.py         # Accuracy, macro-F1, AUROC
│   └── train.py           # Training / validation / checkpointing
├── scripts/
│   ├── train.py           # Command-line interface
│   └── make_synthetic_data.py
├── tests/                 # Synthetic-only verification
└── results/
    └── recorded_experiments.md
```

## Quick start (no participant data required)

Test the program with **synthetic data only**; these samples do not reproduce research accuracy:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
python scripts/make_synthetic_data.py
python scripts/train.py \
    --train-npz data/synthetic/train.npz \
    --val-npz data/synthetic/val.npz \
    --out-dir outputs/synthetic_demo
```

The synthetic demo exercises the model end-to-end; resulting metrics have **no scientific interpretation**. To execute the research experiment, use authorized NPZ files with `X`, `resp_chars`, and (on the training split) `chan_names`. The validation split should use the same original channel order or include its own `chan_names`.

Run the checks with `pip install -r requirements-dev.txt` followed by `pytest -q`.

## Saved experimental observations

See [recorded_experiments.md](results/recorded_experiments.md) for qualified historical metrics (including the 20-epoch configuration's recorded accuracy of 61.30%). Values came from original research records, not from the synthetic demo or a fresh rerun here.

**Evaluation caveat:** A source file used the name `test.npz` for a split evaluated at every epoch and used for checkpoint selection. In this repository it is explicitly treated as **validation**, and no untouched test-set performance is claimed. The original participant-level split procedure is unknown; results are not proof of unseen-participant generalization.

## Provenance and contributions

The original research was conducted as part of a university lab project under **Prof. Yan-Bin Chen**. The researcher's individually authored scripts `new_main.py` and `3-channel.py` contained the original ResNet model, EEG data-loading, training, and validation logic. This codebase is a **post-research refactoring** of those scripts: the architecture and training choices are preserved, while CLI arguments, module boundaries, synthetic fixtures, and code tests were added for easier inspection. The refactoring is not represented as functionality validated during the original study.

The researcher also participated in collaborative EEG preprocessing / bandpower work, but that team's separate code and experiments are **not** presented as independently implemented here.

## Data, ethics, and permissions

The lab has granted permission to share this work, but the repository **intentionally excludes** original EEG recordings, participant-level metadata, intermediate subject data, model checkpoints, and restricted research assets. No personal or clinical conclusions should be drawn from this research prototype. Please do not upload participant data or private model weights when using this project.

No third-party code license is granted by this README. Attribution and redistribution rights for any external materials should be checked before adding them.
