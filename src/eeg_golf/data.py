"""NPZ loading, Fz/Cz/Pz selection, and training-only EEG normalization."""

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

DEFAULT_CHANNELS = ("FZ", "CZ", "PZ")


class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _channel_indices(names: np.ndarray, number_of_channels: int,
                     target_channels: tuple[str, ...]) -> np.ndarray:
    # Original data sometimes contained more channel names than tensor channels.
    if len(names) < number_of_channels:
        raise ValueError("chan_names contains fewer names than the EEG tensor has channels")
    def normalize_name(value: object) -> str:
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8")
        return str(value).strip().upper()
    normalized = [normalize_name(v) for v in names[:number_of_channels]]
    indices = []
    for target in target_channels:
        found = [i for i, name in enumerate(normalized) if name == target.upper()]
        if not found:
            raise ValueError(f"Missing required channel {target!r}")
        indices.append(found[0])  # preserve Fz, Cz, Pz order; take first duplicate
    return np.asarray(indices, dtype=np.int64)


def _load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    # Trusted lab-owned data only. No raw study data are distributed with this repo.
    with np.load(path, allow_pickle=True) as data:
        for key in ("X", "resp_chars"):
            if key not in data:
                raise ValueError(f"{path} is missing required key {key!r}")
        X = np.asarray(data["X"], dtype=np.float32)
        raw_labels = np.asarray(data["resp_chars"]).astype(str).reshape(-1)
        chan_names = np.asarray(data["chan_names"]).copy() if "chan_names" in data else None
    if X.ndim != 3 or X.shape[0] != len(raw_labels):
        raise ValueError("Expected EEG tensor (N, C, T) and N labels")
    if not np.isfinite(X).all():
        raise ValueError("EEG input contains non-finite values")
    labels = (raw_labels == "S").astype(np.int64)
    return X, labels, chan_names


def apply_car(X: np.ndarray) -> np.ndarray:
    """Optional common-average reference across the selected channels."""
    return X - X.mean(axis=1, keepdims=True)


def load_splits(train_path: str | Path, val_path: str | Path,
                target_channels: tuple[str, ...] = DEFAULT_CHANNELS,
                use_car: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized train/validation tensors, labels, and training mean/std.

    Channel indices follow names in the training data. If validation has names,
    map them independently; otherwise assume matching original channel order.
    No claim is made about how the underlying study divided participants.
    """
    Xtr, ytr, train_names = _load_npz(train_path)
    Xva, yva, val_names = _load_npz(val_path)
    if train_names is None:
        raise ValueError("Training NPZ must contain chan_names")
    if val_names is None and Xtr.shape[1] != Xva.shape[1]:
        raise ValueError("Validation lacks chan_names and has a different channel count")
    tr_idx = _channel_indices(train_names, Xtr.shape[1], target_channels)
    va_idx = (_channel_indices(val_names, Xva.shape[1], target_channels)
              if val_names is not None else tr_idx)
    Xtr, Xva = Xtr[:, tr_idx, :], Xva[:, va_idx, :]
    if use_car:
        Xtr, Xva = apply_car(Xtr), apply_car(Xva)
    mean = Xtr.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = (Xtr.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    return (
        ((Xtr - mean) / std).astype(np.float32), ytr,
        ((Xva - mean) / std).astype(np.float32), yva, mean, std,
    )
