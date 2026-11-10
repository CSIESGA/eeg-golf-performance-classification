"""Training/validation workflow migrated from the original Colab research script."""

import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import EEGDataset, load_splits
from .metrics import metrics_from_logits
from .model import ResNetTiny1D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_resnet_tiny(train_path: str, val_path: str, out_dir: str,
                      *, seed: int = 42, use_car: bool = False, epochs: int = 20,
                      batch_size: int = 16, lr: float = 1e-5,
                      base_channels: int = 32, kernel_size: int = 7,
                      num_workers: int = 0, device: str = "auto") -> dict:
    """Train and checkpoint by best validation AUROC.

    Historical scripts called one input 'test.npz' but evaluated it each epoch.
    That split is treated as validation here; metrics are NOT an untouched test score.
    """
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs, batch_size, and learning rate must be positive")
    set_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xva, yva, mean, std = load_splits(train_path, val_path, use_car=use_car)
    if np.unique(ytr).size < 2 or np.unique(yva).size < 2:
        raise ValueError("Both classes must be represented in train and validation splits")
    print(f"Train: {Xtr.shape}; validation: {Xva.shape}")
    train_dl = DataLoader(EEGDataset(Xtr, ytr), batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(EEGDataset(Xva, yva), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model = ResNetTiny1D(in_channels=Xtr.shape[1],
                         base_channels=base_channels, kernel_size=kernel_size).to(resolved_device)
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(n_neg / n_pos), dtype=torch.float32,
                                device=resolved_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best = {"auroc": float("-inf")}
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X, y in train_dl:
            X = X.to(resolved_device)
            y = y.to(resolved_device).float().unsqueeze(1)
            loss = criterion(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(X)
        model.eval()
        logits_list = []
        with torch.no_grad():
            for X, _ in val_dl:
                logits_list.append(model(X.to(resolved_device)).squeeze(1).cpu().numpy())
        metrics = metrics_from_logits(np.concatenate(logits_list), yva)
        print(f"Epoch {epoch:03d}: train_loss={running_loss / len(ytr):.4f} "
              f"val_acc={metrics['acc']:.4f} val_f1_macro={metrics['f1_macro']:.4f} "
              f"val_auroc={metrics['auroc']:.4f}")
        if metrics["auroc"] > best["auroc"]:
            best = {**metrics, "selected_epoch": epoch}
            torch.save({"state_dict": model.state_dict(), "mean": mean,
                        "std": std, "channels": ["FZ", "CZ", "PZ"],
                        "use_car": use_car, "base_channels": base_channels,
                        "kernel_size": kernel_size, "selected_epoch": epoch},
                       out / "best.ckpt")
    (out / "metrics_val.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    # Avoid recording absolute drive paths or subject-level metadata.
    run_config = {"seed": seed, "use_car": use_car, "epochs": epochs,
                  "batch_size": batch_size, "lr": lr, "base_channels": base_channels,
                  "kernel_size": kernel_size, "num_workers": num_workers,
                  "device": resolved_device}
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    return best
