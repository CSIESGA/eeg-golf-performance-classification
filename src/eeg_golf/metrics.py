"""Binary classification evaluation from raw model logits."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have matching shapes")
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    preds = (probs >= 0.5).astype(np.int64)
    auroc = float(roc_auc_score(labels, probs)) if np.unique(labels).size == 2 else float('nan')
    return {
        "auroc": auroc,
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "acc": float(accuracy_score(labels, preds)),
    }
