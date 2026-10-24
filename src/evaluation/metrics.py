"""κ / macro-F1 / accuracy, W trimming, N3+N4 병합."""
import numpy as np
from sklearn.metrics import (cohen_kappa_score, f1_score, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)

STAGE_NAMES_6 = ["W", "N1", "N2", "N3", "N4", "R"]
STAGE_NAMES_5 = ["W", "N1", "N2", "N3", "R"]


def trim_wake(y_true: np.ndarray, pad_epochs: int = 60):
    """첫/마지막 non-W epoch 기준 ±pad_epochs(기본 60 = 30 min) 범위 인덱스 (lo, hi).

    cassette 는 ~20h 녹화로 취침 전후 W 가 과반 -> accuracy 부풀림 (DeepSleepNet 관행 적용).
    """
    sleep_idx = np.where(y_true != 0)[0]
    if len(sleep_idx) == 0:
        return 0, len(y_true)
    lo = max(0, sleep_idx[0] - pad_epochs)
    hi = min(len(y_true), sleep_idx[-1] + pad_epochs + 1)
    return lo, hi


def merge_n3n4(y: np.ndarray) -> np.ndarray:
    """R&K 6-class -> AASM 5-class. N4(4)->N3(3), R(5)->4."""
    y = np.asarray(y).copy()
    y[y == 4] = 3
    y[y == 5] = 4
    return y


def summary_metrics(y_true, y_pred) -> dict:
    return {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def per_class_metrics(y_true, y_pred, n_classes):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "support": s}


def cm(y_true, y_pred, n_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
