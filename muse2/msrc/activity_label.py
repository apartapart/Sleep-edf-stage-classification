"""Activity 기반 pseudo-label (fine-tune 노트북).

activity = mean(|diff(window)|). 전체 분포 30/70 백분위로 sleep-like(0) / awake-like(1).
주의: 신호 미분값으로 라벨 만들고 같은 신호로 학습 -> 순환. 수면 GT 아님.
"""
import numpy as np
from .config import SEQ_LEN, STRIDE_INFER, ACTIVITY_PCT
from .preprocess import preprocess, is_valid
from .windows import load_muse_csv


def get_activity(window):
    return np.mean(np.abs(np.diff(window)))


def get_spike_ratio(window, thr=2.5):
    return np.mean(np.abs(window) > thr)


def collect_all_activity(files, seq_len=SEQ_LEN, stride=STRIDE_INFER):
    acts = []
    for f in files:
        d = preprocess(load_muse_csv(f))
        for i in range(0, len(d) - seq_len, stride):
            w = d[i:i + seq_len]
            if is_valid(w):
                acts.append(get_activity(w))
    return np.array(acts)


def thresholds(files):
    acts = collect_all_activity(files)
    return np.percentile(acts, ACTIVITY_PCT[0]), np.percentile(acts, ACTIVITY_PCT[1])


def load_and_label(csv_path, low_thr, high_thr, seq_len=SEQ_LEN, stride=STRIDE_INFER):
    d = preprocess(load_muse_csv(csv_path))
    X, Y = [], []
    for i in range(0, len(d) - seq_len, stride):
        w = d[i:i + seq_len]
        if not is_valid(w):
            continue
        act = get_activity(w)
        if act < low_thr:
            X.append(w); Y.append(0.0)
        elif act > high_thr:
            X.append(w); Y.append(1.0)
    return np.array(X), np.array(Y)


def augment(x):
    return x * np.random.uniform(0.9, 1.1) + np.random.normal(0, 0.01, x.shape)
