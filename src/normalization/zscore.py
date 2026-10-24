"""채널별 전역 z-score 정규화.

- 통계(mean/std)는 **train 셋에서만** 산출 -> val/test/inference 에 동일 적용 (데이터 누수 방지)
- axis=(0,1): epoch·시간축 전체 평균 -> shape (1, 1, n_channels)
- z-score 는 min-max 대비 이상치 영향 적어 EEG 에 적합
"""
import os
import numpy as np


def fit_zscore(X_train: np.ndarray):
    """train 셋으로 mean/std 산출. 각각 shape (1, 1, n_channels)."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    return mean, std


def apply_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 eps: float = 0.0) -> np.ndarray:
    """(X - mean) / (std + eps).

    학습 노트북은 eps=0, 추론 노트북은 eps=1e-7 사용. 수치적으로 동일.
    """
    n_channels = X.shape[-1]
    mean = np.asarray(mean).reshape(1, 1, n_channels)
    std = np.asarray(std).reshape(1, 1, n_channels)
    return (X - mean) / (std + eps)


def save_stats(mean: np.ndarray, std: np.ndarray, target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    mean_filepath = os.path.join(target_dir, "mean.npy")
    std_filepath = os.path.join(target_dir, "std.npy")
    np.save(mean_filepath, mean)
    np.save(std_filepath, std)
    print(f"Mean and standard deviation files saved to: {target_dir}")
    return mean_filepath, std_filepath


def load_stats(mean_path: str, std_path: str):
    trained_mean = np.load(mean_path)
    trained_std = np.load(std_path)
    return trained_mean, trained_std
