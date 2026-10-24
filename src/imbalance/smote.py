"""SMOTE oversampling (train 셋에만 적용).

주의: raw 시계열(3000x2)을 flatten 후 보간 -> 비현실적 파형 생성/overfitting 위험.
노트북 최종 모델은 SMOTE 를 사용했고 class_weight 는 주석 처리 상태였음.
"""
import numpy as np
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_STATE


def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                random_state: int = RANDOM_STATE):
    """(n, T, C) -> flatten -> SMOTE -> (n', T, C) 복원."""
    orig_shape = X_train.shape[1:]  # (3000, 2)

    # reshape for SMOTE
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

    # SMOTE 적용
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_train)

    # reshape back to original
    X_train = X_resampled.reshape(-1, *orig_shape)
    y_train = y_resampled
    return X_train, y_train
