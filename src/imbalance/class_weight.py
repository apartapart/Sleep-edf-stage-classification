"""클래스 가중치 계산 (SMOTE 대안).

- balanced : n_samples / (n_classes * count_c)  -> 빈도 역수 비례
- sqrt softening : 가중치 효과 완화
- manual : 노트북에서 실험했던 수동 가중치
"""
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def balanced_class_weight(y_train: np.ndarray) -> dict:
    class_labels = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=y_train,
    )
    class_weight_dict = dict(zip(class_labels, class_weights))
    print("계산된 클래스 가중치:", class_weight_dict)
    return class_weight_dict


def sqrt_softened_class_weight(y_train: np.ndarray) -> dict:
    class_weight_dict = balanced_class_weight(y_train)
    class_weight_sqrt = {k: np.sqrt(v) for k, v in class_weight_dict.items()}
    print("완화된 클래스 가중치:", class_weight_sqrt)
    return class_weight_sqrt


MANUAL_CLASS_WEIGHTS = {
    0: 1.5,   # W
    1: 4.0,   # N1
    2: 1.0,   # N2 (기준값)
    3: 3.5,   # N3
    4: 4.5,   # N4
    5: 2.0,   # R
}
