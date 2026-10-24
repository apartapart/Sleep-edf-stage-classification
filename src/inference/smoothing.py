"""예측 하이프노그램 후처리: 이동 윈도우 최빈값 smoothing."""
from collections import Counter
import numpy as np


def smooth_predictions(y_pred: np.ndarray, window_size: int = 5) -> np.ndarray:
    """window 내 most-common 라벨로 치환. 양끝은 edge 패딩."""
    smoothed_pred = []
    padded_y_pred = np.pad(y_pred, (window_size // 2, window_size // 2), mode='edge')

    for i in range(len(y_pred)):
        window = padded_y_pred[i:i + window_size]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed_pred.append(most_common)

    return np.array(smoothed_pred)
