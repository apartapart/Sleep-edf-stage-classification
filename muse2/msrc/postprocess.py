import numpy as np
from .config import HYSTERESIS


def hysteresis(pred, high=HYSTERESIS["high"], low=HYSTERESIS["low"]):
    """확률 -> 상태(0 Sleep / 1 Awake). 전환 떨림 억제."""
    state, out = 0, []
    for p in pred:
        if p > high:
            state = 1
        elif p < low:
            state = 0
        out.append(state)
    return np.array(out)


def minmax_scale(preds):
    """노트북 v2: 파일별 확률을 0~1 로 강제 확장.
    모든 파일에 0 과 1 이 반드시 생김 -> 전부 각성인 파일도 절반이 sleep 으로 찍힘 (README 이슈)."""
    return (preds - np.min(preds)) / (np.max(preds) - np.min(preds) + 1e-6)
