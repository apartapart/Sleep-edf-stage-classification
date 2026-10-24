"""배포용 후처리 규칙 (PDF 10장, VSCode 스트리밍 버전).

- REM 보정: (R, N2, R) 패턴의 가운데 N2 -> R
- 수면 종료 컷: W(0) 가 30 epoch(=15 min) 연속되면 첫 W 이후 30 epoch 까지만 남김
"""
import numpy as np


def fix_isolated_n2_in_rem(pred):
    """[5, 2, 5] -> [5, 5, 5]. in-place 와 동일 동작, list 반환."""
    pred = list(pred)
    for j in range(1, len(pred) - 1):
        if pred[j - 1] == 5 and pred[j] == 2 and pred[j + 1] == 5:
            pred[j] = 5
    return pred


def truncate_after_wake_run(pred, run: int = 30):
    """연속 W 가 run 개 이상이면 첫 W 인덱스 + run 까지 자름.

    원본 로직 그대로: 'first_zero_index' 는 전체에서 **첫 W** (연속 구간 시작이 아님).
    취침 전 W 가 있으면 의도와 다르게 잘릴 수 있음 -> README 이슈 참고.
    """
    pred = list(pred)
    is_zero = (np.array(pred) == 0).astype(int)
    if len(is_zero) > run:
        zero_run = np.max(np.convolve(is_zero, np.ones(run, dtype=int), mode='valid'))
        if zero_run >= run:
            first_zero_index = np.where(is_zero.astype(bool))[0][0]
            pred = pred[:first_zero_index + run]
    return pred
