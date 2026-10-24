"""Sleep-EDF -> 256 Hz, 5 s 윈도우, Wake/Sleep 이진 데이터 (기초 모델 학습 방식 재현).

원본 X_ST7*-PSG.npy(256 Hz) 와 생성 코드는 유실. 아래는 노트북 generator 의 의미를 edf 에서 직접 복원한 것:
  - 100 Hz edf -> mne resample 256 Hz -> (n_sec, 256, 2)
  - 라벨: 초 단위 W 여부. 5 s 창에서 W 가 60% 이상이면 1 (Wake), 아니면 0 (Sleep)
  - 전처리: bandpass(axis=0) -> is_noisy 스킵 -> 스칼라 z-score   (Attention_Final 방식)
"""
import numpy as np
import mne
from .config import FS, SEQ_SEC, SEQ_LEN, STRIDE_SEC, WAKE_RATIO
from .preprocess import bandpass, is_noisy, zscore_scalar


def edf_to_binary_windows(psg_file, hyp_file, channels=("EEG Fpz-Cz", "EEG Pz-Oz"),
                          stage_map=None, fs=FS, stride_sec=STRIDE_SEC):
    stage_map = stage_map or {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
                              "Sleep stage 3": 3, "Sleep stage 4": 4, "Sleep stage R": 5}
    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
    ann = mne.read_annotations(hyp_file)
    raw.set_annotations(ann, verbose=False)
    raw.pick(list(channels))
    raw.resample(fs, verbose=False)
    data = raw.get_data().T                                  # (T, 2), V
    n_sec = data.shape[0] // fs
    data = data[: n_sec * fs].reshape(n_sec, fs, 2)          # X_virtual
    # 초 단위 라벨 (undefined -> -1)
    sec_labels = np.full(n_sec, -1, dtype=int)
    for desc, onset, dur in zip(ann.description, ann.onset, ann.duration):
        s, e = int(onset), min(int(onset + dur), n_sec)
        sec_labels[s:e] = stage_map.get(desc, -1)
    X, Y = [], []
    for idx in range(0, n_sec - SEQ_SEC + 1, stride_sec):
        labs = sec_labels[idx: idx + SEQ_SEC]
        if np.any(labs == -1):
            continue
        label = 1 if np.sum(labs == 0) >= SEQ_SEC * WAKE_RATIO else 0
        sample = data[idx: idx + SEQ_SEC].reshape(SEQ_LEN, 2).astype(np.float32)
        sample = bandpass(sample, axis=0)
        if is_noisy(sample):
            continue
        X.append(zscore_scalar(sample)); Y.append(label)
    return np.array(X, dtype=np.float32), np.array(Y)
