"""전처리 (노트북 동일): butter band-pass 0.5-40 Hz → median 제거 → clip ±100 → z-score."""
import numpy as np
from scipy.signal import butter, filtfilt
from .config import FS, BANDPASS, CLIP


def bandpass(x, axis=0, fs=FS):
    nyq = 0.5 * fs
    b, a = butter(BANDPASS["order"], [BANDPASS["low"] / nyq, BANDPASS["high"] / nyq], btype="band")
    return filtfilt(b, a, x, axis=axis)


def preprocess(x):
    """x: (T, C) 단일 녹화 전체. CSV 추론/fine-tune 노트북 방식 (axis=0)."""
    x = bandpass(x, axis=0)
    x = x - np.median(x, axis=0)
    x = np.clip(x, -CLIP, CLIP)
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)


def preprocess_batch(x):
    """x: (N, T, C) 윈도우 배치. pretrained_2 학습 generator 방식 (axis=1)."""
    x = bandpass(x, axis=1)
    x = x - np.median(x, axis=1, keepdims=True)
    x = np.clip(x, -CLIP, CLIP)
    return ((x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)).astype(np.float32)


def is_noisy(sample, std_thr=10.0, abs_thr=150.0):
    """Attention_Final 학습 시 노이즈 윈도우 제외 규칙. 입력 단위가 V(1e-5) 면 사실상 미작동."""
    return np.std(sample) > std_thr or np.max(np.abs(sample)) > abs_thr


def is_valid(signal, thr=1e-3):
    return np.std(signal) > thr


def zscore_scalar(sample):
    """Attention_Final 학습 방식: 채널 구분 없는 스칼라 mean/std."""
    return (sample - sample.mean()) / (sample.std() + 1e-6)
