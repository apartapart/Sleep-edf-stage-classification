import numpy as np
import pandas as pd
from .config import SEQ_LEN, STRIDE_INFER, MUSE_CHANNELS


def load_muse_csv(path, channels=MUSE_CHANNELS):
    """muselsl 포맷 CSV -> (T, 2) float32 (AF7, AF8)."""
    return pd.read_csv(path)[channels].values.astype(np.float32)


def sliding_windows(d, seq_len=SEQ_LEN, stride=STRIDE_INFER):
    """노트북과 동일: range(0, len(d)-seq_len, stride). 마지막 윈도우 하나 덜 잡음."""
    return np.array([d[i:i + seq_len] for i in range(0, len(d) - seq_len, stride)])
