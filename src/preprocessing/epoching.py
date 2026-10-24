"""연속 신호 -> 30s epoch 분할 -> (X, Y) npy 저장."""
import os
import numpy as np

from src.config import EPOCH_DURATION, UNDEFINED_LABEL


def make_epochs(data: np.ndarray, labels: np.ndarray, sfreq: float,
                epoch_duration: int = EPOCH_DURATION, skip_undefined: bool = True):
    """data: (n_samples, n_channels), labels: (n_samples,)

    - epoch 길이가 모자란 마지막 구간은 버림
    - skip_undefined=True : epoch 내 UNDEFINED_LABEL(-1) 하나라도 있으면 버림 (predict 노트북 방식)
    - skip_undefined=False: 버리지 않음. 학습용 npy 생성(PDF 4-6) 방식 ->
        이때 labeling 은 build_sample_labels(..., undefined_value=0) 로 undefined 를 W 로 매핑해야 동일
    - epoch 라벨 = 최빈값(mode)
    반환: X (n_epochs, samples_per_epoch, n_channels), Y (n_epochs,)
    """
    n_samples = data.shape[0]
    samples_per_epoch = int(sfreq * epoch_duration)
    X_list, Y_list = [], []

    for i in range(0, n_samples, samples_per_epoch):
        if i + samples_per_epoch <= n_samples:
            X_epoch = data[i:i + samples_per_epoch]
            Y_epoch_labels = labels[i:i + samples_per_epoch]
            if skip_undefined and np.any(Y_epoch_labels == UNDEFINED_LABEL):
                continue

            # 최빈값(mode)을 레이블로 사용
            Y_epoch = np.bincount(Y_epoch_labels).argmax()

            X_list.append(X_epoch)
            Y_list.append(Y_epoch)

    X_npy = np.array(X_list)
    Y_npy = np.array(Y_list)
    return X_npy, Y_npy


def fif_to_file_id(fif_file: str) -> str:
    """'ST7242J0-PSG-raw.fif' -> '242J0-PSG'  (노트북 명명 규칙 유지)."""
    return os.path.basename(fif_file).replace("ST7", "").replace("-raw.fif", "")


def save_npy(X: np.ndarray, Y: np.ndarray, fif_file: str, out_dir: str):
    """X_ST7{id}.npy / Y_ST7{id}.npy 로 저장. (x_path, y_path) 반환."""
    file_id = fif_to_file_id(fif_file)

    x_output_path = os.path.join(out_dir, f"X_ST7{file_id}.npy")
    y_output_path = os.path.join(out_dir, f"Y_ST7{file_id}.npy")

    np.save(x_output_path, X)
    np.save(y_output_path, Y)

    print(f"X 데이터가 성공적으로 저장되었습니다: {x_output_path}")
    print(f"Y 데이터가 성공적으로 저장되었습니다: {y_output_path}")
    return x_output_path, y_output_path
