"""Hypnogram annotation -> 샘플 단위 라벨 배열."""
import numpy as np
import mne

from src.config import STAGE_MAP, UNDEFINED_LABEL


def load_fif_with_annotations(fif_file: str, hyp_file: str):
    """fif 로드 + hypnogram annotation 부착. (raw, annotations) 반환."""
    raw = mne.io.read_raw_fif(fif_file, preload=True)
    annotations = mne.read_annotations(hyp_file)
    raw.set_annotations(annotations)
    return raw, annotations


def build_sample_labels(annotations, sfreq: float, n_samples: int,
                        stage_map=STAGE_MAP, undefined_value: int = UNDEFINED_LABEL) -> np.ndarray:
    """annotation(onset, duration, description) 을 샘플 단위 int 라벨로 펼침.

    매핑되지 않는 description(Movement, '?') ->
      undefined_value=-1 : predict/시각화 노트북 방식 (기본)
      undefined_value=0  : 학습용 npy 생성(PDF 4-6) 방식. W 로 취급
    """
    labels = np.zeros(n_samples, dtype=int)
    for desc, onset, duration in zip(annotations.description,
                                     annotations.onset,
                                     annotations.duration):
        stage = stage_map.get(desc, undefined_value)
        start_sample = int(onset * sfreq)
        end_sample = int((onset + duration) * sfreq)
        end_sample = min(end_sample, n_samples)
        labels[start_sample:end_sample] = stage
    return labels


def print_label_counts(labels: np.ndarray) -> None:
    """수면 단계별 샘플 개수 출력."""
    from collections import Counter

    label_counts = Counter(labels)
    sorted_counts = dict(sorted(label_counts.items()))

    print("각 수면 단계의 샘플 개수 (샘플 수):")
    for stage, count in sorted_counts.items():
        if stage == UNDEFINED_LABEL:
            print(f"Movement/Undefined: {count} samples")
        else:
            print(f"Stage {stage}: {count} samples")
