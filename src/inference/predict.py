"""학습된 모델로 새 PSG 한 밤 전체 예측."""
import numpy as np
import mne
import tensorflow as tf

from src.config import STAGE_MAP, EPOCH_DURATION, SAMPLES_PER_EPOCH, CHANNELS


def epochs_from_raw(raw, stage_map=STAGE_MAP, epoch_duration=EPOCH_DURATION):
    """annotation 이 부착된 raw -> (X, y_true).

    y_true 는 epochs.events[:, 2] 사용 -> 드롭된 epoch 이 있어도 X 와 1:1 정렬 보장.
    (노트북은 events_true[:, 2] 를 써서 길이 불일치 시 min_length 로 잘라 밀림 가능)
    tmax=30.0 은 3001 샘플이므로 [:, :3000, :] 로 잘라 맞춤.
    """
    events_true, _ = mne.events_from_annotations(
        raw, event_id=stage_map, chunk_duration=float(epoch_duration), verbose=False
    )
    epochs = mne.Epochs(raw, events_true, tmin=0, tmax=float(epoch_duration),
                        baseline=None, preload=True, verbose=False)
    y_true = epochs.events[:, 2]
    X = epochs.get_data().transpose(0, 2, 1)[:, :SAMPLES_PER_EPOCH, :]
    return X, y_true


def prepare_epochs_for_prediction(fif_file: str, hyp_file: str,
                                  stage_map=STAGE_MAP, epoch_duration=EPOCH_DURATION):
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
    raw.set_annotations(mne.read_annotations(hyp_file))
    return epochs_from_raw(raw, stage_map, epoch_duration)


def prepare_epochs_from_edf(psg_file: str, hyp_file: str, channels=CHANNELS,
                            stage_map=STAGE_MAP, epoch_duration=EPOCH_DURATION):
    """fif 저장 없이 edf 에서 바로 epoch 생성 (일괄 평가용)."""
    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
    raw.set_annotations(mne.read_annotations(hyp_file))
    raw.pick(list(channels))
    return epochs_from_raw(raw, stage_map, epoch_duration)


def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def predict_stages(model, X: np.ndarray, verbose=0) -> np.ndarray:
    y_pred_probs = model.predict(X, verbose=verbose)
    return np.argmax(y_pred_probs, axis=1)
