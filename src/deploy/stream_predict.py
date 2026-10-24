"""배포용 세그먼트 단위 추론 (PDF 9~10장).

흐름: edf -> 채널 2개 -> band-pass 0.5-30 Hz -> N분 세그먼트 -> 30s epoch -> z-score -> model -> 후처리 -> JSON

주의
- band-pass 는 **학습 시 적용되지 않음**. 추론에만 적용 -> train/infer 분포 불일치. 성능 영향 미검증.
- 세그먼트마다 프로세스에서 모델을 다시 로드 (원본 설계). 느리지만 메모리 안전.
"""
import os
import json
import numpy as np
import mne
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor

from src.config import (CHANNELS, SAMPLES_PER_EPOCH, EPOCH_DURATION, DEPLOY_FILTER,
                        DEPLOY_SEGMENT_SEC, DEPLOY_NUM_CORES)
from src.deploy.postprocess import fix_isolated_n2_in_rem, truncate_after_wake_run


def preprocess_segment(segment_data, samples_per_epoch):
    """(C, T) -> (n_epochs, samples_per_epoch, C) float32"""
    n_epochs = segment_data.shape[1] // samples_per_epoch
    epochs = np.array(np.split(segment_data[:, :n_epochs * samples_per_epoch], n_epochs, axis=1))
    return np.transpose(epochs, (0, 2, 1)).astype(np.float32)


def analyze_segment_batch(segment_data_batch, mean, std, model):
    data_for_prediction = np.concatenate(segment_data_batch, axis=0)
    data_for_prediction = (data_for_prediction - mean) / std
    predictions = model.predict(data_for_prediction, verbose=0)
    return np.argmax(predictions, axis=1)


def process_segment_worker(args):
    segment_data, samples_per_epoch, mean, std, model_path = args
    model = tf.keras.models.load_model(model_path)
    preprocessed_segment = preprocess_segment(segment_data, samples_per_epoch)
    predicted_classes = analyze_segment_batch([preprocessed_segment], mean, std, model)
    return predicted_classes.tolist()


def load_raw_for_deploy(edf_file, channels=CHANNELS, apply_filter=True):
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    raw.pick(list(channels))
    if apply_filter:
        raw.filter(**DEPLOY_FILTER, verbose=False)
    return raw


def stream_predict(edf_file, model_path, mean_path, std_path,
                   segment_sec=DEPLOY_SEGMENT_SEC, num_cores=DEPLOY_NUM_CORES,
                   apply_filter=True, postprocess=True):
    """전체 밤 예측 -> list[int]. 마지막 불완전 세그먼트는 버림(원본 동작)."""
    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)

    raw = load_raw_for_deploy(edf_file, apply_filter=apply_filter)
    sfreq = raw.info['sfreq']
    samples_per_segment = int(segment_sec * sfreq)
    samples_per_epoch = int(EPOCH_DURATION * sfreq)
    total_samples = raw.n_times

    tasks = []
    for i in range(0, total_samples, samples_per_segment):
        chunk_end = min(i + samples_per_segment, total_samples)
        if (chunk_end - i) < samples_per_segment:
            continue
        segment_data = raw.get_data(start=i, stop=chunk_end)
        tasks.append((segment_data, samples_per_epoch, mean, std, model_path))

    all_predicted_classes = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for predicted_classes in executor.map(process_segment_worker, tasks):
            all_predicted_classes.extend(predicted_classes)

    if postprocess:
        all_predicted_classes = fix_isolated_n2_in_rem(all_predicted_classes)
        all_predicted_classes = truncate_after_wake_run(all_predicted_classes, run=30)
    return all_predicted_classes, float(sfreq)


def save_prediction_json(edf_file, sfreq, predicted_classes, out_dir):
    base_filename = os.path.splitext(os.path.basename(edf_file))[0]
    json_filepath = os.path.join(out_dir, f"{base_filename}.predict.json")
    json_data = {
        "file": os.path.basename(edf_file),
        "sampling_rate": sfreq,
        "total_epochs": len(predicted_classes),
        "predicted_classes": [int(v) for v in predicted_classes],
    }
    with open(json_filepath, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON 파일 저장 완료: {json_filepath}")
    return json_filepath
