"""CNN + BiLSTM 수면 단계 분류기.

Conv1D x2 (local feature) -> MaxPooling1D (downsample, noise 억제)
-> BiLSTM x2 (temporal dependency) -> Dense -> softmax(6).
Dropout 0.5 / BatchNorm 으로 소규모 데이터 overfitting 방지.
"""
from tensorflow.keras import layers, models

from src.config import INPUT_SHAPE, N_CLASSES


def build_cnn_bilstm(input_shape=INPUT_SHAPE, n_classes=N_CLASSES):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.5),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.5),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model
