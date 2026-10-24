"""CNN-BiGRU-Attention 이진(Wake/Sleep) 모델.

노트북 모델은 Keras 3.13 으로 저장 -> 3.12 환경에서 load_model 실패.
해결: 동일 아키텍처를 코드로 재구성하고 .keras 안의 model.weights.h5 만 로드.
Attention_Final 은 Lambda(reduce_sum axis=1), 나머지 둘은 GlobalAveragePooling1D.
"""
import os
import zipfile
import tempfile
import tensorflow as tf
import keras
from tensorflow.keras import layers, models
from .config import SEQ_LEN


@keras.saving.register_keras_serializable(package="muse2")
class SumPool1D(layers.Layer):
    """Attention_Final 의 Lambda(tf.reduce_sum(axis=1)) 대체. 가중치 없음, 직렬화 안전."""
    def call(self, x):
        return tf.reduce_sum(x, axis=1)

    def compute_output_shape(self, s):
        return (s[0], s[2])

POOL_BY_MODEL = {
    "MUSE2_Attention_Final": "sum",
    "MUSE_pretrained_2": "gap",
    "MUSE_activity_model": "gap",
}


def build_model(pool="gap", seq_len=SEQ_LEN, n_ch=2, lr=1e-4, compile_model=True):
    i = layers.Input((seq_len, n_ch))
    x = layers.Conv1D(32, 7, padding="same", activation="relu")(i)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Bidirectional(layers.GRU(32, return_sequences=True))(x)
    a = layers.Dense(1, activation="tanh")(x)
    a = layers.Softmax(axis=1)(a)
    x = layers.Multiply()([x, a])
    if pool == "sum":
        x = SumPool1D()(x)
    else:
        x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(16, activation="relu")(x)
    o = layers.Dense(1, activation="sigmoid")(x)
    m = models.Model(i, o)
    if compile_model:
        m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def load_muse_model(keras_path, pool=None):
    """원본 .keras(3.13) 에서 weights 만 꺼내 재구성 모델에 로드."""
    name = os.path.splitext(os.path.basename(keras_path))[0]
    pool = pool or POOL_BY_MODEL.get(name, "gap")
    m = build_model(pool, compile_model=False)
    with zipfile.ZipFile(keras_path) as z, tempfile.TemporaryDirectory() as td:
        z.extract("model.weights.h5", td)
        m.load_weights(os.path.join(td, "model.weights.h5"))
    return m
