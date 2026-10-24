"""Activity pseudo-label 로 fine-tune (노트북 재현용 참고 스크립트). 성능 향상 목적 아님.
python muse2/scripts/04_finetune_activity.py --epochs 20
"""
import os, argparse, random, numpy as np, _path  # noqa
import tensorflow as tf
from msrc.model import load_muse_model
from msrc.activity_label import thresholds, load_and_label, augment
from msrc.config import SEQ_LEN

p = argparse.ArgumentParser()
p.add_argument("--base", default="muse2/artifacts/original/MUSE_pretrained_2.keras")
p.add_argument("--out", default="muse2/artifacts/MUSE_activity_model_retrained.keras")
p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch", type=int, default=32)
a = p.parse_args()
train_files = ["muse2/data/bedtime2.csv", "muse2/data/awake_study2.csv"]; val_files = ["muse2/data/awake_study3.csv"]
low_thr, high_thr = thresholds(train_files); print("activity thr:", low_thr, high_thr)

def gen(files):
    while True:
        random.shuffle(files)
        for f in files:
            X, Y = load_and_label(f, low_thr, high_thr)
            if len(X) < a.batch: continue
            idx = np.random.permutation(len(X)); X, Y = X[idx], Y[idx]
            for i in range(0, len(X) - a.batch, a.batch):
                bx = X[i:i + a.batch]
                if np.random.rand() < 0.5: bx = augment(bx)
                yield bx.astype(np.float32), Y[i:i + a.batch].astype(np.float32)
sig = (tf.TensorSpec((a.batch, SEQ_LEN, 2), tf.float32), tf.TensorSpec((a.batch,), tf.float32))
tr = tf.data.Dataset.from_generator(lambda: gen(list(train_files)), output_signature=sig).prefetch(tf.data.AUTOTUNE)
va = tf.data.Dataset.from_generator(lambda: gen(list(val_files)), output_signature=sig).prefetch(tf.data.AUTOTUNE)
m = load_muse_model(a.base); m.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
m.fit(tr, validation_data=va, steps_per_epoch=25, validation_steps=10, epochs=a.epochs,
      callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)])
m.save(a.out); print("saved", a.out)
