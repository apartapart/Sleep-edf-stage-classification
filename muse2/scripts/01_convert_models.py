"""원본 .keras(Keras 3.13) -> 현재 환경 포맷으로 재저장 + 동치 확인.
python muse2/scripts/01_convert_models.py
"""
import os, glob, numpy as np, _path  # noqa
import tensorflow as tf
from msrc.model import load_muse_model

src_dir = "muse2/artifacts/original"; out_dir = "muse2/artifacts"
x = np.random.default_rng(0).standard_normal((8, 1280, 2)).astype("float32")
for p in sorted(glob.glob(f"{src_dir}/*.keras")):
    name = os.path.basename(p)
    m = load_muse_model(p)
    out = os.path.join(out_dir, name); m.save(out)
    m2 = tf.keras.models.load_model(out, compile=False, safe_mode=False)
    d = np.abs(m.predict(x, verbose=0) - m2.predict(x, verbose=0)).max()
    print(f"{name}: params={m.count_params()} resaved -> {out} | reload max|diff|={d:.2e} | out range on noise {m.predict(x, verbose=0).min():.3f}~{m.predict(x, verbose=0).max():.3f}")
