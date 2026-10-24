"""Step 5. 학습 제외 피험자(ST7242J0) 한 밤 전체 예측 + 하이프노그램/혼동행렬.

python scripts/05_predict.py --fif ./data/fif/ST7242J0-PSG-raw.fif --hyp ...-Hypnogram.edf \
    --model ./artifacts/model_4.keras --mean ./artifacts/mean.npy --std ./artifacts/std.npy
"""
import argparse
import os
from collections import Counter
import _path  # noqa: F401

from src.config import RAW_EDF_DIR, STATS_DIR, MODEL_DIR
from src.normalization.zscore import load_stats, apply_zscore
from src.inference.predict import (prepare_epochs_for_prediction, load_model,
                                   predict_stages)
from src.visualization.hypnogram_compare import plot_hypnogram_comparison
from src.visualization.confusion import plot_confusion_sklearn, print_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fif", default="/content/ST7242J0-PSG-raw.fif")
    p.add_argument("--hyp", default=f"{RAW_EDF_DIR}/ST7242JO-Hypnogram.edf")
    p.add_argument("--model", default=os.path.join(MODEL_DIR, "model_4.keras"))
    p.add_argument("--mean", default=os.path.join(STATS_DIR, "mean.npy"))
    p.add_argument("--std", default=os.path.join(STATS_DIR, "std.npy"))
    p.add_argument("--fig-dir", default="results/figures")
    args = p.parse_args()

    X_predict, y_true = prepare_epochs_for_prediction(args.fif, args.hyp)

    print("\n데이터 정규화를 시작합니다...")
    try:
        trained_mean, trained_std = load_stats(args.mean, args.std)
        X_predict = apply_zscore(X_predict, trained_mean, trained_std, eps=1e-7)
        print("데이터 정규화 완료.")
    except FileNotFoundError:
        print("오류: 훈련 데이터의 평균과 표준편차 파일(.npy)을 찾을 수 없습니다.")
        return

    model = load_model(args.model)
    y_pred = predict_stages(model, X_predict)

    print("실제 레이블 수:", Counter(y_true))
    print("예측 레이블 수:", Counter(y_pred))
    print(f"y_true 길이: {len(y_true)}")
    print(f"y_pred 길이: {len(y_pred)}")

    os.makedirs(args.fig_dir, exist_ok=True)
    tag = os.path.basename(args.fif).replace("-raw.fif", "")
    plot_hypnogram_comparison(y_true, y_pred,
                              save_path=os.path.join(args.fig_dir, f"{tag}_hypnogram_compare.png"))
    plot_confusion_sklearn(y_true, y_pred,
                           save_path=os.path.join(args.fig_dir, f"{tag}_confusion.png"))
    print_report(y_true, y_pred)


if __name__ == "__main__":
    main()
