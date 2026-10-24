"""Step 4. 피험자 단위 split -> SMOTE(train) -> z-score(train 통계) -> CNN-BiLSTM 학습 -> 평가.

python scripts/04_train.py --merge-dir ./data/npy --stats-dir ./artifacts --model-out ./artifacts/model.keras
옵션:
  --no-smote          SMOTE 생략
  --class-weight      balanced class_weight 사용 (노트북 최종 설정은 미사용)
  --smooth 5          테스트 예측 smoothing window (노트북 기본 5)
"""
import argparse
import os
from collections import Counter
import _path  # noqa: F401

from src.config import (MERGE_DIR, STATS_DIR, MODEL_DIR, LEARNING_RATE, EPOCHS,
                        BATCH_SIZE, ES_PATIENCE)
from src.data.subjects import first_path_list
from src.data.split import split_subjects
from src.data.loader import load_and_concatenate
from src.imbalance.smote import apply_smote
from src.imbalance.class_weight import balanced_class_weight
from src.normalization.zscore import fit_zscore, apply_zscore, save_stats
from src.models.cnn_bilstm import build_cnn_bilstm
from src.training.train import compile_model, train_model, evaluate_model
from src.inference.predict import predict_stages
from src.inference.smoothing import smooth_predictions
from src.visualization.confusion import plot_confusion_seaborn, print_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--merge-dir", default=MERGE_DIR)
    p.add_argument("--stats-dir", default=STATS_DIR)
    p.add_argument("--model-out", default=os.path.join(MODEL_DIR, "model_5.keras"))
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--patience", type=int, default=ES_PATIENCE)
    p.add_argument("--no-smote", action="store_true")
    p.add_argument("--class-weight", action="store_true")
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--fig-dir", default="results/figures")
    args = p.parse_args()

    # 1) subject-level split
    train_ids, val_ids, test_ids = split_subjects(first_path_list)
    print("train:", train_ids, "\nval:", val_ids, "\ntest:", test_ids)

    X_train, y_train = load_and_concatenate(train_ids, args.merge_dir)
    X_val, y_val = load_and_concatenate(val_ids, args.merge_dir)
    X_test, y_test = load_and_concatenate(test_ids, args.merge_dir)
    print("훈련 데이터 클래스 분포 (원본):", Counter(y_train))

    # 2) imbalance
    if not args.no_smote:
        X_train, y_train = apply_smote(X_train, y_train)
        print("훈련 데이터 클래스 분포 (SMOTE):", Counter(y_train))
    class_weight = balanced_class_weight(y_train) if args.class_weight else None

    # 3) normalization (train 통계만 사용)
    mean, std = fit_zscore(X_train)
    X_train = apply_zscore(X_train, mean, std)
    X_val = apply_zscore(X_val, mean, std)
    X_test = apply_zscore(X_test, mean, std)
    save_stats(mean, std, args.stats_dir)

    # 4) model
    model = build_cnn_bilstm(input_shape=X_train.shape[1:])
    compile_model(model, learning_rate=args.lr)
    train_model(model, X_train, y_train, X_val, y_val,
                epochs=args.epochs, batch_size=args.batch_size,
                patience=args.patience, class_weight=class_weight)
    model.summary()

    # 5) evaluate
    evaluate_model(model, X_test, y_test)
    y_pred = predict_stages(model, X_test)
    y_pred_smoothed = smooth_predictions(y_pred, window_size=args.smooth)

    os.makedirs(args.fig_dir, exist_ok=True)
    plot_confusion_seaborn(y_test, y_pred_smoothed, title='Confusion Matrix (Smoothed)',
                           save_path=os.path.join(args.fig_dir, "train_confusion_smoothed.png"))
    print_report(y_test, y_pred_smoothed, header="Classification Report (Smoothed Predictions):")

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    model.save(args.model_out)
    print("saved:", args.model_out)


if __name__ == "__main__":
    main()
