"""Step 6. cassette 28밤 전부 model_4 로 추론 -> per-night 예측 저장 + 지표 CSV.

python scripts/06_eval_cassette.py --raw-dir /path/sleep-cassette --model model_4.keras --mean mean.npy --std std.npy
출력: results/cassette/{night}_true.npy, {night}_pred.npy, results/cassette/metrics.csv
"""
import argparse
import os
import csv
import _path  # noqa: F401
import numpy as np

from src.data.cassette import pair_cassette_files, subject_of
from src.normalization.zscore import load_stats, apply_zscore
from src.inference.predict import prepare_epochs_from_edf, load_model, predict_stages
from src.evaluation.metrics import trim_wake, merge_n3n4, summary_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mean", required=True)
    p.add_argument("--std", required=True)
    p.add_argument("--out", default="results/cassette")
    p.add_argument("--pad", type=int, default=60, help="W trimming pad (epochs)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.model)
    mean, std = load_stats(args.mean, args.std)
    rows = []
    for night, psg, hyp in pair_cassette_files(args.raw_dir):
        X, y_true = prepare_epochs_from_edf(psg, hyp)
        X = apply_zscore(X, mean, std, eps=1e-7)
        y_pred = predict_stages(model, X)
        np.save(os.path.join(args.out, f"{night}_true.npy"), y_true)
        np.save(os.path.join(args.out, f"{night}_pred.npy"), y_pred)

        lo, hi = trim_wake(y_true, args.pad)
        full = summary_metrics(y_true, y_pred)
        trim = summary_metrics(y_true[lo:hi], y_pred[lo:hi])
        five = summary_metrics(merge_n3n4(y_true[lo:hi]), merge_n3n4(y_pred[lo:hi]))
        row = {"night": night, "subject": subject_of(night),
               "n_full": full["n"], "w_ratio_full": float(np.mean(y_true == 0)),
               "acc_full": full["acc"], "kappa_full": full["kappa"], "mf1_full": full["macro_f1"],
               "n_trim": trim["n"], "acc_trim": trim["acc"], "kappa_trim": trim["kappa"], "mf1_trim": trim["macro_f1"],
               "acc_5c": five["acc"], "kappa_5c": five["kappa"], "mf1_5c": five["macro_f1"]}
        rows.append(row)
        print(f"{night}  trim acc={trim['acc']:.3f} κ={trim['kappa']:.3f} mF1={trim['macro_f1']:.3f} | 5c κ={five['kappa']:.3f}")

    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    for k in ["acc_full", "kappa_full", "acc_trim", "kappa_trim", "mf1_trim", "acc_5c", "kappa_5c", "mf1_5c"]:
        v = np.array([r[k] for r in rows])
        print(f"{k:12s} mean={v.mean():.3f} ± {v.std():.3f}")


if __name__ == "__main__":
    main()
