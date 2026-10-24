"""기초 모델(MUSE2_Attention_Final, Sleep-EDF 95%) 을 cassette 28밤에 외부 검증 (Wake/Sleep 이진).
python muse2/scripts/03_eval_base_on_cassette.py --raw-dir /path/sleep-cassette
"""
import os, argparse, csv, numpy as np, _path  # noqa
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix
from msrc.model import load_muse_model
from msrc.sleepedf_binary import edf_to_binary_windows
from msrc.config import BEST_THRESHOLD
from src.data.cassette import pair_cassette_files  # type: ignore  (상위 레포 모듈)
from src.visualization import style as S  # type: ignore
S.apply()

p = argparse.ArgumentParser(); p.add_argument("--raw-dir", required=True); p.add_argument("--out", default="muse2/results")
a = p.parse_args(); os.makedirs(f"{a.out}/figures", exist_ok=True)
m = load_muse_model("muse2/artifacts/original/MUSE2_Attention_Final.keras")
rows, YT, YP = [], [], []
for night, psg, hyp in pair_cassette_files(a.raw_dir):
    X, y = edf_to_binary_windows(psg, hyp)
    pr = m.predict(X, batch_size=512, verbose=0).ravel()
    r = dict(night=night, n=len(y), wake_ratio=float(y.mean()))
    for tag, thr in [("05", 0.5), ("best", BEST_THRESHOLD)]:
        yp = (pr > thr).astype(int)
        r[f"acc_{tag}"] = accuracy_score(y, yp); r[f"kappa_{tag}"] = cohen_kappa_score(y, yp); r[f"f1_{tag}"] = f1_score(y, yp)
    rows.append(r); YT.append(y); YP.append(pr)
    print(f"{night} n={len(y)} wake={y.mean():.2f} | thr0.5 acc={r['acc_05']:.3f} κ={r['kappa_05']:.3f} | thr{BEST_THRESHOLD} acc={r['acc_best']:.3f} κ={r['kappa_best']:.3f}")
with open(f"{a.out}/cassette_binary_metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for k in ["acc_05", "kappa_05", "f1_05", "acc_best", "kappa_best", "f1_best"]:
    v = np.array([r[k] for r in rows]); print(f"{k:11s} {v.mean():.3f} ± {v.std():.3f}")
yt, pr = np.concatenate(YT), np.concatenate(YP)
np.save(f"{a.out}/cassette_binary_preds.npy", np.stack([yt, pr]))

# fig: 밤별 κ + 합산 혼동행렬 (thr 0.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw=dict(width_ratios=[1.6, 1]))
items = sorted(rows, key=lambda r: r["kappa_05"]); ax1.barh([r["night"] for r in items], [r["kappa_05"] for r in items], color=S.CAT[0], height=0.6)
mu = np.mean([r["kappa_05"] for r in rows]); ax1.axvline(mu, ls="--", color=S.TEXT2); ax1.text(mu + 0.01, -0.9, f"mean {mu:.2f}", color=S.TEXT2, fontsize=9)
ax1.set_xlim(-0.1, 1); ax1.set_xlabel("κ (Wake vs Sleep, thr 0.5)"); ax1.grid(axis="y", visible=False); ax1.set_title("Per-night κ", loc="left")
cm = confusion_matrix(yt, (pr > 0.5).astype(int)); R = cm / cm.sum(1, keepdims=True)
ax2.imshow(R, cmap=S.BLUES, vmin=0, vmax=1); ax2.grid(False)
for i in range(2):
    for j in range(2): ax2.text(j, i, f"{cm[i,j]:,}\n({R[i,j]:.2f})", ha="center", va="center", color="#fff" if R[i,j] > 0.55 else S.TEXT)
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1]); ax2.set_xticklabels(["Sleep", "Awake"]); ax2.set_yticklabels(["Sleep", "Awake"]); ax2.set_xlabel("Predicted"); ax2.set_ylabel("True"); ax2.set_title("Pooled confusion (28 nights)", loc="left")
for s_ in ax2.spines.values(): s_.set_visible(False)
fig.suptitle("MUSE2_Attention_Final (trained on Sleep-EDF telemetry 256 Hz) → cassette external validation", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{a.out}/figures/cassette_binary_eval.png"); plt.close(fig)
