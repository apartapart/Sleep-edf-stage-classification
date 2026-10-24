"""Muse CSV 5개 × 모델 3개 추론 -> 확률 타임라인, 출력 분포, 혼동행렬(파일명 기반 라벨).
python muse2/scripts/02_infer_csv.py
"""
import os, glob, json, hashlib, numpy as np, _path  # noqa
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score
from msrc.model import load_muse_model
from msrc.preprocess import preprocess
from msrc.windows import load_muse_csv, sliding_windows
from msrc.postprocess import hysteresis, minmax_scale
from msrc.config import BEST_THRESHOLD
from src.visualization import style as S  # type: ignore
S.apply()

DATA = "muse2/data"; FIG = "muse2/results/figures"; OUT = "muse2/results"
os.makedirs(FIG, exist_ok=True)
files = ["awake_study.csv", "awake_study2.csv", "awake_study3.csv", "bedtime.csv", "bedtime2.csv"]
models = {n: load_muse_model(f"muse2/artifacts/original/{n}.keras") for n in
          ["MUSE2_Attention_Final", "MUSE_pretrained_2", "MUSE_activity_model"]}

# 파일 중복 검사 (awake_study / bedtime 행 수 동일)
print("md5:", {f: hashlib.md5(open(f"{DATA}/{f}", "rb").read()).hexdigest()[:10] for f in files})

preds = {}  # (model, file) -> probs
for f in files:
    d = preprocess(load_muse_csv(f"{DATA}/{f}"))
    W = sliding_windows(d)
    for n, m in models.items():
        preds[(n, f)] = m.predict(W, batch_size=256, verbose=0).ravel()
    print(f, "windows:", len(W))
np.save(f"{OUT}/csv_preds.npy", {k: v for k, v in preds.items()}, allow_pickle=True)

# --- fig 1: 모델별 출력 분포 (상수 출력 진단)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
for ax, n in zip(axes, models):
    for i, f in enumerate(files):
        ax.hist(preds[(n, f)], bins=60, range=(0, 1), histtype="step", lw=1.4, color=S.CAT[i], label=f.replace(".csv", ""))
    ax.set_title(n, fontsize=10); ax.set_xlabel("P(Awake)"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("windows"); axes[-1].legend(fontsize=7, loc="upper right")
fig.suptitle("Output distribution per model (5 Muse recordings, 5 s windows, 1 s stride)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{FIG}/output_distribution.png"); plt.close(fig)

# --- fig 2: 파일별 타임라인 (3 모델 raw prob)
fig, axes = plt.subplots(len(files), 1, figsize=(14, 2.2 * len(files)), sharex=False)
for ax, f in zip(axes, files):
    for i, n in enumerate(models):
        p = preds[(n, f)]; ax.plot(np.arange(len(p)) / 60, p, lw=1.0, color=S.CAT[i], label=n, alpha=0.9)
    ax.axhline(0.5, color=S.TEXT2, lw=0.8, ls="--"); ax.set_ylim(0, 1); ax.set_ylabel("P(Awake)")
    ax.set_title(f, loc="left", fontsize=10); ax.grid(axis="x", visible=False)
axes[-1].set_xlabel("time (min)"); axes[0].legend(ncol=3, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(f"{FIG}/timeline_raw_prob.png"); plt.close(fig)

# --- 평가: 파일명 라벨 (awake=1, bedtime=0) × 3 후처리 방식
label_of = lambda f: 1 if "awake" in f else 0
rows = []
def evaluate(n, mode):
    yt, yp = [], []
    for f in files:
        p = preds[(n, f)]
        if mode == "thr0.5": s = (p > 0.5).astype(int)
        elif mode == "thr_best": s = (p > BEST_THRESHOLD).astype(int)
        elif mode == "hysteresis": s = hysteresis(p)
        elif mode == "minmax+hyst": s = hysteresis(minmax_scale(p))
        yt += [label_of(f)] * len(s); yp += list(s)
    yt, yp = np.array(yt), np.array(yp)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    rows.append(dict(model=n, mode=mode, acc=accuracy_score(yt, yp), kappa=cohen_kappa_score(yt, yp),
                     awake_recall=cm[1, 1] / cm[1].sum(), sleep_recall=cm[0, 0] / cm[0].sum()))
    return cm
cms = {}
for n in models:
    for mode in ["thr0.5", "thr_best", "hysteresis", "minmax+hyst"]:
        cms[(n, mode)] = evaluate(n, mode)
import csv
with open(f"{OUT}/csv_eval.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print(f"{r['model']:24s} {r['mode']:12s} acc={r['acc']:.3f} κ={r['kappa']:+.3f} awake_rec={r['awake_recall']:.2f} sleep_rec={r['sleep_recall']:.2f}")

# --- fig 3: 혼동행렬 그리드 (3 모델 × 2 모드)
modes = ["thr0.5", "minmax+hyst"]
fig, axes = plt.subplots(len(modes), 3, figsize=(11, 6.4))
for r_, mode in enumerate(modes):
    for c_, n in enumerate(models):
        ax = axes[r_, c_]; cm = cms[(n, mode)]; R = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(R, cmap=S.BLUES, vmin=0, vmax=1); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({R[i,j]:.2f})", ha="center", va="center", fontsize=9, color="#fff" if R[i,j] > 0.55 else S.TEXT)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Sleep", "Awake"]); ax.set_yticklabels(["Sleep", "Awake"])
        ax.set_title(f"{n}\n{mode}", fontsize=9); ax.set_xlabel("Predicted"); ax.set_ylabel("True (filename)")
        for s_ in ax.spines.values(): s_.set_visible(False)
fig.suptitle("Confusion matrices: label = filename (awake_* → Awake, bedtime* → Sleep)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{FIG}/confusion_grid.png"); plt.close(fig)

# --- fig 4: 노트북 v2 재현 (activity 모델, minmax + hysteresis) 타임라인
n = "MUSE_activity_model"
fig, axes = plt.subplots(len(files), 1, figsize=(14, 2.0 * len(files)))
for ax, f in zip(axes, files):
    p = preds[(n, f)]; ps = minmax_scale(p); st = hysteresis(ps); t = np.arange(len(p)) / 60
    ax.plot(t, ps, lw=0.8, color=S.CAT[0], alpha=0.6, label="min-max scaled prob")
    ax.step(t, st, lw=1.6, color=S.CAT[1], label="state (hysteresis)")
    ax.fill_between(t, 0, 1, where=(st == 0), color=S.TEXT2, alpha=0.08, step="pre")
    ax.set_ylim(-0.05, 1.05); ax.set_title(f"{f}  (true label by filename: {'Awake' if label_of(f) else 'Sleep'})", loc="left", fontsize=10); ax.grid(axis="x", visible=False)
axes[0].legend(ncol=2, fontsize=8, loc="upper right"); axes[-1].set_xlabel("time (min)")
fig.suptitle("Notebook v2 reproduction: MUSE_activity_model, per-file min-max + hysteresis", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{FIG}/timeline_activity_minmax.png"); plt.close(fig)
print("figures ->", FIG)
