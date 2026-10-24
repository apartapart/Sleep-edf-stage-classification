"""Step 7. results/cassette/*.npy + metrics.csv -> results/figures/*.png

python scripts/07_make_figures.py --in results/cassette --out results/figures
"""
import argparse
import os
import csv
import _path  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from src.evaluation.metrics import (trim_wake, merge_n3n4, cm, per_class_metrics,
                                    STAGE_NAMES_6, STAGE_NAMES_5)
from src.visualization import style as S


def load_all(in_dir):
    rows = list(csv.DictReader(open(os.path.join(in_dir, "metrics.csv"))))
    nights = {}
    for r in rows:
        n = r["night"]
        yt = np.load(os.path.join(in_dir, f"{n}_true.npy"))
        yp = np.load(os.path.join(in_dir, f"{n}_pred.npy"))
        lo, hi = trim_wake(yt)
        nights[n] = dict(full=(yt, yp), trim=(yt[lo:hi], yp[lo:hi]), lo=lo, hi=hi,
                         **{k: float(v) if k not in ("night", "subject") else v for k, v in r.items()})
    return nights


def concat(nights, key):
    yt = np.concatenate([v[key][0] for v in nights.values()])
    yp = np.concatenate([v[key][1] for v in nights.values()])
    return yt, yp


# ---------------------------------------------------------------- figures ----
def fig_confusion(yt, yp, names, title, path):
    k = len(names)
    C = cm(yt, yp, k)
    R = C / C.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, M, fmt, sub in [(axes[0], C, "{:d}", "counts"), (axes[1], R, "{:.2f}", "row-normalized (recall)")]:
        ax.grid(False)
        im = ax.imshow(M if sub != "counts" else R, cmap=S.BLUES, vmin=0, vmax=1)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels(names); ax.set_yticklabels(names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(sub, fontsize=10, color=S.TEXT2, fontweight="normal")
        for i in range(k):
            for j in range(k):
                v = M[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                        color="#ffffff" if R[i, j] > 0.55 else S.TEXT)
        for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("recall", color=S.TEXT2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.savefig(path); plt.close(fig)


def fig_kappa_per_night(nights, path, key="kappa_trim", title="Cohen's κ per night (W-trimmed, 6-class)"):
    items = sorted(nights.items(), key=lambda kv: kv[1][key])
    labels = [k for k, _ in items]; vals = np.array([v[key] for _, v in items])
    mean = vals.mean()
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.barh(labels, vals, color=S.CAT[0], height=0.62)
    ax.axvline(mean, color=S.TEXT2, lw=1.2, ls="--")
    ax.text(mean + 0.01, -0.9, f"mean {mean:.2f}", color=S.TEXT2, va="center", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=S.TEXT2)
    ax.set_xlim(0, 1); ax.set_xlabel("κ"); ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    fig.savefig(path); plt.close(fig)


def fig_per_class(yt, yp, names, path, title):
    m = per_class_metrics(yt, yp, len(names))
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (lab, key) in enumerate([("precision", "precision"), ("recall", "recall"), ("F1", "f1")]):
        bars = ax.bar(x + (i - 1) * (w + 0.02), m[key], width=w, color=S.CAT[i], label=lab)
        for b, v in zip(bars, m[key]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=S.TEXT2)
    sup = m["support"]
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n(n={s:,})" for n, s in zip(names, sup)])
    ax.set_ylim(0, 1.08); ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper right", fontsize=9)
    ax.set_title(title, loc="left")
    fig.savefig(path); plt.close(fig)


def fig_stage_proportion(yt, yp, names, path):
    k = len(names)
    pt = np.bincount(yt, minlength=k) / len(yt); pp = np.bincount(yp, minlength=k) / len(yp)
    x = np.arange(k); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (vals, lab) in enumerate([(pt, "ground truth"), (pp, "predicted")]):
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, width=w, color=S.CAT[i], label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v*100:.1f}%", ha="center", fontsize=8, color=S.TEXT2)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("share of epochs")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.grid(axis="x", visible=False); ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Stage distribution: ground truth vs predicted (28 nights, W-trimmed)", loc="left")
    fig.savefig(path); plt.close(fig)


def fig_metrics_comparison(nights, path):
    conds = [("full\n(all epochs)", "_full"), ("W-trimmed\n6-class", "_trim"), ("W-trimmed\n5-class (N3+N4)", "_5c")]
    mets = [("accuracy", "acc"), ("macro-F1", "mf1"), ("κ", "kappa")]
    x = np.arange(len(conds)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (lab, mk) in enumerate(mets):
        vals = np.array([[v[mk + suf] for v in nights.values()] for _, suf in conds])
        mu, sd = vals.mean(1), vals.std(1)
        bars = ax.bar(x + (i - 1) * (w + 0.02), mu, width=w, color=S.CAT[i], label=lab,
                      yerr=sd, error_kw=dict(ecolor=S.TEXT2, lw=1, capsize=3))
        for b, m in zip(bars, mu):
            ax.text(b.get_x() + b.get_width() / 2, 0.03, f"{m:.2f}", ha="center", fontsize=8, color="#ffffff")
    ax.set_xticks(x); ax.set_xticklabels([c for c, _ in conds]); ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False); ax.legend(ncol=3, loc="upper right", fontsize=9)
    ax.set_title("Evaluation protocol matters: mean ± std over 28 nights", loc="left")
    fig.savefig(path); plt.close(fig)


def fig_w_inflation(nights, path):
    wr = np.array([v["w_ratio_full"] for v in nights.values()])
    inf = np.array([v["acc_full"] - v["acc_trim"] for v in nights.values()])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(wr, inf, s=42, color=S.CAT[0], edgecolor=S.SURFACE, linewidth=1.5)
    for n, a, b in zip(nights.keys(), wr, inf):
        if b > 0.22 or b < 0.05:
            ax.annotate(n, (a, b), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=S.TEXT2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:+.0f}pt"))
    ax.set_xlabel("share of W epochs in full recording"); ax.set_ylabel("accuracy gain from untrimmed W")
    ax.set_title("Untrimmed wake inflates accuracy", loc="left")
    fig.savefig(path); plt.close(fig)


def _hypno_axis(ax, yt, yp, title):
    t = np.arange(len(yt)) * 0.5  # minutes
    ax.step(t, yt, where="post", color=S.CAT[0], lw=2.0, label="ground truth")
    ax.step(t, yp, where="post", color=S.CAT[1], lw=1.0, label="predicted", alpha=0.95)
    dis = np.where(yt != yp)[0]
    ax.vlines(t[dis], -1.2, -0.8, color=S.MISMATCH, lw=0.6, alpha=0.6)
    ax.set_yticks([-1, 0, 1, 2, 3, 4, 5]); ax.set_yticklabels(["mismatch", "W", "N1", "N2", "N3", "N4", "R"])
    ax.set_ylim(-1.5, 5.5); ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left"); ax.set_xlabel("time since trimmed start (min)")


def fig_hypnogram(nights, name, path):
    yt, yp = nights[name]["trim"]
    fig, ax = plt.subplots(figsize=(14, 3.8))
    _hypno_axis(ax, yt, yp, f"{name}: hypnogram, W-trimmed (acc {nights[name]['acc_trim']:.2f}, κ {nights[name]['kappa_trim']:.2f})")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.savefig(path); plt.close(fig)


def fig_best_worst(nights, path, key="kappa_trim"):
    order = sorted(nights, key=lambda n: nights[n][key])
    picks = [order[-1], order[len(order) // 2], order[0]]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    for ax, n, tag in zip(axes, picks, ["best", "median", "worst"]):
        yt, yp = nights[n]["trim"]
        _hypno_axis(ax, yt, yp, f"{tag}: {n}  (acc {nights[n]['acc_trim']:.2f}, κ {nights[n][key]:.2f})")
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", default="results/cassette")
    p.add_argument("--out", default="results/figures")
    p.add_argument("--single", default="SC4001E0")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    S.apply()
    nights = load_all(args.in_dir)
    yt6, yp6 = concat(nights, "trim")
    yt5, yp5 = merge_n3n4(yt6), merge_n3n4(yp6)
    ytf, ypf = concat(nights, "full")
    o = lambda f: os.path.join(args.out, f)

    fig_confusion(yt6, yp6, STAGE_NAMES_6, "Confusion matrix: 28 cassette nights, W-trimmed, 6-class (R&K)", o("cm_6class_trimmed.png"))
    fig_confusion(yt5, yp5, STAGE_NAMES_5, "Confusion matrix: 28 cassette nights, W-trimmed, 5-class (AASM, N3+N4)", o("cm_5class_trimmed.png"))
    fig_confusion(ytf, ypf, STAGE_NAMES_6, "Confusion matrix: 28 cassette nights, full recording (untrimmed W)", o("cm_6class_full.png"))
    st, sp = nights[args.single]["trim"]
    fig_confusion(st, sp, STAGE_NAMES_6, f"Confusion matrix: {args.single}, W-trimmed, 6-class", o(f"cm_{args.single}_trimmed.png"))
    fig_kappa_per_night(nights, o("kappa_per_night.png"))
    fig_per_class(yt6, yp6, STAGE_NAMES_6, o("per_class_prf_6class.png"), "Per-class precision / recall / F1 (28 nights, W-trimmed, 6-class)")
    fig_per_class(yt5, yp5, STAGE_NAMES_5, o("per_class_prf_5class.png"), "Per-class precision / recall / F1 (28 nights, W-trimmed, 5-class)")
    fig_stage_proportion(yt6, yp6, STAGE_NAMES_6, o("stage_proportion.png"))
    fig_metrics_comparison(nights, o("metrics_by_protocol.png"))
    fig_w_inflation(nights, o("w_inflation.png"))
    fig_hypnogram(nights, args.single, o(f"hypnogram_{args.single}.png"))
    fig_best_worst(nights, o("hypnogram_best_median_worst.png"))
    print("figures written to", args.out)
    for f in sorted(os.listdir(args.out)):
        print(" ", f)


if __name__ == "__main__":
    main()
