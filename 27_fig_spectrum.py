"""Figures for the de-bundled factorial and the learner-plasticity spectrum.

Outputs to results/figures/:
  fig6_factorial.png / .pdf   MLP 2x2 (buffer x fine-tune), AUC + F1 panels
  fig7_spectrum.png  / .pdf   Reset-Harm Index by learner family
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight",
})

PAIR_RHI = [
    ("MLP-32/16\n(non-convex)", "plain", "drift_b2000_ft0"),
    ("MLP-64/32/16\n(non-convex)", "mlp_deep", "mlp_deep_drift"),
    ("SGD\n(convex)", "sgd", "sgd_drift"),
    ("logreg\n(convex)", "logreg", "logreg_drift"),
    ("HT\n(tree)", "ht_plain", "ht_drift"),
    ("HAT\n(tree)", "hat_plain", "hat_drift"),
]


def load():
    df = pd.read_csv(os.path.join(RES, "spectrum_long.csv"))
    return df


def blocks(df, cell):
    return (df[df["cell"] == cell]
            .groupby(["dataset", "scenario", "seed"])[["auc", "f1"]].mean())


def paired(df, a, b, col="auc"):
    A, B = blocks(df, a), blocks(df, b)
    m = A[[col]].rename(columns={col: "a"}).join(
        B[[col]].rename(columns={col: "b"}), how="inner").dropna()
    if m.empty:
        return None
    m = m[m["a"].notna() & m["b"].notna()]
    diff = (m["b"] - m["a"])
    p = np.nan
    if len(m) >= 6 and diff.abs().sum() > 0:
        p = stats.wilcoxon(m["a"], m["b"])[1]
    return dict(n=len(m), a=m["a"].mean(), b=m["b"].mean(),
                diff=diff.mean(), p=p)


def fig_factorial(df):
    cells = [("plain", "plain\n(no reset)"),
             ("drift_b512_ft0", "reset\nbuf512\nno FT"),
             ("drift_b2000_ft0", "reset\nbuf2000\nno FT"),
             ("drift_b512_ft3", "reset\nbuf512\nFT"),
             ("drift_b2000_ft3", "reset\nbuf2000\nFT")]
    auc_m, auc_e, f1_m, f1_e, labs = [], [], [], [], []
    for cell, lab in cells:
        b = blocks(df, cell)
        if b.empty:
            continue
        auc_m.append(b["auc"].mean())
        auc_e.append(b["auc"].std(ddof=1) / max(np.sqrt(len(b)), 1))
        f1_m.append(b["f1"].mean())
        f1_e.append(b["f1"].std(ddof=1) / max(np.sqrt(len(b)), 1))
        labs.append(lab)
    x = np.arange(len(labs))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    cols = ["#9e9e9e", "#c8a26b", "#7fb3d5", "#e59866", "#c0392b"]
    axes[0].bar(x, auc_m, yerr=auc_e, capsize=3, color=cols[:len(labs)])
    axes[0].set_ylabel("post-onset error AUC\n(lower = better)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labs)
    axes[0].set_title("(a) Recovery cost")
    axes[1].bar(x, f1_m, yerr=f1_e, capsize=3, color=cols[:len(labs)])
    axes[1].set_ylabel("overall F1\n(higher = better)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labs)
    axes[1].set_title("(b) Detection performance")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_factorial.png"))
    fig.savefig(os.path.join(FIG, "fig6_factorial.pdf"))
    plt.close(fig)
    print("wrote fig6_factorial")


def fig_spectrum(df):
    tags, diffs, ps = [], [], []
    for tag, a, b in PAIR_RHI:
        r = paired(df, a, b, "auc")
        if r is None or r["n"] < 6:
            continue
        tags.append(tag)
        diffs.append(r["diff"])
        ps.append(r["p"])
    colors = ["#c0392b" if d > 0 else "#2e7d32" for d in diffs]
    order = np.argsort(diffs)[::-1]
    tags = [tags[i] for i in order]
    diffs = [diffs[i] for i in order]
    ps = [ps[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    y = np.arange(len(tags))
    ax.barh(y, diffs, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    for yi, d, p in zip(y, diffs, ps):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(d + (0.002 if d >= 0 else -0.002), yi, star,
                va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(tags)
    ax.set_xlabel("Reset-Harm Index (AUC$_{reset}$ - AUC$_{no-reset}$)")
    ax.set_title("Reset is harmful for some learners, neutral/helpful for others")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_spectrum.png"))
    fig.savefig(os.path.join(FIG, "fig7_spectrum.pdf"))
    plt.close(fig)
    print("wrote fig7_spectrum")


def fig_lr_curve(df):
    s = ["S1_abrupt", "S3_recurrent", "Srare"]
    d = df[df["scenario"].isin(s)]
    cells = [("no FT", "drift_b2000_ft0", np.nan),
             ("5e-4", "drift_ft_lo", 5e-4),
             ("1e-3", "drift_ft_mid", 1e-3),
             ("1e-2", "drift_b2000_ft3", 1e-2)]
    auc_m, auc_e, f1_m, f1_e, xs, labs = [], [], [], [], [], []
    for tag, cell, lr in cells:
        g = d[d["cell"] == cell]
        if g.empty:
            continue
        b = g.groupby(["dataset", "scenario", "seed"])[["auc", "f1"]].mean()
        auc_m.append(b["auc"].mean())
        auc_e.append(b["auc"].std(ddof=1) / max(np.sqrt(len(b)), 1))
        f1_m.append(b["f1"].mean())
        f1_e.append(b["f1"].std(ddof=1) / max(np.sqrt(len(b)), 1))
        xs.append(len(xs))
        labs.append(tag)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    axes[0].errorbar(xs, auc_m, yerr=auc_e, marker="o", capsize=3, color="#c0392b")
    axes[0].set_ylabel("post-onset error AUC"); axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labs); axes[0].set_xlabel("fine-tune lr")
    axes[0].set_title("(a) Recovery cost")
    axes[1].errorbar(xs, f1_m, yerr=f1_e, marker="o", capsize=3, color="#2471a3")
    axes[1].set_ylabel("overall F1"); axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labs); axes[1].set_xlabel("fine-tune lr")
    axes[1].set_title("(b) Detection performance")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_lr_curve.png"))
    fig.savefig(os.path.join(FIG, "fig8_lr_curve.pdf"))
    plt.close(fig)
    print("wrote fig8_lr_curve")


def _delay_cell(m, ft):
    if m == "plain":
        return "plain"
    if m == "drift":
        return "drift_b2000_ft3" if (ft or 0) > 0 else "drift_b2000_ft0"
    return m


def fig_label_delay(df):
    p = os.path.join(RES, "experiment_matrix_delay.csv")
    if not os.path.exists(p):
        return
    dd = pd.read_csv(p)
    if "finetune_epochs" not in dd.columns:
        dd["finetune_epochs"] = 0
    dd["cell"] = [_delay_cell(m, f) for m, f in zip(dd["model"], dd["finetune_epochs"])]
    dd["auc"] = dd["per_onset_err_auc"].map(
        lambda s: float(np.mean([float(x) for x in str(s).split(";")
                                 if x.strip() and x.strip() != "NA"] or [np.nan])))
    dd["f1"] = pd.to_numeric(dd["overall_f1"], errors="coerce")
    cells = [("plain", "plain"), ("drift(buf2000)", "drift_b2000_ft0"),
             ("drift+FT(buf2000)", "drift_b2000_ft3"), ("logreg_drift", "logreg_drift")]
    im_by = {c: df[df["cell"] == c] for _, c in cells}
    dl_by = {c: dd[dd["cell"] == c] for _, c in cells}
    labs = [t for t, c in cells if not dl_by[c].empty and not im_by[c].empty]
    im_f1 = [im_by[c]["f1"].mean() for t, c in cells if t in labs]
    dl_f1 = [dl_by[c]["f1"].mean() for t, c in cells if t in labs]
    im_a = [im_by[c]["auc"].mean() for t, c in cells if t in labs]
    dl_a = [dl_by[c]["auc"].mean() for t, c in cells if t in labs]
    x = np.arange(len(labs))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    w = 0.38
    axes[0].bar(x - w / 2, im_a, w, label="immediate", color="#2471a3")
    axes[0].bar(x + w / 2, dl_a, w, label="label delay 500", color="#c0922b")
    axes[0].set_ylabel("post-onset error AUC")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labs, fontsize=7)
    axes[0].set_title("(a) Recovery cost")
    axes[1].bar(x - w / 2, im_f1, w, label="immediate", color="#2471a3")
    axes[1].bar(x + w / 2, dl_f1, w, label="label delay 500", color="#c0922b")
    axes[1].set_ylabel("overall F1")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labs, fontsize=7)
    axes[1].set_title("(b) Detection performance")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_label_delay.png"))
    fig.savefig(os.path.join(FIG, "fig9_label_delay.pdf"))
    plt.close(fig)
    print("wrote fig9_label_delay")


def main():
    df = load()
    df["cell"] = df["cell"].astype(str)
    fig_factorial(df)
    fig_spectrum(df)
    fig_lr_curve(df)
    fig_label_delay(df)


if __name__ == "__main__":
    main()
