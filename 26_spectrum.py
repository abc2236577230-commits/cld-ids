"""Learner-plasticity spectrum + de-bundled 2x2 factorial analysis.

Reads all results/matrix_*.csv shards plus results/experiment_matrix_v2.csv,
builds a tidy per-run table, and reports:

  A. MLP 2x2 factorial (buffer x fine-tune) on post-onset error AUC and F1.
  B. Reset-Harm Index (RHI) = AUC(reset) - AUC(no-reset) per learner family.
  C. Fine-tune Benefit (FTB) for gradient learners.

Writes results/spectrum_long.csv, results/spectrum_summary.md and prints a
compact report. Purely descriptive + paired Wilcoxon; no causal claim.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def load_all():
    frames = []
    for p in sorted(glob.glob(os.path.join(RES, "matrix_*.csv"))):
        df = pd.read_csv(p)
        frames.append(df)
    for name in ("experiment_matrix_v2.csv", "experiment_matrix.csv"):
        p = os.path.join(RES, name)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    all_df = pd.concat(frames, ignore_index=True)
    if "finetune_epochs" not in all_df.columns:
        all_df["finetune_epochs"] = 0
    all_df["finetune_epochs"] = all_df["finetune_epochs"].fillna(0).astype(int)
    all_df["warmup"] = all_df["warmup"].fillna(512).astype(int)
    key = ["dataset", "scenario", "seed", "model", "warmup", "finetune_epochs"]
    all_df = all_df.drop_duplicates(subset=key, keep="first")
    return all_df


def mean_auc(s):
    vals = []
    for tok in str(s).split(";"):
        tok = tok.strip()
        if tok and tok != "NA":
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else np.nan


def annotate(df):
    def fam(m):
        if m in ("plain", "drift", "periodic"):
            return "MLP"
        if m in ("sgd", "sgd_drift", "sgd_drift_ft", "logreg", "logreg_drift"):
            return "linear"
        if m.startswith("ht"):
            return "HT"
        if m.startswith("hat"):
            return "HAT"
        if m.startswith("gnb"):
            return "NB"
        if m == "arf":
            return "ARF"
        return m

    def reset(m):
        return int(("drift" in m) or ("arf" in m and False))

    df = df.copy()
    df["learner"] = df["model"].map(fam)
    df["reset"] = df["model"].map(lambda m: int(m.endswith("drift") or m.endswith("_ft")))
    df["ft"] = df["finetune_epochs"]
    df["buf"] = df["warmup"]
    df["auc"] = df["per_onset_err_auc"].map(mean_auc)
    df["f1"] = pd.to_numeric(df["overall_f1"], errors="coerce")
    df["gmean"] = pd.to_numeric(df["overall_gmean"], errors="coerce")
    return df


def cell_label(r):
    if r["learner"] == "MLP" and r["model"] == "drift":
        return f"drift_b{r['buf']}_ft{r['ft']}"
    if r["learner"] == "MLP" and r["model"] == "plain":
        return "plain"
    if r["learner"] == "MLP" and r["model"] == "periodic":
        return "periodic"
    return r["model"]


def report_factorial(df, out):
    d = df[df["learner"] == "MLP"].copy()
    d = d[d["model"].isin(["plain", "drift"])]
    d["cell"] = d.apply(cell_label, axis=1)
    piv = (d.pivot_table(index=["dataset", "scenario", "seed"],
                         columns="cell", values="auc", aggfunc="mean"))
    lines = ["", "## A. MLP de-bundled 2x2 (post-onset error AUC)", ""]
    cellcols = [c for c in ["plain", "drift_b512_ft0", "drift_b2000_ft0",
                            "drift_b512_ft3", "drift_b2000_ft3"] if c in piv.columns]
    lines.append("Mean AUC per cell (lower = better recovery):")
    lines.append("")
    lines.append("| cell | mean AUC | mean F1 | n |")
    lines.append("|---|---|---|---|")
    pivf1 = (d.pivot_table(index=["dataset", "scenario", "seed"],
                           columns="cell", values="f1", aggfunc="mean"))
    for c in cellcols:
        lines.append(f"| {c} | {piv[c].mean():.4f} | {pivf1[c].mean():.4f} | {piv[c].notna().sum()} |")

    def paired(a, b):
        m = piv[[a, b]].dropna()
        if len(m) < 6:
            return None
        stat, p = stats.wilcoxon(m[a], m[b])
        return dict(n=len(m), diff=float((m[a] - m[b]).mean()), p=float(p))

    tests = []
    if {"drift_b2000_ft3", "drift_b2000_ft0"}.issubset(piv.columns):
        tests.append(("FT effect (buf2000): ft3 - ft0", "drift_b2000_ft3", "drift_b2000_ft0"))
    if {"drift_b512_ft3", "drift_b512_ft0"}.issubset(piv.columns):
        tests.append(("FT effect (buf512): ft3 - ft0", "drift_b512_ft3", "drift_b512_ft0"))
    if {"drift_b2000_ft0", "drift_b512_ft0"}.issubset(piv.columns):
        tests.append(("buffer effect (ft0): b2000 - b512", "drift_b2000_ft0", "drift_b512_ft0"))
    if {"drift_b2000_ft3", "drift_b512_ft3"}.issubset(piv.columns):
        tests.append(("buffer effect (ft3): b2000 - b512", "drift_b2000_ft3", "drift_b512_ft3"))
    if {"drift_b2000_ft0", "plain"}.issubset(piv.columns):
        tests.append(("reset effect (plain->drift_ft0)", "drift_b2000_ft0", "plain"))
    lines.append("")
    lines.append("Paired Wilcoxon (AUC; positive diff = first cell worse):")
    lines.append("")
    lines.append("| contrast | mean diff | p | n |")
    lines.append("|---|---|---|---|")
    for name, a, b in tests:
        r = paired(a, b)
        if r:
            lines.append(f"| {name} | {r['diff']:+.4f} | {r['p']:.4f} | {r['n']} |")
    out.extend(lines)
    return piv


def report_spectrum(df, out):
    lines = ["", "## B. Reset-Harm Index by learner family", "",
             "RHI = mean AUC(reset) - mean AUC(no-reset); positive = reset raises",
             "post-onset error (reset harmful); negative = reset helps.", ""]
    pairs = [("MLP-32/16 (non-convex)", "plain", "drift_b2000_ft0"),
             ("MLP-64/32/16 (non-convex)", "mlp_deep", "mlp_deep_drift"),
             ("MLP + FT", "drift_b2000_ft0", "drift_b2000_ft3"),
             ("linear SGD (convex)", "sgd", "sgd_drift"),
             ("linear SGD + FT", "sgd_drift", "sgd_drift_ft"),
             ("logreg (convex)", "logreg", "logreg_drift"),
             ("HT (tree)", "ht_plain", "ht_drift"),
             ("HAT (tree)", "hat_plain", "hat_drift"),
             ("NB", "gnb", "gnb_drift"),
             ("periodic vs online", "plain", "periodic")]
    long = df.copy()
    long["cell"] = long.apply(cell_label, axis=1)
    lines.append("| comparison | mean AUC(a) | mean AUC(b) | diff(b-a) | p | n |")
    lines.append("|---|---|---|---|---|---|")
    for tag, a, b in pairs:
        da = long[long["cell"] == a]
        db = long[long["cell"] == b]
        m = (da.groupby(["dataset", "scenario", "seed"])["auc"].mean()
             .rename("a")
             .to_frame()
             .join(db.groupby(["dataset", "scenario", "seed"])["auc"].mean().rename("b"),
                   how="inner")).dropna()
        if m.empty:
            continue
        diff = float((m["b"] - m["a"]).mean())
        p = float(stats.wilcoxon(m["a"], m["b"])[1]) if len(m) >= 6 and (m["b"] - m["a"]).abs().sum() > 0 else np.nan
        lines.append(f"| {tag} | {m['a'].mean():.4f} | {m['b'].mean():.4f} | {diff:+.4f} | {p:.4f} | {len(m)} |")
    out.extend(lines)


def report_lr_curve(df, out):
    """Reset + buffer2000 + 3 epochs, varying fine-tune learning rate."""
    s = ["S1_abrupt", "S3_recurrent", "Srare"]
    d = df[df["scenario"].isin(s) & df["dataset"].isin(["nslkdd", "unsw_full"])]
    cells = [("no FT (ref)", "drift_b2000_ft0"),
             ("FT lr=5e-4", "drift_ft_lo"),
             ("FT lr=1e-3", "drift_ft_mid"),
             ("FT lr=1e-2", "drift_b2000_ft3")]
    out.append("")
    out.append("## C. Fine-tune learning-rate curve (reset + buffer2000 + 3 epochs)")
    out.append("")
    out.append("Restricted to S1_abrupt / S3_recurrent / Srare (lower AUC, higher F1 better).")
    out.append("")
    out.append("| variant | mean AUC | mean F1 | n |")
    out.append("|---|---|---|---|")
    for tag, cell in cells:
        g = d[d["cell"] == cell]
        if g.empty:
            continue
        out.append(f"| {tag} | {g['auc'].mean():.4f} | {g['f1'].mean():.4f} | {len(g)} |")


def report_delay(out):
    """Label-delay hardening: immediate vs 500-sample-delayed updates."""
    p = os.path.join(RES, "experiment_matrix_delay.csv")
    if not os.path.exists(p):
        return
    d = annotate(pd.read_csv(p))
    d["cell"] = d.apply(cell_label, axis=1)
    main_df = annotate(load_all())
    main_df["cell"] = main_df.apply(cell_label, axis=1)
    cells = [("plain", "plain"), ("drift(buf2000)", "drift_b2000_ft0"),
             ("drift+FT(buf2000)", "drift_b2000_ft3"), ("logreg_drift", "logreg_drift")]
    out.append("")
    out.append("## D. Label-delay hardening (updates deferred by 500 samples)")
    out.append("")
    out.append("| variant | immediate AUC | delayed AUC | immediate F1 | delayed F1 |")
    out.append("|---|---|---|---|---|")
    s = ["S1_abrupt", "S3_recurrent", "Srare"]
    d = d[d["scenario"].isin(s)]
    main_df = main_df[main_df["scenario"].isin(s)]
    for tag, cell in cells:
        di = d[d["cell"] == cell].groupby(["dataset", "scenario", "seed"])[["auc", "f1"]].mean()
        im = main_df[main_df["cell"] == cell].groupby(["dataset", "scenario", "seed"])[["auc", "f1"]].mean()
        m = im.join(di, how="inner", lsuffix="_im", rsuffix="_dl").dropna()
        if m.empty:
            continue
        out.append(f"| {tag} | {m['auc_im'].mean():.4f} | {m['auc_dl'].mean():.4f} | "
                   f"{m['f1_im'].mean():.4f} | {m['f1_dl'].mean():.4f} | n={len(m)} |")
    out.append("")
    out.append("Both columns restricted to S1_abrupt / S3_recurrent / Srare.")


def main():
    df = annotate(load_all())
    df["cell"] = df.apply(cell_label, axis=1)
    df[["dataset", "scenario", "seed", "learner", "model", "cell", "reset",
        "ft", "buf", "f1", "auc", "gmean"]].to_csv(
        os.path.join(RES, "spectrum_long.csv"), index=False)
    out = ["# Learner-plasticity spectrum & de-bundled factorial", ""]
    out.append(f"runs loaded: {len(df)}; datasets: {sorted(df['dataset'].unique())}")
    out.append(f"models: {sorted(df['model'].unique())}")
    report_factorial(df, out)
    report_spectrum(df, out)
    report_lr_curve(df, out)
    report_delay(out)

    # compact console summary
    print("\n".join(out))
    with open(os.path.join(RES, "spectrum_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\nwrote results/spectrum_summary.md and results/spectrum_long.csv")


if __name__ == "__main__":
    main()
