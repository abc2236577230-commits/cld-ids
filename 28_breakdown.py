"""Per-dataset robustness breakdown of the spectrum + FT effects."""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

df = pd.read_csv(os.path.join(RES, "spectrum_long.csv"))
PAIRS = [("MLP reset", "plain", "drift_b2000_ft0"),
         ("MLP FT|reset", "drift_b2000_ft0", "drift_b2000_ft3"),
         ("MLP-deep reset", "mlp_deep", "mlp_deep_drift"),
         ("linear SGD reset", "sgd", "sgd_drift"),
         ("linear FT|reset", "sgd_drift", "sgd_drift_ft"),
         ("logreg reset", "logreg", "logreg_drift"),
         ("HT reset", "ht_plain", "ht_drift"),
         ("HAT reset", "hat_plain", "hat_drift"),
         ("NB reset", "gnb", "gnb_drift"),
         ("periodic vs online", "plain", "periodic")]


def blk(d, cell, col):
    return (d[d["cell"] == cell].groupby(["scenario", "seed"])[col].mean())


for ds in ["nslkdd", "unsw_full"]:
    d = df[df["dataset"] == ds]
    print(f"\n===== {ds} (n blocks/cell = {blk(d, 'plain', 'auc').notna().sum()}) =====")
    print(f"{'contrast':22s} {'AUC_a':>7s} {'AUC_b':>7s} {'dAUC':>8s} {'p':>8s} | "
          f"{'F1_a':>7s} {'F1_b':>7s} {'dF1':>8s}")
    for tag, a, b in PAIRS:
        A, B = blk(d, a, "auc"), blk(d, b, "auc")
        FA, FB = blk(d, a, "f1"), blk(d, b, "f1")
        m = pd.DataFrame({"a": A, "b": B}).dropna()
        mf = pd.DataFrame({"a": FA, "b": FB}).dropna()
        if m.empty:
            continue
        diff = m["b"] - m["a"]
        p = stats.wilcoxon(m["a"], m["b"])[1] if len(m) >= 6 and diff.abs().sum() > 0 else np.nan
        df1 = (mf["b"] - mf["a"]).mean() if not mf.empty else np.nan
        print(f"{tag:22s} {m['a'].mean():7.4f} {m['b'].mean():7.4f} {diff.mean():+8.4f} "
              f"{p:8.4f} | {mf['a'].mean():7.4f} {mf['b'].mean():7.4f} {df1:+8.4f}")

# F1 sanity per learner
print("\n===== mean F1 by model (both datasets) =====")
print(df.groupby("model")["f1"].mean().sort_values(ascending=False).round(4).to_string())
print("\n===== mean F1 by model, per dataset =====")
print(df.pivot_table(index="model", columns="dataset", values="f1", aggfunc="mean").round(4).to_string())
