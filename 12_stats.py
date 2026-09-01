"""Aggregate the experiment matrix: mean±std recovery, Wilcoxon tests,
error-AUC and F1 comparisons (drift vs each baseline).

Reads results/matrix_*.csv (shards) -> writes results/stats_summary.md
"""
import csv
import glob
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def model_label(r):
    if r["model"] == "drift" and int(r.get("finetune_epochs", 0) or 0) > 0:
        return "drift_ft"
    return r["model"]


def load_matrix():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "matrix_*.csv"))):
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    for r in rows:
        r["model"] = model_label(r)
    return rows


def parse(s):
    out = []
    for v in s.split(";"):
        v = v.strip()
        if not v or v in ("None", "NA", "nan"):
            out.append(np.nan)
        else:
            out.append(float(v))
    return out


def recovery_of(r, onset_idx):
    vals = parse(r["per_onset_recovery"])
    return vals[onset_idx] if onset_idx < len(vals) else np.nan


def mean_excl_cold(r):
    vals = [v for v in parse(r["per_onset_recovery"])[1:] if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def main():
    rows = load_matrix()
    print(f"loaded {len(rows)} rows")
    os.makedirs(RESULTS, exist_ok=True)
    out = []
    w = out.append

    w("# Recovery time (samples, mean over seeds; drift onsets excl. cold start)")
    w("")
    w("| dataset | scenario | model | rec_mean | rec_std | err_auc_mean | f1_mean |")
    w("|---|---|---|---|---|---|---|")
    keys = {}
    for r in rows:
        keys.setdefault((r["dataset"], r["scenario"], r["model"]), []).append(r)
    for key in sorted(keys):
        grp = keys[key]
        recs = [mean_excl_cold(r) for r in grp]
        recs = [v for v in recs if not np.isnan(v)]
        aucs = [np.nanmean(parse(r["per_onset_err_auc"])[1:]) for r in grp]
        f1s = [float(r["overall_f1"]) for r in grp]
        w(f"| {key[0]} | {key[1]} | {key[2]} | {np.mean(recs):.0f} ± {np.std(recs):.0f} "
          f"| {np.mean(aucs):.4f} | {np.mean(f1s):.4f} |")

    w("")
    w("## Paired Wilcoxon on per-onset recovery time (drift-aware vs baseline)")
    w("")
    pairs = [("drift_ft", "plain"), ("drift_ft", "periodic"), ("drift_ft", "ht_plain"),
             ("drift", "plain"), ("ht_drift", "ht_plain"), ("drift_ft", "arf"),
             ("ht_drift", "arf")]
    for ds in ("unsw_full", "nslkdd"):
        for scn in ("S1_abrupt", "S2_gradual", "S3_recurrent", "S4_inversion", "Srare"):
            w(f"### {ds}/{scn}")
            w("")
            w("| model | onset_idx | n_seeds | rec_mean | baseline_mean | p | wins |")
            w("|---|---|---|---|---|---|---|")
            base_map = {}
            for r in rows:
                if r["dataset"] == ds and r["scenario"] == scn:
                    base_map.setdefault(r["model"], []).append(r)
            models_here = [m for m in ("plain", "drift", "drift_ft", "periodic",
                                       "ht_plain", "ht_drift") if m in base_map]
            if not models_here:
                continue
            n_onsets = max(len(parse(r["per_onset_recovery"])) for r in
                           base_map[models_here[0]])
            for a, b in pairs:
                if a not in base_map or b not in base_map:
                    continue
                for oi in range(1, n_onsets):
                    va = [recovery_of(r, oi) for r in base_map[a]]
                    vb = [recovery_of(r, oi) for r in base_map[b]]
                    va = np.array([v for v in va if not np.isnan(v)])
                    vb = np.array([v for v in vb if not np.isnan(v)])
                    if len(va) < 5 or len(vb) < 5 or len(va) != len(vb):
                        continue
                    _, p = stats.wilcoxon(va, vb, zero_method="wilcox")
                    wins = int((va < vb).sum())
                    w(f"| {a} vs {b} | {oi} | {len(va)} | {va.mean():.0f} | {vb.mean():.0f} "
                      f"| {p:.4f} | {wins}/{len(va)} |")
            w("")

    w("## Overall prequential F1 (mean over seeds)")
    w("")
    w("| dataset | scenario | plain | drift | drift_ft | periodic | ht_plain | ht_drift | arf |")
    w("|---|---|---|---|---|---|---|---|---|")
    for ds in ("unsw_full", "nslkdd"):
        for scn in ("S1_abrupt", "S2_gradual", "S3_recurrent", "S4_inversion", "Srare"):
            f1s = {}
            for r in rows:
                if r["dataset"] == ds and r["scenario"] == scn:
                    f1s.setdefault(r["model"], []).append(float(r["overall_f1"]))
            cells = []
            for m in ("plain", "drift", "drift_ft", "periodic", "ht_plain",
                      "ht_drift", "arf"):
                cells.append(f"{np.mean(f1s[m]):.4f}" if m in f1s else "-")
            w(f"| {ds} | {scn} | " + " | ".join(cells) + " |")

    text = "\n".join(out)
    with open(os.path.join(RESULTS, "stats_summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print("saved -> results/stats_summary.md")


if __name__ == "__main__":
    main()