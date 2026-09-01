"""Comprehensive data audit of results/matrix_*.csv before paper writing.

Checks:
 A. structural: row counts, duplicate keys, seed completeness
 B. config consistency per model group (lr/window/warmup/cooldown/delta/start_after)
 C. field legality: onset-vector lengths, value ranges, monotonic triggers,
    n_triggers == len(triggers), recovery >= window floor, f1/gmean ranges
 D. floor effect: share of recovery == window (saturation diagnostics)
 E. outliers: extreme recoveries / error-AUCs listed explicitly
 F. seed sensitivity: identical outputs across seeds (RNG sanity)
 G. cross-check of stats_summary.md aggregates vs recomputation
"""
import collections
import csv
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

ONSET_N = {"S1_abrupt": 4, "S2_gradual": 3, "S3_recurrent": 6,
           "S4_inversion": 4, "Srare": 3}
WINDOW = 2000


def parse(s):
    out = []
    for v in s.split(";"):
        v = v.strip()
        if not v or v in ("None", "NA", "nan"):
            out.append(np.nan)
        else:
            try:
                out.append(float(v))
            except ValueError:
                out.append(np.nan)
    return out


def label(r):
    return r["model"] + ("_ft" if (r.get("finetune_epochs") or "NA") not in ("NA", "0")
                         else "")


rows = []
for p in sorted(glob.glob(os.path.join(RESULTS, "matrix_*.csv"))):
    shard = os.path.basename(p)
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            r["_shard"] = shard
            r["_label"] = label(r)
            rows.append(r)

issues = []

print(f"[A] rows={len(rows)}")
keys = [(r["dataset"], r["scenario"], str(r["seed"]), r["_label"]) for r in rows]
dup = [k for k, c in collections.Counter(keys).items() if c > 1]
print("  duplicates:", dup if dup else "none")
if dup:
    issues.append(f"A: {len(dup)} duplicate keys")
seen_seeds = collections.defaultdict(set)
for r in rows:
    seen_seeds[(r["dataset"], r["scenario"], r["_label"])].add(str(r["seed"]))
for k, sd in seen_seeds.items():
    if sd != {str(i) for i in range(10)}:
        issues.append(f"A: seed set wrong for {k}: {sorted(sd)}")

BASE = dict(lr="0.0005", window="2000", rec_thr="0.15", cooldown="5000",
            delta="0.002", start_after="5000", use_cat="0")
FT = dict(BASE, warmup="2000", finetune_epochs="3")
cfg_bad = 0
for r in rows:
    spec = FT if r["_label"] == "drift_ft" else BASE
    for field, want in spec.items():
        got = (r.get(field) or "").strip()
        if got != want:
            cfg_bad += 1
            issues.append(f"B: {r['_shard']} {r['_label']} s{r['seed']} "
                          f"{r['scenario']} field {field}={got!r} != {want!r}")
print(f"[B] config drift checked: {'OK' if cfg_bad == 0 else f'{cfg_bad} mismatches'}")

floor_hits = collections.Counter()
floor_tot = collections.Counter()
rec_len_bad, auc_len_bad, trig_bad = [], [], []
steady_na = 0
for r in rows:
    recs = parse(r["per_onset_recovery"])
    aucs = parse(r["per_onset_err_auc"])
    scn = r["scenario"]
    if len(recs) != ONSET_N[scn]:
        rec_len_bad.append((r["_shard"], scn, r["_label"], r["seed"], len(recs)))
    if len(aucs) != ONSET_N[scn]:
        auc_len_bad.append((r["_shard"], scn, r["_label"], r["seed"], len(aucs)))
    for v in recs[1:]:
        if not np.isnan(v):
            if v < WINDOW:
                issues.append(f"C: recovery < window! {r['_shard']} "
                              f"{r['_label']} s{r['seed']} {scn} rec={v}")
            if v == WINDOW:
                floor_hits[(r["dataset"], scn, r["_label"])] += 1
            floor_tot[(r["dataset"], scn, r["_label"])] += 1
    for v in aucs:
        if not np.isnan(v) and not (0 <= v <= 1):
            issues.append(f"C: AUC out of range {r['_shard']} {r['_label']} "
                          f"s{r['seed']} {scn} auc={v}")
    f1v, gmv = float(r["overall_f1"]), float(r["overall_gmean"])
    if not (0 <= f1v <= 1) or not (0 <= gmv <= 1):
        issues.append(f"C: f1/gmean out of range {r['_shard']} {r['_label']}")
    trigs = r.get("trigger_samples", "")
    n_trig = int(r["n_triggers"] or 0)
    tl = [] if not trigs.strip() else [int(x) for x in trigs.split(";")]
    if len(tl) != n_trig:
        trig_bad.append((r["_shard"], r["_label"], r["seed"]))
    if tl != sorted(tl):
        issues.append(f"C: triggers not increasing {r['_shard']} {r['model']}")
    if any(t < 1 or t > int(r["n_samples"]) for t in tl):
        issues.append(f"C: trigger outside stream {r['_shard']} {r['_label']}")
    if (r.get("steady_gmean") or "NA") == "NA":
        steady_na += 1

print("[C] onset-len errors:", rec_len_bad or auc_len_bad or "none")
print("    trigger-count mismatches:", trig_bad or "none")
print("    steady_gmean NA (dead column):", f"{steady_na}/{len(rows)}")

print("[D] floor effect: share of recovery==2000 among post-cold-start onsets")
for k in sorted(floor_tot):
    tot, hit = floor_tot[k], floor_hits.get(k, 0)
    flag = " <-- saturated" if tot and hit / tot > 0.8 else ""
    print(f"    {'/'.join(k)}: {hit}/{tot} = {hit / tot:.0%}{flag}")

print("[E] outliers (post-cold-start):")
n_out = 0
for r in rows:
    recs = parse(r["per_onset_recovery"])[1:]
    aucs = parse(r["per_onset_err_auc"])[1:]
    for i, v in enumerate(recs, start=1):
        if not np.isnan(v) and v >= 8000:
            n_out += 1
            print(f"    BIG-REC {r['_shard']} {r['_label']} ds={r['dataset']} "
                  f"{r['scenario']} seed={r['seed']} onset{i}={v:.0f} "
                  f"f1={r['overall_f1']}")
    for i, v in enumerate(aucs, start=1):
        if not np.isnan(v) and v >= 0.5:
            n_out += 1
            print(f"    BIG-AUC {r['_shard']} {r['_label']} ds={r['dataset']} "
                  f"{r['scenario']} seed={r['seed']} onset{i}={v:.3f} "
                  f"f1={r['overall_f1']}")
if n_out == 0:
    print("    none")

f1_by_seed = collections.defaultdict(list)
for r in rows:
    f1_by_seed[(r["dataset"], r["scenario"], r["_label"])].append(
        float(r["overall_f1"]))
flat = [k for k, vals in f1_by_seed.items() if max(vals) - min(vals) < 1e-9]
for k in flat:
    print(f"    FLAT-SEEDS {k}: values={sorted(set(f1_by_seed[k]))}")
print(f"[F] zero-variance cells: {flat if flat else 'none'}")

agg = collections.defaultdict(list)
rows_no_rec = []
for r in rows:
    recs = [v for v in parse(r["per_onset_recovery"])[1:] if not np.isnan(v)]
    if recs:
        agg[(r["dataset"], r["scenario"], r["_label"])].append(
            float(np.mean(recs)))
    else:
        rows_no_rec.append((r["_shard"], r["_label"], r["seed"],
                            r["scenario"]))
md_lines = open(os.path.join(RESULTS, "stats_summary.md"),
                encoding="utf-8").read().splitlines()
md_rec = {}
for l in md_lines:
    parts = [p.strip() for p in l.split("|")]
    if len(parts) == 8 and "±" in parts[4]:
        md_rec[(parts[1], parts[2], parts[3])] = float(parts[4].split("±")[0])
mism = []
for k, vals in agg.items():
    mine = round(float(np.mean(vals)))
    theirs = md_rec.get(k)
    if theirs is None:
        mism.append(f"G: stats_summary missing {k}")
    elif abs(mine - theirs) > 1:
        mism.append(f"G: recovery mismatch {k}: csv={mine} md={theirs}")
extra = set(md_rec) - set(agg)
print(f"[G] cross-check vs stats_summary.md: {len(mism)} mismatches over "
      f"{len(agg)} cells; md-only rows: {sorted(extra) if extra else 'none'}")
if rows_no_rec:
    print(f"    note: {len(rows_no_rec)} runs with no valid post-cold-start "
          f"recovery (all None, excluded from aggregates):")
    for x in rows_no_rec:
        print(f"      {x}")
issues.extend(mism)

print()
if issues:
    print(f"===== ISSUES ({len(issues)}) =====")
    for x in issues[:60]:
        print(" -", x)
else:
    print("===== NO STRUCTURAL ISSUES =====")
