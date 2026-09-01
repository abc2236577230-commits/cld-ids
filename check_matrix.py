"""Verify experiment matrix completeness: every (dataset, scenario, model)
must have exactly seeds 0-9."""
import collections
import csv
import glob
import os

rows = []
for p in glob.glob(os.path.join(os.path.dirname(__file__), "..", "results", "matrix_*.csv")):
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            ft = r.get("finetune_epochs") or "NA"
            m = r["model"] if ft in ("NA", "0") else "drift_ft"
            rows.append((r["dataset"], r["scenario"], m, int(r["seed"])))

cnt = collections.Counter((d, s, m) for d, s, m, _ in rows)
seeds = collections.defaultdict(set)
for d, s, m, seed in rows:
    seeds[(d, s, m)].add(seed)

models = ["plain", "drift", "drift_ft", "periodic", "ht_plain", "ht_drift", "arf"]
bad = []
for ds in ("unsw_full", "nslkdd"):
    for scn in ("S1_abrupt", "S2_gradual", "S3_recurrent", "S4_inversion", "Srare"):
        cells = []
        for m in models:
            k = (ds, scn, m)
            ok = cnt.get(k, 0) == 10 and seeds[k] == set(range(10))
            if not ok:
                bad.append((k, cnt.get(k, 0), sorted(seeds[k])))
                cells.append(f"{m}:BAD")
            else:
                cells.append(f"{m}:OK")
        print(f"{ds:>9}/{scn:<14}", " ".join(cells))
print("total rows:", len(rows))
print("BAD:", bad if bad else f"none - matrix complete ({len(rows)}/{len(rows)})")
