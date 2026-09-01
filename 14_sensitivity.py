"""Sensitivity grid on unsw_full/S1_abrupt (seed 0) for the drift_ft framework.

Grid: lr, finetune_epochs, window, rec_thr, warmup, delta.
Output: results/sensitivity_grid.csv + rank-stability summary.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runner = importlib.import_module("11_experiment_runner")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

GRID = {
    "lr": [2e-4, 5e-4, 1e-3],
    "finetune_epochs": [1, 3, 5],
    "window": [1000, 2000, 5000],
    "rec_thr": [0.10, 0.15, 0.20],
    "warmup": [512, 2000, 4000],
    "delta": [1e-3, 2e-3, 1e-2],
}
BASE = dict(lr=5e-4, finetune_epochs=3, window=2000, rec_thr=0.15,
            warmup=2000, delta=0.002, cooldown=5000, start_after=5000)


def parse(s):
    out = []
    for v in s.split(";"):
        v = v.strip()
        if not v or v in ("None", "NA", "nan"):
            out.append(np.nan)
        else:
            out.append(float(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="all")
    args = ap.parse_args()

    # plain baseline (fixed config)
    class A:
        dataset = "unsw_full"
        scenario = "S1_abrupt"
        models = ["plain"]
        seeds = [0]
        use_cat = False
        lr, window, rec_thr, warmup, cooldown, delta = 5e-4, 2000, 0.15, 512, 5000, 0.002
        start_after, finetune_epochs, finetune_lr, period = 0, 0, 1e-2, 5000
        out = os.path.join(RESULTS, "sens_plain.csv")
    runner.run_one("unsw_full", "S1_abrupt", 0, "plain", A())
    with open(A.out) as f:
        plain = next(csv.DictReader(f))
    plain_rec = [v for v in parse(plain["per_onset_recovery"])[1:] if not np.isnan(v)]
    print(f"plain recovery mean: {np.mean(plain_rec):.0f}")

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "sensitivity_grid.csv")
    new = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["param", "value", "rec_mean", "rec_std", "f1",
                        "wins_vs_plain", "n_onsets"])
        for param, values in GRID.items():
            if args.grid != "all" and args.grid != param:
                continue
            for v in values:
                cfg = dict(BASE)
                cfg[param] = v
                class C:
                    pass
                c = C()
                for k, vv in cfg.items():
                    setattr(c, k, vv)
                c.dataset, c.scenario = "unsw_full", "S1_abrupt"
                c.models, c.seeds, c.use_cat = ["drift"], [0], False
                c.finetune_lr, c.period = 1e-2, 5000
                c.out = os.path.join(RESULTS, "sens_drift.csv")
                if os.path.exists(c.out):
                    os.remove(c.out)
                runner.run_one("unsw_full", "S1_abrupt", 0, "drift", c)
                with open(c.out) as f:
                    r = list(csv.DictReader(f))[-1]
                rec = [v for v in parse(r["per_onset_recovery"])[1:] if not np.isnan(v)]
                wins = int(np.mean(rec) < np.mean(plain_rec))
                print(f"{param}={v}: rec={np.mean(rec):.0f}±{np.std(rec):.0f} "
                      f"f1={r['overall_f1']} wins_vs_plain={wins}")
                w.writerow([param, v, f"{np.mean(rec):.0f}", f"{np.std(rec):.0f}",
                            r["overall_f1"], wins, len(rec)])
    print("saved -> results/sensitivity_grid.csv")


if __name__ == "__main__":
    main()