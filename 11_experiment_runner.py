"""Unified experiment matrix runner with resume support.

Usage:
  python 11_experiment_runner.py --dataset unsw_sub --scenario S4_inversion --seeds 0 1
  python 11_experiment_runner.py --dataset unsw_full --models plain drift
  python 11_experiment_runner.py --dataset nslkdd --all

Each (dataset, scenario, seed, model) run appends one row to
results/experiment_matrix.csv; already-completed rows are skipped (resume).
"""
import argparse
import csv
import os
import sys
import time
from collections import deque

import numpy as np
from river import preprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import common
from experiments.models import (DriftAware, PeriodicMLP, make_arf, make_drc_ht,
                                make_hat, make_ht, make_mlp, make_mlp_ft,
                                make_gnb, make_logreg, make_periodic, make_sgd, make_sgd_ft)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

BATCH = 64
_LOAD_CACHE = {}
COLUMNS = [
    "dataset", "scenario", "seed", "model", "lr", "window", "rec_thr", "warmup",
    "cooldown", "delta", "start_after", "finetune_epochs", "use_cat", "n_samples",
    "n_triggers", "trigger_samples", "per_onset_recovery", "per_onset_err_auc",
    "overall_f1", "overall_gmean", "steady_gmean", "runtime_s",
]


def done_rows(path):
    """Resume keys: (dataset, scenario, seed, model, warmup, ft) where ft
    normalizes finetune_epochs '0'/missing -> 'NA' (older shards lack the
    column). Must stay in sync with the key built in main()."""
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ft = r.get("finetune_epochs") or "NA"
            if ft == "0":
                ft = "NA"
            keys.add((r["dataset"], r["scenario"], r["seed"], r["model"],
                      r.get("warmup", "512"), ft))
    return keys


def append_row(row, path):
    os.makedirs(RESULTS, exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def encode_row(cat_vals, i, maps, cols):
    xd = {}
    for c in cols:
        xd[c] = maps[c].setdefault(cat_vals[c][i], len(maps[c]))
    return xd


def run_one(dataset, scenario_name, seed, model_name, args):
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # --- load + build scenario (cached per dataset) ---
    cache = _LOAD_CACHE
    if (dataset, args.use_cat) not in cache:
        if dataset == "unsw_full":
            cache[(dataset, args.use_cat)] = common.load_unsw(full=True, use_cat=args.use_cat)
        elif dataset == "unsw_sub":
            cache[(dataset, args.use_cat)] = common.load_unsw(full=False, use_cat=args.use_cat)
        else:
            cache[(dataset, args.use_cat)] = common.load_nslkdd()
    X, cat_vals, y, fam, feats = cache[(dataset, args.use_cat)]
    size = "full" if dataset == "unsw_full" else ("sub" if dataset == "unsw_sub" else "nslkdd")
    sc = common.build_all_scenarios(X, y, fam, rng, size)[scenario_name]
    Xs, ys, onsets = sc["X"], sc["y"], sc["onsets"]
    n = sc["n"]
    cat_cols = list(cat_vals.keys())

    # --- model + scaler ---
    scaler = preprocessing.StandardScaler()
    maps = {c: {} for c in cat_cols}
    cfg = dict(lr=args.lr, window=args.window, rec_thr=args.rec_thr, warmup=args.warmup,
               cooldown=args.cooldown, delta=args.delta, use_cat=int(args.use_cat))

    sk_drift = {"drift", "drift_ft_lo", "drift_ft_mid", "mlp_deep_drift",
                "mlp_deep_drift_ft", "sgd_drift", "sgd_drift_ft"}
    river_drift = {"ht_drift", "hat_drift", "gnb_drift", "logreg_drift"}
    river_plain = {"ht_plain", "hat_plain", "gnb", "logreg"}

    model, fitted, drift_aw, periodic = None, False, None, None
    if model_name == "plain":
        model = make_mlp(lr=args.lr, seed=seed)
    elif model_name == "mlp_deep":
        model = make_mlp(lr=args.lr, hidden=(64, 32, 16), seed=seed)
    elif model_name == "sgd":
        model = make_sgd(lr=args.lr, seed=seed)
    elif model_name == "periodic":
        periodic = PeriodicMLP(period=args.period, window=args.window, lr=1e-3, seed=seed,
                               n_features=len(feats) + len(cat_cols))
    elif model_name == "arf":
        model = make_arf(seed=seed)
    elif model_name in river_plain:
        model = (make_gnb() if model_name == "gnb" else (make_logreg() if model_name == "logreg" else (make_hat() if model_name == "hat_plain" else make_ht())))
    elif model_name in river_drift:
        base = (make_gnb if model_name == "gnb_drift" else (make_logreg if model_name == "logreg_drift" else (make_hat if model_name == "hat_drift" else make_ht)))
        drift_aw = DriftAware(lambda: base(), delta=args.delta,
                              cooldown=args.cooldown, warmup=args.warmup, batch=BATCH,
                              start_after=args.start_after)
        drift_aw.fitted = True  # tree learners predict without training; warm-start n/a
    elif model_name in sk_drift:
        if model_name in ("drift", "drift_ft_lo", "drift_ft_mid"):
            base = lambda: make_mlp(lr=args.lr, seed=seed)
            _ftlr = {"drift_ft_lo": 5e-4, "drift_ft_mid": 1e-3}.get(model_name, args.finetune_lr)
            ft = (lambda: make_mlp_ft(lr=_ftlr, epochs=args.finetune_epochs, batch=BATCH))
        elif model_name in ("mlp_deep_drift", "mlp_deep_drift_ft"):
            base = lambda: make_mlp(lr=args.lr, hidden=(64, 32, 16), seed=seed)
            ft = (lambda: make_mlp_ft(lr=args.finetune_lr, epochs=args.finetune_epochs,
                                      batch=BATCH, hidden=(64, 32, 16)))
        else:
            base = lambda: make_sgd(lr=args.lr, seed=seed)
            ft = (lambda: make_sgd_ft(lr=args.finetune_lr, epochs=args.finetune_epochs))
        drift_aw = DriftAware(base,
                              delta=args.delta, cooldown=args.cooldown,
                              warmup=args.warmup, batch=BATCH,
                              start_after=args.start_after,
                              finetune_epochs=args.finetune_epochs,
                              finetune_lr=args.finetune_lr,
                              make_ft=(ft if args.finetune_epochs > 0 else None))
    else:
        raise ValueError(model_name)

    # --- stream ---
    recent = []
    triggers = []
    label_delay = int(getattr(args, "label_delay", 0) or 0)
    pend = deque()
    sm = common.StreamMetrics(onsets, n, window=args.window, rec_thr=args.rec_thr)
    seg_idx = 0
    for i in range(n):
        if seg_idx < len(onsets) - 1 and i >= onsets[seg_idx + 1]:
            seg_idx += 1
        y_true = int(ys[i])
        xd = encode_row(cat_vals, i, maps, cat_cols)
        for j, f in enumerate(feats):
            xd[f] = float(Xs[i, j])
        xs = scaler.transform_one(xd)
        xn = np.array([xs[f] for f in feats] + [xs[c] for c in cat_cols],
                      dtype=np.float32).reshape(1, -1)
        scaler.learn_one(xd)

        if model_name in ("plain", "sgd", "mlp_deep"):
            yp = model.predict(xn)[0] if fitted else 0
        elif model_name in sk_drift:
            yp = drift_aw.predict(xn)
        elif model_name == "periodic":
            yp = periodic.predict(xn)
        elif model_name in river_plain:
            yp = model.predict_one(xd) or 0
        elif model_name == "arf":
            yp = (model.predict_one(xd) or 0) if fitted else 0
        elif model_name in river_drift:
            yp = drift_aw.predict(xd)
        yp = int(yp)

        # predictions/metrics always use the true label at time t
        sm.update(i, y_true, yp, seg_idx)

        # label-delay gate: model updates (and error-channel detection) only
        # happen once the label is observed, N samples later.
        if label_delay > 0:
            pend.append((xn, xd, y_true, yp))
            if len(pend) <= label_delay:
                continue
            lxn, lxd, ly, lyp = pend.popleft()
        else:
            lxn, lxd, ly, lyp = xn, xd, y_true, yp

        if model_name in sk_drift or model_name in river_drift:
            drift_aw.update_detector(1 if ly != lyp else 0)
            if drift_aw.maybe_reset(i):
                triggers.append(i + 1)

        # learn
        if model_name in ("plain", "sgd", "mlp_deep"):
            recent.append((lxn[0], ly))
            if len(recent) >= BATCH:
                Xb = np.array([v[0] for v in recent], dtype=np.float32)
                yb = np.array([v[1] for v in recent])
                model.partial_fit(Xb, yb, classes=[0, 1])
                fitted = True
                recent = []
        elif model_name in sk_drift:
            drift_aw.push_warm(lxn, ly)
            recent.append((lxn[0], ly))
            if len(recent) >= BATCH:
                Xb = np.array([v[0] for v in recent], dtype=np.float32)
                yb = np.array([v[1] for v in recent])
                drift_aw.model.partial_fit(Xb, yb, classes=[0, 1])
                drift_aw.fitted = True
                recent = []
        elif model_name == "periodic":
            periodic.update(lxn, ly)
        elif model_name in river_plain:
            model.learn_one(lxd, ly)
        elif model_name == "arf":
            model.learn_one(lxd, ly)
            fitted = True
        elif model_name in river_drift:
            drift_aw.model.learn_one(lxd, ly)
            drift_aw.push_warm(lxd, ly)

    res = sm.results()
    row = {
        "dataset": dataset, "scenario": scenario_name, "seed": seed,
        "model": model_name, "lr": args.lr, "window": args.window,
        "rec_thr": args.rec_thr, "warmup": args.warmup, "cooldown": args.cooldown,
        "delta": args.delta, "start_after": args.start_after,
        "finetune_epochs": args.finetune_epochs,
        "use_cat": int(args.use_cat), "n_samples": n,
        "n_triggers": len(triggers),
        "trigger_samples": ";".join(map(str, triggers)),
        "per_onset_recovery": res["recovery"],
        "per_onset_err_auc": res["err_auc"],
        "overall_f1": res["overall_f1"], "overall_gmean": res["overall_gmean"],
        "steady_gmean": res.get("steady_gmean", "NA"),
        "runtime_s": f"{time.time()-t0:.1f}",
    }
    append_row(row, args.out)
    print(f"[{dataset}/{scenario_name}/seed{seed}/{model_name}] {time.time()-t0:.0f}s "
          f"rec={res['recovery']} auc={res['err_auc']} f1={res['overall_f1']} "
          f"trig={len(triggers)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["unsw_full", "unsw_sub", "nslkdd"])
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--models", nargs="+", default=["plain", "drift"],
                    choices=["plain", "mlp_deep", "drift", "drift_ft_lo", "drift_ft_mid",
                             "mlp_deep_drift", "mlp_deep_drift_ft", "periodic",
                             "ht_plain", "ht_drift", "hat_plain", "hat_drift",
                             "gnb", "gnb_drift", "logreg", "logreg_drift",
                             "sgd", "sgd_drift", "sgd_drift_ft", "arf"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--use-cat", action="store_true")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--rec-thr", type=float, default=0.15)
    ap.add_argument("--warmup", type=int, default=512)
    ap.add_argument("--cooldown", type=int, default=5000)
    ap.add_argument("--delta", type=float, default=0.002)
    ap.add_argument("--period", type=int, default=5000)
    ap.add_argument("--start-after", type=int, default=5000,
                    help="drift detection warm-up: ignore triggers before this sample")
    ap.add_argument("--out", default=os.path.join(RESULTS, "experiment_matrix.csv"))
    ap.add_argument("--finetune-epochs", type=int, default=0)
    ap.add_argument("--finetune-lr", type=float, default=1e-2)
    ap.add_argument("--label-delay", type=int, default=0,
                    help="defer model updates by N samples (0 = immediate)")
    args = ap.parse_args()

    scenarios = (["S1_abrupt", "S2_gradual", "S3_recurrent", "S4_inversion"]
                 if args.scenario == "all" else [args.scenario])
    if args.dataset in ("unsw_full", "nslkdd") and "Srare" not in scenarios:
        scenarios.append("Srare")
    done = done_rows(args.out)
    for ds in [args.dataset]:
        for scn in scenarios:
            for seed in args.seeds:
                for m in args.models:
                    ft_token = str(args.finetune_epochs) if args.finetune_epochs else "NA"
                    key = (ds, scn, str(seed), m, str(args.warmup), ft_token)
                    if key in done:
                        print(f"skip {key}")
                        continue
                    run_one(ds, scn, seed, m, args)


if __name__ == "__main__":
    main()

