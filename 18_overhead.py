"""Per-sample overhead benchmark (T5): model-side predict/update latency,
end-to-end throughput, and memory footprint under one real drift onset.

Usage: python 18_overhead.py [--n 60000] [--seed 0]
Writes results/overhead_bench.md
"""
import argparse
import os
import platform
import sys
import time
import tracemalloc

import numpy as np
from river import preprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import common
from experiments.models import DriftAware, PeriodicMLP, make_arf, make_ht, make_mlp

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

BATCH = 64


def bench(model_name, Xs, ys, onsets, feats, cat_cols, cat_vals, maps, n, lr=5e-4):
    scaler = preprocessing.StandardScaler()
    maps = {c: {} for c in cat_cols}
    if model_name == "plain":
        model = make_mlp(lr=lr, seed=0)
        kind = "mlp"
    elif model_name == "drift":
        model = DriftAware(lambda: make_mlp(lr=lr, seed=0), delta=0.002,
                           cooldown=5000, warmup=512, batch=BATCH, start_after=5000)
        kind = "drift_mlp"
    elif model_name == "drift_ft":
        model = DriftAware(lambda: make_mlp(lr=lr, seed=0), delta=0.002,
                           cooldown=5000, warmup=2000, batch=BATCH, start_after=5000,
                           finetune_epochs=3, finetune_lr=1e-2)
        kind = "drift_mlp"
    elif model_name == "periodic":
        model = PeriodicMLP(period=5000, window=2000, lr=1e-3, seed=0,
                            n_features=len(feats) + len(cat_cols))
        kind = "periodic"
    elif model_name == "ht_plain":
        model = make_ht()
        kind = "ht"
    elif model_name == "arf":
        model = make_arf(seed=0)
        kind = "arf"
    elif model_name == "ht_drift":
        model = DriftAware(lambda: make_ht(), delta=0.002, cooldown=5000,
                           warmup=512, batch=BATCH, start_after=5000)
        model.fitted = True
        kind = "drift_ht"
    fitted = kind == "ht"
    recent = []
    triggers = []
    t_pred = t_learn = 0.0
    tracemalloc.start()
    mem_base, _ = tracemalloc.get_traced_memory()
    t0 = time.perf_counter()
    for i in range(n):
        y_true = int(ys[i])
        xd = {}
        for c in cat_cols:
            xd[c] = maps[c].setdefault(cat_vals[c][i], len(maps[c]))
        for j, f in enumerate(feats):
            xd[f] = float(Xs[i, j])
        xs = scaler.transform_one(xd)
        if kind in ("mlp", "drift_mlp", "periodic"):
            xn = np.array([xs[f] for f in feats] + [xs[c] for c in cat_cols],
                          dtype=np.float32).reshape(1, -1)
        else:
            xn = xd
        scaler.learn_one(xd)

        tp = time.perf_counter()
        if kind == "mlp":
            yp = int(model.predict(xn)[0]) if fitted else 0
        elif kind in ("drift_mlp", "drift_ht"):
            yp = int(model.predict(xn))
        elif kind in ("periodic", "arf"):
            yp = int(model.predict_one(xn) or 0) if fitted else 0
        else:
            yp = int(model.predict_one(xn) or 0)
        t_pred += time.perf_counter() - tp

        tl = time.perf_counter()
        err = 1 if y_true != yp else 0
        if kind == "drift_mlp":
            model.update_detector(err)
            if model.maybe_reset(i):
                triggers.append(i + 1)
        elif kind == "drift_ht":
            model.update_detector(err)
            if model.maybe_reset(i):
                triggers.append(i + 1)
        if kind == "mlp":
            recent.append((xn[0], y_true))
            if len(recent) >= BATCH:
                Xb = np.array([v[0] for v in recent], dtype=np.float32)
                yb = np.array([v[1] for v in recent])
                model.partial_fit(Xb, yb, classes=[0, 1])
                fitted = True
                recent = []
        elif kind == "drift_mlp":
            model.push_warm(xn, y_true)
            recent.append((xn[0], y_true))
            if len(recent) >= BATCH:
                Xb = np.array([v[0] for v in recent], dtype=np.float32)
                yb = np.array([v[1] for v in recent])
                if model.fitted:
                    model.model.partial_fit(Xb, yb, classes=[0, 1])
                else:
                    model.model.partial_fit(Xb, yb, classes=[0, 1])
                    model.fitted = True
                recent = []
        elif kind == "periodic":
            model.update(xn, y_true)
        elif kind == "ht":
            model.learn_one(xn, y_true)
        elif kind == "arf":
            model.learn_one(xn, y_true)
            fitted = True
        elif kind == "drift_ht":
            model.model.learn_one(xn, y_true)
            model.push_warm(xn, y_true)
        t_learn += time.perf_counter() - tl
    wall = time.perf_counter() - t0
    mem_peak, _ = tracemalloc.get_traced_memory()
    mem_after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return {
        "us_per_sample": (t_pred + t_learn) / n * 1e6,
        "pred_us": t_pred / n * 1e6,
        "learn_us": t_learn / n * 1e6,
        "wall_us": wall / n * 1e6,
        "throughput_kps": n / wall / 1000.0,
        "mem_base_mb": mem_base / 1e6,
        "mem_peak_mb": mem_peak / 1e6,
        "mem_after_mb": mem_after / 1e6,
        "triggers": len(triggers),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unsw_full")
    ap.add_argument("--scenario", default="S1_abrupt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=60000)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.dataset == "unsw_full":
        X, cat_vals, y, fam, feats = common.load_unsw(full=True)
    elif args.dataset == "unsw_sub":
        X, cat_vals, y, fam, feats = common.load_unsw(full=False)
    else:
        X, cat_vals, y, fam, feats = common.load_nslkdd()
    sc = common.build_all_scenarios(X, y, fam, rng,
                                    "full" if args.dataset == "unsw_full"
                                    else ("sub" if args.dataset == "unsw_sub" else "nslkdd"))[args.scenario]
    Xs, ys, onsets, n = sc["X"], sc["y"], sc["onsets"], min(args.n, sc["n"])
    cat_cols = list(cat_vals.keys())
    print(f"stream={args.dataset}/{args.scenario} n={n} onsets={list(onsets)} "
          f"feats={len(feats)}+{len(cat_cols)}cat")

    order = ["plain", "drift", "drift_ft", "periodic", "ht_plain", "ht_drift", "arf"]
    lines = [
        "# Overhead benchmark (T5)",
        "",
        f"- stream: `{args.dataset}/{args.scenario}` seed {args.seed}, n={n}, "
        f"onsets={list(onsets)}",
        f"- features: {len(feats)} numeric + {len(cat_cols)} categorical; "
        f"{platform.platform()}",
        f"- model-side latency excludes shared pre-processing (scaling/encoding); "
        f"`wall` is end-to-end incl. pre-processing",
        "",
        "| model | us/sample | pred us | learn us | wall us | k samples/s | peak mem MB | triggers |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in order:
        r = bench(m, Xs, ys, onsets, feats, cat_cols, cat_vals, {}, n)
        lines.append(
            f"| {m} | {r['us_per_sample']:.0f} | {r['pred_us']:.0f} | {r['learn_us']:.0f} "
            f"| {r['wall_us']:.0f} | {r['throughput_kps']:.1f} | {r['mem_peak_mb']:.1f} "
            f"| {r['triggers']} |")
        print(m, r)
    out = os.path.join(RESULTS, "overhead_bench.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved ->", out)


if __name__ == "__main__":
    main()
