"""M4: dual-channel integration - KSWIN feature channel + ADWIN error channel.

Runs the drift_ft framework on unsw_full/S1_abrupt (seed 0) with a KSWIN
feature channel monitoring the top-10 variance features (sampled every 5).
At each ADWIN trigger, reports the features whose KSWIN fired within the
previous 5000 samples (online attribution). Consistency is measured against
offline KS localization at the nearest known onset.

Output: results/dual_channel.csv
"""
import collections
import csv
import os
import sys

import numpy as np
from river import drift, preprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import common
from experiments.models import DriftAware, make_mlp

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
BATCH = 64
MONITOR_K = 10
SAMPLE_EVERY = 5
LOOKBACK = 5000
PRIOR_FEATURES = ["sttl", "ct_state_ttl", "dbytes", "rate", "sload"]  # classic drift-sensitive


def main():
    X, cat_vals, y, fam, feats = common.load_unsw(full=True, use_cat=False)
    rng = np.random.default_rng(0)
    sc = common.build_all_scenarios(X, y, fam, rng, "full")["S1_abrupt"]
    Xs, ys, onsets = sc["X"], sc["y"], sc["onsets"]
    var = Xs.var(axis=0)
    top = sorted(range(Xs.shape[1]), key=lambda j: -var[j])[:MONITOR_K]
    top_feats = list(dict.fromkeys([feats[j] for j in top] + PRIOR_FEATURES))

    scaler = preprocessing.StandardScaler()
    d = DriftAware(lambda: make_mlp(lr=5e-4, seed=0), delta=0.002, cooldown=5000,
                   warmup=2000, batch=BATCH, start_after=5000,
                   finetune_epochs=3, finetune_lr=1e-2)
    kswins = {f: drift.KSWIN(alpha=0.005, window_size=100, seed=42) for f in top_feats}
    fires = collections.defaultdict(list)
    recent = []
    triggers = []

    for i in range(sc["n"]):
        y_true = int(ys[i])
        xd = {f: float(Xs[i, j]) for j, f in enumerate(feats)}
        xs = scaler.transform_one(xd)
        xn = np.array([xs[f] for f in feats], dtype=np.float32).reshape(1, -1)
        scaler.learn_one(xd)
        if i % SAMPLE_EVERY == 0:
            for j, f in zip(top, top_feats):
                kswins[f].update(float(Xs[i, j]))
                if kswins[f].drift_detected:
                    fires[f].append(i)
        yp = d.predict(xn)
        err = 1 if y_true != yp else 0
        d.update_detector(err)
        if d.maybe_reset(i):
            triggers.append(i + 1)
        d.push_warm(xn, y_true)
        recent.append((xn[0], y_true))
        if len(recent) >= BATCH:
            Xb = np.array([v[0] for v in recent], dtype=np.float32)
            yb = np.array([v[1] for v in recent])
            if d.fitted:
                d.model.partial_fit(Xb, yb, classes=[0, 1])
            else:
                d.model.partial_fit(Xb, yb, classes=[0, 1])
                d.fitted = True
            recent = []

    # offline KS localization at known onsets (full feature ranking)
    ks = {o: [] for o in onsets}
    for o in onsets:
        if o < 2000:
            continue
        from scipy import stats as st
        pre, post = Xs[o - 2000 : o], Xs[o : o + 2000]
        scores = [(feats[j], float(st.ks_2samp(pre[:, j], post[:, j])[0]))
                  for j in range(Xs.shape[1])]
        scores.sort(key=lambda t: -t[1])
        ks[o] = scores  # full ranking, strongest first

    # online attribution per trigger + consistency (KS percentile of flagged features)
    rows = []
    for t in triggers:
        near = min(onsets, key=lambda o: abs(o - t))
        fired = []
        for f in top_feats:
            c = sum(1 for x in fires[f] if max(0, t - LOOKBACK) <= x < t)
            if c > 0:
                fired.append((f, c))
        fired.sort(key=lambda kv: -kv[1])
        online_top = [f for f, _ in fired[:3]]
        ranking = {f: i for i, (f, _) in enumerate(ks.get(near, []))}
        pct = [1 - ranking[f] / max(len(ranking), 1) for f in online_top if f in ranking]
        mean_pct = float(np.mean(pct)) if pct else None
        rows.append([t, near, ";".join(online_top), mean_pct])
        print(f"trigger@{t} (onset@{near}): online={online_top} "
              f"mean_ks_percentile={mean_pct:.2f}" if mean_pct is not None else
              f"trigger@{t} (onset@{near}): online={online_top} mean_ks_percentile=None")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "dual_channel.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trigger_sample", "nearest_onset", "online_top3", "mean_ks_percentile"])
        w.writerows(rows)
    print(f"saved -> results/dual_channel.csv ({len(triggers)} triggers)")


if __name__ == "__main__":
    main()