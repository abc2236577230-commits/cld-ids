"""UNSW-NB15 streaming baseline: Hoeffding Tree with prequential evaluation.

Usage:
  python 01_stream_baseline.py              # use real CSV at data/UNSW_NB15.csv
  python 01_stream_baseline.py --synthetic  # synthetic drifting stream, no data needed
  python 01_stream_baseline.py --max-rows 500000   # cap rows (memory/time control)
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from river import metrics, tree

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "UNSW_NB15.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

SYN_FEATURES = ["f0", "f1", "f2", "f3", "f4"]


class RollingCM:
    """Rolling confusion matrix over the last `window` predictions (ring buffer)."""

    def __init__(self, window=2000):
        self.window = window
        self.buf = np.zeros((window, 2), dtype=np.int8)
        self.pos = 0
        self.n = 0

    def update(self, y, y_pred):
        self.buf[self.pos] = (y, y_pred)
        self.pos = (self.pos + 1) % self.window
        self.n = min(self.n + 1, self.window)

    def cm(self):
        b = self.buf[: self.n]
        idx = b[:, 0] * 2 + b[:, 1]
        return np.bincount(idx, minlength=4).reshape(2, 2)

    def f1(self):
        tp, fp, fn, tn = self._parts()
        return 2 * tp / max(2 * tp + fp + fn, 1e-9)

    def recall(self, cls=1):
        tp, fp, fn, tn = self._parts()
        return tp / max(tp + fn, 1e-9) if cls == 1 else tn / max(tn + fp, 1e-9)

    def fpr(self, cls=1):
        tp, fp, fn, tn = self._parts()
        return fp / max(fp + tn, 1e-9)

    def _parts(self):
        cm = self.cm()
        tp, fn = cm[1, 1], cm[1, 0]
        fp, tn = cm[0, 1], cm[0, 0]
        return tp, fp, fn, tn


def synthetic_stream(n=20000, seed=42):
    """Three segments with different attack priors and feature shifts => concept drift."""
    rng = np.random.default_rng(seed)
    segs = [(8000, 0.20), (6000, 0.35), (6000, 0.60)]
    for seg, (nseg, p_att) in enumerate(segs):
        for _ in range(nseg):
            is_attack = rng.random() < p_att
            x = {}
            for i, f in enumerate(SYN_FEATURES):
                mu = 3.0 if (is_attack and i < 3) else 0.0
                x[f] = float(rng.normal(mu + seg * 0.3, 1.0))
            yield x, int(is_attack)


def real_stream(path, max_rows):
    """Lazy row-by-row stream from UNSW-NB15 CSV. Numeric columns only."""
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("empty CSV header")
        skip = {"id", "label", "attack_cat", "proto", "service", "state", "label_1"}
        fields = [c.lstrip("\ufeff") for c in reader.fieldnames]
        feats = [c for c in fields if c not in skip]
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            x = {}
            for c in feats:
                v = row[c].strip()
                try:
                    x[c] = float(v)
                except ValueError:
                    pass
            label = row.get("label", "0").strip()
            try:
                y = int(label)
            except ValueError:
                y = 1 if label.lower() in ("attack", "1", "true", "anomaly") else 0
            yield x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--window", type=int, default=2000, help="rolling metric window")
    ap.add_argument("--log-every", type=int, default=2000, help="record a point every N samples")
    args = ap.parse_args()

    if args.synthetic:
        stream_name, gen = "synthetic", synthetic_stream()
    else:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"{DATA_PATH} not found. Download UNSW-NB15 CSV into data/ "
                "or run with --synthetic to demo the pipeline."
            )
        stream_name, gen = os.path.basename(DATA_PATH), real_stream(DATA_PATH, args.max_rows)

    model = tree.HoeffdingTreeClassifier(
        grace_period=50, delta=1e-5, leaf_prediction="mc"
    )
    rolling = RollingCM(window=args.window)
    total = metrics.ClassificationReport()

    curve = []  # (idx, rolling_f1, rolling_recall_attack, rolling_fpr_attack)
    t0 = time.time()
    for i, (x, y) in enumerate(gen):
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        if y_pred is None:
            y_pred = 0
        rolling.update(y, y_pred)
        total.update(y, y_pred)
        if (i + 1) % args.log_every == 0:
            curve.append((i + 1, rolling.f1(), rolling.recall(), rolling.fpr()))
            if (i + 1) % (args.log_every * 5) == 0:
                el = time.time() - t0
                print(f"[{i+1:>9,}] rolling F1={rolling.f1():.4f} "
                      f"attack recall={rolling.recall():.4f} ({el:.0f}s)")

    os.makedirs(FIG_DIR, exist_ok=True)
    np.savetxt(
        os.path.join(RESULTS_DIR, "f1_curve.csv"),
        np.array(curve), delimiter=",", header="sample,rolling_f1,rolling_recall_attack,rolling_fpr_attack",
        comments="",
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    c = np.array(curve)
    ax.plot(c[:, 0], c[:, 1], label="rolling F1")
    ax.plot(c[:, 0], c[:, 2], label="rolling recall (attack)", linestyle="--")
    ax.set_xlabel("samples processed (stream order)")
    ax.set_ylabel("rolling metric (window=%d)" % args.window)
    ax.set_title(f"Prequential evaluation on {stream_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, "fig2_f1_curve.png")
    fig.savefig(out_png, dpi=200)
    print(f"saved figure -> {out_png}")
    print(f"final: F1={rolling.f1():.4f} attack-recall={rolling.recall():.4f} "
          f"attack-FPR={rolling.fpr():.4f}")
    print(total)


if __name__ == "__main__":
    main()