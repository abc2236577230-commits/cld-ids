"""Extended experiment driver (2026-09-13).

Runs the new cells needed for the de-bundled 2x2 factorial and the
learner-plasticity spectrum:

  1. buffer x fine-tune de-confounding (MLP):
       drift @ buffer2000, ft0   (new cell)
       drift @ buffer512,  ft3   (new cell)
     together with existing drift@512,ft0 and drift@2000,ft3 -> full 2x2.
  2. Hoeffding Adaptive Tree: hat_plain, hat_drift
  3. linear gradient learner: sgd, sgd_drift, sgd_drift_ft

Appends rows to results/experiment_matrix_v2.csv (resumable via the
runner's done_rows key). Existing shards are untouched.

Usage:
  python 25_extended_runner.py --dataset unsw_full --wave all
  python 25_extended_runner.py --dataset nslkdd --wave spectrum --seeds 0 1
"""
import argparse
import importlib
import os
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
runner = importlib.import_module("11_experiment_runner")

OUT = os.path.join(HERE, "..", "results", "experiment_matrix_v2.csv")
OUT_DELAY = os.path.join(HERE, "..", "results", "experiment_matrix_delay.csv")
LOG = os.path.join(HERE, "..", "results", "_extended_run.log")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def _start_logging():
    f = open(LOG, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return f

BASE = dict(lr=5e-4, window=2000, rec_thr=0.15, warmup=512, cooldown=5000,
            delta=0.002, use_cat=0, start_after=5000, period=5000,
            finetune_epochs=0, finetune_lr=1e-2, label_delay=0)

# model, overrides
WAVE_BUFFER = [
    ("drift", dict(warmup=2000, finetune_epochs=0, finetune_lr=1e-2, lr=5e-4)),
    ("drift", dict(warmup=512, finetune_epochs=3, finetune_lr=1e-2, lr=5e-4)),
]
WAVE_SPECTRUM = [
    ("hat_plain", dict()),
    ("hat_drift", dict()),
    ("sgd", dict(lr=1e-2)),
    ("sgd_drift", dict(lr=1e-2, warmup=2000)),
    ("sgd_drift_ft", dict(lr=1e-2, warmup=2000, finetune_epochs=3, finetune_lr=1e-1)),
    ("gnb", dict()),
    ("gnb_drift", dict()),
]
# focused validation: 2nd convex learner + fine-tune learning-rate curve
WAVE_VALID = [
    ("logreg", dict()),
    ("logreg_drift", dict()),
    ("drift_ft_lo", dict(warmup=2000, finetune_epochs=3)),    # ft lr = 5e-4
    ("drift_ft_mid", dict(warmup=2000, finetune_epochs=3)),   # ft lr = 1e-3
]
# second non-convex configuration (deeper MLP) -> non-convex vs convex 2v2
WAVE_NONCONV = [
    ("mlp_deep", dict()),
    ("mlp_deep_drift", dict(warmup=2000)),
    ("mlp_deep_drift_ft", dict(warmup=2000, finetune_epochs=3, finetune_lr=1e-2)),
]
# label-delay hardening: updates deferred by DELAY samples
WAVE_DELAY = [
    ("plain", dict()),
    ("drift", dict(warmup=2000)),
    ("drift", dict(warmup=2000, finetune_epochs=3, finetune_lr=1e-2)),
    ("logreg_drift", dict()),
]
DELAY = 500
SCENARIOS = ["S1_abrupt", "S2_gradual", "S3_recurrent", "S4_inversion", "Srare"]
SCENARIOS_VALID = ["S1_abrupt", "S3_recurrent", "Srare"]


def make_args(overrides, out, label_delay=0):
    d = dict(BASE)
    d.update(overrides)
    d["out"] = out
    d["label_delay"] = label_delay
    return SimpleNamespace(**d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["unsw_full", "nslkdd", "all"])
    ap.add_argument("--wave", default="all",
                    choices=["buffer", "spectrum", "valid", "nonconv", "delay", "all"])
    ap.add_argument("--scenarios", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = ap.parse_args()
    _start_logging()
    print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start "
          f"ds={args.dataset} wave={args.wave} =====", flush=True)

    configs = []
    if args.wave in ("buffer", "all"):
        configs += WAVE_BUFFER
    if args.wave in ("spectrum", "all"):
        configs += WAVE_SPECTRUM
    if args.wave in ("valid", "all"):
        configs += WAVE_VALID
    if args.wave in ("nonconv", "all"):
        configs += WAVE_NONCONV
    if args.wave in ("delay", "all"):
        configs += WAVE_DELAY
    if args.scenarios is None:
        args.scenarios = SCENARIOS_VALID if args.wave in ("valid", "delay") else SCENARIOS

    out = OUT_DELAY if args.wave == "delay" else OUT
    delay = DELAY if args.wave == "delay" else 0
    datasets = (["nslkdd", "unsw_full"] if args.dataset == "all"
                else [args.dataset])
    done = runner.done_rows(out)
    total = len(configs) * len(args.scenarios) * len(args.seeds) * len(datasets)
    k = 0
    t0 = time.time()
    print(f"== extended run: ds={args.dataset} wave={args.wave} "
          f"cells={total} existing={len(done)} out={os.path.basename(out)} ==", flush=True)
    for ds in datasets:
        for model, ov in configs:
            run_args = make_args(ov, out, delay)
            for scn in args.scenarios:
                for seed in args.seeds:
                    ft_token = (str(run_args.finetune_epochs)
                                if run_args.finetune_epochs else "NA")
                    key = (ds, scn, str(seed), model,
                           str(run_args.warmup), ft_token)
                    k += 1
                    if key in done:
                        print(f"skip {key}", flush=True)
                        continue
                    runner.run_one(ds, scn, seed, model, run_args)
                    done.add(key)
    print(f"== done {k} cells in {time.time()-t0:.0f}s ==", flush=True)


if __name__ == "__main__":
    main()
