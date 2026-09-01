# CLD-IDS — Closed-Loop Drift-responsive Intrusion Detection System

Code and results for the manuscript:

> **An Online Incremental Intrusion Detection Framework under Concept Drift:
> Design, Mechanisms, and Time-Aware Evaluation**
> (Sensors, under review — details withheld until acceptance)

CLD-IDS is a closed-loop, instance-level framework that couples ADWIN
error-channel detection (gated by warm-up/cooldown) with a family of
update policies (reset / warm-start replay / high-learning-rate
fine-tuning) and a parallel KSWIN-like feature-channel attribution.
Evaluation follows a time-aware prequential protocol: recovery time,
post-onset error AUC, and rolling F1/G-mean over 10 seeds with paired
Wilcoxon tests.

## Environment

- Python 3.14, river 0.26, scikit-learn, pandas, scipy, matplotlib
- CPU-only, no GPU required
- `pip install river scikit-learn pandas scipy matplotlib`

## Data (not bundled — public sources)

| Dataset | Source |
|---|---|
| UNSW-NB15 official (4 CSVs, 2.54M rows) | Zenodo 10140548 — `UNSW-NB15_1..4.csv` |
| UNSW-NB15 175k subset | GitHub mirror — `UNSW_NB15.csv` |
| NSL-KDD | GitHub (Jehuty4949/NSL_KDD) — `NSLKDD_Train/Test.csv` |

Note: the official UNSW-NB15 `attack_cat` is labelled for only 12.6% of
rows (official limitation); this work uses the full 2.54M pool with 100%
per-family retention where `attack_cat` is present (see `results/DATA_AUDIT.md`).

## Reproduce the full matrix (700 runs)

```bash
# UNSW full (4 main scenarios x 10 seeds x 6 configs)
python 11_experiment_runner.py --dataset unsw_full --models plain drift periodic ht_plain ht_drift --seeds 0..9
python 11_experiment_runner.py --dataset unsw_full --models drift --seeds 0..9 --warmup 2000 --finetune-epochs 3
# NSL-KDD (incl. Srare)
python 11_experiment_runner.py --dataset nslkdd --models plain drift drift_ft periodic ht_plain ht_drift --seeds 0..9
# Stats + integrity checks
python 12_stats.py
python check_matrix.py
python audit_data.py
```

## Key configuration (see paper Methods)

- Online MLP: 32-16 hidden, lr=5e-4 (constant), batch=64, max_iter=1/epoch, online StandardScaler
- drift-aware: ADWIN(delta=0.002), warm-up 5000, cooldown 5000, warm-start replay 2000, high-lr (1e-2) fine-tune 3 epochs
- HT: grace=50, delta=1e-5, mc leaf prediction
- ARF: river default 10 trees, mc leaves (strongest baseline: F1 0.9996 main / 0.9871 rare, but slowest: 372 us/sample)
- Evaluation: prequential, rolling window 2000, recovery threshold 0.15, error-AUC horizon 10000, 10 seeds, Wilcoxon Pratt exact

## Results (aggregates)

- `results/stats_summary.md` — main tables + Wilcoxon significance
- `results/overhead_bench.md` — latency/throughput/memory benchmark
- `results/sensitivity_grid.csv` — 6-hyperparameter sensitivity grid
- `results/dual_channel.csv` — error/feature channel alignment
- `results/figures/` — paper figures 1-5 (300 dpi)

## AI-assisted research disclosure

Generative AI was used to assist drafting and language polishing under
author supervision; all scientific claims and numerical results were
verified by the authors.

## License

MIT (see LICENSE).
