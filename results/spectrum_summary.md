# Learner-plasticity spectrum & de-bundled factorial

runs loaded: 2140; datasets: ['nslkdd', 'unsw_full']
models: ['arf', 'drift', 'drift_ft_lo', 'drift_ft_mid', 'gnb', 'gnb_drift', 'hat_drift', 'hat_plain', 'ht_drift', 'ht_plain', 'logreg', 'logreg_drift', 'mlp_deep', 'mlp_deep_drift', 'mlp_deep_drift_ft', 'periodic', 'plain', 'sgd', 'sgd_drift', 'sgd_drift_ft']

## A. MLP de-bundled 2x2 (post-onset error AUC)

Mean AUC per cell (lower = better recovery):

| cell | mean AUC | mean F1 | n |
|---|---|---|---|
| plain | 0.1277 | 0.7848 | 100 |
| drift_b512_ft0 | 0.1898 | 0.7652 | 100 |
| drift_b2000_ft0 | 0.1790 | 0.7688 | 100 |
| drift_b512_ft3 | 0.0750 | 0.8894 | 100 |
| drift_b2000_ft3 | 0.0753 | 0.8897 | 100 |

Paired Wilcoxon (AUC; positive diff = first cell worse):

| contrast | mean diff | p | n |
|---|---|---|---|
| FT effect (buf2000): ft3 - ft0 | -0.1037 | 0.0000 | 100 |
| FT effect (buf512): ft3 - ft0 | -0.1148 | 0.0000 | 100 |
| buffer effect (ft0): b2000 - b512 | -0.0108 | 0.1222 | 100 |
| buffer effect (ft3): b2000 - b512 | +0.0003 | 0.0000 | 100 |
| reset effect (plain->drift_ft0) | +0.0513 | 0.0000 | 100 |

## B. Reset-Harm Index by learner family

RHI = mean AUC(reset) - mean AUC(no-reset); positive = reset raises
post-onset error (reset harmful); negative = reset helps.

| comparison | mean AUC(a) | mean AUC(b) | diff(b-a) | p | n |
|---|---|---|---|---|---|
| MLP-32/16 (non-convex) | 0.1277 | 0.1790 | +0.0513 | 0.0000 | 100 |
| MLP-64/32/16 (non-convex) | 0.0810 | 0.1139 | +0.0329 | 0.0000 | 100 |
| MLP + FT | 0.1790 | 0.0753 | -0.1037 | 0.0000 | 100 |
| linear SGD (convex) | 0.0348 | 0.0268 | -0.0080 | 0.0000 | 100 |
| linear SGD + FT | 0.0268 | 0.0214 | -0.0054 | 0.0000 | 100 |
| logreg (convex) | 0.0548 | 0.0006 | -0.0542 | 0.0000 | 60 |
| HT (tree) | 0.0344 | 0.0338 | -0.0006 | 0.9452 | 100 |
| HAT (tree) | 0.0143 | 0.0257 | +0.0114 | 0.0000 | 100 |
| NB | 0.2518 | 0.1370 | -0.1147 | 0.0000 | 100 |
| periodic vs online | 0.1277 | 0.2218 | +0.0940 | 0.0000 | 100 |

## C. Fine-tune learning-rate curve (reset + buffer2000 + 3 epochs)

Restricted to S1_abrupt / S3_recurrent / Srare (lower AUC, higher F1 better).

| variant | mean AUC | mean F1 | n |
|---|---|---|---|
| no FT (ref) | 0.2059 | 0.6768 | 60 |
| FT lr=5e-4 | 0.1575 | 0.6559 | 60 |
| FT lr=1e-3 | 0.1380 | 0.7086 | 60 |
| FT lr=1e-2 | 0.0839 | 0.8345 | 60 |

## D. Label-delay hardening (updates deferred by 500 samples)

| variant | immediate AUC | delayed AUC | immediate F1 | delayed F1 |
|---|---|---|---|---|
| plain | 0.1385 | 0.1588 | 0.6798 | 0.6526 | n=60 |
| drift(buf2000) | 0.2059 | 0.2371 | 0.6768 | 0.6392 | n=60 |
| drift+FT(buf2000) | 0.0839 | 0.1164 | 0.8345 | 0.7776 | n=60 |
| logreg_drift | 0.0006 | 0.0477 | 0.9986 | 0.8927 | n=60 |

Both columns restricted to S1_abrupt / S3_recurrent / Srare.
