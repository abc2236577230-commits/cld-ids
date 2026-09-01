# Overhead benchmark (T5)

- stream: `unsw_full/S1_abrupt` seed 0, n=60000, onsets=[np.int64(0), np.int64(50000), np.int64(100000), np.int64(140000)]
- features: 38 numeric + 0 categorical; Windows-11-10.0.26200-SP0
- model-side latency excludes shared pre-processing (scaling/encoding); `wall` is end-to-end incl. pre-processing

| model | us/sample | pred us | learn us | wall us | k samples/s | peak mem MB | triggers |
|---|---|---|---|---|---|---|---|
| plain | 300 | 229 | 71 | 509 | 2.0 | 0.3 | 0 |
| drift | 312 | 238 | 74 | 478 | 2.1 | 0.3 | 2 |
| drift_ft | 316 | 237 | 79 | 480 | 2.1 | 1.0 | 2 |
| periodic | 43 | 1 | 42 | 181 | 5.5 | 1.0 | 0 |
| ht_plain | 69 | 13 | 56 | 237 | 4.2 | 0.3 | 0 |
| ht_drift | 80 | 13 | 66 | 273 | 3.7 | 1.2 | 1 |
| arf | 372 | 118 | 255 | 549 | 1.8 | 0.3 | 0 |
