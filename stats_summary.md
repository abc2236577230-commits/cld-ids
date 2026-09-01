# Recovery time (samples, mean over seeds; drift onsets excl. cold start)

| dataset | scenario | model | rec_mean | rec_std | err_auc_mean | f1_mean |
|---|---|---|---|---|---|---|
| nslkdd | S1_abrupt | arf | 2000 ± 0 | 0.0024 | 0.9991 |
| nslkdd | S1_abrupt | drift | 3498 ± 1906 | 0.1250 | 0.8983 |
| nslkdd | S1_abrupt | drift_ft | 2000 ± 0 | 0.0164 | 0.9758 |
| nslkdd | S1_abrupt | ht_drift | 2096 ± 31 | 0.0495 | 0.9814 |
| nslkdd | S1_abrupt | ht_plain | 2643 ± 457 | 0.0608 | 0.9773 |
| nslkdd | S1_abrupt | periodic | 6699 ± 0 | 0.4999 | 0.8101 |
| nslkdd | S1_abrupt | plain | 2984 ± 172 | 0.1251 | 0.9313 |
| nslkdd | S2_gradual | arf | 2000 ± 0 | 0.0001 | 0.9936 |
| nslkdd | S2_gradual | drift | 2275 ± 451 | 0.0530 | 0.8872 |
| nslkdd | S2_gradual | drift_ft | 2000 ± 0 | 0.0002 | 0.9708 |
| nslkdd | S2_gradual | ht_drift | 2000 ± 0 | 0.0150 | 0.9520 |
| nslkdd | S2_gradual | ht_plain | 2000 ± 0 | 0.0312 | 0.9596 |
| nslkdd | S2_gradual | periodic | 2000 ± 0 | 0.0017 | 0.9713 |
| nslkdd | S2_gradual | plain | 2000 ± 0 | 0.0162 | 0.9562 |
| nslkdd | S3_recurrent | arf | 2000 ± 0 | 0.0024 | 0.9988 |
| nslkdd | S3_recurrent | drift | 3646 ± 1626 | 0.1542 | 0.8742 |
| nslkdd | S3_recurrent | drift_ft | 2054 ± 20 | 0.0271 | 0.9717 |
| nslkdd | S3_recurrent | ht_drift | 2260 ± 75 | 0.0642 | 0.9679 |
| nslkdd | S3_recurrent | ht_plain | 2291 ± 193 | 0.0504 | 0.9714 |
| nslkdd | S3_recurrent | periodic | 5899 ± 0 | 0.4199 | 0.7879 |
| nslkdd | S3_recurrent | plain | 2623 ± 125 | 0.1009 | 0.9258 |
| nslkdd | S4_inversion | arf | 2000 ± 0 | 0.0014 | 0.9993 |
| nslkdd | S4_inversion | drift | 2666 ± 425 | 0.1386 | 0.8673 |
| nslkdd | S4_inversion | drift_ft | 2189 ± 10 | 0.0325 | 0.9585 |
| nslkdd | S4_inversion | ht_drift | 2160 ± 24 | 0.0437 | 0.9780 |
| nslkdd | S4_inversion | ht_plain | 2014 ± 7 | 0.0197 | 0.9902 |
| nslkdd | S4_inversion | periodic | 5133 ± 0 | 0.3333 | 0.8333 |
| nslkdd | S4_inversion | plain | 3510 ± 217 | 0.1730 | 0.8845 |
| nslkdd | Srare | arf | 2000 ± 0 | 0.0049 | 0.9876 |
| nslkdd | Srare | drift | 6309 ± 2232 | 0.3989 | 0.2389 |
| nslkdd | Srare | drift_ft | 3392 ± 170 | 0.1443 | 0.4673 |
| nslkdd | Srare | ht_drift | 3449 ± 30 | 0.1304 | 0.6942 |
| nslkdd | Srare | ht_plain | 2271 ± 84 | 0.1001 | 0.7876 |
| nslkdd | Srare | periodic | 2850 ± 0 | 0.1999 | 0.0000 |
| nslkdd | Srare | plain | 2973 ± 361 | 0.2023 | 0.0675 |
| unsw_full | S1_abrupt | arf | 2000 ± 0 | 0.0023 | 0.9996 |
| unsw_full | S1_abrupt | drift | 4090 ± 3740 | 0.1443 | 0.9193 |
| unsw_full | S1_abrupt | drift_ft | 2009 ± 21 | 0.0140 | 0.9864 |
| unsw_full | S1_abrupt | ht_drift | 2136 ± 31 | 0.0430 | 0.9901 |
| unsw_full | S1_abrupt | ht_plain | 2479 ± 380 | 0.0691 | 0.9874 |
| unsw_full | S1_abrupt | periodic | 6699 ± 0 | 0.4999 | 0.9143 |
| unsw_full | S1_abrupt | plain | 2474 ± 178 | 0.0777 | 0.9742 |
| unsw_full | S2_gradual | arf | 2000 ± 0 | 0.0006 | 0.9972 |
| unsw_full | S2_gradual | drift | 2269 ± 481 | 0.0780 | 0.8973 |
| unsw_full | S2_gradual | drift_ft | 2000 ± 0 | 0.0001 | 0.9853 |
| unsw_full | S2_gradual | ht_drift | 2000 ± 0 | 0.0083 | 0.9905 |
| unsw_full | S2_gradual | ht_plain | 2000 ± 0 | 0.0084 | 0.9906 |
| unsw_full | S2_gradual | periodic | 2000 ± 0 | 0.0003 | 0.9927 |
| unsw_full | S2_gradual | plain | 2000 ± 0 | 0.0012 | 0.9825 |
| unsw_full | S3_recurrent | arf | 2000 ± 0 | 0.0024 | 0.9995 |
| unsw_full | S3_recurrent | drift | 4910 ± 5671 | 0.1578 | 0.9230 |
| unsw_full | S3_recurrent | drift_ft | 2023 ± 15 | 0.0217 | 0.9956 |
| unsw_full | S3_recurrent | ht_drift | 2133 ± 25 | 0.0389 | 0.9909 |
| unsw_full | S3_recurrent | ht_plain | 2500 ± 405 | 0.0694 | 0.9804 |
| unsw_full | S3_recurrent | periodic | 6699 ± 0 | 0.4999 | 0.9000 |
| unsw_full | S3_recurrent | plain | 2394 ± 98 | 0.0659 | 0.9872 |
| unsw_full | S4_inversion | arf | 2000 ± 0 | 0.0017 | 0.9994 |
| unsw_full | S4_inversion | drift | 2721 ± 432 | 0.1693 | 0.8613 |
| unsw_full | S4_inversion | drift_ft | 2134 ± 14 | 0.0249 | 0.9751 |
| unsw_full | S4_inversion | ht_drift | 2127 ± 1 | 0.0353 | 0.9864 |
| unsw_full | S4_inversion | ht_plain | 2137 ± 7 | 0.0351 | 0.9894 |
| unsw_full | S4_inversion | periodic | 5133 ± 0 | 0.3333 | 0.9000 |
| unsw_full | S4_inversion | plain | 2915 ± 254 | 0.1147 | 0.9457 |
| unsw_full | Srare | arf | 2000 ± 0 | 0.0051 | 0.9871 |
| unsw_full | Srare | drift | 3341 ± 2233 | 0.2983 | 0.2850 |
| unsw_full | Srare | drift_ft | 2179 ± 155 | 0.0701 | 0.6103 |
| unsw_full | Srare | ht_drift | 2253 ± 330 | 0.0853 | 0.7798 |
| unsw_full | Srare | ht_plain | 2122 ± 232 | 0.0584 | 0.8597 |
| unsw_full | Srare | periodic | 2850 ± 0 | 0.1999 | 0.0000 |
| unsw_full | Srare | plain | 2714 ± 382 | 0.1759 | 0.1928 |

## Paired Wilcoxon on per-onset recovery time (drift-aware vs baseline)

### unsw_full/S1_abrupt

| model | onset_idx | n_seeds | rec_mean | baseline_mean | p | wins |
|---|---|---|---|---|---|---|
| drift_ft vs plain | 1 | 10 | 2000 | 3130 | 0.0020 | 10/10 |
| drift_ft vs plain | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs plain | 3 | 10 | 2028 | 2293 | 0.0020 | 10/10 |
| drift_ft vs periodic | 1 | 10 | 2000 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 2 | 10 | 2000 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 3 | 10 | 2028 | 6699 | 0.0020 | 10/10 |
| drift_ft vs ht_plain | 1 | 10 | 2000 | 2249 | 0.0020 | 10/10 |
| drift_ft vs ht_plain | 2 | 10 | 2000 | 2515 | 0.0078 | 8/10 |
| drift_ft vs ht_plain | 3 | 10 | 2028 | 2674 | 0.0020 | 10/10 |
| drift vs plain | 1 | 10 | 2721 | 3130 | 0.0840 | 7/10 |
| drift vs plain | 2 | 10 | 6870 | 2000 | 0.0312 | 0/10 |
| drift vs plain | 3 | 10 | 2679 | 2293 | 0.4316 | 5/10 |
| ht_drift vs ht_plain | 1 | 10 | 2197 | 2249 | 0.0020 | 10/10 |
| ht_drift vs ht_plain | 2 | 10 | 2166 | 2515 | 0.9102 | 3/10 |
| ht_drift vs ht_plain | 3 | 10 | 2047 | 2674 | 0.0039 | 9/10 |
| drift_ft vs arf | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 3 | 10 | 2028 | 2000 | 0.5000 | 0/10 |
| ht_drift vs arf | 1 | 10 | 2197 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 2 | 10 | 2166 | 2000 | 0.0039 | 0/10 |
| ht_drift vs arf | 3 | 10 | 2047 | 2000 | 0.2500 | 0/10 |

### unsw_full/S2_gradual

| model | onset_idx | n_seeds | rec_mean | baseline_mean | p | wins |
|---|---|---|---|---|---|---|
| drift_ft vs plain | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs plain | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs periodic | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs periodic | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs ht_plain | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs ht_plain | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift vs plain | 1 | 10 | 2304 | 2000 | 1.0000 | 0/10 |
| ht_drift vs ht_plain | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| ht_drift vs ht_plain | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| ht_drift vs arf | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| ht_drift vs arf | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |

### unsw_full/S3_recurrent

| model | onset_idx | n_seeds | rec_mean | baseline_mean | p | wins |
|---|---|---|---|---|---|---|
| drift_ft vs plain | 1 | 10 | 2012 | 3122 | 0.0020 | 10/10 |
| drift_ft vs plain | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs plain | 3 | 10 | 2000 | 2539 | 0.0020 | 10/10 |
| drift_ft vs plain | 4 | 10 | 2037 | 2000 | 0.0156 | 0/10 |
| drift_ft vs plain | 5 | 10 | 2068 | 2307 | 0.0020 | 10/10 |
| drift_ft vs periodic | 1 | 10 | 2012 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 2 | 10 | 2000 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 3 | 10 | 2000 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 4 | 10 | 2037 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 5 | 10 | 2068 | 6699 | 0.0020 | 10/10 |
| drift_ft vs ht_plain | 1 | 10 | 2012 | 2249 | 0.0020 | 10/10 |
| drift_ft vs ht_plain | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs ht_plain | 3 | 10 | 2000 | 2302 | 0.0020 | 10/10 |
| drift_ft vs ht_plain | 4 | 10 | 2037 | 3589 | 0.0020 | 10/10 |
| drift_ft vs ht_plain | 5 | 10 | 2068 | 2357 | 0.7422 | 2/10 |
| drift vs plain | 1 | 10 | 2743 | 3122 | 0.1602 | 7/10 |
| drift vs plain | 2 | 10 | 8348 | 2000 | 0.0078 | 0/10 |
| drift vs plain | 3 | 10 | 2734 | 2539 | 0.9219 | 5/10 |
| drift vs plain | 4 | 10 | 8038 | 2000 | 0.0078 | 0/10 |
| drift vs plain | 5 | 10 | 2689 | 2307 | 0.9414 | 6/10 |
| ht_drift vs ht_plain | 1 | 10 | 2197 | 2249 | 0.0020 | 10/10 |
| ht_drift vs ht_plain | 2 | 10 | 2196 | 2000 | 0.0020 | 0/10 |
| ht_drift vs ht_plain | 3 | 10 | 2057 | 2302 | 0.0020 | 10/10 |
| ht_drift vs ht_plain | 4 | 10 | 2018 | 3589 | 0.0020 | 10/10 |
| ht_drift vs ht_plain | 5 | 10 | 2197 | 2357 | 0.4648 | 2/10 |
| drift_ft vs arf | 1 | 10 | 2012 | 2000 | 0.2500 | 0/10 |
| drift_ft vs arf | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 4 | 10 | 2037 | 2000 | 0.0156 | 0/10 |
| drift_ft vs arf | 5 | 10 | 2068 | 2000 | 0.0312 | 0/10 |
| ht_drift vs arf | 1 | 10 | 2197 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 2 | 10 | 2196 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 3 | 10 | 2057 | 2000 | 0.2500 | 0/10 |
| ht_drift vs arf | 4 | 10 | 2018 | 2000 | 1.0000 | 0/10 |
| ht_drift vs arf | 5 | 10 | 2197 | 2000 | 0.0020 | 0/10 |

### unsw_full/S4_inversion

| model | onset_idx | n_seeds | rec_mean | baseline_mean | p | wins |
|---|---|---|---|---|---|---|
| drift_ft vs plain | 1 | 10 | 2000 | 3032 | 0.0020 | 10/10 |
| drift_ft vs plain | 2 | 10 | 2400 | 3713 | 0.0020 | 10/10 |
| drift_ft vs plain | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs periodic | 1 | 10 | 2000 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 2 | 10 | 2400 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs ht_plain | 1 | 10 | 2000 | 2149 | 0.0020 | 10/10 |
| drift_ft vs ht_plain | 2 | 10 | 2400 | 2262 | 0.0020 | 0/10 |
| drift_ft vs ht_plain | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift vs plain | 1 | 10 | 2744 | 3032 | 0.1680 | 7/10 |
| ht_drift vs ht_plain | 1 | 10 | 2184 | 2149 | 0.0020 | 0/10 |
| ht_drift vs ht_plain | 2 | 10 | 2196 | 2262 | 0.0020 | 10/10 |
| ht_drift vs ht_plain | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 2 | 10 | 2400 | 2000 | 0.0020 | 0/10 |
| drift_ft vs arf | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| ht_drift vs arf | 1 | 10 | 2184 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 2 | 10 | 2196 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |

### unsw_full/Srare

| model | onset_idx | n_seeds | rec_mean | baseline_mean | p | wins |
|---|---|---|---|---|---|---|
| drift_ft vs plain | 1 | 10 | 2358 | 3111 | 0.0371 | 8/10 |
| drift_ft vs plain | 2 | 10 | 2000 | 2316 | 1.0000 | 1/10 |
| drift_ft vs periodic | 1 | 10 | 2358 | 3699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs ht_plain | 1 | 10 | 2358 | 2000 | 0.0039 | 0/10 |
| drift_ft vs ht_plain | 2 | 10 | 2000 | 2244 | 0.1250 | 4/10 |
| drift vs plain | 1 | 10 | 3771 | 3111 | 0.3008 | 6/10 |
| ht_drift vs ht_plain | 1 | 10 | 2477 | 2000 | 0.0020 | 0/10 |
| ht_drift vs ht_plain | 2 | 10 | 2029 | 2244 | 0.3125 | 3/10 |
| drift_ft vs arf | 1 | 10 | 2358 | 2000 | 0.0039 | 0/10 |
| drift_ft vs arf | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| ht_drift vs arf | 1 | 10 | 2477 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 2 | 10 | 2029 | 2000 | 0.5000 | 0/10 |

### nslkdd/S1_abrupt

| model | onset_idx | n_seeds | rec_mean | baseline_mean | p | wins |
|---|---|---|---|---|---|---|
| drift_ft vs plain | 1 | 10 | 2000 | 3403 | 0.0020 | 10/10 |
| drift_ft vs plain | 2 | 10 | 2000 | 2494 | 0.0020 | 10/10 |
| drift_ft vs plain | 3 | 10 | 2000 | 3057 | 0.0020 | 10/10 |
| drift_ft vs periodic | 1 | 10 | 2000 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 2 | 10 | 2000 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 3 | 10 | 2000 | 6699 | 0.0020 | 10/10 |
| drift_ft vs ht_plain | 1 | 10 | 2000 | 2049 | 0.0020 | 10/10 |
| drift_ft vs ht_plain | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs ht_plain | 3 | 10 | 2000 | 3880 | 0.0020 | 10/10 |
| drift vs plain | 1 | 10 | 2454 | 3403 | 0.0020 | 10/10 |
| drift vs plain | 2 | 10 | 5349 | 2494 | 0.0098 | 1/10 |
| drift vs plain | 3 | 10 | 2690 | 3057 | 0.1602 | 7/10 |
| ht_drift vs ht_plain | 1 | 10 | 2197 | 2049 | 0.0020 | 0/10 |
| ht_drift vs ht_plain | 2 | 10 | 2092 | 2000 | 0.0312 | 0/10 |
| ht_drift vs ht_plain | 3 | 10 | 2000 | 3880 | 0.0020 | 10/10 |
| drift_ft vs arf | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| ht_drift vs arf | 1 | 10 | 2197 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 2 | 10 | 2092 | 2000 | 0.0312 | 0/10 |
| ht_drift vs arf | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |

### nslkdd/S2_gradual

| model | onset_idx | n_seeds | rec_mean | baseline_mean | p | wins |
|---|---|---|---|---|---|---|
| drift_ft vs plain | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs plain | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs periodic | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs periodic | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs ht_plain | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs ht_plain | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift vs plain | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift vs plain | 2 | 10 | 2550 | 2000 | 0.2500 | 0/10 |
| ht_drift vs ht_plain | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| ht_drift vs ht_plain | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| ht_drift vs arf | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| ht_drift vs arf | 2 | 10 | 2000 | 2000 | 1.0000 | 0/10 |

### nslkdd/S3_recurrent

| model | onset_idx | n_seeds | rec_mean | baseline_mean | p | wins |
|---|---|---|---|---|---|---|
| drift_ft vs plain | 1 | 10 | 2024 | 3542 | 0.0020 | 10/10 |
| drift_ft vs plain | 2 | 10 | 2053 | 2070 | 0.4316 | 2/10 |
| drift_ft vs plain | 3 | 10 | 2001 | 3250 | 0.0020 | 10/10 |
| drift_ft vs plain | 4 | 10 | 2194 | 2254 | 0.5566 | 7/10 |
| drift_ft vs plain | 5 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs periodic | 1 | 10 | 2024 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 2 | 10 | 2053 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 3 | 10 | 2001 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 4 | 10 | 2194 | 4699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 5 | 10 | 2000 | 4699 | 0.0020 | 10/10 |
| drift_ft vs ht_plain | 1 | 10 | 2024 | 2000 | 1.0000 | 0/10 |
| drift_ft vs ht_plain | 2 | 10 | 2053 | 2000 | 0.0020 | 0/10 |
| drift_ft vs ht_plain | 3 | 10 | 2001 | 3450 | 0.0020 | 10/10 |
| drift_ft vs ht_plain | 4 | 10 | 2194 | 2003 | 0.0039 | 0/10 |
| drift_ft vs ht_plain | 5 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift vs plain | 1 | 10 | 2700 | 3542 | 0.0020 | 10/10 |
| drift vs plain | 2 | 10 | 4237 | 2070 | 0.0039 | 1/10 |
| drift vs plain | 3 | 10 | 2882 | 3250 | 0.0840 | 9/10 |
| drift vs plain | 4 | 10 | 5958 | 2254 | 0.0039 | 1/10 |
| drift vs plain | 5 | 10 | 2451 | 2000 | 0.0312 | 0/10 |
| ht_drift vs ht_plain | 1 | 10 | 2439 | 2000 | 0.0020 | 0/10 |
| ht_drift vs ht_plain | 2 | 10 | 2062 | 2000 | 0.1250 | 0/10 |
| ht_drift vs ht_plain | 3 | 10 | 2177 | 3450 | 0.0020 | 10/10 |
| ht_drift vs ht_plain | 4 | 10 | 2196 | 2003 | 0.0020 | 0/10 |
| ht_drift vs ht_plain | 5 | 10 | 2426 | 2000 | 0.0020 | 0/10 |
| drift_ft vs arf | 1 | 10 | 2024 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 2 | 10 | 2053 | 2000 | 0.0020 | 0/10 |
| drift_ft vs arf | 3 | 10 | 2001 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 4 | 10 | 2194 | 2000 | 0.0039 | 0/10 |
| drift_ft vs arf | 5 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| ht_drift vs arf | 1 | 10 | 2439 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 2 | 10 | 2062 | 2000 | 0.1250 | 0/10 |
| ht_drift vs arf | 3 | 10 | 2177 | 2000 | 0.0039 | 0/10 |
| ht_drift vs arf | 4 | 10 | 2196 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 5 | 10 | 2426 | 2000 | 0.0020 | 0/10 |

### nslkdd/S4_inversion

| model | onset_idx | n_seeds | rec_mean | baseline_mean | p | wins |
|---|---|---|---|---|---|---|
| drift_ft vs plain | 1 | 10 | 2000 | 3508 | 0.0020 | 10/10 |
| drift_ft vs plain | 2 | 10 | 2566 | 5021 | 0.0020 | 10/10 |
| drift_ft vs plain | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs periodic | 1 | 10 | 2000 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 2 | 10 | 2566 | 6699 | 0.0020 | 10/10 |
| drift_ft vs periodic | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs ht_plain | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs ht_plain | 2 | 10 | 2566 | 2042 | 0.0020 | 0/10 |
| drift_ft vs ht_plain | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift vs plain | 1 | 10 | 2394 | 3508 | 0.0020 | 10/10 |
| ht_drift vs ht_plain | 1 | 10 | 2285 | 2000 | 0.0020 | 0/10 |
| ht_drift vs ht_plain | 2 | 10 | 2196 | 2042 | 0.0020 | 0/10 |
| ht_drift vs ht_plain | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 1 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| drift_ft vs arf | 2 | 10 | 2566 | 2000 | 0.0020 | 0/10 |
| drift_ft vs arf | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |
| ht_drift vs arf | 1 | 10 | 2285 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 2 | 10 | 2196 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 3 | 10 | 2000 | 2000 | 1.0000 | 0/10 |

### nslkdd/Srare

| model | onset_idx | n_seeds | rec_mean | baseline_mean | p | wins |
|---|---|---|---|---|---|---|
| drift_ft vs plain | 1 | 10 | 4392 | 3841 | 0.0781 | 1/10 |
| drift_ft vs plain | 2 | 10 | 2392 | 2106 | 0.0840 | 1/10 |
| drift_ft vs periodic | 1 | 10 | 4392 | 3699 | 0.0020 | 0/10 |
| drift_ft vs periodic | 2 | 10 | 2392 | 2000 | 0.0020 | 0/10 |
| drift_ft vs ht_plain | 1 | 10 | 4392 | 2000 | 0.0020 | 0/10 |
| drift_ft vs ht_plain | 2 | 10 | 2392 | 2542 | 0.1055 | 6/10 |
| ht_drift vs ht_plain | 1 | 10 | 4449 | 2000 | 0.0020 | 0/10 |
| ht_drift vs ht_plain | 2 | 10 | 2449 | 2542 | 0.1602 | 7/10 |
| drift_ft vs arf | 1 | 10 | 4392 | 2000 | 0.0020 | 0/10 |
| drift_ft vs arf | 2 | 10 | 2392 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 1 | 10 | 4449 | 2000 | 0.0020 | 0/10 |
| ht_drift vs arf | 2 | 10 | 2449 | 2000 | 0.0020 | 0/10 |

## Overall prequential F1 (mean over seeds)

| dataset | scenario | plain | drift | drift_ft | periodic | ht_plain | ht_drift | arf |
|---|---|---|---|---|---|---|---|---|
| unsw_full | S1_abrupt | 0.9742 | 0.9193 | 0.9864 | 0.9143 | 0.9874 | 0.9901 | 0.9996 |
| unsw_full | S2_gradual | 0.9825 | 0.8973 | 0.9853 | 0.9927 | 0.9906 | 0.9905 | 0.9972 |
| unsw_full | S3_recurrent | 0.9872 | 0.9230 | 0.9956 | 0.9000 | 0.9804 | 0.9909 | 0.9995 |
| unsw_full | S4_inversion | 0.9457 | 0.8613 | 0.9751 | 0.9000 | 0.9894 | 0.9864 | 0.9994 |
| unsw_full | Srare | 0.1928 | 0.2850 | 0.6103 | 0.0000 | 0.8597 | 0.7798 | 0.9871 |
| nslkdd | S1_abrupt | 0.9313 | 0.8983 | 0.9758 | 0.8101 | 0.9773 | 0.9814 | 0.9991 |
| nslkdd | S2_gradual | 0.9562 | 0.8872 | 0.9708 | 0.9713 | 0.9596 | 0.9520 | 0.9936 |
| nslkdd | S3_recurrent | 0.9258 | 0.8742 | 0.9717 | 0.7879 | 0.9714 | 0.9679 | 0.9988 |
| nslkdd | S4_inversion | 0.8845 | 0.8673 | 0.9585 | 0.8333 | 0.9902 | 0.9780 | 0.9993 |
| nslkdd | Srare | 0.0675 | 0.2389 | 0.4673 | 0.0000 | 0.7876 | 0.6942 | 0.9876 |