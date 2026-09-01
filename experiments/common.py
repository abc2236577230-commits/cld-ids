"""Shared components for the experiment pipeline.

- dataset loading (UNSW-NB15 full/subset, NSL-KDD) with optional categorical
  online ordinal encoding
- without-replacement scenario construction (fallback with-replacement for
  scarce families, flagged)
- rolling metrics (error / F1 / G-mean), recovery time (post-onset window),
  post-onset error AUC, steady-state and overall metrics
"""
import collections
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")

UNSW_FULL_COLS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload",
    "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt",
    "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "label",
]
UNSW_FULL_NUM = [c for c in UNSW_FULL_COLS if c not in
                 ("srcip", "sport", "dstip", "dsport", "proto", "state", "service",
                  "stime", "ltime", "attack_cat", "label")]
UNSW_FULL_CAT = ["proto", "state", "service"]
SKIP_SUBSET = {"id", "label", "attack_cat", "proto", "service", "state"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_unsw(full=True, use_cat=False, max_rows=0):
    """Load UNSW-NB15. full=True -> official 4 CSV: keeps rows with attack_cat
    annotation or label==0 (official CSVs leave attack_cat empty for most
    attack rows; documented dataset defect). full=False -> 175k subset."""
    if full:
        parts = [os.path.join(DATA, f"UNSW-NB15_{i}.csv") for i in range(1, 5)]
        for p in parts:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} missing; run download_unsw2.ps1")
        frames = []
        for p in parts:
            chunk_list = []
            for ch in pd.read_csv(p, header=None, dtype={1: "str", 3: "str",
                                                         47: "str"},
                                  chunksize=500000, low_memory=False):
                ch.columns = UNSW_FULL_COLS
                keep = ch["attack_cat"].notna() | (ch["label"].astype(str).str.strip() == "0")
                chunk_list.append(ch[keep])
            frames.append(pd.concat(chunk_list, ignore_index=True))
        df = pd.concat(frames, ignore_index=True)
        if max_rows:
            df = df.iloc[:max_rows]
        if use_cat:
            feats = [c for c in UNSW_FULL_NUM if c in df.columns]
            cat_cols = [c for c in UNSW_FULL_CAT if c in df.columns]
        else:
            feats = [c for c in UNSW_FULL_NUM if c in df.columns]
            cat_cols = []
        X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        cat_vals = {c: df[c].astype(str).str.strip().to_numpy() for c in cat_cols}
        y = (df["label"].astype(str).str.strip() == "1").astype(np.int8).to_numpy()
        fam = df["attack_cat"].fillna("Normal").astype(str).str.strip().to_numpy()
        return X, cat_vals, y, fam, feats
    else:
        df = pd.read_csv(os.path.join(DATA, "UNSW_NB15.csv"), encoding="utf-8-sig")
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        if max_rows:
            df = df.iloc[:max_rows]
        if use_cat:
            feats = [c for c in df.columns if c not in SKIP_SUBSET
                     and c not in ("proto", "service", "state", "attack_cat", "label", "id")]
            cat_cols = [c for c in ("proto", "service", "state") if c in df.columns]
        else:
            feats = [c for c in df.columns if c not in SKIP_SUBSET
                     and c != "attack_cat" and c != "label" and c != "id"]
            cat_cols = []
        X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        cat_vals = {c: df[c].astype(str).str.strip().to_numpy() for c in cat_cols}
        y = (df["label"].astype(str).str.strip() == "1").astype(np.int8).to_numpy()
        fam = df["attack_cat"].astype(str).str.strip().to_numpy()
        return X, cat_vals, y, fam, feats


def load_nslkdd():
    parts = []
    for name in ("NSLKDD_Train.csv", "NSLKDD_Test.csv"):
        parts.append(pd.read_csv(os.path.join(DATA, name), header=None))
    df = pd.concat(parts, ignore_index=True)
    df.columns = [f"f{i}" for i in range(43)]
    y = (df.iloc[:, -2].astype(str).str.strip() != "normal").astype(np.int8).to_numpy()
    fam = df.iloc[:, -2].astype(str).str.strip().to_numpy()
    fam = np.array([next((f for f, names in NSL_FAMILIES.items() if v in names), "normal")
                    for v in fam], dtype=object)
    X = df.iloc[:, :41].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    cat_vals = {c: df[c].astype(str).str.strip().to_numpy() for c in NSL_CAT}
    feats = [f"f{i}" for i in range(41) if f"f{i}" not in NSL_CAT]
    return X, cat_vals, y, fam, feats


NSL_FAMILIES = {
    "dos": ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
            "udpstorm", "processtable", "worm", "mailbomb"],
    "probe": ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"],
    "r2l": ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
            "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
            "httptunnel", "sendmail", "named"],
    "u2r": ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
            "xterm", "ps"],
}
NSL_CAT = ["f1", "f2", "f3"]


# ---------------------------------------------------------------------------
# Scenario construction (without replacement by default)
# ---------------------------------------------------------------------------
def build_scenario(fam, rng, spec, pool_map):
    """spec: list of (kind, n). kind in pool_map or 'inv:<pool>' (label flip)
    or 'ramp_up:<pool>' / 'ramp_down:<pool>'. Returns idx, onsets, flips, flags."""
    idx, onsets, cur, flips, flags = [], [], 0, [], []
    for kind, n in spec:
        if kind.startswith("inv:"):
            pool = kind.split(":", 1)[1]
            onsets.append(cur)
            flips.append((cur, cur + n))
        elif kind.startswith("ramp_"):
            pool = kind.split(":", 1)[1]
        else:
            pool = kind
            onsets.append(cur)
        avail = pool_map[pool]
        replace = len(avail) < n
        flags.append((kind, n, replace))
        if kind.startswith("inv:"):
            pick = rng.choice(avail, size=n, replace=replace)
            idx.extend(int(v) for v in pick)
        elif kind.startswith("ramp_up:"):
            att = rng.choice(pool_map[pool], size=n, replace=replace)
            nor = rng.choice(pool_map["Normal"], size=n, replace=replace)
            for j in range(n):
                idx.append(int(att[j]) if rng.random() < (j + 1) / n else int(nor[j]))
        elif kind.startswith("ramp_down:"):
            att = rng.choice(pool_map[pool], size=n, replace=replace)
            nor = rng.choice(pool_map["Normal"], size=n, replace=replace)
            for j in range(n):
                idx.append(int(att[j]) if rng.random() < 1 - (j + 1) / n else int(nor[j]))
        else:
            idx.extend(int(v) for v in rng.choice(avail, size=n, replace=replace))
        cur += n
    return np.array(idx, dtype=np.int64), np.array(onsets, dtype=np.int64), flips, flags


def scenario_specs(dataset_size="full"):
    """Scenario definitions with segment sizes compatible with pool sizes."""
    if dataset_size == "nslkdd":  # pools: normal 74k, dos ~55k, probe ~14k, r2l ~2k
        return {
            "S1_abrupt": [("normal", 30000), ("dos", 30000), ("normal", 20000),
                          ("probe", 12000)],
            "S2_gradual": [("normal", 20000), ("ramp_up:dos", 20000), ("dos", 20000),
                           ("ramp_down:dos", 20000), ("normal", 15000)],
            "S3_recurrent": [("normal", 25000), ("dos", 20000), ("normal", 25000),
                             ("probe", 12000), ("normal", 25000), ("dos", 20000)],
            "S4_inversion": [("normal", 20000), ("dos", 30000), ("inv:dos", 30000),
                             ("inv:dos", 30000)],
            "Srare": [("normal", 10000), ("r2l", 2000), ("normal", 10000)],
        }
    if dataset_size == "full":  # official UNSW annotated subset pool sizes
        # pools: Normal 2.22M, Generic 215k, Exploits 44.5k, Fuzzers 24.2k,
        #        DoS 16.4k, Recon 14.0k, Analysis 2.7k, Shellcode 1.5k
        return {
            "S1_abrupt": [("Normal", 50000), ("Generic", 50000), ("Normal", 40000),
                          ("Exploits", 40000)],
            "S2_gradual": [("Normal", 40000), ("ramp_up:Generic", 40000), ("Generic", 40000),
                           ("ramp_down:Generic", 40000), ("Normal", 30000)],
            "S3_recurrent": [("Normal", 50000), ("Fuzzers", 20000), ("Normal", 50000),
                             ("Exploits", 40000), ("Normal", 50000), ("Generic", 50000)],
            "S4_inversion": [("Normal", 40000), ("Generic", 50000), ("inv:Generic", 50000),
                             ("inv:Generic", 50000)],
            # rare-family scenario mirroring the NSL-KDD Srare design
            # (10k normal / 2k rare / 10k normal); Analysis pool 2677 >= 2000,
            # so sampling stays without replacement
            "Srare": [("Normal", 10000), ("Analysis", 2000), ("Normal", 10000)],
        }
    else:  # subset 175k pools
        return {
            "S1_abrupt": [("Normal", 30000), ("Generic", 30000), ("Normal", 20000),
                          ("Exploits", 30000)],
            "S2_gradual": [("Normal", 20000), ("ramp_up:Generic", 20000), ("Generic", 20000),
                           ("ramp_down:Generic", 20000), ("Normal", 15000)],
            "S3_recurrent": [("Normal", 25000), ("Fuzzers", 15000), ("Normal", 25000),
                             ("Exploits", 25000), ("Normal", 25000), ("Generic", 25000)],
            "S4_inversion": [("Normal", 20000), ("Generic", 30000), ("inv:Generic", 30000),
                             ("inv:Generic", 30000)],
        }


def build_all_scenarios(X, y, fam, rng, size="full"):
    pools = {}
    for k in np.unique(fam):
        pools[str(k)] = np.where(fam == k)[0]
    if "Normal" not in pools and "normal" in pools:
        pools["Normal"] = pools.pop("normal")
    if size == "nslkdd":
        # aliases so specs can use lower-case family names
        for k in list(pools):
            pools.setdefault(k.lower(), pools[k])
    specs = scenario_specs(size)
    out = {}
    for name, spec in specs.items():
        idx, onsets, flips, flags = build_scenario(fam, rng, spec, pools)
        ysel = y[idx].copy()
        for s, e in flips:
            ysel[s:e] = 1 - ysel[s:e]
        out[name] = {
            "X": X[idx], "y": ysel, "onsets": onsets, "n": idx.size,
            "spec": spec, "flags": flags,
        }
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class RollingCM:
    def __init__(self, window):
        self.window = window
        self.buf = np.zeros((window, 2), dtype=np.int8)
        self.pos = 0
        self.n = 0

    def update(self, y, yp):
        self.buf[self.pos] = (y, yp)
        self.pos = (self.pos + 1) % self.window
        self.n = min(self.n + 1, self.window)

    def cm(self):
        b = self.buf[: self.n]
        return np.bincount(b[:, 0] * 2 + b[:, 1], minlength=4).reshape(2, 2)

    def err(self):
        b = self.buf[: self.n]
        return float(np.mean(b[:, 0] != b[:, 1]))

    def f1(self):
        cm = self.cm()
        tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
        return 2 * tp / max(2 * tp + fp + fn, 1e-9)

    def gmean(self):
        cm = self.cm()
        tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
        tpr = tp / max(tp + fn, 1e-9)
        tnr = tn / max(tn + fp, 1e-9)
        return float(np.sqrt(tpr * tnr))


class StreamMetrics:
    """Collects per-onset recovery (post-onset window error), post-onset error
    AUC over a horizon, and per-segment steady-state G-mean."""

    def __init__(self, onsets, n, window=2000, rec_thr=0.15, auc_horizon=10000):
        self.onsets = list(onsets)
        self.n = n
        self.window = window
        self.rec_thr = rec_thr
        self.auc_horizon = auc_horizon
        self.recovery = {o: None for o in onsets}
        self.err_auc = {o: None for o in onsets}
        self.post = {o: collections.deque(maxlen=window) for o in onsets}
        self.acc = {o: 0.0 for o in onsets}      # cumulative errors since onset
        self.cnt = {o: 0 for o in onsets}
        self._auc_done = {o: False for o in onsets}
        self.total = RollingCM(window * 100)      # overall confusion
        self.steady = {}                          # onset -> gmean over last segment quarter

    def update(self, i, y_true, y_pred, seg_idx):
        err = 1 if y_true != y_pred else 0
        self.total.update(y_true, y_pred)
        for o in self.onsets:
            if i > o:
                self.post[o].append(err)
                self.acc[o] += err
                self.cnt[o] += 1
                if self.recovery[o] is None and len(self.post[o]) == self.window \
                        and np.mean(self.post[o]) <= self.rec_thr:
                    self.recovery[o] = int(i - o)
                if not self._auc_done[o] and self.cnt[o] >= self.auc_horizon:
                    self.err_auc[o] = self.acc[o] / self.auc_horizon
                    self._auc_done[o] = True
        if seg_idx is not None and seg_idx > 0:
            if i >= self.onsets[seg_idx] + self.window * 2:
                self.steady[self.onsets[seg_idx]] = self.total.gmean()

    def results(self):
        return {
            "recovery": ";".join(str(self.recovery[o]) for o in self.onsets),
            "err_auc": ";".join(f"{self.err_auc[o]:.4f}" if self.err_auc[o] is not None else "NA"
                                for o in self.onsets),
            "overall_f1": f"{self.total.f1():.4f}",
            "overall_gmean": f"{self.total.gmean():.4f}",
        }


def ks_localization(Xs, onsets, feats, win=2000):
    """Per-feature KS scores between pre/post-onset windows (offline attribution)."""
    rows = []
    for o in onsets:
        if o < win:
            continue
        pre, post = Xs[o - win : o], Xs[o : o + win]
        scores = [(feats[j], float(stats.ks_2samp(pre[:, j], post[:, j])[0]))
                  for j in range(Xs.shape[1])]
        scores.sort(key=lambda t: -t[1])
        rows.append((o, scores[:5]))
    return rows