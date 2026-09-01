"""Stats for the official full UNSW-NB15: row count, attack_cat coverage, families."""
import pandas as pd
from collections import Counter

total = Counter()
ac = Counter()
n = 0
for i in range(1, 5):
    for chunk in pd.read_csv(f"data/UNSW-NB15_{i}.csv", header=None, chunksize=500000,
                             dtype={1: "str", 3: "str", 47: "str"}):
        n += len(chunk)
        ac["null"] += int(chunk.iloc[:, 47].isna().sum())
        ac["val"] += int(chunk.iloc[:, 47].notna().sum())
        total.update(chunk.iloc[:, 47].dropna().astype(str).str.strip().value_counts().to_dict())
print(f"total rows: {n:,}")
print(f"attack_cat null={ac['null']:,} val={ac['val']:,}")
print("top families:", total.most_common(12))