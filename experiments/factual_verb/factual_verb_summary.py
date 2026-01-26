import os
import sys
import pandas as pd
from functools import reduce

# ============================
# CONFIG
# ============================

# --------------------------------------------------
# PATH HANDLING
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

base_dir = os.path.join(project_root, "Vietfactcheck", "experiments", "factual_verb")
print(base_dir)
splits = ["train", "dev", "test"]

TOP_K = 30

# ============================
# Load CSVs
# ============================

dfs = {}

for split in splits:
    path = os.path.join(base_dir, split, "results", "top_factual_verbs.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = df.rename(columns={"frequency": f"freq_{split}"})
    dfs[split] = df

# ============================
# Merge on verb
# ============================

merged = reduce(lambda l, r: pd.merge(l, r, on="verb", how="outer"), dfs.values())

merged = merged.fillna(0)

# ============================
# Aggregate statistics
# ============================

merged["total_freq"] = merged[[f"freq_{s}" for s in splits]].sum(axis=1)

# stability score: appears in how many splits
merged["split_presence"] = (merged[[f"freq_{s}" for s in splits]] > 0).sum(axis=1)

# mean frequency
merged["mean_freq"] = merged["total_freq"] / len(splits)

# ============================
# Rank factual verbs
# ============================

merged = merged.sort_values(
    by=["split_presence", "total_freq"],
    ascending=False
)

top_verbs = merged.head(TOP_K)

print("\n=== MOST SUITABLE FACTUAL VERBS (Cross-split consensus) ===")
print(top_verbs)

# ============================
# Save results
# ============================

out_path = os.path.join(base_dir, "consensus_factual_verbs.csv")
top_verbs.to_csv(out_path, index=False)

print("\nSaved:")
print(out_path)
