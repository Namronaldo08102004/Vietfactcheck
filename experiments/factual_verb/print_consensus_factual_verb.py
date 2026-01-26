import os
import sys
import pandas as pd

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

project_root = os.getcwd()
consensus_path = os.path.join(
    project_root,
    "consensus_factual_verbs.csv"
)

TOP_K = 20              # adjust if needed
REQUIRE_ALL_SPLITS = True

# ============================
# Load consensus
# ============================

df = pd.read_csv(consensus_path)

# ============================
# Selection criteria
# ============================

if REQUIRE_ALL_SPLITS:
    df = df[df["split_presence"] == 3]

# Top-K by total frequency
df = df.sort_values("total_freq", ascending=False).head(TOP_K)

# ============================
# Build FACTUAL_VERB
# ============================

FACTUAL_VERB = set(df["verb"].tolist())

print("\nFACTUAL_VERB =")
print(FACTUAL_VERB)