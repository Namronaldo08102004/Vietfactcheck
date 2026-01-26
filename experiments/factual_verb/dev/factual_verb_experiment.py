import json
import os
import sys
from collections import Counter, defaultdict
import pandas as pd
from underthesea import pos_tag
from tqdm import tqdm

# --------------------------------------------------
# PATH HANDLING
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

# ============================
# CONFIG
# ============================

JSON_PATH = os.path.join(project_root, "data", "extraction", "dev_synthesis.json")   # your dataset
TOP_K = 20               # number of verbs to show

# ============================
# Load dataset
# ============================

with open(JSON_PATH, "r", encoding="utf8") as f:
    data = json.load(f)

# if your file is JSONL, replace above with:
# data = [json.loads(line) for line in open(JSON_PATH, encoding="utf8")]

# ============================
# Collect all statements
# ============================

all_texts = []
label_texts = defaultdict(list)

for sample in data:
    for s in sample["statements"]:
        txt = s["text"]
        lbl = s["label"]

        all_texts.append(txt)
        label_texts[lbl].append(txt)

print("Total statements:", len(all_texts))

# ============================
# POS tagging → extract verbs
# ============================

verb_counter = Counter()
label_verb_counter = defaultdict(Counter)

for text in tqdm(all_texts):
    tags = pos_tag(text)

    for word, tag in tags:
        if tag.startswith("V"):     # Verb
            verb_counter[word.lower()] += 1

# per label
for lbl, texts in label_texts.items():
    for t in texts:
        tags = pos_tag(t)
        for w, tag in tags:
            if tag.startswith("V"):
                label_verb_counter[lbl][w.lower()] += 1

# ============================
# Global statistics
# ============================

df = pd.DataFrame(
    verb_counter.most_common(TOP_K),
    columns=["verb", "frequency"]
)

print("\n=== TOP FACTUAL VERBS (GLOBAL) ===")
print(df)

# ============================
# Per-label statistics
# ============================

label_tables = {}

for lbl, counter in label_verb_counter.items():
    label_tables[lbl] = pd.DataFrame(
        counter.most_common(15),
        columns=["verb", "frequency"]
    )

    print(f"\n=== LABEL {lbl} ===")
    print(label_tables[lbl])

# ============================
# Save results
# ============================

output_dir = os.path.join(project_root, "experiments", "factual_verb", "dev", "results")

# Create directory tree if not exists
os.makedirs(output_dir, exist_ok=True)

# Save global verbs
df.to_csv(os.path.join(output_dir, "top_factual_verbs.csv"), index=False)

# Save per-label verbs
for lbl, table in label_tables.items():
    table.to_csv(os.path.join(output_dir, f"verbs_label_{lbl}.csv"), index=False)

print("\nSaved:")
print(f"- {os.path.join(output_dir, 'top_factual_verbs.csv')}")
print(f"- {os.path.join(output_dir, 'verbs_label_*.csv')}")
