import json
import os
import sys
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# PATH HANDLING
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

# -----------------------
# Utils
# -----------------------

def tokenize(text):
    # simple word tokenizer
    return re.findall(r"\w+", text.lower())

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------
# Load datasets
# -----------------------

paths = {
    "train": os.path.join(project_root, "Github", "Vietfactcheck", "data", "extraction", "train_synthesis.json"),
    "dev": os.path.join(project_root, "Github", "Vietfactcheck", "data", "extraction", "dev_synthesis.json"),
    "test": os.path.join(project_root, "Github", "Vietfactcheck", "data", "extraction", "test_synthesis.json"),
}

datasets = {}
for split, path in paths.items():
    datasets[split] = load_json(path)

# -----------------------
# Collect Context & Statement text
# -----------------------

all_contexts = []
all_statements = []
all_topics = []

for split, data in datasets.items():
    for item in data:
        # fake context
        all_contexts.append(item["fake_context"])

        # topic
        all_topics.append(item["topic"])

        # statements
        for st in item["statements"]:
            all_statements.append(st["text"])

# -----------------------
# Length statistics
# -----------------------

def compute_lengths(texts):
    lengths = [len(tokenize(t)) for t in texts]
    return {
        "total": len(texts),
        "avg": sum(lengths) / len(lengths),
        "max": max(lengths),
        "min": min(lengths),
        "lengths": lengths,
    }

context_stats = compute_lengths(all_contexts)
statement_stats = compute_lengths(all_statements)

# -----------------------
# Vocabulary size
# -----------------------

context_vocab = set()
for c in all_contexts:
    context_vocab.update(tokenize(c))

statement_vocab = set()
for s in all_statements:
    statement_vocab.update(tokenize(s))

# ==================================================
# Per-split EDA
# ==================================================

def run_split_eda(name, contexts, statements):
    ctx_stats = compute_lengths(contexts)
    st_stats = compute_lengths(statements)

    ctx_vocab = set()
    for c in contexts:
        ctx_vocab.update(tokenize(c))

    st_vocab = set()
    for s in statements:
        st_vocab.update(tokenize(s))

    print(f"\n================ {name.upper()} =================")
    print("---- Context ----")
    print(f"Samples: {ctx_stats['total']}")
    print(f"Avg length: {ctx_stats['avg']:.2f}")
    print(f"Max length: {ctx_stats['max']}")
    print(f"Min length: {ctx_stats['min']}")
    print(f"Vocab size: {len(ctx_vocab)}")

    print("---- Statement ----")
    print(f"Samples: {st_stats['total']}")
    print(f"Avg length: {st_stats['avg']:.2f}")
    print(f"Max length: {st_stats['max']}")
    print(f"Min length: {st_stats['min']}")
    print(f"Vocab size: {len(st_vocab)}")


# -----------------------
# Split-wise collection
# -----------------------

split_data = {}

for split, data in datasets.items():
    contexts = []
    statements = []

    for item in data:
        contexts.append(item["fake_context"])
        for st in item["statements"]:
            statements.append(st["text"])

    split_data[split] = {
        "contexts": contexts,
        "statements": statements
    }

# -----------------------
# Run EDA for each split
# -----------------------

for split in ["train", "dev", "test"]:
    run_split_eda(split, split_data[split]["contexts"], split_data[split]["statements"])


# -----------------------
# Run EDA for TOTAL
# -----------------------

run_split_eda("total", all_contexts, all_statements)
# --------------------------------
# Count topics per split
# --------------------------------
split_counters = {}
for split, data in datasets.items():
    split_counters[split] = Counter([item["topic"] for item in data])

# Union of all topics
topics = sorted(set().union(*[c.keys() for c in split_counters.values()]))

# Sort topics by total frequency (nice ordering)
topics = sorted(
    topics,
    key=lambda t: split_counters["train"].get(t, 0)
                + split_counters["dev"].get(t, 0)
                + split_counters["test"].get(t, 0),
    reverse=True
)

train_vals = [split_counters["train"].get(t, 0) for t in topics]
dev_vals   = [split_counters["dev"].get(t, 0) for t in topics]
test_vals  = [split_counters["test"].get(t, 0) for t in topics]

# --------------------------------
# Plot
# --------------------------------

y = np.arange(len(topics))
height = 0.25

plt.figure(figsize=(10, 6))

plt.barh(y - height, train_vals, height, label="Train")
plt.barh(y, dev_vals, height, label="Dev")
plt.barh(y + height, test_vals, height, label="Test")

plt.yticks(y, topics)
plt.xlabel("Number of Samples")
plt.ylabel("Topics")
plt.title("Topic Distribution Across Train / Dev / Test Sets")

plt.legend()
plt.tight_layout()
plt.show()
