import json
import os
import sys
import unicodedata
from collections import defaultdict


# ------------------------------------------------------------------
# Path handling to ensure 'src' imports work correctly
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# ------------------------------------------------------------------ 
# Topic normalization mapping # Keys: raw topic variants
# # Values: canonical topic name 
# ------------------------------------------------------------------ 
TOPIC_NORMALIZATION_MAP = { 
    # Science & Technology 
    "công nghệ": "khoa học công nghệ", 
    "khoa học công nghệ": "khoa học công nghệ",
    "khoa học": "khoa học công nghệ", 
    "số hóa": "khoa học công nghệ", 
    # Education 
    "giáo dục - hướng nghiệp": "giáo dục", 
    "giáo dục": "giáo dục",
    "khoa giáo": "giáo dục", 
    # Real estate 
    "bất động sản": "bất động sản", 
    "nhà đất": "bất động sản", 
    # Health 
    "sức khỏe": "sức khỏe", 
    "sức khoẻ": "sức khỏe", 
    "y tế": "sức khỏe", 
    # Military 
    "người lính": "quân sự", 
    "quân sự": "quân sự", 
    # Society & Culture 
    "xã hội": "văn hóa - xã hội", 
    "văn hóa": "văn hóa - xã hội", 
    "văn hoá": "văn hóa - xã hội", 
    # World / International 
    "thế giới": "thế giới", 
    "quốc tế": "thế giới", 
    # Economy / Business 
    "kinh doanh": "kinh tế", 
    "kinh tế": "kinh tế", 
}

def normalize_topic(topic: str) -> str: 
    """ Normalize topic string: 
        - Unicode normalize (NFC) 
        - Lowercase 
        - Map to canonical topic name if available 
    """ 
    topic = unicodedata.normalize("NFC", topic.lower()) 
    return TOPIC_NORMALIZATION_MAP.get(topic, topic)

def group_data_by_full_context(json_path: str) -> dict:
    """
    Group data by full_context.
    For each full_context, collect:
      - statements with their labels
      - normalized topic (first seen)

    Returns:
        dict: {full_context: grouped_object}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped_data = {}

    for item in data:
        full_context = item.get("Context")
        if not full_context:
            continue

        statement = item.get("Statement")
        label = item.get("labels")
        topic = item.get("Topic")

        if not statement:
            continue

        if full_context not in grouped_data:
            grouped_data[full_context] = {
                "full_context": full_context,
                "topic": normalize_topic(topic) if topic else None,
                # dùng dict để tránh duplicate statement
                "statements": {},
            }

        # dùng statement text làm key để tránh trùng
        if statement not in grouped_data[full_context]["statements"]:
            grouped_data[full_context]["statements"][statement] = {
                "text": statement,
                "label": label
            }

    # --------------------------------------------------
    # Remove contexts with only 1 statement
    # --------------------------------------------------
    filtered_grouped_data = {}

    for ctx, obj in grouped_data.items():
        if len(obj["statements"]) > 1:
            obj["statements"] = list(obj["statements"].values())
            filtered_grouped_data[ctx] = obj

    return filtered_grouped_data

def save_grouped_full_context(
    grouped_data: dict,
    output_path: str
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(grouped_data.values()), f, ensure_ascii=False, indent=2)

    print(f"Saved {len(grouped_data)} grouped contexts to {output_path}")

# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    dev_data_path = os.path.join(
        PROJECT_ROOT, "data/retrieval/dev_data.json"
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "experiments/synthesis_data_generation/dev/results/dev_parse_data/dev_grouped_full_context.json"
    )

    grouped_contexts = group_data_by_full_context(dev_data_path)

    save_grouped_full_context(
        grouped_contexts,
        output_path=output_dir
    )
