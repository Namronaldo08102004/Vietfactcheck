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
# Topic normalization mapping
# Keys: raw topic variants
# Values: canonical topic name
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
    """
    Normalize topic string:
    - Unicode normalize (NFC)
    - Lowercase
    - Map to canonical topic name if available
    """
    topic = unicodedata.normalize("NFC", topic.lower())
    return TOPIC_NORMALIZATION_MAP.get(topic, topic)


def group_data_by_topic(json_path: str) -> dict:
    """
    Group input JSON data by normalized topic.

    Args:
        json_path (str): Path to input JSON file

    Returns:
        dict: {topic_name: list_of_items}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped_data = defaultdict(list)

    for item in data:
        topic = item.get("Topic")
        if not topic:
            continue

        normalized_topic = normalize_topic(topic)
        grouped_data[normalized_topic].append(item)

    return grouped_data


def save_grouped_topics(
    grouped_data: dict,
    output_dir: str = "topics"
) -> None:
    """
    Save grouped data into separate JSON files by topic.

    Args:
        grouped_data (dict): Output from group_data_by_topic
        output_dir (str): Directory to save topic files
    """
    os.makedirs(output_dir, exist_ok=True)

    for topic, items in grouped_data.items():
        output_path = os.path.join(output_dir, f"{topic}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

        print("=" * 60)
        print(f"Topic: {topic}")
        print(f"Saved {len(items)} items to {output_path}")


# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    dev_data_path = os.path.join(PROJECT_ROOT, "data/retrieval/dev_data.json")
    dev_synthesis_data_path = os.path.join(PROJECT_ROOT, "experiments/synthesis_data_generation/dev/results/dev_parse_data")
    grouped_topics = group_data_by_topic(dev_data_path)
    save_grouped_topics(grouped_topics, output_dir=dev_synthesis_data_path)
