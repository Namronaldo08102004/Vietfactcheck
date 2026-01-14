import os
import json
import re
from typing import List, Dict, Any
import sys

from dotenv import load_dotenv
from google import genai

# ------------------------------------------------------------------
# Path handling to ensure 'src' imports work correctly
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))


# ======================================================================
# CONFIGURATION
# ======================================================================

# Load variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set")

MODEL_NAME = "models/gemini-2.5-flash"  # Free tier, model tồn tại thật
BASE_INDEX = 1

# Initialize Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

SYNTHESIS_PROMPT = """
Bạn là một chuyên gia tạo dữ liệu tổng hợp kiểm tra sự thật.

Chủ đề: {topic}

Cho bạn các ví dụ gốc sau đây về các câu tuyên bố liên quan đến chủ đề đã cho:
{examples}

Tạo ra {n} ví dụ MỚI với:
- Cùng Chủ đề
- Chỉ số khác
- Ngữ cảnh khác
- Bằng chứng khác với số lượng ngẫu nhiên (1 - 6 câu)
- Nhãn khác (1 / 0 / 2)
- Các câu tuyên bố phải thực tế và hợp lý
- Không sao chép câu từ các ví dụ gốc

CHỈ TRẢ VỀ JSON THUẦN (KHÔNG markdown, KHÔNG giải thích).

Định dạng:
[
  {{
    "index": 0,
    "Statement": "...",
    "Context": "...",
    "Evidence_list": ["..."],
    "labels": 0,
    "Topic": "{topic}"
  }}
]
"""


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def extract_json_from_text(text: str) -> Any:
    """
    Extract and parse JSON from Gemini output.
    Removes ```json or ``` fences if they exist.

    Args:
        text (str): Raw model output

    Returns:
        Parsed JSON object

    Raises:
        json.JSONDecodeError: If parsing fails
    """
    text = text.strip()

    # Remove Markdown code fences if present
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


def build_examples_text(data: List[Dict[str, Any]], max_examples: int = 5) -> str:
    """
    Build bullet-point example statements for prompt conditioning.

    Args:
        data (list): Original topic data
        max_examples (int): Number of examples to include

    Returns:
        str: Formatted example text
    """
    return "\n".join(
        f"- {item['Statement']}"
        for item in data[:max_examples]
        if "Statement" in item
    )


# ======================================================================
# CORE FUNCTION
# ======================================================================

def generate_synthetic_data(
    topic_json_path: str,
    output_path: str,
    n_samples: int = 5,
) -> bool:
    """
    Generate synthetic fact-checking samples for a given topic using Gemini.

    Args:
        topic_json_path (str): Path to topic-specific JSON file
        output_path (str): Path to save generated synthetic data
        n_samples (int): Number of synthetic samples to generate

    Returns:
        bool: True if generation succeeded, False otherwise
    """
    print(f"\n▶ Processing topic file: {topic_json_path}")

    # --------------------------------------------------------------
    # Load topic data
    # --------------------------------------------------------------
    try:
        with open(topic_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"❌ Failed to load JSON: {exc}")
        return False

    if not data:
        print("⚠️ Topic file is empty, skipping.")
        return False

    topic = data[0].get("Topic")
    if not topic:
        print("⚠️ Missing 'Topic' field, skipping.")
        return False

    # --------------------------------------------------------------
    # Prepare prompt
    # --------------------------------------------------------------
    examples_text = build_examples_text(data)

    if not examples_text.strip():
        print("⚠️ No valid example statements found, skipping.")
        return False

    prompt = SYNTHESIS_PROMPT.format(
        topic=topic,
        examples=examples_text,
        n=n_samples,
    )

    # --------------------------------------------------------------
    # Call Gemini API
    # --------------------------------------------------------------
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        raw_output = response.text
    except Exception as exc:
        print(f"❌ Gemini API call failed: {exc}")
        return False

    # --------------------------------------------------------------
    # Parse JSON output safely
    # --------------------------------------------------------------
    try:
        synthetic_data = extract_json_from_text(raw_output)
    except Exception:
        print("❌ Invalid JSON returned by Gemini")
        print("----- RAW OUTPUT START -----")
        print(raw_output)
        print("----- RAW OUTPUT END -----")
        return False

    if not isinstance(synthetic_data, list) or not synthetic_data:
        print("⚠️ Model output is not a non-empty list")
        return False

    # --------------------------------------------------------------
    # Post-process: validate & fix index / topic
    # --------------------------------------------------------------
    processed_data = []

    for idx, item in enumerate(synthetic_data):
        if not isinstance(item, dict):
            continue

        # Minimal schema validation
        if "Statement" not in item or "labels" not in item:
            continue

        item["index"] = BASE_INDEX + idx
        item["Topic"] = topic

        processed_data.append(item)

    if not processed_data:
        print("⚠️ No valid samples after validation, skipping.")
        return False

    # --------------------------------------------------------------
    # Save output (avoid silent overwrite)
    # --------------------------------------------------------------
    if os.path.exists(output_path):
        print(f"⚠️ Output file already exists, overwriting: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Saved {len(processed_data)} synthetic samples "
        f"to {output_path}"
    )

    return True


def process_all_topic_files(
    topic_dir: str,
    output_path: str,
    output_suffix: str = "_synthetic.json",
    n_samples: int = 5,
) -> None:
    """
    Find all topic JSON files in a directory and generate synthetic data
    for each of them.

    Args:
        topic_dir (str): Directory containing topic JSON files
        output_suffix (str): Suffix for generated output files
        n_samples (int): Number of synthetic samples per topic
    """
    if not os.path.isdir(topic_dir):
        raise NotADirectoryError(f"{topic_dir} is not a directory")

    json_files = sorted(
        f for f in os.listdir(topic_dir)
        if f.endswith(".json") and not f.endswith(output_suffix)
    )

    if not json_files:
        print("⚠️ No topic JSON files found.")
        return

    print(f"📂 Found {len(json_files)} topic files")

    success_count = 0

    for filename in json_files:
        input_path = os.path.join(topic_dir, filename)
        output_dir = os.path.join(
            output_path,
            filename.replace(".json", output_suffix),
        )
        success = generate_synthetic_data(
            topic_json_path=input_path,
            output_path=output_dir,
            n_samples=n_samples,
        )

        if success:
            success_count += 1

    print(
        f"\n✅ Finished processing: "
        f"{success_count}/{len(json_files)} succeeded"
    )

# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    topic_json_path = os.path.join(PROJECT_ROOT, "experiments/synthesis_data_generation/dev/results/dev_parse_data")
    output_path = os.path.join(PROJECT_ROOT, "experiments/synthesis_data_generation/dev/results/dev_synthesis_data")

    process_all_topic_files(
        topic_dir=topic_json_path,
        output_path=output_path,
        n_samples=10,
    )