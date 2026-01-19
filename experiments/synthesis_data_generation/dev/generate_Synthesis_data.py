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

CONTEXT_GENERATION_PROMPT = """
Bạn là chuyên gia tạo ngữ cảnh cho dữ liệu kiểm tra sự thật.

Chủ đề: {topic}

Cho các CÂU TUYÊN BỐ:
{statements} và nhãn của chúng {labels}

Hãy tạo ra MỘT đoạn CONTEXT GIẢ sao cho:
- Phù hợp với chủ đề
- Logic, tự nhiên như bài báo
- Sao chép nguyên văn statements nhưng bổ sung thêm các từ nối để câu văn trôi chảy, mạch lạc

CHỈ TRẢ VỀ MỘT CHUỖI VĂN BẢN (string).
KHÔNG JSON, KHÔNG markdown, KHÔNG giải thích.
"""


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def generate_fake_context(
    topic: str,
    statements: List[str],
    labels: List[int],
) -> str:
    prompt = CONTEXT_GENERATION_PROMPT.format(
        topic=topic,
        statements="\n".join(f"- {s}" for s in statements),
        labels="\n".join(f"- {l}" for l in labels),
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text.strip()

def process_grouped_full_context(
    input_path: str,
    output_path: str,
    max_items: int = None,
):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    success = 0

    for idx, item in enumerate(data):
        if max_items and idx >= max_items:
            break

        try:
            # ----------------------------------------
            # Extract statements + labels from new format
            # ----------------------------------------
            statements_text = [s["text"] for s in item["statements"]]
            labels = [s["label"] for s in item["statements"]]

            fake_context = generate_fake_context(
                topic=item["topic"],
                statements=statements_text,
                labels=labels,
            )

            results.append({
                "topic": item["topic"],
                "Context": fake_context,
                # giữ nguyên structure để dùng về sau
                "Statement_list": [s["text"] for s in item["statements"]],
            })

            success += 1
            print(f"✅ Generated context {success}")

        except Exception as e:
            print(f"❌ Failed at index {idx}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Done: {success}/{len(data)} samples")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    input_file = os.path.join(
        PROJECT_ROOT,
        "experiments/synthesis_data_generation/dev/results/dev_parse_data/dev_grouped_full_context.json"
    )

    output_file = os.path.join(
        PROJECT_ROOT,
        "experiments/synthesis_data_generation/dev/results/dev_synthesis_data/dev_fake_context.json"
    )

    process_grouped_full_context(
        input_path=input_file,
        output_path=output_file,
        max_items=20,   # để test trước
    )
