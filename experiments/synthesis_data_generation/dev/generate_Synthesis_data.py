import os
import json
import time
import sys
from typing import List
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from google import genai

# ------------------------------------------------------------------
# Path handling
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# ======================================================================
# CONFIGURATION
# ======================================================================

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set")

MODEL_NAME = "models/gemini-2.5-pro" # Đã cập nhật model name mới nhất
client = genai.Client(api_key=GOOGLE_API_KEY)

CONTEXT_GENERATION_PROMPT = """
Bạn là chuyên gia tạo ngữ cảnh cho dữ liệu kiểm tra sự thật.
Chủ đề: {topic}
Cho các CÂU TUYÊN BỐ:
{statements}

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

def generate_fake_context(topic: str, statements: List[str]) -> str:
    """Hàm gọi API Gemini để sinh text"""
    prompt = CONTEXT_GENERATION_PROMPT.format(
        topic=topic,
        statements="\n".join(f"- {s}" for s in statements)
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    
    if not response.text:
        raise RuntimeError("Gemini returned an empty response")
        
    return response.text.strip()

def process_grouped_full_context(
    input_path: str,
    output_path: str,
    max_items: int | None = None,
    max_retries: int = 5,
    time_sleep_retry: int = 15
):
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if max_items:
        data = data[:max_items]
    
    total = len(data)
    results = []

    # 2. Resume Logic: Nếu file output đã tồn tại, load dữ liệu cũ
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with out_path.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
                # Tạo map để check nhanh dựa trên topic/statements (hoặc ID nếu có)
                results = existing_data
                print(f"[+] Resuming: Found {len(results)} existing records.")
        except Exception as e:
            print(f"[!] Could not resume: {e}")

    def save_progress(data_to_save):
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

    # 3. Processing loop
    try:
        with tqdm(total=total, unit="rec", desc="Generating Context") as pbar:
            # Nếu resume, skip pbar tới vị trí hiện tại
            pbar.update(len(results))

            for idx in range(len(results), total):
                item = data[idx]
                statements_text = [s["text"] for s in item["statements"]]
                
                success = False
                attempts = 0

                while not success and attempts < max_retries:
                    try:
                        fake_context = generate_fake_context(
                            topic=item["topic"],
                            statements=statements_text
                        )
                        
                        # Thêm kết quả mới vào list
                        results.append({
                            "topic": item["topic"],
                            "Context": fake_context,
                            "Statement_list": statements_text,
                        })
                        
                        # Lưu ngay lập tức (Checkpoint)
                        save_progress(results)
                        success = True
                        pbar.update(1)

                    except Exception as e:
                        attempts += 1
                        err_msg = str(e)
                        
                        # Kiểm tra lỗi Rate Limit hoặc Server
                        if any(code in err_msg for code in ("429", "ResourceExhausted", "503", "500")):
                            print(f"\n[!] Rate limit/Server error (Attempt {attempts}/{max_retries}). Sleeping {time_sleep_retry}s...")
                            time.sleep(time_sleep_retry)
                        else:
                            print(f"\n[!] Unrecoverable error at index {idx}: {e}")
                            save_progress(results) # Lưu những gì đã làm được
                            raise e # Dừng chương trình nếu là lỗi logic/auth

            print(f"\n🎉 Done: Processed {len(results)} samples")

    except KeyboardInterrupt:
        print("\n[!] Process interrupted by user. Saving progress...")
        save_progress(results)
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] Critical error: {e}")
        save_progress(results)
        raise

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
        max_items=150, 
        max_retries=10,
        time_sleep_retry=10
    )