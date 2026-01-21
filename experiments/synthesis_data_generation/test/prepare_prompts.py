import os
import json

# ------------------------------------------------------------------
# Thiết lập đường dẫn tương tự như code của bạn
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

# Định nghĩa Template Prompt
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

def generate_prompt_list(input_path: str, output_path: str):
    # 1. Đọc file input
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file tại: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_prepared_prompts = []

    print(f"🔄 Đang xử lý {len(data)} bản ghi...")

    # 2. Duyệt qua từng item và gán vào template
    for idx, item in enumerate(data):
        try:
            # Trích xuất thông tin
            topic = item.get("topic", "Không có chủ đề")
            statements_list = item.get("statements", [])
            
            statements_text = "\n".join(f"- {s['text']}" for s in statements_list)

            # Điền vào template
            final_prompt = CONTEXT_GENERATION_PROMPT.format(
                topic=topic,
                statements=statements_text
            )

            # Lưu vào danh sách (kèm một chút metadata để dễ quản lý)
            all_prepared_prompts.append({
                "id": idx + 1,
                "topic": topic,
                "prepared_prompt": final_prompt.strip()
            })

        except Exception as e:
            print(f"❌ Lỗi tại bản ghi thứ {idx}: {e}")

    # 3. Lưu kết quả ra file mới
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_prepared_prompts, f, ensure_ascii=False, indent=4)

    print(f"✅ Đã lưu {len(all_prepared_prompts)} prompt vào: {output_path}")

# ======================================================================
# CHẠY XỬ LÝ
# ======================================================================

if __name__ == "__main__":
    input_file = os.path.join(
        PROJECT_ROOT,
        "experiments/synthesis_data_generation/test/results/test_parse_data/test_grouped_full_context.json"
    )

    # File đầu ra chứa danh sách các prompt đã gán thông tin
    output_file = os.path.join(
        PROJECT_ROOT,
        "experiments/synthesis_data_generation/test/results/test_prompts/test_prepared_prompts.json"
    )

    generate_prompt_list(input_file, output_file)