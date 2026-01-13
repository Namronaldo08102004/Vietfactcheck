import sys
import os
import json
from tqdm import tqdm

# Đảm bảo đường dẫn để import từ 'src' chính xác
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from src.settings import settings
from src.components.vectorDB import VietnameseVectorDB
from src.modules.document_retrieval import DocumentRetrievalModule

def run_retrieval_experiment():
    print("--- Running Document Retrieval Experiment: Retrieval Only (Unified Index) ---")

    # 1. Cấu hình đường dẫn
    # Mặc dù đánh giá trên train, nhưng ta build system từ toàn bộ tập dữ liệu để mô phỏng thực tế
    test_data_path = os.path.join(project_root, "data/retrieval/test_data.json")
    all_data_paths = [
        test_data_path,
        os.path.join(project_root, "data/retrieval/dev_data.json"),
        os.path.join(project_root, "data/retrieval/train_data.json")
    ]
    
    # Lọc các đường dẫn tồn tại thực tế
    existing_paths = [p for p in all_data_paths if os.path.exists(p)]
    db_name = "test_vectorDB" 
    
    # 2. Khởi tạo Vector Database
    db = VietnameseVectorDB(
        db_name = db_name,
        storage_dir = settings.STORAGE_DIR,
        model_name = settings.EMBEDDING_MODEL,
        truncation_dim = settings.TRUNCATION_DIM
    )

    retrieval_module = DocumentRetrievalModule(db)

    # 3. Load hoặc Build Index (Sử dụng List các đường dẫn theo logic mới)
    if not db.load():
        print(f"🚀 Building unified index for {db_name} from: {existing_paths}")
        retrieval_module.build_system(existing_paths)
    else:
        print(f"✅ Loaded existing index: {db_name}")

    # 4. Tải dữ liệu để đánh giá (Tập test)
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    correct_count = 0
    total_count = len(test_records)
    results_log = []

    # Trọng số mặc định như trong module (0.3, 0.3, 0.4)
    CURRENT_WEIGHTS = (0.3, 0.3, 0.4)

    print(f"Testing accuracy on {total_count} records using weights: {CURRENT_WEIGHTS}")

    # 5. Vòng lặp đánh giá
    for record in tqdm(test_records, desc="Evaluating Top-1 Accuracy"):
        query = record["Statement"]
        ground_truth_url = record["Url"]

        # Truy xuất Top 1 URL với tham số trọng số mới
        predicted_urls = retrieval_module.get_top_k_url(
            query, 
            top_k = 1, 
            weights = CURRENT_WEIGHTS
        )
        predicted_url = predicted_urls[0] if predicted_urls else None

        # Kiểm tra tính đúng đắn
        is_correct = (predicted_url == ground_truth_url)
        if is_correct:
            correct_count += 1
        
        results_log.append({
            "statement": query,
            "ground_truth": ground_truth_url,
            "predicted": predicted_url,
            "is_correct": is_correct
        })

    # 6. Tính toán và lưu kết quả
    accuracy = (correct_count / total_count) * 100
    print("\n" + "="*50)
    print(f"FINAL RETRIEVAL ACCURACY: {accuracy:.2f}% ({correct_count}/{total_count})")
    print("="*50)

    output_log = os.path.join(current_dir, "results", "retrieval_only_results.json")
    with open(output_log, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "retrieval_only",
            "weights_used": CURRENT_WEIGHTS,
            "accuracy": accuracy, 
            "details": results_log
        }, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    run_retrieval_experiment()