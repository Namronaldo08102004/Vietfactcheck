import sys
import os
import json
from tqdm import tqdm

# Path handling to ensure 'src' imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from src.settings import settings
from src.components.vectorDB import VietnameseVectorDB
from src.components.reranker import VietnameseReranker
from src.modules.document_retrieval import DocumentRetrievalModule

def run_retrieval_cer_experiment():
    print("--- Running Document Retrieval Experiment: Retrieval + CER (Top 3 -> Top 1) ---")

    # 1. Cấu hình đường dẫn dữ liệu (Hợp nhất tập Train, Dev, Test)
    test_data_path = os.path.join(project_root, "data/retrieval/test_data.json")
    all_data_paths = [
        test_data_path,
        os.path.join(project_root, "data/retrieval/dev_data.json"),
        os.path.join(project_root, "data/retrieval/train_data.json")
    ]
    existing_paths = [p for p in all_data_paths if os.path.exists(p)]
    db_name = "test_vectorDB" 
    
    # 2. Khởi tạo components
    db = VietnameseVectorDB(
        db_name = db_name,
        storage_dir = settings.STORAGE_DIR,
        model_name = settings.EMBEDDING_MODEL,
        truncation_dim = settings.TRUNCATION_DIM
    )
    
    retrieval_module = DocumentRetrievalModule(db)
    reranker = VietnameseReranker(model_name = 'AITeamVN/Vietnamese_Reranker')

    # 3. Load hoặc Build Index từ danh sách đường dẫn
    if not db.load():
        print(f"🚀 Building unified index for {db_name} from: {existing_paths}")
        retrieval_module.build_system(existing_paths)
    else:
        print(f"✅ Loaded existing unified index: {db_name}")
    
    # 4. Tạo mapping URL -> Context từ toàn bộ dữ liệu để Reranker có đủ thông tin
    url_to_context = {}
    for path in existing_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for record in data:
                url_to_context[record["Url"]] = record["Context"]

    # Tải dữ liệu để đánh giá (Chỉ đánh giá trên tập test)
    with open(test_data_path, "r", encoding = "utf-8") as f:
        test_records = json.load(f)

    for top_k in range(6, 11):
        correct_count = 0
        total_count = len(test_records)
        results_log = []
        
        # Định nghĩa trọng số cho bước Retrieval ban đầu
        CURRENT_WEIGHTS = (0.3, 0.3, 0.4)

        print(f"Testing on {total_count} records using weights: {CURRENT_WEIGHTS}")

        # 5. Vòng lặp đánh giá
        for record in tqdm(test_records, desc="Evaluating CER Accuracy"):
            query = record["Statement"]
            ground_truth_url = record["Url"]

            # Bước A: Retrieval - Lấy Top 3 URLs sử dụng trọng số tùy chỉnh
            top_k_urls = retrieval_module.get_top_k_url(query, top_k = top_k, weights = CURRENT_WEIGHTS)

            # Bước B: Chuẩn bị dữ liệu cho Reranker
            class Candidate:
                def __init__(self, url, content):
                    self.page_content = content
                    self.url = url

            candidates = [Candidate(u, url_to_context[u]) for u in top_k_urls if u in url_to_context]

            # Bước C: Reranking với Cross-Encoder
            # Logic: Cross-Encoder chấm điểm cặp (Query, Full Context)
            reranked_results = reranker.rerank(query, candidates)

            # Bước D: Chọn Top 1 sau khi Rerank
            predicted_url = None
            if reranked_results:
                predicted_url = reranked_results[0]["document"].url

            # Kiểm tra độ chính xác
            is_correct = (predicted_url == ground_truth_url)
            if is_correct:
                correct_count += 1
            
            results_log.append({
                "statement": query,
                "ground_truth": ground_truth_url,
                "predicted": predicted_url,
                "is_correct": is_correct,
                "top_k_before_rerank": top_k_urls
            })

        # 6. Tính toán và lưu kết quả
        accuracy = (correct_count / total_count) * 100
        print("\n" + "="*50)
        print(f"RESULTS FOR: Retrieval + CER (Top-{top_k} Unified Index)")
        print(f"Final Top-1 Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
        print("="*50)

        output_log = os.path.join(current_dir, "results", "top_k", "retrieval_and_cer_results_top_{}.json".format(top_k))
        with open(output_log, "w", encoding="utf-8") as f:
            json.dump({
                "experiment": "retrieval_with_cer",
                "weights_used": CURRENT_WEIGHTS,
                "accuracy": accuracy, 
                "details": results_log
            }, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    run_retrieval_cer_experiment()