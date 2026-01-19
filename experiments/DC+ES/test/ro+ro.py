import sys
import os
import json
from tqdm import tqdm

# --- XỬ LÝ ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from src.settings import settings
from src.components.vectorDB import VietnameseVectorDB
from src.modules.document_retrieval import DocumentRetrievalModule
from src.modules.evidence_selection import EvidenceSelectionModule

def calculate_f2(precision, recall):
    """
    Tính toán F2-score. Ưu tiên Recall để tránh bỏ sót bằng chứng quan trọng.
    """
    if precision + recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)

def run_pipeline_evidence_experiment():
    print("--- Running Pipeline: Document Retrieval (Retrieve Only) -> Evidence Selection (Retrieve Only) ---")

    # 1. Cấu hình đường dẫn và Database
    test_data_path = os.path.join(project_root, "data/retrieval/test_data.json")
    all_data_paths = [
        os.path.join(project_root, "data/retrieval/train_data.json"),
        os.path.join(project_root, "data/retrieval/dev_data.json"),
        test_data_path
    ]
    existing_paths = [p for p in all_data_paths if os.path.exists(p)]
    db_name = "test_vectorDB" 
    
    # 2. Khởi tạo Components
    db = VietnameseVectorDB(
        db_name = db_name,
        storage_dir = settings.STORAGE_DIR,
        model_name = settings.EMBEDDING_MODEL,
        truncation_dim = settings.TRUNCATION_DIM
    )
    
    retrieval_module = DocumentRetrievalModule(db)
    evidence_module = EvidenceSelectionModule(db)

    # 3. Load hoặc Build Index
    if not db.load():
        print(f"🚀 Building unified index for {db_name} from: {existing_paths}")
        retrieval_module.build_system(existing_paths)
    else:
        print(f"✅ Loaded existing unified index: {db_name}")

    # 4. Tải dữ liệu đánh giá (Tập Test)
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    total_f2, total_precision, total_recall = 0.0, 0.0, 0.0
    count = len(test_records)
    results_log = []

    # Cấu hình tham số cho pipeline
    DOC_WEIGHTS = (0.3, 0.3, 0.4)       # Trọng số tìm Document
    EVIDENCE_WEIGHTS = (0.2, 0.2, 0.6)  # Trọng số chọn Evidence
    EVIDENCE_TOP_K = 3

    # 5. Vòng lặp Pipeline
    for record in tqdm(test_records, desc="Running Pipeline"):
        query = record["Statement"]
        gt_url = record["Url"]
        gt_evidences = record.get("Evidence_List", [])
        if not gt_evidences:
            gt_evidences = [record.get("Evidence", "")]

        # --- BƯỚC 1: DOCUMENT RETRIEVAL ---
        # Lấy Top-1 URL dự đoán
        predicted_urls = retrieval_module.get_top_k_url(query, top_k=1, weights=DOC_WEIGHTS)
        predicted_url = predicted_urls[0] if predicted_urls else None

        # --- BƯỚC 2: EVIDENCE SELECTION ---
        # Quan trọng: Sử dụng predicted_url thay vì ground_truth_url
        retrieved_docs = []
        if predicted_url:
            retrieved_docs = evidence_module.select_top_k_evidence(
                query, 
                predicted_url, 
                top_k = EVIDENCE_TOP_K, 
                weights = EVIDENCE_WEIGHTS
            )
        
        retrieved_texts = [doc.page_content for doc in retrieved_docs]

        # --- BƯỚC 3: TÍNH TOÁN METRIC (Cho Evidence Selection) ---
        relevant_chunks_found = 0
        for text in retrieved_texts:
            for gt in gt_evidences:
                if text in gt or gt in text:
                    relevant_chunks_found += 1
                    break
        
        gt_items_found = 0
        for gt in gt_evidences:
            for text in retrieved_texts:
                if text in gt or gt in text:
                    gt_items_found += 1
                    break

        precision = relevant_chunks_found / EVIDENCE_TOP_K if retrieved_texts else 0.0
        recall = gt_items_found / len(gt_evidences) if gt_evidences else 0.0
        f2 = calculate_f2(precision, recall)

        # Tích lũy Macro-Average
        total_precision += precision
        total_recall += recall
        total_f2 += f2

        # Lưu log chi tiết: Lưu cả URL dự đoán để Nam dễ debug
        results_log.append({
            "statement": query,
            "retrieved_url": predicted_url,
            "is_correct_url": (predicted_url == gt_url),
            "evidence_metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f2_score": round(f2, 4)
            }
        })

    # 6. Tổng hợp kết quả
    avg_precision = (total_precision / count) * 100
    avg_recall = (total_recall / count) * 100
    avg_f2 = (total_f2 / count) * 100

    print("\n" + "="*60)
    print(f"PIPELINE RESULTS: EVIDENCE SELECTION PERFORMANCE")
    print(f"(Based on predicted Top-1 Documents)")
    print("-" * 60)
    print(f"Average Precision: {avg_precision:.2f}%")
    print(f"Average Recall:    {avg_recall:.2f}%")
    print(f"Average F2-Score:  {avg_f2:.2f}%")
    print("="*60)

    # Lưu kết quả
    output_dir = os.path.join(current_dir, "results")
    os.makedirs(output_dir, exist_ok = True)
    output_log = os.path.join(current_dir, "results", "ro+ro_results.json")
    with open(output_log, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "ro+ro",
            "macro_avg_metrics": {
                "f2": avg_f2,
                "precision": avg_precision,
                "recall": avg_recall
            },
            "details": results_log
        }, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    run_pipeline_evidence_experiment()