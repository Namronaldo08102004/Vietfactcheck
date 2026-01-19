import sys
import os
import json
from tqdm import tqdm

# --- XỬ LÝ PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from src.settings import settings
from src.components.vectorDB import VietnameseVectorDB
from src.components.reranker import VietnameseReranker
from src.modules.document_retrieval import DocumentRetrievalModule
from src.modules.evidence_selection import EvidenceSelectionModule

def calculate_f2(precision, recall):
    """
    Tính toán F2-score: Ưu tiên Recall hơn Precision.
    """
    if precision + recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)

def run_pipeline_evidence_rerank_experiment():
    print("--- Running Pipeline: Document Retrieval (Retrieve Only) -> Evidence Selection (Retrieve + Rerank) ---")
    
    # 1. Cấu hình thực nghiệm
    dev_data_path = os.path.join(project_root, "data/retrieval/dev_data.json")
    all_data_paths = [
        os.path.join(project_root, "data/retrieval/train_data.json"),
        dev_data_path,
        os.path.join(project_root, "data/retrieval/test_data.json")
    ]
    existing_paths = [p for p in all_data_paths if os.path.exists(p)]
    db_name = "dev_vectorDB" 
    
    # Tham số logic từ feedback của Nam
    THRESHOLD_1 = 0.75      # Ngưỡng tuyệt đối để chọn ngay bằng chứng
    THRESHOLD_2 = 0.05      # Ngưỡng khoảng cách (gap) giữa các bậc điểm
    TOP_K_CANDIDATES = 10   # Số lượng ứng viên trích xuất trước khi Rerank
    
    # 2. Khởi tạo Components
    db = VietnameseVectorDB(
        db_name = db_name,
        storage_dir = settings.STORAGE_DIR,
        model_name = settings.EMBEDDING_MODEL,
        truncation_dim = settings.TRUNCATION_DIM
    )
    
    retrieval_module = DocumentRetrievalModule(db)
    evidence_module = EvidenceSelectionModule(db)
    reranker = VietnameseReranker(model_name = 'AITeamVN/Vietnamese_Reranker')

    # 3. Load Index
    if not db.load():
        print(f"🚀 Building unified index...")
        retrieval_module.build_system(existing_paths)
    else:
        print(f"✅ Loaded existing unified index.")

    # 4. Tải dữ liệu đánh giá
    with open(dev_data_path, "r", encoding="utf-8") as f:
        dev_records = json.load(f)

    total_f2, total_precision, total_recall = 0.0, 0.0, 0.0
    count = len(dev_records)
    results_log = []

    # 5. Vòng lặp thực nghiệm Pipeline
    for record in tqdm(dev_records, desc="Processing Pipeline"):
        query = record["Statement"]
        gt_url = record["Url"]
        gt_evidences = record.get("Evidence_List", [])
        if not gt_evidences:
            gt_evidences = [record.get("Evidence", "")]

        # --- BƯỚC 1: DOCUMENT RETRIEVAL (TOP-1 ONLY) ---
        predicted_urls = retrieval_module.get_top_k_url(query, top_k=1, weights=(0.3, 0.3, 0.4))
        predicted_url = predicted_urls[0] if predicted_urls else None

        # --- BƯỚC 2: EVIDENCE SELECTION (RETRIEVE + RERANK) ---
        selected_texts = []
        if predicted_url:
            # A. Retrieve 10 ứng viên từ URL vừa tìm được
            retrieved_docs = evidence_module.select_top_k_evidence(
                query, predicted_url, top_k = TOP_K_CANDIDATES, weights = (0.2, 0.2, 0.6)
            )
            
            # B. Reranking với Cross-Encoder
            reranked_results = reranker.rerank(query, retrieved_docs)
            
            # C. Áp dụng Hierarchical Selection (Logic phân cấp)
            selected_entries = []
            has_high_score = any(res['rerank_score'] > THRESHOLD_1 for res in reranked_results)
            
            if has_high_score:
                selected_entries = [res for res in reranked_results if res['rerank_score'] > THRESHOLD_1]
            else:
                if reranked_results:
                    selected_entries.append(reranked_results[0])
                    for i in range(1, len(reranked_results)):
                        prev_score = reranked_results[i-1]['rerank_score']
                        curr_score = reranked_results[i]['rerank_score']
                        if (prev_score - curr_score) < THRESHOLD_2:
                            selected_entries.append(reranked_results[i])
                        else: break
            
            selected_texts = [entry['document'].page_content for entry in selected_entries]

        # --- BƯỚC 3: TÍNH TOÁN METRIC ---
        relevant_chunks_found = 0
        for text in selected_texts:
            for gt in gt_evidences:
                if text in gt or gt in text:
                    relevant_chunks_found += 1
                    break
        
        gt_items_found = 0
        for gt in gt_evidences:
            for text in selected_texts:
                if text in gt or gt in text:
                    gt_items_found += 1
                    break

        precision = relevant_chunks_found / len(selected_texts) if selected_texts else 0.0
        recall = gt_items_found / len(gt_evidences) if gt_evidences else 0.0
        f2 = calculate_f2(precision, recall)

        total_precision += precision
        total_recall += recall
        total_f2 += f2

        results_log.append({
            "statement": query,
            "retrieved_url": predicted_url,
            "is_correct_url": (predicted_url == gt_url),
            "evidence_selection": {
                "f2_score": round(f2, 4),
                "selected_count": len(selected_texts),
                "precision": round(precision, 4),
                "recall": round(recall, 4)
            }
        })

    # 6. Thống kê kết quả cuối cùng (Chỉ báo cáo Metric cho Evidence)
    avg_f2 = (total_f2 / count) * 100
    avg_precision = (total_precision / count) * 100
    avg_recall = (total_recall / count) * 100

    print("\n" + "="*60)
    print(f"PIPELINE RESULTS: EVIDENCE SELECTION (RETRIEVE + RERANK)")
    print(f"(Based on predicted Top-1 Documents from Retrieval-Only)")
    print("-" * 60)
    print(f"Macro-Avg F2-Score:  {avg_f2:.2f}%")
    print(f"Macro-Avg Precision: {avg_precision:.2f}%")
    print(f"Macro-Avg Recall:    {avg_recall:.2f}%")
    print("="*60)

    # Lưu log
    output_dir = os.path.join(current_dir, "results")
    os.makedirs(output_dir, exist_ok = True)
    output_log = os.path.join(current_dir, "results", "ro+cer_results.json")
    with open(output_log, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "ro+cer",
            "configs": {"threshold_abs": THRESHOLD_1, "threshold_gap": THRESHOLD_2},
            "metrics": {"f2": avg_f2, "precision": avg_precision, "recall": avg_recall},
            "details": results_log
        }, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run_pipeline_evidence_rerank_experiment()