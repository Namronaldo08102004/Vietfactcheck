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
    Tính toán F2-score: F2 = (5 * P * R) / (4 * P + R).
    Ưu tiên Recall để đảm bảo không bỏ lỡ bằng chứng cần thiết cho việc kiểm chứng.
    """
    if precision + recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)

def run_full_rerank_pipeline():
    print("--- Running Full Pipeline: Document Retrieval (Rerank) -> Evidence Selection (Rerank) ---")

    # 1. Cấu hình đường dẫn và Tham số
    dev_data_path = os.path.join(project_root, "data/retrieval/dev_data.json")
    all_data_paths = [
        os.path.join(project_root, "data/retrieval/train_data.json"),
        dev_data_path,
        os.path.join(project_root, "data/retrieval/test_data.json")
    ]
    existing_paths = [p for p in all_data_paths if os.path.exists(p)]
    db_name = "dev_vectorDB" 
    
    # Logic thresholds cho Evidence
    THRESHOLD_ABS = 0.75
    THRESHOLD_GAP = 0.05
    DOC_TOP_K_RETR = 3
    EVI_TOP_K_RETR = 10
    
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

    # 3. Load Index và tạo URL Mapping
    if not db.load():
        print(f"🚀 Building unified index...")
        retrieval_module.build_system(existing_paths)
    else:
        print(f"✅ Loaded existing index.")

    url_to_context = {}
    for path in existing_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for record in data:
                url_to_context[record["Url"]] = record["Context"]

    # 4. Tải dữ liệu đánh giá
    with open(dev_data_path, "r", encoding="utf-8") as f:
        dev_records = json.load(f)

    total_f2, total_precision, total_recall = 0.0, 0.0, 0.0
    count = len(dev_records)
    results_log = []

    # Lớp giả lập Document cho Reranker
    class CandidateDoc:
        def __init__(self, url, content):
            self.page_content = content
            self.url = url

    # 5. Vòng lặp thực nghiệm End-to-End
    for record in tqdm(dev_records, desc="Processing Full Pipeline"):
        query = record["Statement"]
        gt_url = record["Url"]
        gt_evidences = record.get("Evidence_List", []) or [record.get("Evidence", "")]

        # --- BƯỚC 1: DOCUMENT RETRIEVAL (WITH RERANK) ---
        raw_doc_urls = retrieval_module.get_top_k_url(query, top_k=DOC_TOP_K_RETR, weights=(0.3, 0.3, 0.4))
        doc_candidates = [CandidateDoc(u, url_to_context[u]) for u in raw_doc_urls if u in url_to_context]
        reranked_docs = reranker.rerank(query, doc_candidates)
        predicted_url = reranked_docs[0]["document"].url if reranked_docs else None

        # --- BƯỚC 2: EVIDENCE SELECTION (WITH RERANK) ---
        selected_texts = []
        if predicted_url:
            # A. Retrieve ứng viên từ bài báo đã tìm thấy
            raw_evidences = evidence_module.select_top_k_evidence(
                query, predicted_url, top_k = EVI_TOP_K_RETR, weights = (0.2, 0.2, 0.6)
            )
            # B. Rerank ứng viên bằng chứng
            reranked_evidences = reranker.rerank(query, raw_evidences)
            
            # C. Logic phân cấp chọn bằng chứng
            selected_entries = []
            has_high_score = any(res['rerank_score'] > THRESHOLD_ABS for res in reranked_evidences)
            
            if has_high_score:
                selected_entries = [res for res in reranked_evidences if res['rerank_score'] > THRESHOLD_ABS]
            else:
                if reranked_evidences:
                    selected_entries.append(reranked_evidences[0])
                    for i in range(1, len(reranked_evidences)):
                        if (reranked_evidences[i-1]['rerank_score'] - reranked_evidences[i]['rerank_score']) < THRESHOLD_GAP:
                            selected_entries.append(reranked_evidences[i])
                        else: break
            
            selected_texts = [entry['document'].page_content for entry in selected_entries]

        # --- BƯỚC 3: TÍNH TOÁN METRIC ---
        relevant_found = 0
        for text in selected_texts:
            if any(text in gt or gt in text for gt in gt_evidences):
                relevant_found += 1
        
        gt_found = 0
        for gt in gt_evidences:
            if any(text in gt or gt in text for text in selected_texts):
                gt_found += 1

        precision = relevant_found / len(selected_texts) if selected_texts else 0.0
        recall = gt_found / len(gt_evidences) if gt_evidences else 0.0
        f2 = calculate_f2(precision, recall)

        total_precision += precision
        total_recall += recall
        total_f2 += f2

        results_log.append({
            "statement": query,
            "predicted_url": predicted_url,
            "is_correct_url": (predicted_url == gt_url),
            "evidence_selection": {
                "f2_score": round(f2, 4),
                "selected_count": len(selected_texts),
                "precision": round(precision, 4),
                "recall": round(recall, 4)
            }
        })

    # 6. Tổng kết
    avg_f2 = (total_f2 / count) * 100
    avg_p = (total_precision / count) * 100
    avg_r = (total_recall / count) * 100

    print("\n" + "="*60)
    print(f"FULL RERANKED PIPELINE RESULTS (ViFactCheck Dataset)")
    print(f"Metrics calculated for Evidence Selection based on Predicted Docs")
    print("-" * 60)
    print(f"Macro-Avg F2-Score:  {avg_f2:.2f}%")
    print(f"Macro-Avg Precision: {avg_p:.2f}%")
    print(f"Macro-Avg Recall:    {avg_r:.2f}%")
    print("="*60)

    # Lưu log
    output_dir = os.path.join(current_dir, "results")
    os.makedirs(output_dir, exist_ok = True)
    output_log = os.path.join(current_dir, "results", "cer+cer_results.json")
    with open(output_log, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "cer+cer",
            "dataset": "ViFactCheck",
            "metrics": {"f2": avg_f2, "precision": avg_p, "recall": avg_r},
            "details": results_log
        }, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run_full_rerank_pipeline()