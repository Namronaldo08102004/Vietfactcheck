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
from src.components.reranker import VietnameseReranker
from src.modules.document_retrieval import DocumentRetrievalModule
from src.modules.evidence_selection import EvidenceSelectionModule

def calculate_f2(precision, recall):
    """
    Calculates the F2-score. Formula: F2 = (5 * P * R) / (4 * P + R)
    """
    if precision + recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)

def run_evidence_cer_experiment():
    print("--- Running Evidence Selection: Retrieval + CER (Unified Index & Dynamic Selection) ---")

    # 1. Cấu hình thực nghiệm
    test_data_path = os.path.join(project_root, "data/retrieval/test_data.json")
    all_data_paths = [
        test_data_path,
        os.path.join(project_root, "data/retrieval/dev_data.json"),
        os.path.join(project_root, "data/retrieval/train_data.json")
    ]
    existing_paths = [p for p in all_data_paths if os.path.exists(p)]
    db_name = "test_vectorDB" 
    
    # Ngưỡng logic theo yêu cầu của người dùng
    THRESHOLD_1 = 0.75    # Ngưỡng tuyệt đối
    THRESHOLD_2 = 0.05    # Ngưỡng khoảng cách (gap)
    TOP_K_RETRIEVAL = 10  # Số lượng ứng viên lấy ra trước khi Rerank
    
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

    # 3. Load hoặc Build Unified Index
    if not db.load():
        print(f"🚀 Building unified index for {db_name} from: {existing_paths}")
        retrieval_module.build_system(existing_paths)
    else:
        print(f"✅ Loaded existing unified index: {db_name}")

    # 4. Tải dữ liệu đánh giá
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    total_f2, total_precision, total_recall = 0.0, 0.0, 0.0
    count = len(test_records)
    results_log = []

    # 5. Vòng lặp đánh giá
    for record in tqdm(test_records, desc = "Evaluating CER Evidence"):
        query = record["Statement"]
        gt_url = record["Url"]
        gt_evidences = record.get("Evidence_List", [])
        if not gt_evidences:
            gt_evidences = [record.get("Evidence", "")]

        # Bước A: Truy xuất Top-10 ứng viên (Candidate Retrieval)
        # Sử dụng trọng số mặc định cho bằng chứng (0.2, 0.2, 0.6)
        retrieved_docs = evidence_module.select_top_k_evidence(
            query, gt_url, top_k = TOP_K_RETRIEVAL, weights = (0.2, 0.2, 0.6)
        )

        # Bước B: Reranking với Cross-Encoder
        # Trả về danh sách dicts {"document": doc_obj, "rerank_score": float}
        reranked_results = reranker.rerank(query, retrieved_docs)

        # Bước C: Áp dụng logic chọn bằng chứng phân cấp (Hierarchical Selection)
        selected_entries = []
        
        # Quy tắc 1: Nếu có ít nhất 1 bằng chứng > THRESHOLD_1, lấy tất cả các bằng chứng đó
        has_high_score = any(res['rerank_score'] > THRESHOLD_1 for res in reranked_results)
        
        if has_high_score:
            selected_entries = [res for res in reranked_results if res['rerank_score'] > THRESHOLD_1]
        else:
            # Quy tắc 2: Nếu không, chọn theo khoảng cách điểm (Gap) THRESHOLD_2
            if reranked_results:
                selected_entries.append(reranked_results[0])
                for i in range(1, len(reranked_results)):
                    prev_score = reranked_results[i-1]['rerank_score']
                    curr_score = reranked_results[i]['rerank_score']
                    
                    if (prev_score - curr_score) < THRESHOLD_2:
                        selected_entries.append(reranked_results[i])
                    else:
                        break # Dừng lại khi khoảng cách vượt quá ngưỡng

        # Bước D: Tính toán các chỉ số (F2-Score)
        selected_texts = [entry['document'].page_content for entry in selected_entries]
        
        # Tính Precision
        relevant_chunks_found = 0
        for text in selected_texts:
            for gt in gt_evidences:
                if text in gt or gt in text:
                    relevant_chunks_found += 1
                    break
        precision = relevant_chunks_found / len(selected_texts) if selected_texts else 0.0
        
        # Tính Recall
        gt_items_found = 0
        for gt in gt_evidences:
            for text in selected_texts:
                if text in gt or gt in text:
                    gt_items_found += 1
                    break
        recall = gt_items_found / len(gt_evidences) if gt_evidences else 0.0
        
        f2 = calculate_f2(precision, recall)

        # Tích lũy Macro-Average
        total_precision += precision
        total_recall += recall
        total_f2 += f2

        # Lưu log chi tiết kèm điểm số Rerank
        results_log.append({
            "statement": query,
            "f2_score": round(f2, 4),
            "selected_count": len(selected_entries),
            "selected_details": [
                {
                    "content": entry['document'].page_content,
                    "rerank_score": round(entry['rerank_score'], 4)
                } 
                for entry in selected_entries
            ]
        })

    # 6. Thống kê kết quả
    avg_f2 = (total_f2 / count) * 100
    print("\n" + "="*50)
    print(f"CER EVIDENCE SELECTION RESULTS (Unified Index)")
    print(f"Macro-Avg F2-Score: {avg_f2:.2f}%")
    print("="*50)

    # Lưu file kết quả JSON
    output_log = os.path.join(current_dir, "results", "evidence_cer_results.json")
    with open(output_log, "w", encoding = "utf-8") as f:
        json.dump({
            "experiment": "evidence_retrieval_and_cer",
            "threshold_1": THRESHOLD_1,
            "threshold_2": THRESHOLD_2,
            "macro_avg_f2": avg_f2, 
            "details": results_log
        }, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    run_evidence_cer_experiment()