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
from src.modules.evidence_selection import EvidenceSelectionModule

def calculate_f2(precision, recall):
    """
    Calculates the F2-score using the formula: F2 = (5 * P * R) / (4 * P + R).
    """
    if precision + recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)

def run_evidence_f2_experiment():
    print("--- Running Evidence Selection Experiment: Retrieval Only (Unified Index) ---")

    # 1. Cấu hình đường dẫn dữ liệu (Hợp nhất tập Train, Dev, Test)
    test_data_path = os.path.join(project_root, "data/retrieval/test_data.json")
    all_data_paths = [
        test_data_path,
        os.path.join(project_root, "data/retrieval/dev_data.json"),
        os.path.join(project_root, "data/retrieval/train_data.json")
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

    # 3. Load hoặc Build Index từ danh sách đường dẫn
    if not db.load():
        print(f"🚀 Building unified index for {db_name} from: {existing_paths}")
        retrieval_module.build_system(existing_paths)
    else:
        print(f"✅ Loaded existing unified index: {db_name}")

    # 4. Tải dữ liệu để đánh giá (Tập Test)
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)
    
    for top_k in range(1, 11):  # Thử từ Top-1 đến Top-10
        total_f2, total_precision, total_recall = 0.0, 0.0, 0.0
        count = len(test_records)
        results_log = []
        
        # Trọng số ưu tiên Embedding cho bằng chứng (theo module evidence_selection)
        EVIDENCE_WEIGHTS = (0.2, 0.2, 0.6)
        print(f"🧪 Evaluating on {count} records using Top-{top_k} and weights {EVIDENCE_WEIGHTS}...")

        # 5. Vòng lặp đánh giá
        for record in tqdm(test_records, desc="Calculating F2-Scores"):
            query = record["Statement"]
            gt_url = record["Url"]
            
            # Lấy nhãn bằng chứng chuẩn (Ground Truth)
            gt_evidences = record.get("Evidence_List", [])
            if not gt_evidences:
                gt_evidences = [record.get("Evidence", "")]

            # Phase 1: Truy xuất Top-K bằng chứng với trọng số động
            retrieved_docs = evidence_module.select_top_k_evidence(
                query, 
                gt_url, 
                top_k = top_k, 
                weights = EVIDENCE_WEIGHTS
            )
            retrieved_texts = [doc.page_content for doc in retrieved_docs]

            # Phase 2: So khớp chuỗi (X in Y or Y in X)
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

            # Phase 3: Tính toán chỉ số Precision, Recall và F2
            precision = relevant_chunks_found / top_k if top_k > 0 else 0.0
            recall = gt_items_found / len(gt_evidences) if gt_evidences else 0.0
            f2 = calculate_f2(precision, recall)

            # Tích lũy để tính Macro-Average
            total_precision += precision
            total_recall += recall
            total_f2 += f2

            results_log.append({
                "statement": query,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f2_score": round(f2, 4)
            })

        # 6. Thống kê cuối cùng
        avg_precision = (total_precision / count) * 100
        avg_recall = (total_recall / count) * 100
        avg_f2 = (total_f2 / count) * 100

        print("\n" + "="*50)
        print(f"RESULTS FOR: Evidence Selection (Top-{top_k} Only)")
        print(f"Average Precision: {avg_precision:.2f}%")
        print(f"Average Recall:    {avg_recall:.2f}%")
        print(f"Average F2-Score:  {avg_f2:.2f}%")
        print("="*50)

        # Lưu log chi tiết
        output_log = os.path.join(current_dir, "results", "top_k", "evidence_f2_results_top_{}.json".format(top_k))
        with open(output_log, "w", encoding = "utf-8") as f:
            json.dump({
                "macro_avg_f2": avg_f2,
                "macro_avg_precision": avg_precision,
                "macro_avg_recall": avg_recall,
                "details": results_log
            }, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    run_evidence_f2_experiment()