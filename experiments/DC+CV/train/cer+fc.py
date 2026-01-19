import sys
import os
import json
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- CẤU HÌNH PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from src.settings import settings
from src.components.vectorDB import VietnameseVectorDB
from src.components.reranker import VietnameseReranker
from src.modules.document_retrieval import DocumentRetrievalModule
from src.modules.claim_verification import ClaimVerificationModule

def run_integrated_rerank_verification():
    print("\n--- Pipeline: Retrieval + Rerank (Top-1) + Claim Verification ---")
    
    # 1. Khởi tạo dữ liệu và DB
    data_path = os.path.join(project_root, "data/retrieval/train_data.json")
    all_data_paths = [
        data_path,
        os.path.join(project_root, "data/retrieval/dev_data.json"),
        os.path.join(project_root, "data/retrieval/test_data.json")
    ]
    existing_paths = [p for p in all_data_paths if os.path.exists(p)]
    db_name = "train_vectorDB"

    # 2. Khởi tạo các Components Retrieval
    db = VietnameseVectorDB(
        db_name=db_name,
        storage_dir=settings.STORAGE_DIR,
        model_name=settings.EMBEDDING_MODEL,
        truncation_dim=settings.TRUNCATION_DIM
    )
    retrieval_module = DocumentRetrievalModule(db)
    reranker = VietnameseReranker(model_name='AITeamVN/Vietnamese_Reranker')

    if not db.load():
        retrieval_module.build_system(existing_paths)
    
    # Mapping URL -> Context để Reranker và Verifier sử dụng
    url_to_context = {}
    for path in existing_paths:
        with open(path, "r", encoding="utf-8") as f:
            for record in json.load(f):
                url_to_context[record["Url"]] = record["Context"]

    # 3. Danh sách 6 PLMs dùng cho Verification
    plm_models = [
        "Namronaldo2004/Vifactcheck-xlm-roberta-base",
        "Namronaldo2004/Vifactcheck-xlm-roberta-large",
        "Namronaldo2004/Vifactcheck-ViBERT",
        "Namronaldo2004/Vifactcheck-mBERT",
        "Namronaldo2004/Vifactcheck-phoBERT-base",
        "Namronaldo2004/Vifactcheck-phoBERT-large"
    ]

    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 4. Thực hiện Pipeline
    # Lớp giả lập Document để tương thích với hàm rerank của bạn
    class Candidate:
        def __init__(self, url, content):
            self.page_content = content
            self.url = url

    for model_path in plm_models:
        model_name_safe = model_path.replace("/", "_")
        print(f"\n🚀 Pipeline với Verifier: {model_path}")
        
        try:
            verifier = ClaimVerificationModule(model_path=model_path)
            y_true, y_pred = [], []
            details = []

            for record in tqdm(records, desc=f"Processing {model_name_safe}"):
                query = str(record["Statement"])
                gt_label = int(record["labels"])

                # BƯỚC A: Retrieval Top-3
                top_3_urls = retrieval_module.get_top_k_url(query, top_k=3, weights=(0.3, 0.3, 0.4))

                # BƯỚC B: Reranking lấy Top-1
                candidates = [Candidate(u, url_to_context[u]) for u in top_3_urls if u in url_to_context]
                reranked = reranker.rerank(query, candidates)
                
                best_context = ""
                predicted_url = None
                if reranked:
                    predicted_url = reranked[0]["document"].url
                    best_context = reranked[0]["document"].page_content

                # BƯỚC C: Verification
                result = verifier.verify_claim(query, full_context=best_context)
                pred_label = result["label_code"]

                y_true.append(gt_label)
                y_pred.append(pred_label)

                details.append({
                    "statement": query,
                    "retrieved_url": predicted_url,
                    "true_label": gt_label,
                    "pred_label": pred_label,
                    "is_correct": gt_label == pred_label
                })

            # 5. Thống kê & Lưu trữ
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            
            print(f"Accuracy: {acc:.4f} | F1-Macro: {f1:.4f}")
            
            output_dir = os.path.join(current_dir, "results")
            os.makedirs(output_dir, exist_ok = True)
            output_file = os.path.join(current_dir, "results", f"cer+fc_{model_name_safe}.json")
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump({"verification_model": model_path, 
                           "metrics": {"acc": acc, "f1": f1}, 
                           "details": details}, f_out, ensure_ascii=False, indent=4)

            del verifier
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Lỗi tại mô hình {model_path}: {e}")

if __name__ == "__main__":
    run_integrated_rerank_verification()