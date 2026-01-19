import sys
import os
import json
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

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
from src.modules.claim_verification import ClaimVerificationModule

def run_advanced_full_pipeline():
    print("\n--- Pipeline: Retrieval (Only) -> Selection (Rerank) -> Verification (Gold PLMs) ---")
    
    # 1. Cấu hình thực nghiệm
    data_path = os.path.join(project_root, "data/retrieval/test_data.json")
    all_data_paths = [data_path, 
                      os.path.join(project_root, "data/retrieval/dev_data.json"),
                      os.path.join(project_root, "data/retrieval/train_data.json")]
    existing_paths = [p for p in all_data_paths if os.path.exists(p)]
    db_name = "test_vectorDB"

    # Ngưỡng logic Selection (theo thực nghiệm trước đó của bạn)
    THRESHOLD_1 = 0.75  
    THRESHOLD_2 = 0.05  
    TOP_K_SELECTION_CANDIDATES = 10 

    # 2. Khởi tạo Components
    db = VietnameseVectorDB(
        db_name = db_name,
        storage_dir = settings.STORAGE_DIR,
        model_name = settings.EMBEDDING_MODEL,
        truncation_dim = settings.TRUNCATION_DIM
    )
    retrieval_module = DocumentRetrievalModule(db)
    reranker = VietnameseReranker(model_name = 'AITeamVN/Vietnamese_Reranker')
    evidence_module = EvidenceSelectionModule(db)

    if not db.load():
        retrieval_module.build_system(existing_paths)
    
    # 3. Danh sách 6 PLMs chuyên biệt cho Gold Evidence
    plm_models = [
        "Namronaldo2004/Vifactcheck-xlm-roberta-base-gold-evidence",
        "Namronaldo2004/Vifactcheck-xlm-roberta-large-gold-evidence",
        "Namronaldo2004/Vifactcheck-ViBERT-gold-evidence",
        "Namronaldo2004/Vifactcheck-mBERT-gold-evidence",
        "Namronaldo2004/Vifactcheck-phoBERT-base-gold-evidence",
        "Namronaldo2004/Vifactcheck-phoBERT-large-gold-evidence"
    ]

    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 4. Chạy thực nghiệm Full Pipeline
    for model_path in plm_models:
        model_name_safe = model_path.replace("/", "_")
        print(f"\n🚀 Evaluating Integrated Pipeline with: {model_path}")
        
        try:
            verifier = ClaimVerificationModule(model_path = model_path)
            y_true, y_pred = [], []
            results_log = []

            for record in tqdm(records, desc=f"Processing {model_name_safe}"):
                claim = str(record["Statement"])
                label_true = int(record["labels"])
                gt_url = record["Url"]

                # --- STEP 1: DOCUMENT RETRIEVAL (TOP-1 ONLY) ---
                # Trọng số mặc định: (0.3, 0.3, 0.4)
                predicted_urls = retrieval_module.get_top_k_url(claim, top_k=1, weights=(0.3, 0.3, 0.4))
                predicted_url = predicted_urls[0] if predicted_urls else None

                # --- STEP 2: EVIDENCE SELECTION (RETRIEVE + RERANK) ---
                selected_evidence_text = ""
                if predicted_url:
                    # Lấy ứng viên bằng chứng (Weights: 0.2, 0.2, 0.6)
                    candidates = evidence_module.select_top_k_evidence(
                        claim, predicted_url, top_k=TOP_K_SELECTION_CANDIDATES, weights=(0.2, 0.2, 0.6)
                    )
                    
                    # Rerank ứng viên bằng Cross-Encoder
                    reranked_results = reranker.rerank(claim, candidates)
                    
                    # Logic chọn bằng chứng phân cấp (Hierarchical Selection)
                    selected_entries = []
                    has_high_score = any(res['rerank_score'] > THRESHOLD_1 for res in reranked_results)
                    
                    if has_high_score:
                        selected_entries = [res for res in reranked_results if res['rerank_score'] > THRESHOLD_1]
                    else:
                        if reranked_results:
                            selected_entries.append(reranked_results[0])
                            for i in range(1, len(reranked_results)):
                                if (reranked_results[i-1]['rerank_score'] - reranked_results[i]['rerank_score']) < THRESHOLD_2:
                                    selected_entries.append(reranked_results[i])
                                else:
                                    break
                    
                    selected_evidence_text = " ".join([entry['document'].page_content for entry in selected_entries])

                # --- STEP 3: CLAIM VERIFICATION ---
                result = verifier.verify_claim(claim, full_context = selected_evidence_text)
                label_pred = result["label_code"]

                y_true.append(label_true)
                y_pred.append(label_pred)
                
                results_log.append({
                    "statement": claim,
                    "is_retrieval_correct": (predicted_url == gt_url),
                    "selected_evidence": selected_evidence_text,
                    "true_label": label_true,
                    "pred_label": label_pred
                })

            # 5. Thống kê & Lưu log
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            
            print(f"Accuracy: {acc:.4f} | F1-Macro: {f1_macro:.4f}")
            print(classification_report(y_true, y_pred, digits=4, target_names=["Supported", "Refuted", "NEI"]))
            
            output_dir = os.path.join(current_dir, "results")
            os.makedirs(output_dir, exist_ok = True)
            output_file = os.path.join(current_dir, "results", f"ro+cer+ge_{model_name_safe}.json")
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump({"model_info": model_path, "metrics": {"accuracy": acc, "f1_macro": f1_macro}, "details": results_log}, f_out, ensure_ascii=False, indent=4)

            del verifier
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error at {model_path}: {e}")

if __name__ == "__main__":
    run_advanced_full_pipeline()