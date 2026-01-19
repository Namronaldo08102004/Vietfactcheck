import sys
import os
import json
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

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

# Lớp giả lập Document để tương thích với cấu trúc của Reranker và Selection
class CandidateDoc:
    def __init__(self, url, content):
        self.page_content = content
        self.url = url

def run_complex_pipeline():
    print("\n--- Pipeline: Retrieval+Rerank (Top-1) -> Selection (Top-3) -> Verification ---")
    
    # 1. Cấu hình thực nghiệm
    data_path = os.path.join(project_root, "data/retrieval/train_data.json")
    all_data_paths = [data_path, 
                      os.path.join(project_root, "data/retrieval/dev_data.json"),
                      os.path.join(project_root, "data/retrieval/test_data.json")]
    existing_paths = [p for p in all_data_paths if os.path.exists(p)]
    db_name = "train_vectorDB"

    # 2. Khởi tạo Components Truy xuất
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
    
    # Tạo mapping URL -> Context để Reranker sử dụng
    url_to_context = {}
    for path in existing_paths:
        with open(path, "r", encoding="utf-8") as f:
            for record in json.load(f):
                url_to_context[record["Url"]] = record["Context"]

    # 3. Danh sách 6 PLMs Gold Evidence
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

    # 4. Chạy thực nghiệm
    for model_path in plm_models:
        model_name_safe = model_path.replace("/", "_")
        print(f"\n🚀 Pipeline Evaluation: {model_path}")
        
        try:
            verifier = ClaimVerificationModule(model_path = model_path)
            y_true, y_pred = [], []
            pipeline_details = []

            for record in tqdm(records, desc=f"Processing {model_name_safe}"):
                claim = str(record["Statement"])
                label_true = int(record["labels"])
                gt_url = record["Url"]

                # --- BƯỚC 1: RETRIEVAL TOP-3 ---
                top_3_urls = retrieval_module.get_top_k_url(claim, top_k=3, weights=(0.3, 0.3, 0.4))

                # --- BƯỚC 2: RERANKING LẤY TOP-1 ---
                candidates = [CandidateDoc(u, url_to_context[u]) for u in top_3_urls if u in url_to_context]
                reranked_results = reranker.rerank(claim, candidates)
                
                best_url = None
                if reranked_results:
                    best_url = reranked_results[0]["document"].url

                # --- BƯỚC 3: EVIDENCE SELECTION (TOP-3) ---
                selected_docs = []
                if best_url:
                    selected_docs = evidence_module.select_top_k_evidence(
                        claim, best_url, top_k=3, weights=(0.2, 0.2, 0.6)
                    )
                evidence_text = " ".join([d.page_content for d in selected_docs])

                # --- BƯỚC 4: VERIFICATION ---
                result = verifier.verify_claim(claim, full_context = evidence_text)
                label_pred = result["label_code"]

                y_true.append(label_true)
                y_pred.append(label_pred)
                
                pipeline_details.append({
                    "statement": claim,
                    "reranked_url": best_url,
                    "retrieval_correct": (best_url == gt_url),
                    "true_label": label_true,
                    "pred_label": label_pred
                })

            # 5. Lưu kết quả
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            print(f"Accuracy: {acc:.4f} | F1-Macro: {f1_macro:.4f}")
            
            output_dir = os.path.join(current_dir, "results")
            os.makedirs(output_dir, exist_ok = True)
            output_file = os.path.join(current_dir, "results", f"cer+ro+ge_{model_name_safe}.json")
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump({"model_info": model_path, "metrics": {"acc": acc, "f1": f1_macro}, "details": pipeline_details}, f_out, ensure_ascii=False, indent=4)

            del verifier
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error at {model_path}: {e}")

if __name__ == "__main__":
    run_complex_pipeline()