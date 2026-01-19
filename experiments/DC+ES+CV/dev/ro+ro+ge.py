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
from src.modules.document_retrieval import DocumentRetrievalModule
from src.modules.evidence_selection import EvidenceSelectionModule
from src.modules.claim_verification import ClaimVerificationModule

def run_full_pipeline_experiment():
    print("\n--- Running Full Pipeline: Retrieval (Top-1) -> Selection (Top-3) -> Verification ---")
    
    # 1. Cấu hình dữ liệu
    data_path = os.path.join(project_root, "data/retrieval/dev_data.json")
    all_data_paths = [
        data_path,
        os.path.join(project_root, "data/retrieval/train_data.json"),
        os.path.join(project_root, "data/retrieval/test_data.json")
    ]
    existing_paths = [p for p in all_data_paths if os.path.exists(p)]
    db_name = "dev_vectorDB"
    if not os.path.exists(data_path):
        print(f"❌ Không tìm thấy file dữ liệu tại: {data_path}")
        return

    # 2. Khởi tạo Components
    db = VietnameseVectorDB(
        db_name = db_name,
        storage_dir = settings.STORAGE_DIR,
        model_name = settings.EMBEDDING_MODEL,
        truncation_dim = settings.TRUNCATION_DIM
    )
    retrieval_module = DocumentRetrievalModule(db)
    evidence_module = EvidenceSelectionModule(db)

    if not db.load():
        print(f"🚀 Building unified index for {db_name}...")
        retrieval_module.build_system(existing_paths)
    else:
        print(f"✅ Loaded existing unified index: {db_name}")

    # 3. Danh sách 6 PLMs được fine-tune trên Gold Evidence
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

    # Cấu hình trọng số thực nghiệm từ các bước trước
    RETRIEVAL_WEIGHTS = (0.3, 0.3, 0.4) #
    EVIDENCE_WEIGHTS = (0.2, 0.2, 0.6)  #
    TOP_K_EVIDENCE = 3

    # 4. Chạy thực nghiệm Full Pipeline
    for model_path in plm_models:
        model_name_safe = model_path.replace("/", "_")
        print(f"\n🚀 Full Pipeline Evaluation with Verifier: {model_path}")
        
        try:
            verifier = ClaimVerificationModule(model_path = model_path)
            y_true, y_pred = [], []
            pipeline_details = []

            for record in tqdm(records, desc=f"Pipeline {model_name_safe}"):
                claim = str(record["Statement"])
                label_true = int(record["labels"])
                gt_url = record["Url"]

                # --- STEP 1: DOCUMENT RETRIEVAL (TOP-1) ---
                predicted_urls = retrieval_module.get_top_k_url(claim, top_k=1, weights=RETRIEVAL_WEIGHTS)
                predicted_url = predicted_urls[0] if predicted_urls else None

                # --- STEP 2: EVIDENCE SELECTION (TOP-3) ---
                # Lọc bằng chứng từ bài báo vừa tìm được
                selected_docs = []
                if predicted_url:
                    selected_docs = evidence_module.select_top_k_evidence(
                        claim, 
                        predicted_url, 
                        top_k=TOP_K_EVIDENCE, 
                        weights=EVIDENCE_WEIGHTS
                    )
                
                # Kết hợp các câu bằng chứng thành một đoạn context duy nhất
                selected_evidence_text = " ".join([doc.page_content for doc in selected_docs])

                # --- STEP 3: CLAIM VERIFICATION ---
                # Kiểm chứng dựa trên đoạn bằng chứng đã lọc (thay vì Full Context)
                result = verifier.verify_claim(claim, full_context = selected_evidence_text)
                label_pred = result["label_code"]

                y_true.append(label_true)
                y_pred.append(label_pred)
                
                pipeline_details.append({
                    "statement": claim,
                    "gt_url": gt_url,
                    "retrieved_url": predicted_url,
                    "retrieved_evidence": selected_evidence_text,
                    "true_label": label_true,
                    "pred_label": label_pred,
                    "is_correct": (label_true == label_pred)
                })

            # 5. Thống kê chỉ số cuối cùng
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')

            print(f"\n📊 FINAL RESULTS for {model_name_safe}:")
            print(f"Accuracy: {acc:.4f} | F1-Macro: {f1_macro:.4f}")
            print(classification_report(y_true, y_pred, digits=4, target_names=["Supported", "Refuted", "NEI"]))
            
            # Lưu log kết quả
            output_dir = os.path.join(current_dir, "results")
            os.makedirs(output_dir, exist_ok = True)
            output_file = os.path.join(current_dir, "results", f"ro+ro+ge_{model_name_safe}.json")
            
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump({
                    "model_info": model_path,
                    "metrics": {"accuracy": acc, "f1_macro": f1_macro},
                    "details": pipeline_details
                }, f_out, ensure_ascii=False, indent=4)
            
            # Dọn dẹp tài nguyên GPU
            del verifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error in Full Pipeline for {model_path}: {e}")

if __name__ == "__main__":
    run_full_pipeline_experiment()