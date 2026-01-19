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
from src.modules.claim_verification import ClaimVerificationModule

def run_retrieval_verification_pipeline():
    print("\n--- Running Integrated Experiment: Document Retrieval (Top-1) + Claim Verification (Full Context) ---")
    
    # 1. Cấu hình đường dẫn dữ liệu
    # Sử dụng file retrieval/train_data.json như bạn yêu cầu
    data_path = os.path.join(project_root, "data/retrieval/train_data.json")
    all_data_paths = [
        data_path,
        os.path.join(project_root, "data/retrieval/dev_data.json"),
        os.path.join(project_root, "data/retrieval/test_data.json")
    ]
    existing_paths = [p for p in all_data_paths if os.path.exists(p)]
    db_name = "train_vectorDB"

    if not os.path.exists(data_path):
        print(f"❌ Không tìm thấy file dữ liệu tại: {data_path}")
        return

    # 2. Khởi tạo Document Retrieval Module
    db = VietnameseVectorDB(
        db_name = db_name,
        storage_dir = settings.STORAGE_DIR,
        model_name = settings.EMBEDDING_MODEL,
        truncation_dim = settings.TRUNCATION_DIM,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    retrieval_module = DocumentRetrievalModule(db)

    # Load hoặc Build Index
    if not db.load():
        print(f"🚀 Building unified index for {db_name}...")
        retrieval_module.build_system(existing_paths)
    else:
        print(f"✅ Loaded existing index: {db_name}")

    # 3. Tạo mapping URL -> Context để lấy nội dung bài báo sau khi retrieve
    url_to_context = {}
    for path in existing_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for record in data:
                # Lưu Context tương ứng với mỗi URL
                url_to_context[record["Url"]] = record["Context"]

    # 4. Danh sách 6 mô hình PLM đã fine-tune
    plm_models = [
        "Namronaldo2004/Vifactcheck-xlm-roberta-base",
        "Namronaldo2004/Vifactcheck-xlm-roberta-large",
        "Namronaldo2004/Vifactcheck-ViBERT",
        "Namronaldo2004/Vifactcheck-mBERT",
        "Namronaldo2004/Vifactcheck-phoBERT-base",
        "Namronaldo2004/Vifactcheck-phoBERT-large"
    ]

    # Load dữ liệu đánh giá
    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    RETRIEVAL_WEIGHTS = (0.3, 0.3, 0.4)

    # 5. Vòng lặp qua từng mô hình Verification
    for model_path in plm_models:
        model_name_safe = model_path.replace("/", "_")
        print(f"\n🚀 Pipeline Testing with Verification Model: {model_path}")
        
        try:
            verifier = ClaimVerificationModule(model_path = model_path)
            y_true, y_pred = [], []
            pipeline_details = []

            for record in tqdm(records, desc=f"Processing {model_name_safe}"):
                claim = str(record["Statement"])
                label_true = int(record["labels"])
                ground_truth_url = record["Url"]

                # --- BƯỚC 1: DOCUMENT RETRIEVAL (TOP-1) ---
                predicted_urls = retrieval_module.get_top_k_url(
                    claim, 
                    top_k=1, 
                    weights=RETRIEVAL_WEIGHTS
                )
                predicted_url = predicted_urls[0] if predicted_urls else None

                # --- BƯỚC 2: LẤY CONTEXT TƯƠNG ỨNG ---
                # Nếu không retrieve được hoặc URL không có trong mapping, dùng string rỗng
                retrieved_context = url_to_context.get(predicted_url, "")

                # --- BƯỚC 3: CLAIM VERIFICATION ---
                # Sử dụng nội dung bài báo vừa truy xuất được để verify
                result = verifier.verify_claim(claim, full_context = retrieved_context)
                label_pred = result["label_code"]

                y_true.append(label_true)
                y_pred.append(label_pred)
                
                pipeline_details.append({
                    "statement": claim,
                    "ground_truth_url": ground_truth_url,
                    "retrieved_url": predicted_url,
                    "retrieval_correct": (predicted_url == ground_truth_url),
                    "true_label": label_true,
                    "pred_label": label_pred,
                    "verification_correct": (label_true == label_pred)
                })

            # Tính toán chỉ số cho bước Verification trong pipeline
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')

            print(f"\n📊 Kết quả Pipeline ({model_name_safe}):")
            print(f"Verification Accuracy: {acc:.4f} | F1-Macro: {f1_macro:.4f}")
            print(classification_report(y_true, y_pred, digits=4, target_names=["Supported", "Refuted", "NEI"]))
            
            # Lưu kết quả
            output_dir = os.path.join(current_dir, "results")
            os.makedirs(output_dir, exist_ok = True)
            output_file = os.path.join(current_dir, "results", f"ro+fc_{model_name_safe}.json")
            
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump({
                    "verification_model": model_path,
                    "retrieval_weights": RETRIEVAL_WEIGHTS,
                    "metrics": {"accuracy": acc, "f1_macro": f1_macro},
                    "details": pipeline_details
                }, f_out, ensure_ascii=False, indent=4)
            
            # Giải phóng bộ nhớ GPU
            del verifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
           
        except Exception as e:
            print(f"❌ Error in pipeline for {model_path}: {e}")

if __name__ == "__main__":
    run_retrieval_verification_pipeline()