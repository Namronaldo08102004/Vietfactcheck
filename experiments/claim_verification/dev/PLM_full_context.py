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

from src.modules.claim_verification import ClaimVerificationModule

# ==========================================
# --- HÀM CHẠY VALIDATION ---
# ==========================================

def run_all_plm_gold_evidence():
    print("\n--- Running Claim Verification: 6 PLMs + Gold Evidence ---")
    
    data_path = os.path.join(project_root, "data", "verification", "dev.json")
    
    if not os.path.exists(data_path):
        print(f"❌ Không tìm thấy file dữ liệu tại: {data_path}")
        return

    plm_models = [
        "tranthaihoa/xlm_base_full",
        "tranthaihoa/xlm_large_full",
        "tranthaihoa/ViBERT_Full",
        "tranthaihoa/mBert_Full",
        "tranthaihoa/phobert_base_Context",
        "tranthaihoa/phobert_large_Context"
    ]

    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    for model_path in plm_models:
        model_name_safe = model_path.replace("/", "_")
        print(f"\n🚀 Testing Model: {model_path}")
        
        try:
            verifier = ClaimVerificationModule(model_path = model_path)
            y_true, y_pred = [], []
            results_details = [] # Lưu chi tiết từng câu

            for record in tqdm(records, desc = f"Processing {model_name_safe}"):
                claim = str(record["Statement"])
                gold_evidence = str(record.get("Context", ""))
                label_true = int(record["labels"])

                result = verifier.verify_claim(claim, full_context = gold_evidence)
                label_pred = result["label_code"]

                y_true.append(label_true)
                y_pred.append(label_pred)
                
                # Lưu log chi tiết
                results_details.append({
                    "statement": claim,
                    "evidence": gold_evidence,
                    "true_label": label_true,
                    "pred_label": label_pred,
                    "is_correct": label_true == label_pred
                })

            # 1. Tính toán các chỉ số
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_micro = f1_score(y_true, y_pred, average='micro')

            # 2. In báo cáo ra console
            print(f"\n📊 Kết quả {model_name_safe}:")
            print(f"Accuracy: {acc:.4f} | F1-Macro: {f1_macro:.4f}")
            target_names = ["Supported", "Refuted", "NEI"] 
            print(classification_report(y_true, y_pred, digits=4, target_names=target_names))
            
            # 3. LƯU FILE KẾT QUẢ
            output_dir = os.path.join(project_root, "experiments", "claim_verification", "dev", "results")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"results_full_context_{model_name_safe}.json")
            
            final_output = {
                "model_info": {
                    "path": model_path,
                    "name_safe": model_name_safe
                },
                "metrics": {
                    "accuracy": acc,
                    "f1_macro": f1_macro,
                    "f1_micro": f1_micro
                },
                "details": results_details # Danh sách chi tiết từng câu dự đoán
            }

            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump(final_output, f_out, ensure_ascii=False, indent=4)
            
            print(f"💾 Đã lưu kết quả tại: {output_file}")
            
            # Dọn dẹp GPU
            del verifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error testing {model_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_all_plm_gold_evidence()