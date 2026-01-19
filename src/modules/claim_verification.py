import torch
from typing import List, Optional, Dict, Any
from src.components.PLM import PLMHandler 

class ClaimVerificationModule:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load handler (đã được tối ưu ở bước 1)
        self.handler = PLMHandler(model_path, self.device)
        
        # Lấy label_map từ config của model nếu có, nếu không thì dùng mặc định
        if hasattr(self.handler.model.config, 'id2label'):
            self.label_map = self.handler.model.config.id2label
        else:
            self.label_map = {0: "Supported", 1: "Refuted", 2: "Not Enough Information"}

    def verify_claim(self, claim: str, 
                     full_context: Optional[str] = None, 
                     evidences: Optional[List[Any]] = None) -> Dict[str, Any]:
        
        # Chuẩn bị context
        if evidences:
            texts = [doc.page_content.strip() for doc in evidences]
            context = ' '.join([s if s.endswith('.') else s + '.' for s in texts])
        else:
            context = full_context if full_context else ""

        # Dự đoán
        prediction = self.handler.predict(claim, context)
                    
        return {
            "label_code": prediction,
            "label_name": self.label_map.get(prediction) or self.label_map.get(prediction),
            "claim": claim,
            "context_used": "Evidence-based" if evidences else "Full-context"
        }