import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class PLMHandler:
    def __init__(self, model_path: str, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Tự động load cả Encoder và Classifier head đã train từ Hub
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict(self, claim: str, context: str) -> int:
        # Tokenize đầu vào
        inputs = self.tokenizer.encode_plus(
            claim,
            text_pair=context,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # Model trả về logits trực tiếp
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Lấy index của nhãn có xác suất cao nhất
            predicted_class_id = torch.argmax(logits, dim=1).item()
            return predicted_class_id