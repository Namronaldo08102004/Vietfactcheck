import os
import json
import sys
import tempfile
from typing import List

# --------------------------------------------------
# PATH HANDLING
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))
presumm_root = os.path.join(project_root, "src", "components", "presumm")

from src.components.BERTSum import PresummPackage

class BERTSumClaimExtractor:
    """
    High-level wrapper around PresummPackage.

    Usage:
        extractor = BERTSumClaimExtractor(model_path=...)
        claims = extractor("your input text")
    """

    def __init__(
        self,
        model_path: str,
        alpha: float = 0.95,
        visible_gpus: str = "0"
    ):
        self.presumm = PresummPackage()

        self.model_path = model_path
        self.alpha = alpha
        self.visible_gpus = visible_gpus

    def __call__(self, text: str) -> List[str]:
        return self.extract(text)

    def batch_extract(self, texts: List[str]) -> List[List[str]]:
        """
        Run BERTSum ONCE for multiple documents.
        Returns list of predicted sentence lists.
        """
        import tempfile
        import json
        import os

        if not texts:
            return []

        # build temporary dataset
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")

        samples = [{"Context": t} for t in texts]
        json.dump(samples, tmp, ensure_ascii=False)
        tmp.close()

        configs = {
            "task": "ext",
            "mode": "test_text",
            "test_from": self.model_path,
            "text_src": tmp.name,
            "result_path": tempfile.mkdtemp(),
            "alpha": self.alpha,
            "log_file": os.path.join(tempfile.mkdtemp(), "test.log"),
            "visible_gpus": self.visible_gpus
        }

        # ONE model load here
        sent_with_score = self.presumm.train_main(configs)

        # only return selected sentences
        return [x[0] for x in sent_with_score]


    def extract(self, text: str) -> List[str]:
        """
        Input: raw string
        Output: list of extracted claims (sentences)
        """

        if not text.strip():
            return []

        # ------------------------------------------------
        # Presumm expects JSON file → create temp dataset
        # ------------------------------------------------
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")

        sample = [{
            "Context": text
        }]

        json.dump(sample, tmp, ensure_ascii=False)
        tmp.close()

        # ------------------------------------------------
        # Build configs
        # ------------------------------------------------
        configs = {
            "task": "ext",
            "mode": "test_text",
            "test_from": self.model_path,
            "text_src": tmp.name,
            "result_path": tempfile.mkdtemp(),
            "alpha": self.alpha,
            "log_file": os.path.join(tempfile.mkdtemp(), "test.log"),
            "visible_gpus": self.visible_gpus
        }

        # ------------------------------------------------
        # Run BERTSum via PresummPackage
        # ------------------------------------------------
        sent_with_score = self.presumm.train_main(configs)

        # ------------------------------------------------
        # Extract selected sentences
        # ------------------------------------------------
        claims = sent_with_score[0][0]

        return claims
