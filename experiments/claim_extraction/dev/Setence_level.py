import os
import sys
import json
from typing import List, Dict
from collections import defaultdict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import py_vncorenlp

# ==================================================
# PATH SETUP
# ==================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================
# MODEL CONFIG
# ==================================================
ENCODER_NAME = "vinai/phobert-base"
NUM_LABELS = 2   # 0: non-claim, 1: claim

# ==================================================
# SENTENCE-LEVEL CLAIM EXTRACTOR (HF)
# ==================================================

class VietnameseSentenceLevelClaimExtractorHF:
    """
    Sentence-level Claim Extraction using HuggingFace models
    """

    def __init__(
        self,
        model_name_or_path: str,
        vncorenlp_dir: str = None,
        threshold: float = 0.5
    ):
        """
        model_name_or_path:
            - HuggingFace Hub id
            - or local directory saved by save_pretrained()
        """

        if vncorenlp_dir is None:
            vncorenlp_dir = os.path.join(project_root, "vncorenlp")

        self.annotator = py_vncorenlp.VnCoreNLP(
            save_dir=vncorenlp_dir,
            annotators=["wseg"]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=NUM_LABELS
        )

        self.model.to(DEVICE)
        self.model.eval()

        self.threshold = threshold

    # --------------------------------------------------
    # Sentence segmentation
    # --------------------------------------------------
    def _split_sentences(self, text: str) -> List[str]:
        annotated = self.annotator.annotate_text(text)

        if isinstance(annotated, dict):
            annotated = list(annotated.values())

        sentences = []
        for sent in annotated:
            words = [w["wordForm"] for w in sent]
            sentences.append(" ".join(words))

        return sentences

    # --------------------------------------------------
    # Claim extraction
    # --------------------------------------------------
    def extract(self, text: str) -> List[Dict]:
        sentences = self._split_sentences(text)
        outputs = []

        for sent in sentences:
            inputs = self.tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                claim_prob = probs[0, 1].item()

            if claim_prob >= self.threshold:
                outputs.append({
                    "text": sent,
                    "claim_probability": claim_prob
                })

        return outputs


# ==================================================
# SEMANTIC EVALUATION
# ==================================================

from sentence_transformers import SentenceTransformer, util

SEM_MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
semantic_model = SentenceTransformer(SEM_MODEL_NAME)

def semantic_similarity(a: str, b: str) -> float:
    emb_a = semantic_model.encode(a, convert_to_tensor=True)
    emb_b = semantic_model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item()


def evaluate_sentence_level_claim_extraction(
    records,
    extractor,
    output_json_path: str,
    threshold: float = 0.75,
    model_info: str = "sentence-level-phobert",
    evaluate_model: str = "semantic_simcse_phobert",
):
    hits, total = 0, 0
    sims, ranks = [], []
    total_claims = 0
    document_results = []

    for rec in records:
        context = rec.get("Context", "")
        statement = rec.get("Statement", "")
        index = rec.get("index")

        if not context.strip() or not statement.strip():
            continue

        extracted = extractor.extract(context)
        pred_claims = [c["text"] for c in extracted]

        total += 1
        total_claims += len(pred_claims)

        if not pred_claims:
            document_results.append({
                "index": index,
                "statement": statement,
                "predicted_claims": [],
                "hit": 0
            })
            continue

        scores = [
            semantic_similarity(statement, c)
            for c in pred_claims
        ]

        best_sim = max(scores)
        rank = scores.index(best_sim) + 1
        hit = int(best_sim >= threshold)

        hits += hit
        sims.append(best_sim)
        ranks.append(rank)

        document_results.append({
            "index": index,
            "statement": statement,
            "predicted_claims": pred_claims,
            "num_predicted_claims": len(pred_claims),
            "hit": hit,
            "best_similarity": best_sim,
            "rank": rank,
        })

    dataset_metrics = {
        "claim_hit_rate": hits / max(total, 1),
        "avg_best_similarity": sum(sims) / max(len(sims), 1),
        "mean_rank": sum(ranks) / max(len(ranks), 1),
        "avg_claims_per_doc": total_claims / max(total, 1),
        "num_docs": total,
    }

    output = {
        "model_info": model_info,
        "evaluate_model": evaluate_model,
        "semantic_model": SEM_MODEL_NAME,
        "threshold": threshold,
        "dataset_metrics": dataset_metrics,
        "document_results": document_results,
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return dataset_metrics


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":

    # Có thể là:
    # 1) "your-hf-username/phobert-claim-sentence"
    # 2) hoặc local dir: models/claim_sentence_phobert/
    # MODEL_NAME_OR_PATH = os.path.join(
    #     project_root,
    #     "models",
    #     "sentence_level_claim_phobert"
    # )

    MODEL_NAME_OR_PATH = 'vinai/phobert-base'

    extractor = VietnameseSentenceLevelClaimExtractorHF(
        model_name_or_path=MODEL_NAME_OR_PATH,
        threshold=0.5
    )

    input_path = os.path.join(
        project_root,
        "data",
        "verification",
        "dev.json"
    )

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    output_eval_path = os.path.join(
        project_root,
        "experiments",
        "claim_extraction",
        "dev",
        "results",
        "sentence_level_claim_extraction_hf_eval.json"
    )

    metrics = evaluate_sentence_level_claim_extraction(
        records=records,
        extractor=extractor,
        output_json_path=output_eval_path
    )

    print("\n📊 SENTENCE-LEVEL CLAIM EXTRACTION (HF)")
    for k, v in metrics.items():
        print(f"{k:25}: {v:.4f}" if isinstance(v, float) else f"{k:25}: {v}")
    print(f"\n📁 Saved to: {output_eval_path}")