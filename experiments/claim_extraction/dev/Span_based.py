import os
import sys
import json
from typing import List, Dict
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import py_vncorenlp

from sentence_transformers import SentenceTransformer, util

# ==================================================
# PATH SETUP
# ==================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================
# LABEL SETUP (BIO)
# ==================================================
LABELS = ["O", "B-CLAIM", "I-CLAIM"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ==================================================
# SPAN-BASED CLAIM EXTRACTOR
# ==================================================

class VietnameseSpanBasedClaimExtractorHF:
    """
    Span-based Claim Extraction using BIO tagging
    """

    def __init__(
        self,
        model_name_or_path: str,
        vncorenlp_dir: str = None
    ):
        if vncorenlp_dir is None:
            vncorenlp_dir = os.path.join(project_root, "vncorenlp")

        self.annotator = py_vncorenlp.VnCoreNLP(
            save_dir=vncorenlp_dir,
            annotators=["wseg"]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )

        self.model.to(DEVICE)
        self.model.eval()

    # --------------------------------------------------
    # Sentence segmentation
    # --------------------------------------------------
    def _split_sentences(self, text: str) -> List[List[str]]:
        annotated = self.annotator.annotate_text(text)

        if isinstance(annotated, dict):
            annotated = list(annotated.values())

        sentences = []
        for sent in annotated:
            tokens = [w["wordForm"] for w in sent]
            if tokens:
                sentences.append(tokens)

        return sentences

    # --------------------------------------------------
    # Span extraction
    # --------------------------------------------------
    def extract(self, text: str) -> List[Dict]:
        sentences = self._split_sentences(text)
        extracted_spans = []

        for tokens in sentences:
            inputs = self.tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                predictions = torch.argmax(logits, dim=-1)[0].cpu().tolist()

            spans = []
            current_span = []

            # Skip [CLS] token → predictions[1:]
            for token, label_id in zip(tokens, predictions[1:len(tokens)+1]):
                label = ID2LABEL[label_id]

                if label == "B-CLAIM":
                    if current_span:
                        spans.append(" ".join(current_span))
                    current_span = [token]

                elif label == "I-CLAIM" and current_span:
                    current_span.append(token)

                else:
                    if current_span:
                        spans.append(" ".join(current_span))
                        current_span = []

            if current_span:
                spans.append(" ".join(current_span))

            for span in spans:
                extracted_spans.append({
                    "text": span
                })

        return extracted_spans

# ==================================================
# SEMANTIC EVALUATION (Span-based)
# ==================================================

SEM_MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
semantic_model = SentenceTransformer(SEM_MODEL_NAME)

def semantic_similarity(a: str, b: str) -> float:
    emb_a = semantic_model.encode(a, convert_to_tensor=True)
    emb_b = semantic_model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item()


def evaluate_span_based_claim_extraction(
    records,
    extractor,
    output_json_path: str,
    threshold: float = 0.75,
    model_info: str = "span-based-phobert",
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
                "num_predicted_claims": 0,
                "hit": 0
            })
            continue

        scores = [semantic_similarity(statement, c) for c in pred_claims]

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

    # MODEL_NAME_OR_PATH = os.path.join(
    #     project_root,
    #     "models",
    #     "span_based_claim_phobert"
    # )
    MODEL_NAME_OR_PATH = 'vinai/phobert-base'

    extractor = VietnameseSpanBasedClaimExtractorHF(
        model_name_or_path=MODEL_NAME_OR_PATH
    )

    input_path = os.path.join(
        project_root,
        "data",
        "verification",
        "dev.json"
    )

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("\n🔎 SPAN-BASED CLAIM EXTRACTION (DEMO)\n")

    for rec in records[:3]:
        spans = extractor.extract(rec["Context"])
        print("=" * 60)
        print("STATEMENT:", rec["Statement"])
        for s in spans:
            print("CLAIM SPAN:", s["text"])

    print("\n📊 SPAN-BASED CLAIM EXTRACTION (EVALUATION)\n")

    output_eval_path = os.path.join(
        project_root,
        "experiments",
        "claim_extraction",
        "dev",
        "results",
        "span_based_claim_extraction_semantic.json"
    )

    metrics = evaluate_span_based_claim_extraction(
        records=records,
        extractor=extractor,
        output_json_path=output_eval_path
    )

    print(f"Claim Hit Rate        : {metrics['claim_hit_rate']:.2%}")
    print(f"Avg Best Similarity   : {metrics['avg_best_similarity']:.4f}")
    print(f"Mean Rank             : {metrics['mean_rank']:.2f}")
    print(f"Avg Claims / Document : {metrics['avg_claims_per_doc']:.2f}")
    print(f"Evaluated Documents   : {metrics['num_docs']}")
    print(f"\n📁 Saved to: {output_eval_path}")
