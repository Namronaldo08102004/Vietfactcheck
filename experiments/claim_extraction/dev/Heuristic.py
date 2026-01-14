import os
import sys
import json
from typing import List, Dict
import py_vncorenlp
from collections import defaultdict

# --------------------------------------------------
# PATH HANDLING
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))


class VietnameseHeuristicClaimExtractor:
    """
    Heuristic + POS-based Claim Extraction for Vietnamese news.
    """

    FACTUAL_VERBS = {
        "là", "đã", "đang", "sẽ", "có", "đạt",
        "tăng", "giảm", "chiếm", "gây", "dẫn",
        "ảnh hưởng", "ghi nhận", "xảy ra"
    }

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(project_root, "vncorenlp")

        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"VnCoreNLP model not found at {model_dir}. "
                f"Please run py_vncorenlp.download_model(save_dir='{model_dir}')"
            )

        self.annotator = py_vncorenlp.VnCoreNLP(
            save_dir=model_dir,
            annotators=["wseg", "pos", "ner"],
        )

    def _is_declarative(self, sentence: str) -> bool:
        return (
            sentence.strip().endswith(".")
            and "?" not in sentence
            and "!" not in sentence
        )

    def _has_factual_verb(self, pos_tags: List[str], words: List[str]) -> bool:
        for word, pos in zip(words, pos_tags):
            if pos.startswith("V") and word.lower() in self.FACTUAL_VERBS:
                return True
        return False

    def _has_entity_or_number(self, pos_tags: List[str], ner_tags: List[str]) -> bool:
        for pos, ner in zip(pos_tags, ner_tags):
            if pos == "M":          # number
                return True
            if ner != "O":          # named entity
                return True
        return False

    def _normalize_sentences(self, annotated):
        """
        Normalize all known py_vncorenlp output formats
        to List[List[token_dict]]
        """

        # Case 1: list of sentences
        if isinstance(annotated, list):
            return annotated

        if isinstance(annotated, dict):

            # Case 2: dict[int -> list[token]]  (MOST COMMON ON WINDOWS)
            if all(isinstance(k, int) for k in annotated.keys()):
                return list(annotated.values())

            # Case 3: dict with sentences
            if "sentences" in annotated and annotated["sentences"]:
                return annotated["sentences"]

            # Case 4: nested annotations
            if "annotations" in annotated:
                for ann in annotated["annotations"]:
                    if "sentences" in ann:
                        return ann["sentences"]

            # Case 5: flat tokens with sentenceIndex
            if "tokens" in annotated:
                sent_map = defaultdict(list)
                for tok in annotated["tokens"]:
                    idx = tok.get("sentenceIndex", 0)
                    sent_map[idx].append(tok)
                return list(sent_map.values())

        raise ValueError(
            f"Unsupported VnCoreNLP output format: {type(annotated)}"
        )


    def extract(self, text: str):
        output = []
        annotated = self.annotator.annotate_text(text)

        sentences = self._normalize_sentences(annotated)

        for sent in sentences:
            words = [w["wordForm"] for w in sent]
            pos_tags = [w["posTag"] for w in sent]
            ner_tags = [w.get("nerLabel", "O") for w in sent]

            sentence_text = " ".join(words)

            score = 0
            if self._is_declarative(sentence_text):
                score += 1
            if self._has_factual_verb(pos_tags, words):
                score += 1
            if self._has_entity_or_number(pos_tags, ner_tags):
                score += 1

            if score >= 2:
                output.append({
                    "text": sentence_text,
                    "score": score,
                    "pos_tags": pos_tags,
                    "ner_tags": ner_tags
                })

        return output



def extract_claims_from_json(
    json_path: str,
    extractor: VietnameseHeuristicClaimExtractor,
) -> List[Dict]:
    """
    Read a JSON file and extract claims from each Context field.

    Returns:
        List[Dict]: list of records with extracted claims
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for record in data:
        context = record.get("Context", "")
        if not context.strip():
            continue

        claims = extractor.extract(context)

        results.append({
            "index": record.get("index"),
            "topic": record.get("Topic"),
            "claims": claims,
            "num_claims": len(claims),
        })

    return results

# ==================================================
# CLAIM EXTRACTION METRICS
# ==================================================

from sentence_transformers import SentenceTransformer, util

SEM_MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
semantic_model = SentenceTransformer(SEM_MODEL_NAME)

def semantic_similarity(a: str, b: str) -> float:
    emb_a = semantic_model.encode(a, convert_to_tensor=True)
    emb_b = semantic_model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item()

def evaluate_single_record(
    statement: str,
    extracted_claims: List[str],
    threshold: float = 0.75,
):
    """
    Evaluate whether extracted claims hit the gold Statement
    """
    if not extracted_claims:
        return {
            "hit": 0,
            "best_sim": 0.0,
            "rank": None,
        }

    sims = [
        semantic_similarity(statement, c)
        for c in extracted_claims
    ]

    best_sim = max(sims)
    best_rank = sims.index(best_sim) + 1

    hit = int(best_sim >= threshold)

    return {
        "hit": hit,
        "best_sim": best_sim,
        "rank": best_rank,
    }


def evaluate_and_log_claim_extraction_semantic(
    records,
    extractor,
    output_json_path: str,
    threshold: float = 0.75,
    model_info: str = "heuristic",
    evaluate_model: str = "semantic_simcse_phobert",
):
    """
    Semantic evaluation + detailed JSON logging
    """

    document_results = []

    hits = 0
    total = 0
    sims = []
    ranks = []
    total_claims = 0

    for rec in records:
        context = rec.get("Context", "")
        statement = rec.get("Statement", "")
        index = rec.get("index")

        if not context.strip() or not statement.strip():
            continue

        extracted = extractor.extract(context)
        pred_claims = [c["text"] for c in extracted]

        eval_result = evaluate_single_record(
            statement=statement,
            extracted_claims=pred_claims,
            threshold=threshold
        )

        hits += eval_result["hit"]
        total += 1
        total_claims += len(pred_claims)

        if eval_result["best_sim"] > 0:
            sims.append(eval_result["best_sim"])
        if eval_result["rank"] is not None:
            ranks.append(eval_result["rank"])

        document_results.append({
            "index": index,
            "statement": statement,
            "predicted_claims": pred_claims,
            "num_predicted_claims": len(pred_claims),
            "hit": eval_result["hit"],
            "best_similarity": eval_result["best_sim"],
            "rank": eval_result["rank"],
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

if __name__ == "__main__":
    extractor = VietnameseHeuristicClaimExtractor()

    json_input_path = os.path.join(
        project_root,
        "data",
        "verification",
        "dev.json"
    )

    # Load raw records
    with open(json_input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # -------------------------------
    # Extract claims (demo)
    # -------------------------------
    print("\n📊 CLAIM EXTRACTION EVALUATION (SEMANTIC + LOGGING)")

    output_eval_path = os.path.join(
        project_root,
        "experiments",
        "claim_extraction",
        "dev",
        "results",
        "claim_extraction_heuristic_with_semantic_eval.json"
    )

    metrics = evaluate_and_log_claim_extraction_semantic(
        records=records,
        extractor=extractor,
        output_json_path=output_eval_path,
        threshold=0.75,
        model_info="heuristic",
        evaluate_model="semantic_simcse_phobert"
    )

    print(f"Claim Hit Rate        : {metrics['claim_hit_rate']:.2%}")
    print(f"Avg Best Similarity   : {metrics['avg_best_similarity']:.4f}")
    print(f"Mean Rank             : {metrics['mean_rank']:.2f}")
    print(f"Avg Claims / Document : {metrics['avg_claims_per_doc']:.2f}")
    print(f"Evaluated Documents   : {metrics['num_docs']}")
    print(f"\n📁 Saved to: {output_eval_path}")