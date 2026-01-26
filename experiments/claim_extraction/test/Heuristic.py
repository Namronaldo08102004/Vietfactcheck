import os
import sys
import json
from typing import List, Dict
import py_vncorenlp
from collections import defaultdict
import re

from nltk.translate.chrf_score import sentence_chrf
from nltk.tokenize import sent_tokenize


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

    def _normalize_sentence_text(self, text: str) -> str:
        """
        Convert VnCoreNLP word-segmented text to readable Vietnamese
        """

        # 1. replace underscore with space
        text = text.replace("_", " ")

        # 2. normalize spaces around punctuation
        text = re.sub(r"\s+([,.!?;:)])", r"\1", text)
        text = re.sub(r"([(])\s+", r"\1", text)

        # 3. normalize multiple spaces
        text = re.sub(r"\s+", " ", text)

        # 4. normalize dash spacing
        text = re.sub(r"\s*-\s*", " - ", text)

        # 5. strip
        return text.strip()

    def extract(self, text: str):
        output = []
        annotated = self.annotator.annotate_text(text)

        sentences = self._normalize_sentences(annotated)

        for sent in sentences:
            words = [w["wordForm"] for w in sent]
            pos_tags = [w["posTag"] for w in sent]
            ner_tags = [w.get("nerLabel", "O") for w in sent]

            raw_sentence = " ".join(words)
            sentence_text = self._normalize_sentence_text(raw_sentence)

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

def get_topk_chrf_sentences(
    claim: str,
    context: str,
    k: int = 5
):
    sentences = sent_tokenize(context)
    scored = []

    for sent in sentences:
        score = sentence_chrf(claim, sent)
        scored.append((sent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]   # [(sentence, chrf_score)]

def precision_at_k(predicted: List[str], gold: List[str], k: int):
    if not predicted:
        return 0.0

    pred_k = predicted[:k]
    gold_set = set(gold)

    hit = sum(1 for s in pred_k if s in gold_set)
    return hit / k

def evaluate_single_record_chrf(
    statement: str,
    context: str,
    predicted_claims: List[str],
    ks=(1, 3, 5)
):
    gold_scored = get_topk_chrf_sentences(statement, context, max(ks))
    gold_sents = [s for s, _ in gold_scored]

    metrics = {}
    for k in ks:
        metrics[f"precision@{k}"] = precision_at_k(
            predicted_claims,
            gold_sents,
            k
        )

    return {
        "gold_chrf_sentences": gold_scored,
        **metrics
    }

def evaluate_and_log_claim_extraction_chrf(
    records,
    extractor,
    output_json_path: str,
    ks=(1, 3, 5),
    model_info: str = "heuristic",
    evaluate_model: str = "chrf_sentence_retrieval",
):
    precision_sum = {k: 0.0 for k in ks}
    document_results = []

    total_docs = 0
    total_claims = 0
    total_statements = 0

    for rec in records:
        context = rec.get("fake_context", "")
        statements = rec.get("statements", [])
        index = rec.get("index")
        topic = rec.get("topic")

        if not context.strip() or not statements:
            continue

        # -----------------------------------
        # Extract ONCE per document
        # -----------------------------------
        extracted = extractor.extract(context)

        # support both:
        #  - ["sent1", "sent2"]
        #  - [{"text": "..."}]
        if extracted and isinstance(extracted[0], dict):
            pred_claims = [c["text"] for c in extracted]
        else:
            pred_claims = extracted

        total_docs += 1
        total_claims += len(pred_claims)

        per_doc_results = []

        # -----------------------------------
        # Evaluate EACH gold statement
        # -----------------------------------
        for st in statements:
            gold_text = st.get("text", "")
            label = st.get("label")

            if not gold_text.strip():
                continue

            total_statements += 1

            eval_result = evaluate_single_record_chrf(
                statement=gold_text,
                context=context,
                predicted_claims=pred_claims,
                ks=ks
            )

            for k in ks:
                precision_sum[k] += eval_result[f"precision@{k}"]

            per_doc_results.append({
                "gold_statement": gold_text,
                "label": label,
                "gold_chrf_sentences": [
                    {"text": s, "chrf": sc}
                    for s, sc in eval_result["gold_chrf_sentences"]
                ],
                **{f"precision@{k}": eval_result[f"precision@{k}"] for k in ks}
            })

        document_results.append({
            "index": index,
            "topic": topic,
            "predicted_claims": pred_claims,
            "num_predicted_claims": len(pred_claims),
            "statement_results": per_doc_results
        })

    # -----------------------------------
    # IMPORTANT: average over STATEMENTS
    # -----------------------------------
    dataset_metrics = {
        f"precision@{k}": precision_sum[k] / max(total_statements, 1)
        for k in ks
    }

    dataset_metrics.update({
        "avg_claims_per_doc": total_claims / max(total_docs, 1),
        "num_docs": total_docs,
        "num_statements": total_statements
    })

    output = {
        "model_info": model_info,
        "evaluate_model": evaluate_model,
        "dataset_metrics": dataset_metrics,
        "document_results": document_results
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
        "extraction",
        "test_synthesis.json"
    )

    with open(json_input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("\n📊 CLAIM EXTRACTION EVALUATION (chrF + Precision@k)")

    output_eval_path = os.path.join(
        project_root,
        "experiments",
        "claim_extraction",
        "test",
        "results",
        "claim_extraction_heuristic_chrf_eval.json"
    )

    metrics = evaluate_and_log_claim_extraction_chrf(
        records=records,
        extractor=extractor,
        output_json_path=output_eval_path,
        ks=(1, 3, 5),
        model_info="heuristic",
        evaluate_model="chrf_sentence_retrieval"
    )

    for k, v in metrics.items():
        if k.startswith("precision"):
            print(f"{k:15}: {v:.4f}")

    print(f"Avg Claims / Doc : {metrics['avg_claims_per_doc']:.2f}")
    print(f"Evaluated Docs   : {metrics['num_docs']}")
    print(f"\n📁 Saved to: {output_eval_path}")
