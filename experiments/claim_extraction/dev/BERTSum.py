import os
import sys
import json
from typing import List

from nltk.translate.chrf_score import sentence_chrf
from nltk.tokenize import sent_tokenize

# --------------------------------------------------
# PATH HANDLING
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

import src.components.presumm.model as _models
sys.modules["models"] = _models

from src.modules.claim_extraction import BERTSumClaimExtractor

# ==================================================
# Metrics helpers
# ==================================================

def get_topk_chrf_sentences(claim: str, context: str, k: int = 5):
    sentences = sent_tokenize(context)

    scored = []
    for sent in sentences:
        scored.append((sent, sentence_chrf(claim, sent)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def precision_at_k(predicted: List[str], gold: List[str], k: int):
    if not predicted:
        return 0.0

    pred_k = predicted[:k]
    gold_set = set(gold)
    hit = sum(1 for s in pred_k if s in gold_set)

    return hit / k


def evaluate_single_record_chrf(statement, context, predicted_claims, ks=(1, 3, 5)):
    gold_scored = get_topk_chrf_sentences(statement, context, max(ks))
    gold_sents = [s for s, _ in gold_scored]

    metrics = {}
    for k in ks:
        metrics[f"precision@{k}"] = precision_at_k(predicted_claims, gold_sents, k)

    return {
        "gold_chrf_sentences": gold_scored,
        **metrics
    }


# ==================================================
# Main evaluation
# ==================================================

def evaluate_and_log_claim_extraction_chrf(
    records,
    extractor,
    output_json_path,
    ks=(1, 3, 5),
    model_info="bertsum",
    evaluate_model="chrf_sentence_retrieval",
):

    precision_sum = {k: 0.0 for k in ks}
    document_results = []

    total_docs = 0
    total_claims = 0
    total_statements = 0

    # ------------------------------------------------
    # 1) Collect ALL contexts first
    # ------------------------------------------------
    valid_records = []
    all_contexts = []

    for rec in records:
        ctx = rec.get("fake_context", "")
        stmts = rec.get("statements", [])
        if ctx.strip() and stmts:
            valid_records.append(rec)
            all_contexts.append(ctx)

    # ------------------------------------------------
    # 2) Run BERTSum ONCE
    # ------------------------------------------------
    print(f"Running BERTSum on {len(all_contexts)} documents ...")
    all_pred_claims = extractor.batch_extract(all_contexts)

    # ------------------------------------------------
    # 3) Evaluate using cached predictions
    # ------------------------------------------------
    for rec, pred_claims in zip(valid_records, all_pred_claims):

        # flatten safety
        if len(pred_claims) == 1 and isinstance(pred_claims[0], list):
            pred_claims = pred_claims[0]

        context = rec["fake_context"]
        statements = rec["statements"]
        index = rec.get("index")
        topic = rec.get("topic")

        total_docs += 1
        total_claims += len(pred_claims)

        per_doc_results = []

        for st in statements:
            gold_text = st.get("text", "")
            if not gold_text.strip():
                continue

            total_statements += 1

            eval_result = evaluate_single_record_chrf(
                gold_text, context, pred_claims, ks
            )

            for k in ks:
                precision_sum[k] += eval_result[f"precision@{k}"]

            per_doc_results.append({
                "gold_statement": gold_text,
                "label": st.get("label"),
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

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return dataset_metrics

# ==================================================
# Entry
# ==================================================

if __name__ == "__main__":

    json_input_path = os.path.join(project_root, "data", "extraction", "dev_synthesis.json")

    with open(json_input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    extractor = BERTSumClaimExtractor(
        model_path=os.path.join(project_root, "src", "weights", "BERTSum", "bertext_cnndm_transformer.pt")
    )

    output_eval_path = os.path.join(
        project_root,
        "experiments",
        "claim_extraction",
        "dev",
        "results",
        "claim_extraction_bertsum_chrf_eval.json"
    )

    metrics = evaluate_and_log_claim_extraction_chrf(
        records=samples,
        extractor=extractor,
        output_json_path=output_eval_path,
    )

    for k, v in metrics.items():
        if k.startswith("precision"):
            print(f"{k:15}: {v:.4f}")

    print(f"Avg Claims / Doc : {metrics['avg_claims_per_doc']:.2f}")
    print(f"Evaluated Docs   : {metrics['num_docs']}")
