import os
import sys
import json
from typing import List, Dict

import numpy as np
from nltk.translate.chrf_score import sentence_chrf
sys.path.append('C:/Users/lebat/Documents/Github/Vietfactcheck/src/components/BERTSum')
from presumm import train
from nltk.tokenize import sent_tokenize

# --------------------------------------------------
# PATH HANDLING
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))


class BERTSumSentenceExtractor:
    """
    Extract claims using BERTSum selected sentences.
    """

    def extract(self, sample: Dict) -> List[str]:
        return sample.get("sents_selected_by_bertsum", [])

def extract_claims_from_json(
    json_path: str,
    extractor: BERTSumSentenceExtractor,
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
    model_info: str = "bertsum",
    evaluate_model: str = "chrf_sentence_retrieval",
):
    precision_sum = {k: 0.0 for k in ks}
    document_results = []
    total = 0
    total_claims = 0

    for rec in records:
        context = rec.get("Context", "")
        statement = rec.get("Statement", "")
        index = rec.get("index")

        if not context.strip() or not statement.strip():
            continue

        # 🔥 BERTSum extractor
        pred_claims = extractor.extract(rec)

        eval_result = evaluate_single_record_chrf(
            statement=statement,
            context=context,
            predicted_claims=pred_claims,
            ks=ks
        )

        for k in ks:
            precision_sum[k] += eval_result[f"precision@{k}"]

        total += 1
        total_claims += len(pred_claims)

        document_results.append({
            "index": index,
            "statement": statement,
            "predicted_claims": pred_claims,
            "num_predicted_claims": len(pred_claims),
            "gold_chrf_sentences": [
                {"text": s, "chrf": sc}
                for s, sc in eval_result["gold_chrf_sentences"]
            ],
            **{f"precision@{k}": eval_result[f"precision@{k}"] for k in ks}
        })

    dataset_metrics = {
        f"precision@{k}": precision_sum[k] / max(total, 1)
        for k in ks
    }
    dataset_metrics.update({
        "avg_claims_per_doc": total_claims / max(total, 1),
        "num_docs": total
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

def sentence_ranking_by_BertSum(all_avail_url_files): 
    # load configs 
    configs = dict() 
    configs['task'] = 'ext' 
    configs['mode'] = 'test_text' 
    configs['test_from'] = 'C:/Users/lebat/Documents/Github/Vietfactcheck/src/components/BERTSum/presumm/save_model/bertext_cnndm_transformer.pt' 
    configs['text_src'] = 'C:/Users/lebat/Documents/Github/Vietfactcheck/data/verification/dev.json' 
    configs['result_path'] = 'C:/Users/lebat/Documents/Github/Vietfactcheck/src/components/BERTSum/presumm/results/ootb_output' 
    configs['alpha'] = 0.95 
    configs['log_file'] = 'C:/Users/lebat/Documents/Github/Vietfactcheck/src/components/BERTSum/presumm/logs/test.log' 
    configs['visible_gpus'] = '0' 
    # use BertSum to select candidate central sentences 
    sent_with_score = train.main(configs) 

    with open(all_avail_url_files, 'r', encoding='utf-8') as f: 
        samples = json.load(f) 
    
    for idx, sample in enumerate(samples): 
        # split the fulltext into sentences 
        fulltext = sample['Context'] 
        if fulltext[0] in ["“", "'", "”"] and fulltext[-1] in ["“", "'", "”"]: 
            fulltext = fulltext[1:-1] 
        sentences = sent_tokenize(fulltext) 
        
        sample['sents_id_selected_by_bertsum'] = sent_with_score[idx][2] 
        sample['sents_selected_by_bertsum'] = sent_with_score[idx][0] 
        sample['sents_with_scores_by_bertsum'] = sent_with_score[idx][1].tolist() 
        sample['sents_order_by_bertsum'] = sent_with_score[idx][4] 
        sample['sent_texts_order_by_bertsum'] = sent_with_score[idx][5] 
        sample['sentences'] = sentences 
        
        print("Sample id={} with {} sentences, {} candidate central sentences selected by BertSum".format( idx, len(sentences), len(sample['sents_id_selected_by_bertsum']))) 
        print(sentences) 
    
    return samples

if __name__ == "__main__":

    # 1️⃣ chạy BERTSum
    json_input_path = os.path.join(
        project_root, "data", "verification", "dev.json"
    )

    samples = sentence_ranking_by_BertSum(json_input_path)

    # 2️⃣ dùng BERTSum làm extractor
    extractor = BERTSumSentenceExtractor()

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
        ks=(1, 3, 5),
        model_info="bertsum",
        evaluate_model="chrf_sentence_retrieval"
    )

    for k, v in metrics.items():
        if k.startswith("precision"):
            print(f"{k:15}: {v:.4f}")

    print(f"Avg Claims / Doc : {metrics['avg_claims_per_doc']:.2f}")
    print(f"Evaluated Docs   : {metrics['num_docs']}")