from typing import List, Dict, Any
from src.components.document import Document
from src.components.vectorDB import VietnameseVectorDB
from src.modules.document_retrieval import vietnamese_tokenizer

class EvidenceSelectionModule:
    def __init__(self, db: VietnameseVectorDB):
        self.db = db
        self.tokenizer = vietnamese_tokenizer

    def select_top_k_evidence(self, query: str, target_url: str, top_k: int = 3, 
                               weights: tuple = (0.2, 0.2, 0.6)) -> List[Document]:
        target_indices = [i for i, d in enumerate(self.db.corpus) if d.metadata['context_url'] == target_url]
        if not target_indices: return []

        s_bm25, s_tfidf, s_emb = self.db.compute_all_scores(query, self.tokenizer)
        w_bm25, w_tfidf, w_emb = weights
        
        results = []
        for idx in target_indices:
            score = (s_bm25[idx] * w_bm25) + (s_tfidf[idx] * w_tfidf) + (s_emb[idx] * w_emb)
            results.append((self.db.corpus[idx], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in results[:top_k]]

    def dynamic_select_evidence(self, query: str, target_url: str, reranker, 
                                initial_top_k: int = 10, t1: float = 0.75, t2: float = 0.05):
        """Logic for dynamic evidence selection based on CER scores."""
        # Step 1: Get initial candidates
        candidates = self.select_top_k_evidence(query, target_url, top_k=initial_top_k)
        # Step 2: Rerank
        reranked = reranker.rerank(query, candidates)
        
        # Step 3: Hierarchical selection logic
        has_high_score = any(res['rerank_score'] > t1 for res in reranked)
        if has_high_score:
            return [res['document'] for res in reranked if res['rerank_score'] > t1]
        
        selected = []
        if reranked:
            selected.append(reranked[0])
            for i in range(1, len(reranked)):
                if (reranked[i-1]['rerank_score'] - reranked[i]['rerank_score']) < t2:
                    selected.append(reranked[i])
                else: break
        return [s['document'] for s in selected]