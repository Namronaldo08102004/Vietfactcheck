from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class VietnameseReranker:
    def __init__(self, model_name: str = 'AITeamVN/Vietnamese_Reranker'):
        """
        Initialize the CrossEncoder reranker.
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name, tokenizer_args = {"use_fast": False})

    def rerank(self, query: str, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on the query.
        'documents' can be a list of Document objects or strings.
        """
        if not documents:
            return []

        # Prepare pairs for CrossEncoder (Query, Document_Content)
        # Handle cases where documents might be the Document class or raw strings
        pairs = []
        for doc in documents:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            pairs.append([query, content])

        # Predict scores
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Create results list with original objects and their new scores
        reranked_results = []
        for i in range(len(documents)):
            reranked_results.append({
                "document": documents[i],
                "rerank_score": float(scores[i])
            })

        # Sort by score descending
        reranked_results.sort(key = lambda x: x["rerank_score"], reverse=True)

        return reranked_results