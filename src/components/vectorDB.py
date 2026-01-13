import os
import pickle
from pathlib import Path
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class VietnameseVectorDB:
    def __init__(self, db_name: str, storage_dir: str, model_name: str, truncation_dim: int):
        self.db_path = Path(storage_dir) / f"{db_name}.pkl"
        self.model_name = model_name
        self.truncation_dim = truncation_dim
        self.model = None
        
        # Core Index Components
        self.corpus = []
        self.doc_embeddings = None
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def load_model(self):
        """Loads the SentenceTransformer model if not already in memory."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, truncate_dim=self.truncation_dim)

    def save(self):
        """Persists the current index components to a pickle file."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {
            "corpus": self.corpus,
            "doc_embeddings": self.doc_embeddings,
            "bm25": self.bm25,
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "tfidf_matrix": self.tfidf_matrix
        }
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        """Loads the index components from storage."""
        if not self.db_path.exists():
            return False
        with open(self.db_path, "rb") as f:
            data = pickle.load(f)
        for key, value in data.items():
            setattr(self, key, value)
        self.load_model()
        return True

    def compute_all_scores(self, query: str, tokenizer_func):
        """Calculates normalized BM25, TF-IDF, and Embedding scores for the query."""
        # 1. BM25 Scores
        query_tokens = tokenizer_func(query)
        bm25_raw = self.bm25.get_scores(query_tokens)
        
        # 2. TF-IDF Scores
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_raw = (self.tfidf_matrix * query_tfidf.T).toarray().flatten()

        # 3. Embedding Scores (Cosine Similarity)
        query_emb = self.model.encode(query, convert_to_tensor = True)
        emb_raw = F.cosine_similarity(query_emb.unsqueeze(0), self.doc_embeddings).cpu().numpy()

        # Min-Max Normalization Helper
        def normalize(scores):
            s_min, s_max = scores.min(), scores.max()
            return (scores - s_min) / (s_max - s_min) if s_max > s_min else scores

        return normalize(bm25_raw), normalize(tfidf_raw), emb_raw