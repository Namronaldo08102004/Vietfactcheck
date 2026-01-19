import os
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class VietnameseVectorDB:
    def __init__(
        self,
        db_name: str,
        storage_dir: str,
        model_name: str,
        truncation_dim: int,
        device: str = None
    ):
        self.db_path = Path(storage_dir) / f"{db_name}.pkl"
        self.model_name = model_name
        self.truncation_dim = truncation_dim

        # Device handling
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Model
        self.model = None

        # Index components
        self.corpus = []
        self.doc_embeddings = None   # torch.Tensor [N, D] (CPU)
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    def load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(
                self.model_name,
                truncate_dim=self.truncation_dim,
                device=str(self.device)
            )

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------
    def save(self):
        os.makedirs(self.db_path.parent, exist_ok=True)
        data = {
            "corpus": self.corpus,
            "doc_embeddings": self.doc_embeddings.cpu()
            if self.doc_embeddings is not None else None,
            "bm25": self.bm25,
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "tfidf_matrix": self.tfidf_matrix
        }
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        if not self.db_path.exists():
            return False

        with open(self.db_path, "rb") as f:
            data = pickle.load(f)

        for k, v in data.items():
            setattr(self, k, v)

        # Ensure embeddings are torch tensor on CPU
        if self.doc_embeddings is not None:
            self.doc_embeddings = torch.as_tensor(self.doc_embeddings)

        self.load_model()
        return True

    # --------------------------------------------------
    # Scoring
    # --------------------------------------------------
    def compute_all_scores(self, query: str, tokenizer_func):
        """
        Returns:
            bm25_norm, tfidf_norm, emb_norm
        """

        # ---------- 1. BM25 ----------
        query_tokens = tokenizer_func(query)
        bm25_raw = self.bm25.get_scores(query_tokens)

        # ---------- 2. TF-IDF ----------
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_raw = (self.tfidf_matrix @ query_tfidf.T).toarray().ravel()

        # ---------- 3. Embedding ----------
        self.load_model()

        query_emb = self.model.encode(
            query,
            convert_to_tensor=True,
            device=str(self.device)
        )

        doc_emb = self.doc_embeddings.to(query_emb.device)

        emb_raw = F.cosine_similarity(
            query_emb.unsqueeze(0),
            doc_emb
        ).detach().cpu().numpy()

        # ---------- Normalization ----------
        def normalize(x):
            x_min, x_max = x.min(), x.max()
            return (x - x_min) / (x_max - x_min) if x_max > x_min else x

        return (
            normalize(bm25_raw),
            normalize(tfidf_raw),
            emb_raw
        )
