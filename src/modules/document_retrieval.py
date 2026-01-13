import json
from typing import List, Tuple
from tqdm import tqdm
from underthesea import word_tokenize
from src.components.document import Document
from src.components.vectorDB import VietnameseVectorDB
from rank_bm25 import BM25L
from sklearn.feature_extraction.text import TfidfVectorizer

def vietnamese_tokenizer(text: str) -> List[str]:
    return word_tokenize(text.lower(), format = "text").split()

class DocumentRetrievalModule:
    def __init__(self, db: VietnameseVectorDB):
        self.db = db
        self.tokenizer = vietnamese_tokenizer

    def build_system(self, data_paths: List[str]):
        """Builds a unified index from train, dev, and test files with deduplication."""
        all_raw_data = []
        for path in data_paths:
            with open(path, "r", encoding="utf-8") as f:
                all_raw_data.extend(json.load(f))
        
        # Deduplication by URL
        unique_data = {item['Url']: item for item in all_raw_data}.values()
        
        corpus, tokenized_corpus, texts_for_tfidf = [], [], []
        for record in tqdm(unique_data, desc="Indexing Combined Corpus"):
            chunks = record.get("splited_sentences", [record.get("Context", "")])
            for idx, text in enumerate(chunks):
                if not text.strip(): continue
                corpus.append(Document(text, {"context_url": record['Url'], "chunk_id": idx}))
                tokenized_corpus.append(self.tokenizer(text))
                texts_for_tfidf.append(text)

        self.db.corpus = corpus
        self.db.bm25 = BM25L(tokenized_corpus)
        self.db.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, norm='l2')
        self.db.tfidf_matrix = self.db.tfidf_vectorizer.fit_transform(texts_for_tfidf)
        self.db.load_model()
        self.db.doc_embeddings = self.db.model.encode(
            [d.page_content for d in corpus], show_progress_bar=True, convert_to_tensor=True
        )
        self.db.save()

    def get_top_k_url(self, query: str, top_k: int = 1, weights: Tuple[float, float, float] = (0.3, 0.3, 0.4)) -> List[str]:
        s_bm25, s_tfidf, s_emb = self.db.compute_all_scores(query, self.tokenizer)
        w_bm25, w_tfidf, w_emb = weights
        
        url_max_scores = {}
        for i, doc in enumerate(self.db.corpus):
            url = doc.metadata['context_url']
            total_score = (s_bm25[i] * w_bm25) + (s_tfidf[i] * w_tfidf) + (s_emb[i] * w_emb)
            if url not in url_max_scores or total_score > url_max_scores[url]:
                url_max_scores[url] = total_score
        
        sorted_urls = sorted(url_max_scores.items(), key=lambda x: x[1], reverse=True)
        return [url for url, score in sorted_urls[:top_k]]