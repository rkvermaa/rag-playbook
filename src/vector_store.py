import numpy as np


class SimpleVectorStore:
    def __init__(self):
        self.texts = []
        self.embeddings = []
        self.chunk_ids = []  # NEW    
        
    def add(self, text: str, embedding: list):
        """Add a text and its embedding to the store."""
        self.texts.append(text)
        self.embeddings.append(embedding)
    
    def add_batch(self, texts: list, embeddings: list):
        start_id = len(self.texts)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        self.chunk_ids.extend(range(start_id, start_id + len(texts)))  # NEW

    def get_chunk_by_id(self, chunk_id: int):
        """Get chunk by ID."""
        if 0 <= chunk_id < len(self.texts):
            return self.texts[chunk_id]
        return None
    
    def cosine_similarity(self, a: list, b: list) -> float:
        """Calculate cosine similarity between two vectors."""
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query_embedding: list, top_k: int = 5) -> list:
        similarities = [
            self.cosine_similarity(query_embedding, emb)
            for emb in self.embeddings
        ]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [
            {
                "text": self.texts[i],
                "score": similarities[i],
                "chunk_id": self.chunk_ids[i]  # NEW
            }
            for i in top_indices
        ]
        return results
    

class FusionVectorStore:
    """Hybrid search: dense embeddings + sparse BM25."""

    def __init__(self):
        self.texts = []
        self.embeddings = []
        self.chunk_ids = []
        self.bm25 = None

    def add_batch(self, texts: list, embeddings: list):
        from rank_bm25 import BM25Okapi  # Import only when needed

        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        start_id = len(self.chunk_ids)
        self.chunk_ids.extend(range(start_id, start_id + len(texts)))

        tokenized = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query_embedding: list, top_k: int = 5, query_text: str = "") -> list:
        """Hybrid search with Reciprocal Rank Fusion."""
        dense_scores = [
            np.dot(query_embedding, np.array(emb)) /
            (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            for emb in self.embeddings
        ]
        dense_ranked = np.argsort(dense_scores)[::-1]

        query_tokens = query_text.lower().split() if query_text else []
        sparse_scores = (
            self.bm25.get_scores(query_tokens)
            if query_tokens else [0] * len(self.texts)
        )
        sparse_ranked = np.argsort(sparse_scores)[::-1]

        rrf_scores = {}
        k = 60

        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

        for rank, idx in enumerate(sparse_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

        fused_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                'text': self.texts[idx],
                'score': score,
                'chunk_id': self.chunk_ids[idx]
            }
            for idx, score in fused_ranked[:top_k]
        ]
