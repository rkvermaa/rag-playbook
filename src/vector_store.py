import numpy as np

class SimpleVectorStore:
    def __init__(self):
        self.texts = []
        self.embeddings = []
    
    def add(self, text: str, embedding: list):
        """Add a text and its embedding to the store."""
        self.texts.append(text)
        self.embeddings.append(embedding)
    
    def add_batch(self, texts: list, embeddings: list):
        """Add multiple texts and embeddings."""
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
    
    def cosine_similarity(self, a: list, b: list) -> float:
        """Calculate cosine similarity between two vectors."""
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query_embedding: list, top_k: int = 5) -> list:
        """Find top-k most similar texts."""
        similarities = [
            self.cosine_similarity(query_embedding, emb) 
            for emb in self.embeddings
        ]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            {"text": self.texts[i], "score": similarities[i]} 
            for i in top_indices
        ]
        return results