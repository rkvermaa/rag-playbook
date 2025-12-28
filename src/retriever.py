from src.embedding import get_embedding

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve most relevant chunks for a query."""
        query_embedding = get_embedding(query)
        results = self.vector_store.search(query_embedding, top_k)
        return results