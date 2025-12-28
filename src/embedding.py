from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

# Load model once
model = SentenceTransformer(EMBEDDING_MODEL)

def get_embeddings(texts: list) -> list:
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()

def get_embedding(text: str) -> list:
    """Generate embedding for a single text."""
    return model.encode(text).tolist()