import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.chunking import load_pdf, recursive_chunking
from src.embedding import get_embeddings
from src.vector_store import FusionVectorStore
from src.retriever import Retriever
from src.generator import generate_response

def build_rag_pipeline(pdf_path: str):
    """Example RAG pipeline combining multiple techniques."""

    text = load_pdf(pdf_path)
    chunks = recursive_chunking(text, max_chunk_size=800)

    embeddings = get_embeddings(chunks)

    store = FusionVectorStore()
    store.add_batch(chunks, embeddings)

    return Retriever(store)

def ask(retriever, question: str):
    results = retriever.retrieve_with_rerank(question, top_k=3)
    return generate_response(question, results)

if __name__ == "__main__":
    retriever = build_rag_pipeline("data/Attention Is All You Need.pdf")
    question = "How is attention calculated in transformers?"
    answer = ask(retriever, question)
    print(answer)