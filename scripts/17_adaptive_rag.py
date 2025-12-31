
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.chunking import load_pdf, chunk_text
from src.embedding import get_embeddings
from src.vector_store import SimpleVectorStore
from src.retriever import Retriever, adaptive_rag
from src.generator import generate_response

def build_rag_pipeline(pdf_path: str):
    """Build the complete RAG pipeline."""
    
    # Step 1: Load and chunk
    print("📄 Loading PDF...")
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Step 2: Generate embeddings
    print("🔢 Generating embeddings...")
    embeddings = get_embeddings(chunks)
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    # Step 3: Store in vector store
    print("🗄️ Building vector store...")
    store = SimpleVectorStore()
    store.add_batch(chunks, embeddings)
    print("✅ Vector store ready")
    
    # Step 4: Create retriever
    retriever = Retriever(store)
    
    return retriever

def ask(retriever, question: str):
    """Ask a question and get an answer."""

    # Use adaptive RAG to get answer
    result = adaptive_rag(question, retriever)

    print(f"\n🎯 Strategy used: {result['strategy']}")

    return result['answer']

# Run it
if __name__ == "__main__":
    retriever = build_rag_pipeline("data/Attention Is All You Need.pdf")
    
    question = "How is Scaled Dot-Product Attention calculated?"
    print(f"\n❓ Question: {question}")
    
    answer = ask(retriever, question)
    print(f"\n💡 Answer: {answer}")