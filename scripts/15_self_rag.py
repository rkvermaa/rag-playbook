
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.chunking import load_pdf, chunk_text
from src.embedding import get_embeddings
from src.vector_store import SimpleVectorStore
from src.retriever import Retriever
from src.generator import self_rag_generate , generate_response

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

    # Self-RAG: critique, retrieve, generate
    results = self_rag_generate(question, retriever)

    print(f"\n✅ Critique: {results['critique']}")
    print(f"📊 Retrieved: {results['retrieved']}")

    # Show retrieved chunks if any
    if results['retrieved']:
        print("\n📚 Retrieved chunks:")
        retrieved_chunks = results.get('chunks', [])
        for i, r in enumerate(retrieved_chunks):
            if isinstance(r, dict):
                print(f"  [{i+1}] (score: {r.get('score', 0):.3f}) {r.get('text', '')[:100]}...")

    print(f"\n💡 Answer: {results['answer']}")

    return results['answer']

# Run it
if __name__ == "__main__":
    retriever = build_rag_pipeline("data/Attention Is All You Need.pdf")

    question = "How is Scaled Dot-Product Attention calculated?"
    print(f"\n❓ Question: {question}")

    answer = ask(retriever, question)