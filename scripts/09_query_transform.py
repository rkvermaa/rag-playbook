import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.chunking import load_pdf, chunk_text
from src.embedding import get_embeddings
from src.vector_store import SimpleVectorStore
from src.retriever import Retriever, step_back_prompting, decompose_query
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

# Run it
if __name__ == "__main__":
    # Build RAG system
    retriever = build_rag_pipeline("data/Attention Is All You Need.pdf")

    print("\n" + "="*60)
    print("Strategy 1: Step-back prompting (for overly specific queries)")
    print("="*60)

    original_query = "What is the formula for scaled dot-product attention?"
    print(f"\n❓ Original query: {original_query}")

    broader_query = step_back_prompting(original_query)
    print(f"🔄 Broader query: {broader_query}")

    results = retriever.retrieve(broader_query, top_k=5)
    answer = generate_response(original_query, results[:3])
    print(f"\n💡 Answer: {answer}")

    print("\n" + "="*60)
    print("Strategy 2: Decomposition (for multi-part questions)")
    print("="*60)

    complex_query = "How do transformers work and why are they better than RNNs?"
    print(f"\n❓ Complex query: {complex_query}")

    sub_queries = decompose_query(complex_query)
    print(f"\n🔄 Sub-queries:")
    for i, sub_q in enumerate(sub_queries, 1):
        print(f"  {i}. {sub_q}")

    all_results = []
    for sub_q in sub_queries:
        results = retriever.retrieve(sub_q, top_k=2)
        all_results.extend(results)

    # Combine and generate answer
    answer = generate_response(complex_query, all_results)
    print(f"\n💡 Answer: {answer}")
