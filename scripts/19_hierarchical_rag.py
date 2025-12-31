import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.chunking import load_pdf, recursive_chunking
from src.hierarchical_rag import HierarchicalRAG
from src.generator import generate_response

# Build hierarchical index
text = load_pdf("data/Attention Is All You Need.pdf")
chunks = recursive_chunking(text, max_chunk_size=600)

print("🏗️  Building hierarchical index...")
h_rag = HierarchicalRAG()
h_rag.build_hierarchy(text, chunks)
print("✅ Hierarchy built (summaries + details)")

# Query
question = "What is this paper about?"  # High-level query
results = h_rag.retrieve(question, top_k=3)

print(f"\n📚 Retrieved {len(results)} chunks")

answer = generate_response(question, results)
print(f"\n💡 Answer: {answer}")
