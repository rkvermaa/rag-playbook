import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.chunking import load_pdf, recursive_chunking
from src.graph_rag import KnowledgeGraph
from src.generator import generate_response

# Build knowledge graph
text = load_pdf("data/Attention Is All You Need.pdf")
chunks = recursive_chunking(text)

print("🔨 Building knowledge graph...")
kg = KnowledgeGraph()
kg.build_graph(chunks)
print(f"✅ Graph built: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")

# Query using graph
question = "How does attention mechanism relate to transformers and translation tasks?"
results = kg.query_graph(question, chunks, max_hops=2)

print(f"\n📚 Found {len(results)} relevant chunks via graph traversal")

answer = generate_response(question, results[:3])
print(f"\n💡 Answer: {answer}")