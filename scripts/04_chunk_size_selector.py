import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.chunking import load_pdf, evaluate_chunk_sizes

text = load_pdf("data/Attention Is All You Need.pdf")

test_queries = [
    "What is the attention mechanism?",
    "How does multi-head attention work?",
    "What are the advantages of transformers?"
]

results = evaluate_chunk_sizes(text, test_queries)

for size, metrics in results.items():
    print(f"Size {size}: Score {metrics['avg_score']:.3f}, "
          f"{metrics['num_chunks']} chunks")
