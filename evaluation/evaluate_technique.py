import json
from src.chunking import load_pdf, recursive_chunking
from src.embedding import get_embeddings
from src.vector_store import SimpleVectorStore
from src.retriever import Retriever
from src.generator import generate_response
from evaluation.metrics import (
    recall_at_k, precision_at_k, mrr, hit_rate_at_k,
    evaluate_faithfulness, evaluate_answer_relevancy, evaluate_context_relevancy
)

# Load evaluation dataset
with open('evaluation/val.json', 'r') as f:
    val_data = json.load(f)

# Build RAG system (use the technique you want to test)
text = load_pdf("data/Attention Is All You Need.pdf")
chunks = recursive_chunking(text, max_chunk_size=800)  # ← Change to test different chunking
embeddings = get_embeddings(chunks)

store = SimpleVectorStore()
store.add_batch(chunks, embeddings)
retriever = Retriever(store)

# Store metrics
results = {
    'retrieval': {'recall@5': [], 'precision@5': [], 'mrr': [], 'hit_rate@5': []},
    'generation': {'faithfulness': [], 'answer_relevancy': [], 'context_relevancy': []}
}

# Run evaluation
for item in val_data:
    question = item['question']

    # Retrieve
    retrieved = retriever.retrieve(question, top_k=5)

    # Generate
    answer = generate_response(question, retrieved[:3])

    # Retrieval metrics (simplified)
    # In practice, you would label relevant chunk IDs per question
    results['retrieval']['hit_rate@5'].append(
        1.0 if retrieved else 0.0
    )

    # Generation metrics
    results['generation']['faithfulness'].append(
        evaluate_faithfulness(answer, retrieved[:3])
    )

    results['generation']['answer_relevancy'].append(
        evaluate_answer_relevancy(question, answer)
    )

    results['generation']['context_relevancy'].append(
        evaluate_context_relevancy(question, retrieved[:3])
    )

# Aggregate results
print("\n📊 Evaluation Results:")

print("\nRetrieval Metrics:")
for metric, values in results['retrieval'].items():
    if values:
        print(f"  {metric}: {sum(values) / len(values):.3f}")

print("\nGeneration Metrics:")
for metric, values in results['generation'].items():
    print(f"  {metric}: {sum(values) / len(values):.3f}")
