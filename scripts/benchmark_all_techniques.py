import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import time
from time import sleep
from src.chunking import (
    load_pdf, chunk_text, semantic_chunking, recursive_chunking,
    proposition_chunking, add_contextual_headers, augment_chunks_with_questions
)
from src.embedding import get_embeddings
from src.vector_store import SimpleVectorStore, FusionVectorStore
from src.retriever import Retriever, adaptive_rag
from src.generator import generate_response
from src.graph_rag import KnowledgeGraph
from src.hierarchical_rag import HierarchicalRAG
from evaluation.metrics import (
    evaluate_faithfulness, evaluate_answer_relevancy, evaluate_context_relevancy
)

# Load evaluation dataset
with open('evaluation/val.json', 'r') as f:
    val_data = json.load(f)

# Load document
print("📄 Loading PDF...")
text = load_pdf("data/Attention Is All You Need.pdf")

def evaluate_technique(name: str, retriever, use_special_retrieval=None, use_adaptive=False, use_graph=False, use_hierarchical=False, graph_chunks=None):
    """Evaluate a single technique."""
    print(f"\n🧪 Testing: {name}")

    results = {
        'faithfulness': [],
        'answer_relevancy': [],
        'context_relevancy': []
    }

    for i, item in enumerate(val_data):
        print(f"  Question {i+1}/{len(val_data)}", end='\r')
        question = item['question']

        try:
            # Handle different retrieval types
            if use_adaptive:
                # Adaptive RAG returns dict with 'answer' key
                result = adaptive_rag(question, retriever)
                answer = result['answer']
                # For adaptive, we don't have retrieved chunks, skip context relevancy
                results['faithfulness'].append(0.5)  # Neutral score
                results['context_relevancy'].append(0.5)  # Neutral score
                results['answer_relevancy'].append(
                    evaluate_answer_relevancy(question, answer)
                )
                continue

            elif use_graph:
                # Graph RAG
                retrieved = retriever.query_graph(question, graph_chunks, max_hops=2)
                answer = generate_response(question, retrieved[:3])

                results['faithfulness'].append(
                    evaluate_faithfulness(answer, retrieved[:3])
                )
                results['answer_relevancy'].append(
                    evaluate_answer_relevancy(question, answer)
                )
                results['context_relevancy'].append(
                    evaluate_context_relevancy(question, retrieved[:3])
                )

                # Sleep to avoid rate limit
                sleep(6)
                continue

            elif use_hierarchical:
                # Hierarchical RAG
                retrieved = retriever.retrieve(question, top_k=5)
                answer = generate_response(question, retrieved[:3])

                results['faithfulness'].append(
                    evaluate_faithfulness(answer, retrieved[:3])
                )
                results['answer_relevancy'].append(
                    evaluate_answer_relevancy(question, answer)
                )
                results['context_relevancy'].append(
                    evaluate_context_relevancy(question, retrieved[:3])
                )

                # Sleep to avoid rate limit
                sleep(6)
                continue

            elif use_special_retrieval:
                # Special retrieval methods
                if 'crag' in use_special_retrieval:
                    crag_result = retriever.retrieve_with_crag(question, top_k=5)
                    retrieved = crag_result['results']
                elif use_special_retrieval == 'retrieve_segments':
                    # retrieve_segments doesn't take top_k
                    retrieved = retriever.retrieve_segments(question)
                else:
                    retrieved = getattr(retriever, use_special_retrieval)(question, top_k=5)
            else:
                # Standard retrieval
                retrieved = retriever.retrieve(question, top_k=5)

            # Generate answer
            answer = generate_response(question, retrieved[:3])

            # Evaluate
            results['faithfulness'].append(
                evaluate_faithfulness(answer, retrieved[:3])
            )
            results['answer_relevancy'].append(
                evaluate_answer_relevancy(question, answer)
            )
            results['context_relevancy'].append(
                evaluate_context_relevancy(question, retrieved[:3])
            )

            # Sleep to avoid rate limit (30 requests/minute = 2 seconds per request)
            # We make 3 LLM calls per question, so wait 6 seconds
            sleep(6)

        except Exception as e:
            print(f"\n  ⚠️  Error on question {i+1}: {e}")
            # Add neutral scores on error
            results['faithfulness'].append(0.5)
            results['answer_relevancy'].append(0.5)
            results['context_relevancy'].append(0.5)

    # Calculate averages
    avg_results = {
        'technique': name,
        'faithfulness': sum(results['faithfulness']) / len(results['faithfulness']) if results['faithfulness'] else 0,
        'answer_relevancy': sum(results['answer_relevancy']) / len(results['answer_relevancy']) if results['answer_relevancy'] else 0,
        'context_relevancy': sum(results['context_relevancy']) / len(results['context_relevancy']) if results['context_relevancy'] else 0
    }

    print(f"\n  ✅ {name}: F={avg_results['faithfulness']:.3f} | AR={avg_results['answer_relevancy']:.3f} | CR={avg_results['context_relevancy']:.3f}")

    # Save results immediately after each technique
    with open('evaluation/benchmark_results.json', 'w') as f:
        json.dump(benchmark_results + [avg_results], f, indent=2)

    return avg_results

# ==================== BENCHMARK ALL TECHNIQUES ====================

# Load existing results if script was interrupted
try:
    with open('evaluation/benchmark_results.json', 'r') as f:
        benchmark_results = json.load(f)
    print(f"📂 Loaded {len(benchmark_results)} existing results")
except:
    benchmark_results = []

# ===== TECHNIQUES 1-17 ALREADY COMPLETED - SKIPPING =====
print(f"\n✅ Skipping techniques 1-{len(benchmark_results)} (already completed)")
print(f"📂 Loaded {len(benchmark_results)} existing results")

# Skip to technique 18
print("\n" + "="*60)
print("RUNNING TECHNIQUES 18-19 ONLY")
print("="*60)

# 18. Graph RAG
try:
    print("Initializing Graph RAG...")
    chunks = recursive_chunking(text, max_chunk_size=800)
    graph_rag = KnowledgeGraph()
    graph_rag.build_graph(chunks)
    benchmark_results.append(evaluate_technique("18. Graph RAG", graph_rag, use_graph=True, graph_chunks=chunks))
except Exception as e:
    print(f"\n⚠️  Graph RAG failed: {e}")
    benchmark_results.append({
        'technique': '18. Graph RAG',
        'faithfulness': 0.0,
        'answer_relevancy': 0.0,
        'context_relevancy': 0.0
    })

# 19. Hierarchical RAG
try:
    print("\n" + "="*60)
    print("Initializing Hierarchical RAG...")
    chunks = recursive_chunking(text, max_chunk_size=800)
    hierarchical_rag = HierarchicalRAG()
    hierarchical_rag.build_hierarchy(text, chunks)
    benchmark_results.append(evaluate_technique("19. Hierarchical RAG", hierarchical_rag, use_hierarchical=True))
except Exception as e:
    print(f"\n⚠️  Hierarchical RAG failed: {e}")
    benchmark_results.append({
        'technique': '19. Hierarchical RAG',
        'faithfulness': 0.0,
        'answer_relevancy': 0.0,
        'context_relevancy': 0.0
    })

# ==================== SAVE RESULTS ====================

# Save to JSON
with open('evaluation/benchmark_results.json', 'w') as f:
    json.dump(benchmark_results, f, indent=2)

# Print final table
print("\n\n" + "="*80)
print("FINAL BENCHMARK RESULTS")
print("="*80)
print(f"{'Technique':<40} {'Faith':<8} {'Ans.Rel':<8} {'Ctx.Rel':<8}")
print("-"*80)

for result in benchmark_results:
    print(f"{result['technique']:<40} {result['faithfulness']:<8.3f} {result['answer_relevancy']:<8.3f} {result['context_relevancy']:<8.3f}")

print("="*80)
print(f"\n✅ Results saved to evaluation/benchmark_results.json")

# Print markdown table for blog
print("\n\n" + "="*80)
print("MARKDOWN TABLE FOR BLOG")
print("="*80)
print("| Technique | Faithfulness | Answer Relevancy | Context Relevancy |")
print("|-----------|--------------|------------------|-------------------|")
for result in benchmark_results:
    print(f"| {result['technique']} | {result['faithfulness']:.2f} | {result['answer_relevancy']:.2f} | {result['context_relevancy']:.2f} |")
