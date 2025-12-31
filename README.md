# RAG Playbook

A complete collection of 19 RAG (Retrieval-Augmented Generation) techniques with hands-on implementations and evaluation framework.

## What's Inside

- **19 RAG techniques** - From simple baseline to advanced architectures
- **Evaluation framework** - 7 metrics to measure what actually works
- **Benchmark scripts** - Test all techniques on your own data

## Quick Start

```bash
# Clone the repo
git clone https://github.com/rkvermaa/rag-playbook.git
cd rag-playbook

# Install all dependencies
uv sync

# Run your first RAG pipeline
uv run python scripts/01_simple_rag.py
```

That's it! `uv sync` installs everything. Start testing and editing code for your use case.

## Usage

```bash
# Run any technique
uv run python scripts/01_simple_rag.py  # Simple RAG
uv run python scripts/10_reranker.py    # Reranker
uv run python scripts/18_graph_rag.py   # Graph RAG

# Create evaluation dataset
uv run python scripts/create_val_dataset.py

# Benchmark all 19 techniques
uv run python scripts/benchmark_all_techniques.py
```

## Project Structure

```
scripts/          # All 19 RAG techniques (01-19)
src/              # Core modules (chunking, retriever, generator)
evaluation/       # Metrics and benchmark tools
data/             # Your PDFs go here
```

## Requirements

- Python 3.12+
- uv package manager
- Groq API key (add to `config.py`)

## Techniques Covered

Simple RAG → Semantic Chunking → Recursive Chunking → Proposition Chunking → Context-Enriched RAG → Contextual Headers → Document Augmentation → Query Transform → Reranker → RSE → Contextual Compression → Fusion RAG → HyDE → Self-RAG → CRAG → Adaptive RAG → Graph RAG → Hierarchical RAG

## Contact

**Ravi Kumar Verma**
Email: rkverma87@gmail.com
GitHub: [@rkvermaa](https://github.com/rkvermaa)

---

*Don't trust benchmarks. Test on YOUR data. What works for one use case might fail for another.*
