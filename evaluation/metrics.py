import numpy as np
from openai import OpenAI
from config import GROQ_API_KEY, GROQ_BASE_URL

# Initialize LLM client for generation metrics
client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

# ========== RETRIEVAL METRICS ==========

def recall_at_k(retrieved_chunks: list, relevant_chunks: list, k: int) -> float:
    """Calculate Recall@K: what % of relevant chunks did we retrieve?"""
    retrieved_ids = set([c['chunk_id'] for c in retrieved_chunks[:k]])
    relevant_ids = set([c['chunk_id'] for c in relevant_chunks])

    if not relevant_ids:
        return 0.0

    overlap = retrieved_ids.intersection(relevant_ids)
    return len(overlap) / len(relevant_ids)

def precision_at_k(retrieved_chunks: list, relevant_chunks: list, k: int) -> float:
    """Calculate Precision@K: what % of retrieved chunks are relevant?"""
    retrieved_ids = set([c['chunk_id'] for c in retrieved_chunks[:k]])
    relevant_ids = set([c['chunk_id'] for c in relevant_chunks])

    if not retrieved_ids:
        return 0.0

    overlap = retrieved_ids.intersection(relevant_ids)
    return len(overlap) / len(retrieved_ids)

def mrr(retrieved_chunks: list, relevant_chunks: list) -> float:
    """Calculate Mean Reciprocal Rank: how high is the first relevant result?"""
    relevant_ids = set([c['chunk_id'] for c in relevant_chunks])

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk['chunk_id'] in relevant_ids:
            return 1.0 / rank

    return 0.0

def hit_rate_at_k(retrieved_chunks: list, relevant_chunks: list, k: int) -> float:
    """Calculate Hit Rate@K: did we get at least one relevant chunk?"""
    retrieved_ids = set([c['chunk_id'] for c in retrieved_chunks[:k]])
    relevant_ids = set([c['chunk_id'] for c in relevant_chunks])

    return 1.0 if retrieved_ids.intersection(relevant_ids) else 0.0

# ========== GENERATION METRICS (LLM-as-Judge) ==========

def evaluate_faithfulness(answer: str, context: list) -> float:
    """Use LLM to judge if the answer is faithful to the context (no hallucinations)."""
    context_text = "\n\n".join([c['text'] for c in context])

    prompt = f"""Evaluate if the answer is faithful to the context (no hallucinations).

Context:
{context_text}

Answer: {answer}

Is the answer fully supported by the context?
Respond with a score from 0.0 (completely unfaithful) to 1.0 (completely faithful).
Return ONLY the number."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Faster model with higher limits
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        score = float(response.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.0

def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """Use LLM to judge if the answer addresses the question."""
    prompt = f"""Does this answer address the question?

Question: {question}

Answer: {answer}

Score from 0.0 (completely irrelevant) to 1.0 (perfectly addresses the question).
Return ONLY the number."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        score = float(response.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.0

def evaluate_context_relevancy(question: str, context: list) -> float:
    """Use LLM to judge if the retrieved context is relevant to the question."""
    context_text = "\n\n".join([c['text'][:200] + '...' for c in context])

    prompt = f"""Is this retrieved context relevant for answering the question?

Question: {question}

Context:
{context_text}

Score from 0.0 (completely irrelevant) to 1.0 (highly relevant).
Return ONLY the number."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        score = float(response.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.0