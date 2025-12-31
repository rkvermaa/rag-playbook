from src.embedding import get_embedding
from openai import OpenAI
from config import GROQ_API_KEY, GROQ_BASE_URL
from sentence_transformers import CrossEncoder

# Lazy-loaded reranker to avoid startup cost
_reranker = None
query_client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

def generate_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical answer to improve retrieval."""
    prompt = f"""Generate a detailed, hypothetical answer to this question.
Make it realistic and use technical terminology.

Question: {query}

Hypothetical answer:"""

    response = query_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()

def evaluate_retrieval_quality(query: str, chunks: list) -> str:
    """Evaluate if retrieved chunks are good enough to answer the query."""
    context_text = "\n\n".join([c['text'] for c in chunks])

    prompt = f"""Evaluate if these retrieved documents can answer the question well.

Question: {query}

Documents:
{context_text[:400]}...

Respond with ONLY:
- 'HIGH' if documents clearly answer the question
- 'MEDIUM' if documents partially answer the question
- 'LOW' if documents don't answer the question"""

    response = query_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    return response.upper()

def web_search_fallback(query: str) -> list:
    """Fallback to web search when document retrieval fails."""
    # Placeholder - integrate with a real web search API
    # (DuckDuckGo, Brave Search, Google Custom Search, etc.)
    print(f"⚠️  Document retrieval insufficient. Falling back to web search for: {query}")

    return [{
        'text': f"[Web search results for: {query}]",
        'score': 0.5
    }]
    
def get_reranker():
    """Lazy load the reranker model."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve most relevant chunks for a query."""
        query_embedding = get_embedding(query)
        results = self.vector_store.search(query_embedding, top_k)
        return results
    
    def retrieve_with_context(self, query: str, top_k: int = 3) -> list:
        """Retrieve chunks with their neighbors for context."""
        results = self.retrieve(query, top_k)
        enriched_results = []

        for result in results:
            chunk_id = result['chunk_id']
            context_parts = []

            prev = self.vector_store.get_chunk_by_id(chunk_id - 1)
            if prev:
                context_parts.append(f"[Previous]: {prev}")

            context_parts.append(f"[Main]: {result['text']}")

            next_chunk = self.vector_store.get_chunk_by_id(chunk_id + 1)
            if next_chunk:
                context_parts.append(f"[Next]: {next_chunk}")

            enriched_results.append({
                'text': '\n\n'.join(context_parts),
                'score': result['score']
            })

        return enriched_results
    
    def retrieve_with_rerank(
        self,
        query: str,
        top_k: int = 5,
        candidate_multiplier: int = 3
    ) -> list:
        """Two-stage retrieval: embeddings → rerank."""
        # Stage 1: retrieve more candidates
        candidates = self.retrieve(query, top_k * candidate_multiplier)

        # Stage 2: rerank candidates
        reranker = get_reranker()
        pairs = [[query, c['text']] for c in candidates]
        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {'text': c['text'], 'score': float(score)}
            for c, score in ranked[:top_k]
        ]
        
    def retrieve_segments(self, query: str, similarity_threshold: float = 0.7) -> list:
        """Extract continuous segments of relevant chunks."""
        # Retrieve a larger candidate set
        all_results = self.retrieve(query, top_k=20)

        # Filter by similarity threshold
        relevant = [r for r in all_results if r['score'] >= similarity_threshold]
        if not relevant:
            return []

        # Group consecutive chunks
        segments = []
        current_segment = [relevant[0]]

        for i in range(1, len(relevant)):
            prev_id = current_segment[-1].get('chunk_id', -1)
            curr_id = relevant[i].get('chunk_id', -2)

            if curr_id == prev_id + 1:
                current_segment.append(relevant[i])
            else:
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                current_segment = [relevant[i]]

        if len(current_segment) >= 2:
            segments.append(current_segment)

        # Merge segments into continuous text
        segment_texts = []
        for segment in segments:
            text = ' '.join([c['text'] for c in segment])
            avg_score = sum(c['score'] for c in segment) / len(segment)
            segment_texts.append({'text': text, 'score': avg_score})

        return sorted(segment_texts, key=lambda x: x['score'], reverse=True)
    
    def retrieve_compressed(self, query: str, top_k: int = 5) -> list:
        """Retrieve and compress to only relevant parts."""
        results = self.retrieve(query, top_k)
        compressed = []

        for result in results:
            prompt = f"""Extract ONLY sentences relevant to the query. Keep verbatim.

    Query: {query}
    Context: {result['text']}

    Relevant sentences:"""

            response = query_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300
            )

            compressed_text = response.choices[0].message.content.strip()
            if compressed_text and compressed_text != "NONE":
                compressed.append({
                    'text': compressed_text,
                    'score': result['score']
                })

        return compressed  
    
    def retrieve_fusion(self, query: str, top_k: int = 5) -> list:
        """Hybrid retrieval using FusionVectorStore (dense + sparse)."""
        query_embedding = get_embedding(query)
        return self.vector_store.search(
            query_embedding,
            top_k,
            query_text=query
        )  
        
    def retrieve_with_hyde(self, query: str, top_k: int = 5) -> list:
        """Retrieve using HyDE: generate hypothetical answer first."""
        # Generate hypothetical answer
        hypothetical_answer = generate_hypothetical_answer(query)

        # Embed the hypothetical answer instead of the query
        from src.embedding import get_embedding
        hyde_embedding = get_embedding(hypothetical_answer)

        # Retrieve using the hypothetical answer's embedding
        results = self.vector_store.search(hyde_embedding, top_k)
        return results  
    
    def retrieve_with_crag(self, query: str, top_k: int = 5) -> dict:
        """CRAG: Evaluate retrieval quality and fall back to web if needed."""
        # Retrieve from documents
        results = self.retrieve(query, top_k)

        # Evaluate quality
        quality = evaluate_retrieval_quality(query, results)

        if quality == 'HIGH':
            return {
                'source': 'documents',
                'quality': 'high',
                'results': results
            }

        elif quality == 'MEDIUM':
            # Filter and compress chunks
            compressed = self.retrieve_compressed(query, top_k)
            return {
                'source': 'documents',
                'quality': 'medium',
                'results': compressed
            }

        else:  # LOW quality
            # Fall back to web search
            web_results = web_search_fallback(query)
            return {
                'source': 'web',
                'quality': 'low',
                'results': web_results
            }

def step_back_prompting(query: str) -> str:
    """Generate broader, more conceptual version of query."""
    prompt = f"""Given this specific question, generate a more general question about the underlying concept.

Specific: {query}

General question:"""

    response = query_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def decompose_query(query: str) -> list:
    """Break complex query into simpler sub-queries."""
    prompt = f"""Break this complex question into 2–4 simpler sub-questions.

Complex: {query}

Sub-questions (numbered):"""

    response = query_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    sub_queries = []
    for line in response.choices[0].message.content.split('\n'):
        if line.strip() and line[0].isdigit():
            sub_q = line.split('.', 1)[-1].strip()
            if sub_q:
                sub_queries.append(sub_q)
    return sub_queries

def classify_query_complexity(query: str) -> str:
    """Classify query as simple, medium, or complex."""
    prompt = f"""Classify this query's complexity for a RAG system.

Question: {query}

Respond with ONLY:
- 'SIMPLE' if it's factual and the LLM likely knows the answer
- 'MEDIUM' if it needs document retrieval but is straightforward
- 'COMPLEX' if it requires multiple retrievals, reasoning, or decomposition"""

    response = query_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    return response.upper()


def adaptive_rag(query: str, retriever) -> dict:
    """Route query to the appropriate RAG strategy based on complexity."""
    from src.generator import generate_response

    complexity = classify_query_complexity(query)

    if 'SIMPLE' in complexity:
        # Direct LLM, no retrieval
        from openai import OpenAI
        from config import GROQ_API_KEY, GROQ_BASE_URL

        client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
        answer = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": query}],
            temperature=0
        ).choices[0].message.content

        return {
            'strategy': 'direct',
            'answer': answer
        }

    elif 'MEDIUM' in complexity:
        # Standard RAG
        results = retriever.retrieve(query, top_k=5)
        answer = generate_response(query, results)

        return {
            'strategy': 'standard_rag',
            'answer': answer
        }

    else:  # COMPLEX
        # Advanced: decompose + retrieve + rerank
        sub_queries = decompose_query(query)

        all_results = []
        for sub_q in sub_queries:
            # Retrieve for each sub-query
            results = retriever.retrieve(sub_q, top_k=3)
            all_results.extend(results)

        # Remove duplicates
        seen = set()
        unique_results = []
        for r in all_results:
            if r['text'] not in seen:
                seen.add(r['text'])
                unique_results.append(r)

        answer = generate_response(query, unique_results[:5])

        return {
            'strategy': 'advanced_rag',
            'answer': answer
        }
