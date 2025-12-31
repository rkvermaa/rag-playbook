from openai import OpenAI
from config import GROQ_API_KEY, GROQ_BASE_URL, GENERATION_MODEL

client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

def generate_response(query: str, context: list) -> str:
    """Generate answer using retrieved context."""
    
    # Format context
    context_text = "\n\n".join([chunk["text"] for chunk in context])
    
    prompt = f"""Answer the question based on the provided context. 
If the context doesn't contain the answer, say "I don't have enough information to answer this."

Context:
{context_text}

Question: {query}

Answer:"""
    
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

def self_rag_generate(query: str, retriever, max_iterations: int = 2) -> dict:
    """Self-RAG: LLM decides when to retrieve and critiques itself."""
    from openai import OpenAI
    from config import GROQ_API_KEY, GROQ_BASE_URL

    client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

    # Step 1: Should we retrieve?
    decision_prompt = f"""Given this question, do you need to retrieve external documents to answer it, or can you answer directly?

Question: {query}

Respond with ONLY 'RETRIEVE' or 'DIRECT'."""

    decision = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": decision_prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    if "DIRECT" in decision.upper():
        # No retrieval needed
        answer = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": query}],
            temperature=0
        ).choices[0].message.content

        return {
            "answer": answer,
            "retrieved": False,
            "critique": "No retrieval needed"
        }

    # Step 2: Retrieve
    results = retriever.retrieve(query, top_k=5)

    # Step 3: Critique retrieval quality
    context_text = "\n\n".join([r['text'] for r in results])

    critique_prompt = f"""Are these retrieved documents relevant to answering the question?

Question: {query}

Retrieved documents:
{context_text[:500]}...

Respond with 'RELEVANT' or 'NOT_RELEVANT'."""

    critique = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": critique_prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    if "NOT_RELEVANT" in critique.upper():
        # Retrieval failed
        return {
            "answer": "I don't have enough relevant information to answer this question.",
            "retrieved": True,
            "critique": "Retrieved documents not relevant"
        }

    # Step 4: Generate answer
    gen_prompt = f"""Answer the question based on the provided context.

Context:
{context_text}

Question: {query}

Answer:"""

    answer = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": gen_prompt}],
        temperature=0
    ).choices[0].message.content

    # Step 5: Validate answer is supported
    validation_prompt = f"""Is this answer supported by the provided context?

Context:
{context_text[:300]}...

Answer: {answer}

Respond with 'SUPPORTED' or 'NOT_SUPPORTED'."""

    validation = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": validation_prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    return {
        "answer": answer,
        "retrieved": True,
        "critique": validation
    }
