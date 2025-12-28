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