import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.chunking import load_pdf, recursive_chunking
from openai import OpenAI
from config import GROQ_API_KEY, GROQ_BASE_URL
import json
import random

client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

def generate_question_answer_pair(chunk: str, difficulty: str) -> dict:
    """Generate a Q&A pair from a chunk."""
    prompt = f"""Based on this text, generate 1 {difficulty} question and its ideal answer.

Text: {chunk}

Return ONLY valid JSON in this exact format:
{{"question": "your question here", "answer": "your answer here", "difficulty": "{difficulty}"}}

Do not include any other text, just the JSON."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Stronger model for quality
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    try:
        content = response.choices[0].message.content.strip()
        # Try to extract JSON if wrapped in markdown code blocks
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:].strip()

        result = json.loads(content)

        # Validate required fields
        if 'question' in result and 'answer' in result:
            result['difficulty'] = difficulty
            return result
        else:
            return None
    except Exception as e:
        print(f"\n  ⚠️  JSON parse error: {e}")
        return None

# Load and chunk document
print("📄 Loading and chunking PDF...")
text = load_pdf("data/Attention Is All You Need.pdf")
chunks = recursive_chunking(text, max_chunk_size=800)

# Sample chunks (don't use all - too many questions)
sampled_chunks = random.sample(chunks, min(50, len(chunks)))  # Increased to 50 to get ~25-30 good ones

val_dataset = []
difficulties = ['easy', 'medium', 'hard']
target_count = 25  # Target number of questions

for i, chunk in enumerate(sampled_chunks):
    if len(val_dataset) >= target_count:
        break

    difficulty = difficulties[i % len(difficulties)]

    print(f"Generating Q&A {len(val_dataset)+1}/{target_count} ({difficulty})...", end='\r')

    # Retry up to 2 times if JSON parsing fails
    for attempt in range(2):
        qa_pair = generate_question_answer_pair(chunk, difficulty)
        if qa_pair:
            qa_pair['chunk_source'] = chunk[:100] + '...'
            val_dataset.append(qa_pair)
            break

# Save dataset
with open('evaluation/val.json', 'w') as f:
    json.dump(val_dataset, f, indent=2)

print(f"\n✅ Created evaluation dataset with {len(val_dataset)} questions")
print(f"   Easy: {sum(1 for q in val_dataset if q['difficulty'] == 'easy')}")
print(f"   Medium: {sum(1 for q in val_dataset if q['difficulty'] == 'medium')}")
print(f"   Hard: {sum(1 for q in val_dataset if q['difficulty'] == 'hard')}")
