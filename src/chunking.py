
import fitz
from config import CHUNK_SIZE, CHUNK_OVERLAP
import numpy as np
from src.embedding import model
from openai import OpenAI
from config import GROQ_API_KEY, GROQ_BASE_URL

client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

def load_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    
    # Remove empty chunks
    chunks = [c for c in chunks if c]
    return chunks

def semantic_chunking(text: str, threshold: float = 0.6) -> list:
    """Split text by semantic similarity between sentences."""
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    if not sentences:
        return []

    embeddings = model.encode(sentences, show_progress_bar=False)
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )

        if similarity >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def recursive_chunking(text: str, max_chunk_size: int = 1000,
                      separators: list = None) -> list:
    """Recursively split text using hierarchical separators."""
    if separators is None:
        separators = ['\n\n', '\n', '. ', ' ']

    chunks = []

    def split_text(text, sep_index):
        if sep_index >= len(separators) or len(text) <= max_chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return

        separator = separators[sep_index]
        splits = text.split(separator)
        current_chunk = []
        current_size = 0

        for split in splits:
            if current_size + len(split) <= max_chunk_size:
                current_chunk.append(split)
                current_size += len(split) + len(separator)
            else:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    if len(chunk_text) > max_chunk_size:
                        split_text(chunk_text, sep_index + 1)
                    else:
                        chunks.append(chunk_text.strip())
                current_chunk = [split]
                current_size = len(split)

        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) > max_chunk_size:
                split_text(chunk_text, sep_index + 1)
            else:
                chunks.append(chunk_text.strip())

    split_text(text, 0)
    return chunks

def evaluate_chunk_sizes(text: str, test_queries: list,
                        sizes: list = [200, 400, 600, 800, 1000]) -> dict:
    """Test different chunk sizes and return quality metrics."""
    from src.embedding import get_embeddings
    from src.vector_store import SimpleVectorStore
    from src.retriever import Retriever

    results = {}

    for size in sizes:
        chunks = chunk_text(text, chunk_size=size)
        embeddings = get_embeddings(chunks)

        store = SimpleVectorStore()
        store.add_batch(chunks, embeddings)
        retriever = Retriever(store)

        avg_score = 0
        for query in test_queries:
            retrieved = retriever.retrieve(query, top_k=3)
            avg_score += sum(r['score'] for r in retrieved) / len(retrieved)

        avg_score /= len(test_queries)

        results[size] = {
            'avg_score': avg_score,
            'num_chunks': len(chunks)
        }

    return results


def proposition_chunking(text: str, max_chars_per_chunk: int = 1500) -> list:
    """Break text into atomic factual propositions using LLM."""
    import time

    # First, split text into manageable chunks (not paragraphs, but fixed-size chunks)
    # This handles PDFs that don't have proper paragraph breaks
    chunks = []
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_chars_per_chunk and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    print(f"🔄 Processing {len(chunks)} chunks for proposition extraction...")
    all_propositions = []

    for i, chunk in enumerate(chunks):
        try:
            prompt = f"""Break this text into atomic factual propositions.
Each proposition should be a single, self-contained statement.
Return ONLY the propositions, one per line.

Text: {chunk}"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            propositions_text = response.choices[0].message.content

            for line in propositions_text.split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    prop = line.split('.', 1)[-1].strip()
                    if prop:
                        all_propositions.append(prop)
                elif line and not line[0].isdigit() and len(line) > 10:
                    # Sometimes LLM doesn't number them
                    all_propositions.append(line)

            print(f"  ✓ Chunk {i+1}/{len(chunks)} → {len(all_propositions)} propositions so far")

            # Rate limiting: wait 1 second between requests to avoid rate limits
            if i < len(chunks) - 1:
                time.sleep(1)

        except Exception as e:
            print(f"⚠️  Error processing chunk {i+1}: {str(e)[:100]}")
            continue

    return all_propositions

def augment_chunks_with_questions(chunks: list, num_questions: int = 3) -> list:
    """Generate questions for each chunk to improve retrieval."""
    augmented = []

    for chunk in chunks:
        prompt = f"""Generate {num_questions} questions that this text answers.
Make them specific and diverse. Return ONLY the questions, numbered.

Text: {chunk[:500]}"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        questions = []
        for line in response.choices[0].message.content.split('\n'):
            if line.strip() and line[0].isdigit():
                q = line.split('.', 1)[-1].strip()
                if q:
                    questions.append(f"Q: {q}")

        augmented_chunk = '\n'.join(questions[:num_questions]) + f"\n\nContent: {chunk}"
        augmented.append(augmented_chunk)

    return augmented

def add_contextual_headers(chunks: list, document_text: str) -> list:
    """Prepend hierarchical context to chunks based on markdown headers."""
    import re

    lines = document_text.split('\n')
    headers_map = {}
    current_h1, current_h2, current_h3 = None, None, None
    char_count = 0

    for line in lines:
        if line.startswith('# '):
            current_h1 = line[2:].strip()
            current_h2, current_h3 = None, None
        elif line.startswith('## '):
            current_h2 = line[3:].strip()
            current_h3 = None
        elif line.startswith('### '):
            current_h3 = line[4:].strip()

        headers_map[char_count] = (current_h1, current_h2, current_h3)
        char_count += len(line) + 1

    enriched_chunks = []
    for chunk in chunks:
        chunk_pos = document_text.find(chunk[:50])
        if chunk_pos == -1:
            enriched_chunks.append(chunk)
            continue

        h1, h2, h3 = None, None, None
        for pos in sorted(headers_map.keys(), reverse=True):
            if pos <= chunk_pos:
                h1, h2, h3 = headers_map[pos]
                break

        header_parts = []
        if h1:
            header_parts.append(h1)
        if h2:
            header_parts.append(h2)
        if h3:
            header_parts.append(h3)

        if header_parts:
            header = " > ".join(header_parts)
            enriched_chunk = f"[{header}]\n\n{chunk}"
        else:
            enriched_chunk = chunk

        enriched_chunks.append(enriched_chunk)

    return enriched_chunks