from openai import OpenAI
from config import GROQ_API_KEY, GROQ_BASE_URL
from src.embedding import get_embeddings
from src.vector_store import SimpleVectorStore

client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

class HierarchicalRAG:
    """Multi-level indexing: summaries + detailed chunks."""

    def __init__(self):
        self.summary_store = SimpleVectorStore()
        self.detail_store = SimpleVectorStore()
        self.section_map = {}  # Map summary IDs to detailed chunk IDs

    def create_summary(self, text: str, level: str = "document") -> str:
        """Generate summary of text."""
        prompt = f"""Summarize this {level} in 2-3 sentences.

Text: {text[:1000]}...

Summary:"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150
        ).choices[0].message.content

        return response.strip()

    def build_hierarchy(self, document_text: str, chunks: list):
        """Build hierarchical index."""
        # Level 1: Document summary
        doc_summary = self.create_summary(document_text, "document")

        # Level 2: Section summaries (every 5 chunks = 1 section)
        section_summaries = []
        section_size = 5

        for i in range(0, len(chunks), section_size):
            section_chunks = chunks[i:i + section_size]
            section_text = ' '.join(section_chunks)
            section_summary = self.create_summary(section_text, "section")

            section_summaries.append(section_summary)
            self.section_map[len(section_summaries) - 1] = list(
                range(i, min(i + section_size, len(chunks)))
            )

        # Index summaries
        all_summaries = [doc_summary] + section_summaries
        summary_embeddings = get_embeddings(all_summaries)
        self.summary_store.add_batch(all_summaries, summary_embeddings)

        # Index detailed chunks
        detail_embeddings = get_embeddings(chunks)
        self.detail_store.add_batch(chunks, detail_embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Hierarchical retrieval: summaries first, then details."""
        from src.embedding import get_embedding

        # Step 1: Search summaries
        query_emb = get_embedding(query)
        summary_results = self.summary_store.search(query_emb, top_k=2)

        # Step 2: Identify relevant sections
        relevant_chunk_ids = set()

        for result in summary_results:
            summary_text = result['text']

            for section_id, chunk_ids in self.section_map.items():
                section_summary = self.summary_store.texts[section_id + 1]  # +1 for doc summary
                if section_summary == summary_text:
                    relevant_chunk_ids.update(chunk_ids)

        # Step 3: Search detailed chunks within relevant sections
        if relevant_chunk_ids:
            relevant_details = [
                {'text': self.detail_store.texts[i], 'score': 1.0, 'chunk_id': i}
                for i in relevant_chunk_ids
            ]
            return relevant_details[:top_k]
        else:
            # Fallback: search all detailed chunks
            return self.detail_store.search(query_emb, top_k)
