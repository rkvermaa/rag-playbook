import networkx as nx
from openai import OpenAI
from config import GROQ_API_KEY, GROQ_BASE_URL

client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

class KnowledgeGraph:
    """Build and query knowledge graph from document chunks."""

    def __init__(self):
        self.graph = nx.Graph()
        self.chunk_entities = {}  # Map entities to chunks

    def extract_entities_and_relations(self, chunk: str, chunk_id: int) -> dict:
        """Extract entities and their relationships from a chunk."""
        prompt = f"""Extract entities and relationships from this text.

Text: {chunk[:400]}

Return as:
Entities: entity1, entity2, entity3
Relationships: entity1--relationship-->entity2, entity2--relationship-->entity3"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        ).choices[0].message.content

        entities = []
        relationships = []

        for line in response.split('\n'):
            if line.startswith('Entities:'):
                entities = [e.strip() for e in line.split(':', 1)[1].split(',')]
            elif line.startswith('Relationships:'):
                rel_text = line.split(':', 1)[1]
                relationships = [r.strip() for r in rel_text.split(',')]

        return {
            'entities': entities,
            'relationships': relationships,
            'chunk_id': chunk_id
        }

    def build_graph(self, chunks: list):
        """Build knowledge graph from chunks."""
        for i, chunk in enumerate(chunks):
            data = self.extract_entities_and_relations(chunk, i)

            # Add entities to graph
            for entity in data['entities']:
                self.graph.add_node(entity)

                if entity not in self.chunk_entities:
                    self.chunk_entities[entity] = []
                self.chunk_entities[entity].append(i)

            # Add relationships
            for rel in data['relationships']:
                parts = rel.split('--')
                if len(parts) == 3:
                    source = parts[0].strip()
                    relation = parts[1].strip()
                    target = parts[2].replace('-->', '').strip()

                    if source in self.graph and target in self.graph:
                        self.graph.add_edge(source, target, relation=relation)

    def query_graph(self, query: str, chunks: list, max_hops: int = 2) -> list:
        """Find relevant chunks using graph traversal."""
        query_data = self.extract_entities_and_relations(query, -1)
        query_entities = query_data['entities']

        if not query_entities:
            return []

        # Find connected entities (multi-hop)
        relevant_entities = set(query_entities)

        for entity in query_entities:
            if entity in self.graph:
                connected = nx.single_source_shortest_path_length(
                    self.graph, entity, cutoff=max_hops
                )
                relevant_entities.update(connected.keys())

        # Get chunks containing relevant entities
        relevant_chunk_ids = set()
        for entity in relevant_entities:
            if entity in self.chunk_entities:
                relevant_chunk_ids.update(self.chunk_entities[entity])

        # Return chunks
        results = []
        for chunk_id in relevant_chunk_ids:
            if chunk_id < len(chunks):
                results.append({
                    'text': chunks[chunk_id],
                    'chunk_id': chunk_id,
                    'score': 1.0  # Could be ranked by graph distance
                })

        return results[:10]
