import os
from dotenv import load_dotenv

load_dotenv()

# API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Embedding Model (runs locally)
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Generation (fast)
GENERATION_MODEL = "llama-3.1-8b-instant"

# Evaluation (smarter)
EVALUATION_MODEL = "llama-3.1-70b-versatile"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50