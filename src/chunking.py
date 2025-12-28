
import fitz
from config import CHUNK_SIZE, CHUNK_OVERLAP

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