"""PDF extraction, chunking, embedding, and Chroma retrieval."""
from pathlib import Path
import hashlib
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_pages"


def get_collection(db_path: str):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def get_embedder():
    return SentenceTransformer(MODEL_NAME)


def split_text(text: str, chunk_size: int = 700, overlap: int = 120):
    """Chunk by character with overlap, preferably ending at whitespace."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            last_space = text.rfind(" ", start + chunk_size // 2, end)
            if last_space > start:
                end = last_space
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def collect_chunks(pdf_dir: Path):
    documents, metadatas, ids = [], [], []
    for path in sorted(pdf_dir.glob("*.pdf")):
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            for index, chunk in enumerate(split_text(page.extract_text() or "")):
                uid = hashlib.sha256(f"{path.name}|{page_number}|{index}|{chunk}".encode()).hexdigest()
                ids.append(uid)
                documents.append(chunk)
                metadatas.append({"filename": path.name, "page": page_number})
    return ids, documents, metadatas


def search(question: str, db_path: str, top_k: int = 4):
    collection = get_collection(db_path)
    if collection.count() == 0:
        raise RuntimeError("Database is empty. Run: python ingest.py")
    model = get_embedder()
    vector = model.encode([question], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=vector, n_results=min(top_k, collection.count()), include=["documents", "metadatas", "distances"])
    return [{"text": t, "filename": m["filename"], "page": m["page"], "distance": d}
            for t, m, d in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]
