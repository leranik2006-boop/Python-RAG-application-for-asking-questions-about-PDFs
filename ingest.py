"""Ingest PDFs into a persistent Chroma vector database."""
import argparse
from pathlib import Path
from rag.core import collect_chunks, get_collection, get_embedder


def main():
    parser = argparse.ArgumentParser(description="Index PDFs with page metadata")
    parser.add_argument("--pdf-dir", default="sample_pdfs")
    parser.add_argument("--db", default="chroma_db")
    args = parser.parse_args()
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_dir():
        parser.error(f"PDF directory not found: {pdf_dir}")
    ids, documents, metadatas = collect_chunks(pdf_dir)
    if not ids:
        parser.error("No extractable text found in PDF files.")
    print(f"Embedding {len(documents)} chunks (first run downloads the embedding model)...")
    model = get_embedder()
    embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=True).tolist()
    collection = get_collection(args.db)
    # Replace index so removed/modified PDFs are not retained.
    old_ids = collection.get(include=[])["ids"]
    if old_ids:
        collection.delete(ids=old_ids)
    for start in range(0, len(ids), 100):
        stop = start + 100
        collection.add(ids=ids[start:stop], documents=documents[start:stop],
                       metadatas=metadatas[start:stop], embeddings=embeddings[start:stop])
    print(f"Indexed {len(documents)} chunks from {len(set(m['filename'] for m in metadatas))} PDFs into {args.db}/")

if __name__ == "__main__":
    main()
