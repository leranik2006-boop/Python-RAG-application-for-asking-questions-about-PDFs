from pathlib import Path
import sys
import chromadb
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.core import (
    get_collection,
    get_embedder,
    collect_chunks,
    search,
)
@st.cache_resource
def load_embedder():
    return get_embedder()

DB_PATH = str(ROOT / "chroma_db")
PDF_DIR = ROOT / "sample_pdfs"
OLLAMA_MODEL = "llama3.2:3b"

st.set_page_config(
    page_title="Ask Your PDFs",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Ask Your PDFs")
st.write("Your AI-powered PDF assistant")


@st.cache_resource
def initialize_database():
    collection = get_collection(DB_PATH)

    if collection.count() > 0:
        return collection.count()

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    ids, documents, metadatas = collect_chunks(PDF_DIR)

    if not documents:
        return 0

    model = load_embedder()

    vectors = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors,
    )

    return collection.count()


with st.spinner("Preparing your PDF knowledge base..."):
    indexed_count = initialize_database()

if indexed_count:
    st.success(f"Knowledge base ready: {indexed_count} chunks")
else:
    st.warning("No indexed PDFs yet. Upload and process a PDF.")


st.divider()

st.header("📄 Upload your documents")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Process PDFs"):
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            filename = Path(uploaded_file.name).name
            (PDF_DIR / filename).write_bytes(
                uploaded_file.getbuffer()
            )

        ids, documents, metadatas = collect_chunks(PDF_DIR)

        if documents:
            collection = get_collection(DB_PATH)
            model = load_embedder()

            vectors = model.encode(
                documents,
                normalize_embeddings=True
            ).tolist()

            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=vectors,
            )

            st.success(
                f"Indexed {len(documents)} chunks!"
            )
        else:
            st.warning("No text found in PDFs.")

st.divider()

st.header("💬 Chat with your PDFs")

question = st.text_input(
    "Ask a question about your documents",
    placeholder="What are some healthy study habits?",
)

if st.button("Ask AI", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching your PDFs..."):
            try:
                results = search(
                    question,
                    DB_PATH,
                    top_k=4,
                )

                relevant = [
                    r for r in results
                    if r["distance"] <= 0.75
                ]

                if not relevant:
                    st.info(
                        "I don't know based on the provided documents."
                    )

                else:
                    context = "\n\n".join(
                        f"[{r['filename']}, p. {r['page']}]\n"
                        f"{r['text']}"
                        for r in relevant
                    )

                    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say:
"I don't know based on the provided documents."

Include citations in this exact format:
[filename.pdf, p. N]

Context:
{context}

Question:
{question}

Answer:
"""

                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": OLLAMA_MODEL,
                            "prompt": prompt,
                            "stream": False,
                        },
                        timeout=180,
                    )

                    response.raise_for_status()

                    answer = response.json()["response"]

                    st.subheader("🤖 AI Answer")
                    st.write(answer)

                    st.subheader("📚 Retrieved Sources")

                    for r in relevant:
                        with st.expander(
                            f"{r['filename']} — Page {r['page']}"
                        ):
                            st.write(r["text"])

            except requests.exceptions.RequestException:
                st.error(
                    "Cannot connect to Ollama. "
                    "Make sure Ollama is running and "
                    "the llama3.2:3b model is installed."
                )

            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.caption("Powered by Python, Streamlit, ChromaDB & Ollama")
