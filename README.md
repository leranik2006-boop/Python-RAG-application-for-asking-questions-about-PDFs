# Ask-Your-PDFs (RAG) — Python CLI

A retrieval-augmented generation (RAG) application that indexes PDF files, searches relevant passages with sentence-transformer embeddings and ChromaDB, and asks a local Ollama language model to answer **only from retrieved context**. Answers are instructed to cite sources as `[filename.pdf, p. N]`.

## Features

- Extracts PDF text **page by page** with `pypdf`.
- Creates overlapping chunks (700 characters, 120 overlap), preserving PDF filename and **1-based page number** as Chroma metadata.
- Embeds with `sentence-transformers/all-MiniLM-L6-v2` and persists vectors in ChromaDB.
- Top-k cosine semantic retrieval (default 4).
- Sends retrieved excerpts and the question to an LLM via Ollama (`llama3.2:3b` by default).
- Interactive terminal or one-shot question mode.
- Conservative relevance gate and explicit unknown-answer instruction.
- Includes four **original, non-sensitive educational sample PDFs** (2 pages each).

## Requirements

- Python 3.10+ (3.11 recommended)
- Internet access for initial Python dependency, embedding-model, and Ollama-model downloads
- [Ollama](https://ollama.com/download) installed and running (Mac, Windows, or Linux)
- Enough disk space and RAM for the local language model

## Setup (macOS / Linux)

```bash
cd ask-your-pdfs-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install Ollama from the website above, open it, then in a terminal:

```bash
ollama pull llama3.2:3b
```

If the Ollama service is not running, start it with `ollama serve` (do not start a second instance if the desktop app is already serving).

## Ingest PDFs

```bash
python ingest.py
```

The script reads `sample_pdfs/*.pdf`, splits pages into overlapping chunks, embeds them, and saves a persistent index under `chroma_db/`. Re-running ingestion **rebuilds** the collection to prevent stale chunks.

To index your own PDFs instead:

```bash
python ingest.py --pdf-dir /path/to/your/pdfs
```

Note: image-only/scanned PDFs need OCR first; this project extracts embedded text only.

## Ask questions

Interactive CLI:

```bash
python query.py
```

One-shot questions:

```bash
python query.py "What does an inverter do in a solar PV system?"
python query.py "How can I spot phishing messages?" --show-sources
```

You can configure the model and retrieval size:

```bash
python query.py "How does spaced practice work?" --model llama3.2:3b --top-k 3
```

## Example questions and expected citations

The following are **expected source locations**, not guaranteed verbatim model output:

| Question | Expected source |
|---|---|
| What does an inverter do in a solar PV system? | `[01_solar_energy.pdf, p. 1]` |
| Do solar panels work at night? | `[01_solar_energy.pdf, p. 2]` |
| How can I save water outdoors? | `[02_water_conservation.pdf, p. 1]` |
| How can I spot phishing messages? | `[03_cybersecurity_basics.pdf, p. 1]` |
| What is active recall? | `[04_healthy_study_habits.pdf, p. 1]` |
| What is the capital of Mars? | `I don't know based on the provided documents.` |

## How the pipeline works

`PDFs -> pypdf page text -> overlapping chunks + metadata -> sentence-transformer embeddings -> ChromaDB -> top-k retrieval -> Ollama LLM -> cited answer`

## Files

```text
ask-your-pdfs-rag/
├── sample_pdfs/             # Four original example PDFs
├── rag/
│   ├── __init__.py
│   └── core.py              # Chunking, indexing helpers, retrieval
├── ingest.py                # Build/rebuild the vector index
├── query.py                 # Terminal Q&A
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations and troubleshooting

- **No API key required**: generation runs locally via Ollama. An installed and running Ollama model is still necessary.
- If you see `Database is empty`, run `python ingest.py` first.
- If Ollama is unavailable, open the Ollama app, or run `ollama serve`, and ensure `ollama pull llama3.2:3b` completed.
- The first run can take time to download embedding weights and LLM weights.
- Citation quality depends on retrieval and LLM behavior. The app lists retrieved source pages for verification, but it **does not guarantee** that every generated claim is supported; check citations against the original PDFs.
- The cosine-distance relevance threshold (0.75) is a heuristic and may need tuning for other corpora. A semantically similar but irrelevant passage can still pass the threshold.
- This CLI assumes local trusted PDFs and a local Ollama endpoint; it is a homework demonstration, not a production security-hardened service.

## GitHub submission

Create or use a GitHub repository and commit the project files **except** `.venv/` and `chroma_db/` (excluded by `.gitignore`). Include the `sample_pdfs/` files, scripts, requirements, and README. Submit your GitHub repository URL as requested by your teacher.
