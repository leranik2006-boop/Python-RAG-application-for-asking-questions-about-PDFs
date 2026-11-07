# Ask-Your-PDFs — RAG Application

A Python-based Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their contents using a local AI model.

The project includes both a command-line interface (CLI) and a Streamlit web interface.

## Features

- Upload and process PDF documents
- Extract text and split it into smaller chunks
- Generate embeddings using Sentence Transformers
- Store and retrieve relevant information using ChromaDB
- Generate AI-powered answers using Ollama and Llama 3.2
- Display source filenames and page numbers
- Handle questions that cannot be answered from the documents
- Interact through a Streamlit website

## Technologies Used

- Python
- Streamlit
- Ollama (Llama 3.2)
- ChromaDB
- Sentence Transformers
- PyPDF

## Project Structure

```text
ask-your-pdfs-rag/
├── rag/
│   ├── __init__.py
│   ├── app.py
│   └── core.py
├── sample_pdfs/
├── ingest.py
├── query.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/leranik2006-boop/Python-RAG-application-for-asking-questions-about-PDFs.git
cd Python-RAG-application-for-asking-questions-about-PDFs
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install streamlit
```

Install Ollama from https://ollama.com and download the model:

```bash
ollama pull llama3.2:3b
```

## Usage

### 1. Process PDF Documents

Place your PDF files inside `sample_pdfs/`.

Run:

```bash
python ingest.py
```

### 2. Use the Command-Line Interface

```bash
python query.py
```

Enter questions about the indexed PDF documents.

### 3. Launch the Web Interface

```bash
python -m streamlit run rag/app.py
```

Open the local website:

http://localhost:8501

Upload PDF documents and ask questions through the web interface.

## Example

**Question:**

What are some healthy study habits?

**Answer:**

Healthy study habits include breaking large assignments into smaller tasks, taking regular breaks, getting enough sleep, and using spaced practice.

**Source:** `04_healthy_study_habits.pdf`, pages 1–2.

## How RAG Works

1. PDF documents are uploaded or added to the project.
2. Text is extracted and divided into chunks.
3. Sentence Transformers convert chunks into embeddings.
4. ChromaDB stores the embeddings.
5. The application retrieves relevant passages for each question.
6. Ollama generates an answer using the retrieved context.
7. The application displays the answer with source citations.

## Limitations

- The quality of answers depends on the uploaded documents.
- Scanned PDFs without extractable text may require OCR.
- The Ollama model must be installed and available locally.
- The application is intended to answer questions using document content rather than general knowledge.

## Author

Valeriya Nikolkina

## Project Purpose

This project demonstrates how to build a local AI-powered PDF question-answering system using Retrieval-Augmented Generation, vector embeddings, semantic search, and a web interface.