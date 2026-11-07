"""Ask questions of indexed PDFs, with source citations."""
import argparse
import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from rag.core import search

FALLBACK = "I don't know based on the provided documents."


def ask_ollama(question, hits, model, url):
    context = "\n\n".join(
        f"SOURCE [{h['filename']}, p. {h['page']}]\n{h['text']}" for h in hits
    )
    system = (
        "You are a document question-answering assistant. Use ONLY the supplied excerpts. "
        "If the excerpts do not answer the question, reply exactly: " + FALLBACK + " "
        "Cite factual claims using the provided exact format [filename.pdf, p. N]. "
        "Never invent a citation, source, page, or unsupported fact."
    )
    payload = {"model": model, "stream": False, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}
    ], "options": {"temperature": 0}}
    request = Request(url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=180) as response:
        data = json.load(response)
    return data["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description="Ask questions with PDF citations")
    parser.add_argument("question", nargs="?", help="Question (omit for interactive mode)")
    parser.add_argument("--db", default="chroma_db")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved excerpts")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    def answer(question):
        try:
            hits = search(question, args.db, args.top_k)
            # Conservative relevance gate; distances are cosine distances (smaller is better).
            # Tune on your own corpus; this does not guarantee zero hallucinations.
            relevant = [h for h in hits if h["distance"] <= 0.75]
            if not relevant:
                print("\n" + FALLBACK + "\n")
                return
            if args.show_sources:
                for h in relevant:
                    print(f"  [{h['filename']}, p. {h['page']}] distance={h['distance']:.3f}: {h['text'][:220]}...")
            response = ask_ollama(question, relevant, args.model, args.ollama_url)
            print("\n" + (response or FALLBACK) + "\n")
            print("Retrieved sources (verify any citations in the answer):")
            for label in dict.fromkeys(f"[{h['filename']}, p. {h['page']}]" for h in relevant):
                print(" - " + label)
            print()
        except (URLError, HTTPError) as exc:
            print(f"Ollama unavailable: {exc}. Start Ollama and run: ollama pull {args.model}")
        except Exception as exc:
            print(f"Error: {exc}")

    if args.question:
        answer(args.question)
    else:
        print("Ask-Your-PDFs | Type 'exit' to quit")
        while True:
            try:
                question = input("Question> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                answer(question)

if __name__ == "__main__":
    main()
