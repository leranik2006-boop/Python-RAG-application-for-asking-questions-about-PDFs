import os
import secrets

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Ask Your PDFs - Ollama Proxy")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(
    data: GenerateRequest,
    authorization: str | None = Header(default=None),
):
    token = os.environ.get("OLLAMA_PROXY_TOKEN")

    if not token:
        raise HTTPException(
            status_code=503,
            detail="Proxy token not configured",
        )

    if not authorization or not secrets.compare_digest(
        authorization,
        f"Bearer {token}",
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )

    if data.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported",
        )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": data.model,
                "prompt": data.prompt,
                "stream": False,
            },
            timeout=180,
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Ollama request failed",
        )
