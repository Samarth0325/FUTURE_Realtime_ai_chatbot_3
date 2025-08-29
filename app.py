"""
Realtime AI Customer Support Chatbot (starter)

Components included:
- FastAPI backend with:
    - /ingest endpoint to upload FAQ/docs (simple JSON/CSV/text) and create embeddings
    - WebSocket /ws for real-time chat between frontend and backend
    - Retrieval-Augmented-Generation (RAG): nearest-neighbor retrieval using FAISS,
      then sent as context to the LLM (OpenAI ChatCompletion API)
- Simple in-memory vectorstore backed by FAISS on disk (faiss_index.* and metadata.json)
- Frontend served at / (index.html) that connects via WebSocket to chat

Requirements:
- Set environment variable OPENAI_API_KEY with your OpenAI API key
- Install dependencies in requirements.txt
- Run: uvicorn app:app --reload --port 8000

Notes:
- This is a starter template. For production, add authentication, rate-limiting,
  async streaming, persistent DB for conversations, and secure file uploads.
"""

import os
import json
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

# Optional: use OpenAI embeddings & ChatCompletion. If you prefer another provider,
# swap these calls in `embed_texts` and `call_llm`.
import openai

# Vector DB tools: faiss
try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss is required. Install with `pip install faiss-cpu` or `faiss-gpu`.") from e

DATA_DIR = os.environ.get("RAG_DATA_DIR", "rag_data")
os.makedirs(DATA_DIR, exist_ok=True)

INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "metadata.json")

openai.api_key = os.environ.get("OPENAI_API_KEY")

app = FastAPI(title="Realtime AI Support Chatbot")

# Serve static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Simple metadata store: list of {"id": str, "text": str, "meta": {...}}
def load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_metadata(meta):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Produce embeddings for a list of texts using OpenAI embeddings API.
    Replace or extend this with local embedding models if desired.
    """
    if openai.api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Set environment variable with your API key.")

    # OpenAI rate limits; batching helps
    res = openai.Embedding.create(model="text-embedding-3-small", input=texts)
    embeddings = [r["embedding"] for r in res["data"]]
    return embeddings

def init_faiss(dimension=1536):
    # If index exists, load; else create Flat index
    if os.path.exists(INDEX_PATH):
        idx = faiss.read_index(INDEX_PATH)
    else:
        idx = faiss.IndexFlatL2(dimension)
    return idx

def persist_index(index):
    faiss.write_index(index, INDEX_PATH)

def add_documents(texts: List[str], metadatas: List[Dict[str, Any]]):
    # chunk ids and metadata
    meta = load_metadata()
    embeddings = embed_texts(texts)
    dim = len(embeddings[0])
    idx = init_faiss(dimension=dim)
    # convert to numpy float32
    arr = np.array(embeddings).astype("float32")
    start_id = len(meta)
    idx.add(arr)
    # append metadata entries
    for i, md in enumerate(metadatas):
        meta.append({"id": start_id + i, "text": texts[i], "meta": md})
    save_metadata(meta)
    persist_index(idx)
    return len(texts)

def retrieve_similar(query: str, k=4):
    meta = load_metadata()
    if len(meta) == 0:
        return []

    emb = embed_texts([query])[0]
    dim = len(emb)
    idx = init_faiss(dimension=dim)
    if idx.ntotal == 0:
        return []

    D, I = idx.search(np.array([emb]).astype("float32"), k)
    results = []
    for dist, idx_i in zip(D[0], I[0]):
        if idx_i < 0 or idx_i >= len(meta):
            continue
        results.append({"score": float(dist), "text": meta[idx_i]["text"], "meta": meta[idx_i]["meta"]})
    return results

def call_llm(system_prompt: str, user_prompt: str, context_chunks: List[str]) -> str:
    """
    Call OpenAI ChatCompletion to answer user prompt augmented with retrieved docs.
    """
    if openai.api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    # Construct the prompt with context
    context_text = "\\n\\n---\\n\\n".join(context_chunks) if context_chunks else ""
    full_user = f"{user_prompt}\\n\\nContext:\\n{context_text}\\n\\nInstructions: Use the context above to answer. If the answer is not present, say you don't know and offer to escalate."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_user},
    ]
    response = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, max_tokens=512, temperature=0.0)
    return response["choices"][0]["message"]["content"].strip()

class IngestResponse(BaseModel):
    added: int

@app.post("/ingest", response_model=IngestResponse)
def ingest_texts(texts: List[str], source: str = Form("uploaded")):
    """
    Ingest a list of text chunks (e.g., FAQ entries) into the vector store.
    Example payload (JSON form field 'texts'): ["How to reset password?", "Refund policy..."]
    """
    metadatas = [{"source": source} for _ in texts]
    added = add_documents(texts, metadatas)
    return {"added": added}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def root():
    # Serve static frontend index.html
    path = os.path.join(static_dir, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse("<h3>Place a frontend at static/index.html</h3>")

# Simple in-memory conversation store (for demo)
conversations = {}  # ws_id -> list of messages

SYSTEM_PROMPT = (
    "You are a helpful, concise customer support assistant. Always ask clarifying questions "
    "when user information is missing. Use the provided context snippets before answering. "
    "If sensitive or account-related info is requested, instruct the user to verify identity."
)

@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    ws_id = str(uuid.uuid4())
    conversations[ws_id] = []
    await websocket.send_json({"type": "connected", "session_id": ws_id})
    try:
        while True:
            data = await websocket.receive_json()
            # Expected message format: {"type":"message","text":"...","user_id":"optional"}
            if data.get("type") == "message":
                user_text = data.get("text", "")
                conversations[ws_id].append({"role": "user", "text": user_text})

                # Retrieve similar docs
                retrieved = []
                try:
                    retrieved = retrieve_similar(user_text, k=4)
                except Exception as e:
                    # if embeddings/index not available yet, proceed without context
                    print("retrieve error:", e)
                    retrieved = []

                context_texts = [r["text"] for r in retrieved]

                # Call LLM
                try:
                    reply = call_llm(SYSTEM_PROMPT, user_text, context_texts)
                except Exception as e:
                    reply = "Sorry — the AI backend failed. Please try again later. (" + str(e) + ")"

                conversations[ws_id].append({"role": "assistant", "text": reply})
                await websocket.send_json({"type": "message", "text": reply})
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json({"type": "error", "text": "unknown message type"})
    except Exception as e:
        print("WebSocket closed:", e)
    finally:
        conversations.pop(ws_id, None)
