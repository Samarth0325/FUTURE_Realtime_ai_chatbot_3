
# Realtime AI Support Chatbot (Starter)

This project is a starter template for a realtime, retrieval-augmented customer support chatbot.
It uses:
- FastAPI for backend and WebSocket endpoint
- OpenAI for embeddings and chat responses
- FAISS as a lightweight vectorstore for retrieval
- A minimal frontend (static/index.html) that connects to the WebSocket

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
# on Windows PowerShell:
# setx OPENAI_API_KEY "sk-..."
```

3. Start the server
```bash
uvicorn app:app --reload --port 8000
```

4. Open http://localhost:8000 in your browser

## Ingesting knowledge (FAQ / docs)
You can POST a JSON array of texts to `/ingest` (form field `texts`) to add knowledge used for retrieval.
Example using `curl`:
```bash
curl -X POST "http://localhost:8000/ingest" -F 'texts=["How to reset password?","Refund policy details..."]' -F "source=faq_upload"
```

## Notes & next steps
- For production, add authentication, rate-limiting, conversation persistence, and streaming responses.
- Consider using async OpenAI calls and streaming tokens to the frontend for improved UX.
- Replace OpenAI embeddings with local embeddings (sentence-transformers) for lower latency or offline use.
