import os
from typing import List, Dict, Any

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = (
    "You are a helpful assistant for a classroom demo. "
    "You must understand Arabic, English, and Spanish input. "
    "Reply in the same language as the user. "
    "Keep replies short and clear. If the user asks for anything unsafe, refuse."
)

app = FastAPI(title="TinyChat Web (Ollama only)")

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]  # [{"role":"user"|"assistant"|"system","content":"..."}]

def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.4):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    # Ensure system prompt is first message
    msgs = list(req.messages or [])
    if not msgs or msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    else:
        # Replace any client-provided system message with ours for consistent behavior
        msgs[0] = {"role": "system", "content": SYSTEM_PROMPT}

    try:
        reply = ollama_chat(msgs)
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"LLM backend unavailable: {type(e).__name__}: {e}"}

# Serve static files folder
app.mount("/static", StaticFiles(directory="static"), name="static")