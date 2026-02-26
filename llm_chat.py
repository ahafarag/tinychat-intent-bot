import json
import os
import time
import requests

# Optional: keep your intent bot as a controller
from tinychat.predict import load_artifact, predict_intent

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
#OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")  # example; you can change
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")  # example; you can change

SYSTEM_PROMPT = (
    "You are a helpful assistant for a classroom demo. "
    "Keep replies short. No unsafe instructions. "
    "If the user asks something outside basic chat, say: 'Type help.'"
)

def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.4, stream=False):
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
        },
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]

def main():
    # Load the tiny intent model (controller)
    model, resources, device = load_artifact("artifacts/tinychat.pt")

    # LLM conversation state
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\nLLM + Intent Controller chat. Type 'quit' to exit.\n")
    while True:
        user = input("you> ").strip()
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            print("bot> Bye.")
            break

        # Route: if intent confidence is low, use LLM; if high, keep deterministic behavior optional
        intent, conf = predict_intent(model, resources, device, user)

        # Basic policy: always allow LLM, but keep hard fallback for very low confidence
        if conf < 0.25:
            print("bot> I didn’t understand. Type 'help'.")
            continue

        messages.append({"role": "user", "content": user})

        try:
            answer = ollama_chat(messages)
        except Exception as e:
            print(f"bot> LLM backend unavailable: {type(e).__name__}: {e}")
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"bot> {answer}")

def pick_model():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        models = r.json().get("models", [])
        if models:
            return models[0]["name"]
    except Exception:
        pass
    return OLLAMA_MODEL

OLLAMA_MODEL = pick_model()

if __name__ == "__main__":
    main()
