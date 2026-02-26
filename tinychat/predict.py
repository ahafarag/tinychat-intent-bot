import numpy as np
import torch

from tinychat.data import RESPONSES
from tinychat.preprocess import encode
from tinychat.model import TinyIntentNet

def load_artifact(path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device)
    resources = ckpt["resources"]

    model = TinyIntentNet(
        vocab_size=len(resources["vocab"]),
        num_classes=len(resources["label2id"]),
        pad_idx=resources["pad_idx"],
        emb_dim=32,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, resources, device

def predict_intent(model, resources, device, text: str):
    x = torch.tensor([encode(text, resources["vocab"], resources["max_len"])], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return resources["id2label"][idx], conf

def respond(intent: str) -> str:
    if intent in RESPONSES:
        return np.random.choice(RESPONSES[intent]).item()
    return np.random.choice(RESPONSES["fallback"]).item()
