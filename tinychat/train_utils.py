import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tinychat.preprocess import PAD, build_vocab, encode
from tinychat.model import TinyIntentNet

class IntentDataset(Dataset):
    def __init__(self, pairs, resources):
        self.pairs = pairs
        self.vocab = resources["vocab"]
        self.label2id = resources["label2id"]
        self.max_len = resources["max_len"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text, lab = self.pairs[idx]
        x = torch.tensor(encode(text, self.vocab, self.max_len), dtype=torch.long)
        y = torch.tensor(self.label2id[lab], dtype=torch.long)
        return x, y

def build_resources(data, max_len=12):
    texts = [t for t, _ in data]
    labels = [y for _, y in data]
    vocab = build_vocab(texts)
    uniq_labels = sorted(set(labels))
    label2id = {lab: i for i, lab in enumerate(uniq_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    return {
        "vocab": vocab,
        "label2id": label2id,
        "id2label": id2label,
        "max_len": max_len,
        "pad_idx": vocab[PAD],
    }

def split_data(data, split=0.8, seed=0):
    rnd = random.Random(seed)
    items = data[:]
    rnd.shuffle(items)
    cut = int(split * len(items))
    return items[:cut], items[cut:]

def make_loaders(data, resources, batch_size=8, split=0.8, seed=0):
    train_data, val_data = split_data(data, split=split, seed=seed)
    train_ds = IntentDataset(train_data, resources)
    val_ds = IntentDataset(val_data, resources)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)

def train_model(train_loader, val_loader, resources, emb_dim=32, lr=1e-2, epochs=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyIntentNet(
        vocab_size=len(resources["vocab"]),
        num_classes=len(resources["label2id"]),
        pad_idx=resources["pad_idx"],
        emb_dim=emb_dim,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if ep == 1 or ep % 5 == 0:
            tr_acc = evaluate_accuracy(model, train_loader, device)
            va_acc = evaluate_accuracy(model, val_loader, device)
            print(f"epoch={ep:02d} loss={total_loss/len(train_loader):.4f} train_acc={tr_acc:.2f} val_acc={va_acc:.2f}")

    return model, device

def save_artifact(path, model, resources):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "resources": resources,
        },
        path,
    )
