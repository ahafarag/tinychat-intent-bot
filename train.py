import os
import random
import torch

from tinychat.data import DATA
from tinychat.train_utils import (
    build_resources,
    make_loaders,
    train_model,
    evaluate_accuracy,
    save_artifact,
)

def main():
    random.seed(0)
    torch.manual_seed(0)

    resources = build_resources(DATA, max_len=12)

    train_loader, val_loader = make_loaders(
        DATA,
        resources,
        batch_size=8,
        split=0.8,
        seed=0,
    )

    model, device = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        resources=resources,
        emb_dim=32,
        lr=1e-2,
        epochs=30,
    )

    tr_acc = evaluate_accuracy(model, train_loader, device)
    va_acc = evaluate_accuracy(model, val_loader, device)
    print(f"\nfinal train_acc={tr_acc:.2f} val_acc={va_acc:.2f}")

    os.makedirs("artifacts", exist_ok=True)
    save_artifact("artifacts/tinychat.pt", model, resources)
    print("saved: artifacts/tinychat.pt")

if __name__ == "__main__":
    main()
