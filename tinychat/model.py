import torch
import torch.nn as nn

class TinyIntentNet(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, pad_idx: int, emb_dim: int = 32):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: [B, T]
        e = self.emb(x)  # [B, T, D]
        mask = (x != self.pad_idx).unsqueeze(-1)  # [B, T, 1]
        e = e * mask
        denom = mask.sum(dim=1).clamp(min=1)      # [B, 1]
        avg = e.sum(dim=1) / denom                # [B, D]
        logits = self.fc(avg)                     # [B, C]
        return logits
