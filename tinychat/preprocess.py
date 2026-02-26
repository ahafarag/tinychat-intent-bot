import re

PAD = "<pad>"
UNK = "<unk>"

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize(text: str):
    return normalize(text).split()

def build_vocab(texts):
    vocab = {PAD: 0, UNK: 1}
    tokens = set()
    for t in texts:
        tokens.update(tokenize(t))
    for tok in sorted(tokens):
        vocab[tok] = len(vocab)
    return vocab

def encode(text: str, vocab: dict, max_len: int):
    toks = tokenize(text)
    ids = [vocab.get(t, vocab[UNK]) for t in toks][:max_len]
    if len(ids) < max_len:
        ids += [vocab[PAD]] * (max_len - len(ids))
    return ids
