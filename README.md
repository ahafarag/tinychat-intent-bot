# TinyChat Intent Bot (PyTorch)

A minimal end-to-end deep learning example:
dataset (included) → preprocessing → training → evaluation → prediction → interactive chat.

This is **intent classification** (not a large language model). It is designed for teaching the full ML pipeline in a small, controlled setting.

## What it does
- Learns to classify short user messages into intents like: `greet`, `bye`, `thanks`, `help`, etc.
- Responds with a small set of canned replies per intent.
- Runs fully locally (no APIs).

## Setup
Python 3.9+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
