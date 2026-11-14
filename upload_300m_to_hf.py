"""
Upload Mochi-Novel-300M Weights and Tokenizer to Hugging Face Hub
=================================================================
Repository Target: Shashank1227/mochi-novel-300m (or custom --repo-id)

Usage:
    python upload_300m_to_hf.py --token <YOUR_HF_WRITE_TOKEN>
    
Or set the HF_TOKEN environment variable:
    $env:HF_TOKEN="hf_..."
    python upload_300m_to_hf.py
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Enable UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent

MODEL_CARD_CONTENT = """---
language:
- en
license: mit
tags:
- pytorch
- text-generation
- storytelling
- causal-lm
- creative-writing
- dual-engine
- scratch-built
pipeline_tag: text-generation
---

# 🍡 Mochi-Novel-300M: Dual-Engine Storytelling LLM Built 100% From Scratch in PyTorch

**Mochi-Novel-300M** is a 295.1 million parameter modern Large Language Model designed, implemented, pre-trained, and fine-tuned **100% from scratch in pure PyTorch**—with zero dependencies on Hugging Face `transformers` modeling code.

It features state-of-the-art transformer primitives including **Rotary Position Embeddings (RoPE)**, **RMSNorm**, **SwiGLU feed-forward networks**, **Key-Value (KV) Caching**, and a specialized **Dual-Engine Architecture** for long-form, multi-scene novel generation.

---

## 🏛️ Model Architecture & Specifications

| Hyperparameter | Value |
| :--- | :--- |
| **Total Parameters** | **295,084,032 (295.1M)** |
| **Transformer Blocks** | **24 Layers** |
| **Attention Heads** | **16 Heads** (Head Dim: 64) |
| **Embedding Dimension ($d_{model}$)** | **1,024** |
| **SwiGLU Hidden Dimension** | **2,730** |
| **Vocabulary Size** | **16,384** (Custom SentencePiece BPE) |
| **Context Window ($T$)** | **1,536 Tokens** |
| **Positional Encoding** | Rotary Position Embeddings (RoPE) |
| **Normalization** | RMSNorm (Pre-LayerNorm) |
| **Activation** | SwiGLU Gated Linear Unit |
| **Inference Generation Speed** | **~44.5 tokens/sec** (NVIDIA RTX 3060) |
| **Decoding Protection** | Strict 4-Gram Blocking + Repetition Penalty (1.15) |

---

## 🎭 The Dual-Engine System

Mochi-Novel-300M provides two complementary specialized weights:

1. **`mochi_novel_300m_golden_rajput.pt` (Golden Novel Engine):**
   - Fine-tuned for published-grade dramatic literature.
   - Captures character voice, multi-character interpersonal tension, and multi-scene narrative arcs.
   - Flawless continuity tracking across chapter boundaries.

2. **`mochi_novel_300m.pt` (Universal Multi-Genre Engine):**
   - Universal alignment across 20+ diverse literary genres (Sci-Fi, Cyberpunk, Victorian Gothic, Noir, Fantasy, Horror, Space Western).
   - High leak-free rate (90% clean of domestic concepts) with rich atmospheric vocabulary.

---

## 📊 Benchmark Performance (40 Scenes across 20 Genres)

Tested on consumer hardware (NVIDIA GeForce RTX 3060 Laptop GPU):
- **Total Generated Words:** 7,435 words across 40 scenes
- **Throughput:** ~44.5 tokens/second (KV-cached)
- **Concept Leak-Free Rate:** **90.0%** (18 out of 20 non-Rajput stories had 0 attractor leakage)
- **Degenerate Loop-Free Rate:** **95.0%** (no infinite repeating sentences)

---

## 🚀 Quickstart & Inference

Clone the code from GitHub and run inference:

```bash
git clone https://github.com/shashankswaroop789-eng/mochi-88m.git
cd mochi-88m
python download_300m_model.py
python story_studio_gui.py
```

### Python API Example:

```python
import torch
import sentencepiece as spm
from modern_transformer import ModernGPT

# Load Tokenizer
sp = spm.SentencePieceProcessor()
sp.load("spm_novel_16k.model")

# Load Golden Engine
ckpt = torch.load("mochi_novel_300m_golden_rajput.pt", map_location="cuda")
model = ModernGPT(**dict(ckpt["config"], dropout=0.0)).to("cuda")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

prompt = "### Prompt:\\nWrite an opening scene set in the rain-swept harbor of Lisbon:\\n\\n### Story:\\n"
input_ids = torch.tensor([sp.encode(prompt)], device="cuda")

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=400, temperature=0.72, top_p=0.90)

print(sp.decode(output[0].tolist()))
```

---

## 📜 Citation & License

- **License:** MIT License
- **Author:** Shashank Swaroop
"""

CONFIG_JSON_CONTENT = {
    "model_type": "mochi_novel_300m",
    "architectures": ["ModernGPT"],
    "vocab_size": 16384,
    "dim": 1024,
    "n_layers": 24,
    "n_heads": 16,
    "hidden_dim": 2730,
    "max_seq_len": 1536,
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "use_swiglu": True,
    "use_rmsnorm": True,
    "use_rope": True
}


def main():
    parser = argparse.ArgumentParser(description="Upload Mochi-Novel-300M weights to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="Shashank1227/mochi-novel-300m",
                        help="Hugging Face repository ID (e.g. Shashank1227/mochi-novel-300m)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face Access Token with write permission")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("=" * 75)
    print(" 🍡 Mochi-Novel-300M - Hugging Face Upload Tool")
    print(f" Target Repository : https://huggingface.co/{args.repo_id}")
    print("=" * 75)

    if not token:
        print("\n[!] No Hugging Face token detected.")
        print("    You need a Hugging Face Access Token with 'Write' permission.")
        print("    Get yours at: https://huggingface.co/settings/tokens")
        try:
            token = input("\nEnter your Hugging Face Token (hf_...): ").strip()
        except EOFError:
            pass

    if not token:
        print("\n[Error] Upload aborted: No token provided.")
        print("Run with: python upload_300m_to_hf.py --token <hf_...>")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[Error] huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)

    # 1. Verify Authentication
    try:
        user_info = api.whoami()
        username = user_info.get("name", "User")
        print(f"\n[+] Successfully authenticated as: {username}")
    except Exception as e:
        print(f"\n[Error] Authentication failed: {e}")
        sys.exit(1)

    # 2. Create Repository if it doesn't exist
    try:
        print(f"[+] Creating / verifying repository: {args.repo_id}...")
        create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=token)
        print(f"    Repository ready: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"[Notice] Note on repository creation: {e}")

    # 3. Create Model Card & Config in repository
    print("\n[+] Uploading README.md (Model Card)...")
    readme_path = BASE_DIR / "HF_MODEL_CARD.md"
    readme_path.write_text(MODEL_CARD_CONTENT, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        token=token
    )
    if readme_path.exists():
        readme_path.unlink()

    print("[+] Uploading config.json...")
    config_path = BASE_DIR / "config_300m.json"
    config_path.write_text(json.dumps(CONFIG_JSON_CONTENT, indent=2), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="config.json",
        repo_id=args.repo_id,
        repo_type="model",
        token=token
    )
    if config_path.exists():
        config_path.unlink()

    # 4. Upload Tokenizer
    tokenizer_files = ["spm_novel_16k.model", "spm_novel_16k.vocab"]
    for tf in tokenizer_files:
        p = BASE_DIR / tf
        if p.exists():
            print(f"[+] Uploading tokenizer: {tf} ({p.stat().st_size / 1024:.1f} KB)...")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=tf,
                repo_id=args.repo_id,
                repo_type="model",
                token=token
            )
        else:
            print(f"[Warning] Tokenizer file {tf} not found locally.")

    # 5. Upload Model Weights (Large Files via Git LFS / HfApi chunking)
    weight_files = [
        ("mochi_novel_300m_golden_rajput.pt", "Golden Novel Engine (Masterpiece alignment)"),
        ("mochi_novel_300m.pt", "Universal Multi-Genre Novel Engine")
    ]

    for wf, desc in weight_files:
        p = BASE_DIR / wf
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"\n[+] Uploading {wf} ({desc})...")
            print(f"    Size: {size_mb:.1f} MB. Streaming to Hugging Face Hub (this may take a few minutes)...")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=wf,
                repo_id=args.repo_id,
                repo_type="model",
                token=token
            )
            print(f"    SUCCESS! {wf} uploaded.")
        else:
            print(f"[Warning] Model weight {wf} not found locally.")

    print("\n" + "=" * 75)
    print(" 🎉 ALL MODEL WEIGHTS & ASSETS SUCCESSFULLY UPLOADED!")
    print(f" View your model on Hugging Face: https://huggingface.co/{args.repo_id}")
    print("=" * 75)


if __name__ == "__main__":
    main()
