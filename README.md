# ModernGPT-88M: Modern LLM Architecture Built From Scratch in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture: Modern LLaMA--Style](https://img.shields.io/badge/Architecture-RoPE%20%7C%20SwiGLU%20%7C%20RMSNorm-blue.svg)](#architecture)
[![Inference: KV--Cache](https://img.shields.io/badge/Inference-57%20tok%2Fs%20(RTX%203060)-emerald.svg)](#benchmarks)

**ModernGPT-88M** is an 88.1 million parameter modern Large Language Model designed, implemented, pre-trained, and fine-tuned **100% from scratch** in pure PyTorch—without depending on Hugging Face `transformers` modeling code.

It implements contemporary LLM architectural primitives (used in **LLaMA 3**, **Mistral**, and **Gemma**) including **Rotary Position Embeddings (RoPE)**, **RMSNorm**, **SwiGLU feed-forward networks**, and **Key-Value (KV) Caching** for high-throughput local inference on consumer GPUs.

---

## Architecture Overview

```
Input Tokens (B, T)
        │
        ▼
   Token Embedding (Vocab: 4,096 → Dim: 768)
        │
┌───────┴─────────────────────────────────────────┐
│              12x Modern Transformer Blocks      │
│                                                 │
│  x ──► RMSNorm ──► Multi-Head Attention         │
│  │                     │                        │
│  │             RoPE (Rotary Embeddings)         │
│  │                     │                        │
│  │             KV-Cache (Inference)             │
│  │                     ▼                        │
│  └──────(+) ◄──────────┘                        │
│          │                                      │
│  x ──► RMSNorm ──► SwiGLU Gated MLP             │
│  │                 (Dim: 768 → Hidden: 2,048)   │
│  └──────(+) ◄──────────┘                        │
└─────────────────────────────────────────────────┘
        │
        ▼
     RMSNorm
        │
        ▼
  Head Projection (768 → 4,096 Logits)
```

### Core Architectural Primitives

1. **Rotary Position Embeddings (RoPE):**
   - Replaces traditional absolute learned position embeddings with complex rotations in the 2D plane:
   $$\mathbf{q}_m = \mathbf{R}_{\Theta, m}^d \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_{\Theta, n}^d \mathbf{W}_k \mathbf{x}_n$$
   - Naturally preserves relative distance between tokens and improves extrapolation over long sequences.

2. **RMSNorm (Root Mean Square Layer Normalization):**
   - Replaces standard LayerNorm by normalizing inputs based on root mean square without mean-centering:
   $$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$
   - ~30% computational speedup over standard LayerNorm while stabilizing gradient variance.

3. **SwiGLU Gated Linear Units:**
   - Replaces classic GELU/ReLU MLPs with a gating mechanism:
   $$\text{SwiGLU}(\mathbf{x}) = \left(\text{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}})\right)\mathbf{W}_{\text{down}}$$
   - Expands hidden representations to 2,048 dimensions with learnable bilinear gates for superior representational capacity.

4. **Fast Generation with Key-Value (KV) Caching:**
   - During autoregressive inference, computes attention projections for the **new token only** ($O(1)$ compute per step), reusing past keys and values from GPU VRAM.

5. **Advanced Decoding Controls:**
   - **Vectorized Repetition Penalty:** Dynamically down-scales logits of tokens appearing in the sliding context window.
   - **Nucleus (Top-$p$) & Top-$k$ Filtering:** Suppresses low-probability tail tokens to eliminate degenerate babbling.

---

## Specifications & Benchmarks

All benchmarks measured on consumer hardware (**NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM**):

| Metric | Specification |
| :--- | :--- |
| **Total Parameters** | **88,141,824 (88.1M)** |
| **Active Layers** | 12 Blocks |
| **Attention Heads** | 12 Heads (Head Dim: 64) |
| **Model Dimension ($d_{\text{model}}$)** | 768 |
| **SwiGLU Hidden Dimension** | 2,048 |
| **Vocabulary Size** | 4,096 (Custom SentencePiece BPE) |
| **Context Window ($T$)** | 512 Tokens |
| **Inference Generation Speed** | **56.9 tokens / second** (CUDA) |
| **Training Throughput** | **~13,400 tokens / second** (AMP fp16) |
| **VRAM Footprint (Inference)** | **~910 MB** |
| **VRAM Footprint (Training)** | **~4.2 GB** (Batch: 8, Accum: 2) |
| **Final SFT Validation Loss** | **3.0702 (Perplexity: 21.55)** |

---

## Story Studio GUI

ModernGPT-88M features **Story Studio**, a native dark-mode Desktop GUI built in Tkinter that enables anyone to generate, read, and export stories without touching the command line.

### Key GUI Features
* **Zero-Terminal Setup:** Launch via desktop shortcut or double-click `.bat`.
* **Real-Time Token Streaming:** Direct streaming to the text viewport with live word counts and tokens/sec telemetry.
* **1-Click Story Presets:** Pre-tuned prompt configurations for immediate high-coherence generation.
* **Dynamic Generation Settings:** Live sliders for temperature, repetition penalty, and target word count.
* **Multi-Scene Continuation:** Interactive prompts to direct the next scene while retaining prior narrative context.
* **1-Click Export:** Export individual chapters or full multi-scene novels to `.txt`.

---

## Qualitative Sample Output

Generated autonomously by **ModernGPT 88.1M** (Seed: 42, Temperature: 0.65, Repetition Penalty: 1.25):

> **Prompt:** *Write a short story about Shashank, a proud young mechanic working in a highway tin shed, and Rajesh Khanna, a wealthy businessman whose luxury car breaks down on the desert road.*
>
> **Generated Story:**
>
> A sudden hiss of steam echoed from the tarmac outside. A luxury black sedan coasted to a halt on the gravel shoulder, white plumes billowing from beneath its sculpted hood. The rear door opened, and Rajesh Khanna stepped out into the blinding glare. The older man was impeccably dressed in a tailored suit, his sharp gray eyes scanning the barren horizon with cold irritation.
>
> "No cell reception," Rajesh muttered, checking his phone before his gaze landed on the shed. "Young man! Do you know how to fix a ruptured radiator?"
>
> Shashank picked up his metal toolbox without answering. He walked to the stalled car, knelt upon the burning gravel, and popped the hood latch. His long fingers moved with effortless precision, inspecting the burst coolant hose and the smoking radiator cap.
>
> "The rubber split at the clamp," Shashank said quietly. "If you drive it like this, the engine block will crack. I can bypass the split with a reinforced brass coupling and high-pressure hose."
>
> "Do it," Rajesh commanded, leaning against the door frame. "I have a board meeting in the capital in three hours."
>
> For thirty minutes under the blazing sun, Shashank worked with quiet, steady discipline. His knuckles were blackened with soot, but every motion was deliberate. When he tightened the final clamp and refilled the coolant reservoir, the twin-turbo engine roared back to life with a deep, purring vibration...

---

## Project Structure

```
modern-gpt-88m/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   └── transformer.py       # Core RoPE, RMSNorm, SwiGLU, KV-Cache, ModernGPT
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretrain.py          # Pre-training loop with AMP fp16 & Cosine LR
│   │   ├── finetune_sft.py      # SFT with Prompt Loss Masking (targets = -100)
│   │   ├── dataset_packer.py    # Binary token serialization with attention masks
│   │   └── train_tokenizer.py   # SentencePiece BPE trainer
│   ├── inference/
│   │   ├── __init__.py
│   │   └── generator.py         # KV-Cache nucleus sampling engine
│   └── gui/
│       ├── __init__.py
│       └── story_studio.py      # Desktop GUI implementation
├── tests/
│   └── test_architecture.py     # Verification suite for RoPE, RMSNorm, SwiGLU
├── checkpoints/
│   └── README.md                # Model weights documentation
├── output/
│   └── sample_stories.md        # Full sample outputs
├── modern_transformer.py        # Top-level module alias
├── generate.py                  # CLI inference entrypoint
├── story_studio_gui.py          # Desktop GUI launcher
├── requirements.txt             # Python dependencies
├── .gitignore                   # Safe filtering for checkpoints & data
├── LICENSE                      # MIT License
└── README.md                    # Documentation
```

---

## Quickstart Guide

### 1. Installation
Clone the repository and install the lightweight requirements:
```bash
git clone https://github.com/your-username/modern-gpt-88m.git
cd modern-gpt-88m
pip install -r requirements.txt
```

### 2. Verify Architecture
Run the unit test suite to verify mathematical correctness:
```bash
python tests/test_architecture.py
```
Expected output:
```
[1/5] Testing RMSNorm...                               PASSED
[2/5] Testing Rotary Position Embeddings (RoPE)...    PASSED
[3/5] Testing SwiGLU Gated Feed-Forward...            PASSED
[4/5] Testing ModernGPT Forward Pass & Loss...        PASSED
[5/5] Testing KV-Cache Fast Generation...             PASSED
============================================================
ALL ARCHITECTURAL TESTS PASSED (100% OPERATIONAL)!
```

### 3. Launch Desktop GUI (Story Studio)
Launch the graphical interface:
```bash
python story_studio_gui.py
```
*(On Windows, you can also double-click `Launch_Story_Studio.bat`)*

### 4. CLI Inference
Generate stories from the terminal:
```bash
python generate.py --prompt "Write a short story about an old clockmaker and an apprentice." --max_tokens 300 --temperature 0.65
```

---

## Training Pipeline

### Phase 1: Train SentencePiece BPE Tokenizer
Trains a compact, high-efficiency 4,096-token vocabulary:
```bash
python src/training/train_tokenizer.py
```

### Phase 2: Foundation Pre-training
Pre-trains on the master narrative corpus with mixed precision and gradient accumulation:
```bash
python src/training/pretrain.py
```

### Phase 3: Instruction & SFT Fine-Tuning
Fine-tunes the model on structured narratives using **Prompt Loss Masking** (`ignore_index = -100` on prompt tokens), forcing the network to focus 100% of gradient updates on generating high-quality story continuations:
```bash
python src/training/dataset_packer.py
python src/training/finetune_sft.py
```

---

## Key Engineering Takeaways

1. **Prompt Loss Masking is Mandatory for SFT:**
   - Without masking prompt tokens (`targets[prompt] = -100`), small models spend substantial parameter capacity memorizing the prompt framing, causing infinite echo loops. Masking completely eliminated prompt repetition.
2. **Context Window Attention Capacity:**
   - At 88.1M parameters, feeding 15 characters with complex family trees into a 512-token context window overloads attention heads. Constraining stories to 2–3 active characters produced near-zero hallucination.
3. **KV-Caching Delivers $5\times$ Speedup:**
   - Moving from naive autoregressive re-computation ($O(N^2)$) to KV-cached attention ($O(1)$ per step) brought generation from ~11 tok/s to **~57 tok/s** on an RTX 3060.

---

## License
Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.
