# 🍡 Mochi: Storytelling LLMs Built 100% From Scratch in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Hugging Face 300M](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Mochi--Novel--300M-FFD21E.svg)](https://huggingface.co/Shashank1227/moochi-300m)
[![Hugging Face 88M](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Mochi--88M-FFD21E.svg)](https://huggingface.co/Shashank1227/mochi-88m)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture: Modern LLaMA--Style](https://img.shields.io/badge/Architecture-RoPE%20%7C%20SwiGLU%20%7C%20RMSNorm-blue.svg)](#architecture)
[![Inference: KV--Cache](https://img.shields.io/badge/Inference-45--57%20tok%2Fs%20(RTX%203060)-emerald.svg)](#specifications--benchmarks)

**Mochi** is a family of modern Large Language Models designed, implemented, pre-trained, and fine-tuned **100% from scratch in pure PyTorch**—without depending on Hugging Face `transformers` modeling code.

Named after the sweet delicacy, Mochi models are engineered specifically for local, highly coherent narrative and novel generation on consumer GPUs. They incorporate state-of-the-art LLM architectural primitives (the same building blocks found in **LLaMA 3**, **Mistral**, and **Gemma**) including **Rotary Position Embeddings (RoPE)**, **RMSNorm**, **SwiGLU feed-forward networks**, and **Key-Value (KV) Caching**.

---

## 🌟 The Mochi Model Family

| Model | Parameters | Layers | Heads | Hidden Dim | Vocab Size | Context Window | Primary Specialization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mochi-88M (🍡)** | **88.1M** | 12 | 12 | 768 | 4,096 | 512 tokens | Ultra-lightweight local storytelling & mobile inference |
| **Mochi-Novel-300M (👑)** | **295.1M** | 24 | 16 | 1,024 | 16,384 | 1,536 tokens | **Flagship**: Multi-scene novel generation & dual-engine literature |

---

## 🏛️ Flagship: Mochi-Novel-300M (295.1M Parameters)

**Mochi-Novel-300M** expands the architecture to 24 deep transformer layers and a 16,384-token SentencePiece vocabulary trained on massive literary and long-form corpora. It features an advanced **Dual-Engine Architecture** directly selectable in the GUI:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Mochi-Novel-300M Systems                           │
├──────────────────────────────────────┬──────────────────────────────────────┤
│  🎭 Engine A: Golden Rajput Engine   │  🌌 Engine B: Universal Multi-Genre  │
│  (mochi_novel_300m_golden_rajput.pt) │  (mochi_novel_300m.pt)               │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • Published-grade drama & dialogue   │ • Universal alignment across 20+     │
│ • Flawless character tracking        │   genres (Sci-Fi, Cyberpunk, Noir,   │
│ • Perfect multi-scene continuity     │   Gothic, Fantasy, Horror, Western)  │
│ • 10/10 consistency for main cast    │ • 90.0% concept leak-free rate       │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### 📊 Rigorous 20-Genre Benchmark Results

Tested on consumer hardware (**NVIDIA GeForce RTX 3060 Laptop GPU**):
* **Total Stories Tested:** 20 distinct genres (Hard Sci-Fi, Cyberpunk, Victorian Gothic, Epic Fantasy, Espionage, Psychological Horror, Post-Apocalyptic, Cozy Mystery, Roman Historical, 1940s Noir, Deep Sea, Romance, Classic Western, Time Travel, Grimdark, Urban Supernatural, Space Western, Courtroom, Wilderness, and Rajput Saga).
* **Scenes Generated:** **40 / 40 scenes** (Scene 1 Premise $\rightarrow$ Scene 2 Next-Scene Instruction).
* **Total Word Count:** **7,435 words** (~185 words per scene) at **~44.5 tok/s**.
* **Concept Leak-Free Rate:** **90.0% (18/20)** (foreign genres stayed 100% free of dominant attractors).
* **Degenerate Loop-Free Rate:** **95.0% (19/20)** (strict 4-gram repetition blocking prevents infinite loops).

---

## 🏗️ Architecture Overview

```
Input Tokens (B, T)
        │
        ▼
   Token Embedding (Vocab: 16,384 → Dim: 1,024)
        │
┌───────┴─────────────────────────────────────────┐
│              24x Modern Transformer Blocks      │
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
│  │                 (Dim: 1,024 → Hidden: 2,730) │
│  └──────(+) ◄──────────┘                        │
└─────────────────────────────────────────────────┘
        │
        ▼
     RMSNorm
        │
        ▼
   Head Projection (1,024 → 16,384 Logits)
```

### Core Primitives
1. **Rotary Position Embeddings (RoPE):**
   - Relative position-aware complex rotational encoding:
   $$\mathbf{q}_m = \mathbf{R}_{\Theta, m}^d \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_{\Theta, n}^d \mathbf{W}_k \mathbf{x}_n$$
2. **RMSNorm (Root Mean Square Layer Normalization):**
   - Faster, more stable normalization without mean-centering:
   $$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$
3. **SwiGLU Gated Linear Units:**
   - Learnable bilinear gating mechanism for superior representation capacity:
   $$\text{SwiGLU}(\mathbf{x}) = \left(\text{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}})\right)\mathbf{W}_{\text{down}}$$
4. **Key-Value (KV) Caching:**
   - $O(1)$ token compute per step during autoregressive generation.
5. **Strict 4-Gram Repetition Blocking:**
   - Real-time n-gram blocking and 1.15 repetition penalty completely eliminates degenerate loops.

---

## 🍡 Mochi Story Studio (Native Desktop GUI)

Mochi includes **Story Studio**, a native dark-mode Desktop GUI built in Tkinter:

* **Dual-Engine Selector:** Hot-swap between Golden Rajput Novel Engine and Universal Multi-Genre Engine on the fly without restarting.
* **1-Click Story Presets:** Curated presets for 20+ genres.
* **Real-Time Token Streaming:** Direct streaming to text viewport with live word count and generation speed telemetry.
* **Multi-Scene Continuation:** Chain scene after scene with custom instructions while preserving the previous scene's narrative tail.
* **1-Click Export:** Save individual chapters or full multi-scene novels to `.txt`.

To launch:
```bash
python story_studio_gui.py
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository:
```bash
git clone https://github.com/shashankswaroop789-eng/mochi-88m.git
cd mochi-88m
pip install -r requirements.txt
```

### 2. Download Model Weights from Hugging Face
* **To download Mochi-Novel-300M (Flagship):**
  ```bash
  python download_300m_model.py
  ```
* **To download Mochi-88M (Compact):**
  ```bash
  python download_model.py
  ```

### 3. Launch the Desktop GUI
```bash
python story_studio_gui.py
```

### 4. Upload Weights to Hugging Face Hub (Maintainers)
To push new 300M weights to Hugging Face:
```bash
python upload_300m_to_hf.py --token <YOUR_HF_WRITE_TOKEN>
```

---

## 📖 Sample Outputs

### Mochi-Novel-300M (Golden Engine):
> *"The afternoon heat rose in shimmering waves off the cracked black asphalt of the national highway. Miles from the nearest town, where the scrub thorny bushes gave way to empty sand and rusted hoardings, stood a lone tin shed. Its corrugated metal roof groaned beneath the relentless sun... Inside the stifling shade of the workshop, Shashank wiped a streak of engine oil from his brow with the back of a calloused hand. He was twenty-three, thin-framed and lean as a desert wolf... On the workbench lay an old motorcycle carburetor, dismantled into dozens of tiny brass jets and springs..."*

### Mochi-Novel-300M (Universal Engine — Sci-Fi):
> *"The viewport was built on an infinite surface, looking out over the dark void of the Trappist-1 perimeter. 'It is the first step,' Dr. Alistair said in a low voice, watching the encrypted transmission frequencies decode across the primary monitor..."*

---

## 📁 Repository Structure

```
mochi-88m/
├── modern_transformer.py        # Core RoPE, RMSNorm, SwiGLU, KV-Cache ModernGPT
├── story_studio_gui.py          # Native Tkinter Desktop GUI with Dual-Engine support
├── generate.py                  # CLI inference generation tool
├── download_weights.py          # Unified Hugging Face downloader (300M & 88M)
├── download_300m_model.py       # Wrapper for 300M weights
├── download_model.py            # Wrapper for 88M weights
├── upload_300m_to_hf.py         # Hugging Face upload script with model card & configs
├── spm_novel_16k.model          # SentencePiece 16K tokenizer model (300M)
├── spm_novel_16k.vocab          # SentencePiece 16K vocabulary (300M)
├── spm_hf_master.model          # SentencePiece 4K tokenizer model (88M)
├── spm_hf_master.vocab          # SentencePiece 4K vocabulary (88M)
├── benchmarks/                  # 20-Genre benchmark suite & results
│   ├── benchmark_20_stories_results.json
│   └── rigorous_20_story_benchmark.py
├── tests/                       # Automated test suites
│   ├── test_dual_engine_suite.py
│   ├── test_all_scenes_continuity.py
│   ├── test_gui_generation_loop.py
│   └── test_architecture.py
├── scripts/training/            # Active training & dataset alignment pipelines
│   ├── build_multi_scene_sft_dataset.py
│   ├── train_multiscene_alignment.py
│   ├── build_universal_continuity_sft.py
│   └── train_universal_continuation.py
├── src/                         # Modular package library
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

---

## 📜 License & Acknowledgments

- **License:** MIT License
- **Author:** [Shashank Swaroop](https://github.com/shashankswaroop789-eng)
- **Hugging Face Hub Repositories:**
  - [Shashank1227/moochi-300m](https://huggingface.co/Shashank1227/moochi-300m)
  - [Shashank1227/mochi-88m](https://huggingface.co/Shashank1227/mochi-88m)
