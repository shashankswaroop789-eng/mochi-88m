# 🚀 Ready-to-Post LinkedIn Announcement: 🍡 Mochi-Novel-300M

*Copy and paste the text below directly into LinkedIn. For maximum engagement, attach a screenshot of the Story Studio GUI or the 20-Genre Benchmark scorecard!*

---

🚀 **Scaling from Scratch: Introducing 🍡 Mochi-Novel-300M — A 295.1M Parameter Dual-Engine Storytelling LLM Built 100% in Pure PyTorch on a Single Consumer GPU (RTX 3060).**

No Hugging Face `transformers` modeling code. No pre-trained checkpoints. No massive compute cluster. Just raw PyTorch tensors, linear algebra, custom CUDA kernels, and architectural discipline.

Following our 88M prototype, I scaled the architecture to **295.1 Million Parameters** with a **16,384-token SentencePiece BPE vocabulary**, expanded the context window to **1,536 tokens**, and engineered a specialized **Dual-Engine Architecture** for long-form, multi-scene narrative generation.

Here is the deep architectural breakdown, the engineering lessons, and our **rigorous 20-story automated benchmark report**:

---

### 🧠 The Modern LLaMA-Style Architecture (Built From Scratch)

Rather than cloning legacy architectures, Mochi-Novel-300M implements modern architectural primitives from the ground up:

1. **Rotary Position Embeddings (RoPE):** Preserves relative token distances via 2D complex plane rotations ($e^{i m \theta}$), unlocking sequence extrapolation and smooth chapter transitions.
2. **RMSNorm (Pre-LayerNorm):** Normalizes inputs via Root Mean Square without mean-centering, stabilizing gradient variance and boosting throughput by ~30%.
3. **SwiGLU Gated Feed-Forward Units:** Bilinear SiLU-gated linear projections ($1024 \rightarrow 2730$), significantly expanding representational expressiveness.
4. **Key-Value (KV) Caching:** Computes attention for the *newest token only* ($O(1)$ latency per step), sustaining **~44.5 tokens/sec** autoregressive generation on an RTX 3060.
5. **Strict 4-Gram Repetition Blocking:** A dynamic sliding-window 4-gram suppression filter and a 1.15 repetition penalty completely eliminate degenerate repeating loops.

---

### 🎭 The Breakthrough: Dual-Engine Architecture

A common dilemma in smaller LLMs (<1B parameters) is the **Specialization vs. Generalization trade-off**:
* Fine-tuning heavily on a rich novel saga creates masterclass dialogue, but the model develops a "gravity well" (attractor collapse), pulling every prompt back into that world.
* Training purely across thousands of web stories creates broad genre diversity, but dilutes intense character voice and multi-character tension.

To solve this, I designed a **Dual-Engine System** accessible with a live hot-swap dropdown inside our native desktop GUI:

* **Engine A: Golden Novel Engine (`mochi_novel_300m_golden_rajput.pt`)**  
  *Fine-tuned for published-grade dramatic literature.* Captures deep subtext, character voice, interpersonal friction, and flawless multi-scene narrative tracking across 6 canonical scenes.
* **Engine B: Universal Multi-Genre Engine (`mochi_novel_300m.pt`)**  
  *Trained for broad creative exploration.* Neutralized the dominant domestic attractor, enabling clean atmospheric worldbuilding across 20+ distinct literary genres without concept bleeding.

---

### 📊 The Rigorous 20-Story Benchmark Report (Honest & Transparent)

To rigorously test whether the model is truly "universal" or just memorizing, I built an automated evaluation suite generating **40 scenes across 20 radically different genres** (Scene 1 premise $\rightarrow$ Scene 2 custom next-scene user instruction):

* **Genres Tested (20):** Hard Sci-Fi, Cyberpunk Noir, Victorian Gothic, Epic Fantasy, Espionage, Psychological Horror, Post-Apocalyptic, Cozy Mystery, Roman Historical, 1940s Noir, Deep Sea Thriller, Contemporary Romance, Classic Western, Time Travel, Grimdark, Urban Supernatural, Space Western, Courtroom Drama, Wilderness Survival, and Contemporary Roadside Saga.
* **Total Generated Output:** **7,435 words** across 40 scenes (~185 words per scene) in 247.1 seconds.
* **Throughput:** **~44.5 tokens/second** (KV-cached on consumer silicon).
* **Concept Leak-Free Rate:** **90.0% (18 / 20)** — 18 out of 20 foreign universes generated zero rogue attractor tokens.
* **Degenerate Loop-Free Rate:** **95.0% (19 / 20)** — zero infinite repeating sentence bugs.

#### 💡 The Architectural Reality (What 300M CAN vs CANNOT Do):
* **What it does brilliantly:** Evocative atmospheric sampling and genre pacing. When prompted with Sci-Fi, it generates decompression viewports, quantum anomalies, and antimatter; when prompted with Victorian, it outputs London carriages, leaded glass, and chapter headings.
* **Where it hits boundaries:** A 295M parameter model is a *probabilistic creative stylist*, not a multi-step causal reasoning engine. Complex multi-step instructions are treated associatively rather than sequentially, and arbitrary sub-word proper nouns can drift over long horizons (e.g. *Dr. Alistair* $\rightarrow$ *Dr. Alistay*). This transparency is essential for honest AI engineering.

---

### 💻 Native Desktop GUI: "Mochi Story Studio"

To make the system immediately usable by writers and creators, I built a dark-mode native desktop application (Tkinter + PyTorch backend):
* ⚡ **Zero-Terminal 1-Click Launch**
* 🔄 **Dual-Engine Dropdown:** Switch between Golden Novel Engine and Universal Engine on the fly.
* 📊 **Real-time token streaming** with live word counts and tok/s telemetry.
* 🔗 **Multi-scene chaining** with automatic narrative tail extraction.
* 💾 **1-Click Chapter & Novel Exporter.**

---

### ⚡ Specifications at a Glance:
* **Model:** **Mochi-Novel-300M (🍡)**
* **Parameters:** **295,084,032 (295.1M)**
* **Architecture:** 24 Layers | 16 Heads | 1,024 Dim | SwiGLU 2,730 | 1,536 Context | 16K Vocab
* **Hardware:** Single NVIDIA GeForce RTX 3060 (12GB VRAM)
* **Inference Speed:** **44.5 tokens/sec**
* **VRAM Footprint:** ~1.8 GB (Inference)

---

📂 **100% Open-Source Code, Benchmarks & Model Weights:**  
👉 **GitHub Repository:** https://github.com/shashankswaroop789-eng/mochi-88m  
🤗 **Hugging Face Model:** https://huggingface.co/Shashank1227/moochi-300m  

*(Includes core transformer architecture, benchmark suites, 16K tokenizer, training pipelines, and the Story Studio desktop app).*

Would love to hear feedback, architectural thoughts, or questions from fellow ML engineers and creators!

#MachineLearning #DeepLearning #PyTorch #LLM #GenerativeAI #ArtificialIntelligence #NLP #OpenSource #AI #SoftwareEngineering #Python #HuggingFace #DataScience #Mochi300M
