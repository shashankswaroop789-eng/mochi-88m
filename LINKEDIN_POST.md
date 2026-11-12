# Ready-to-Post LinkedIn Announcement: 🍡 Mochi-88M

*Copy and paste the text below directly into LinkedIn. For maximum engagement, attach a 15–30 second screen recording of the **Mochi Story Studio GUI** streaming words live at 57 tok/s!*

---

🚀 **Introducing 🍡 Mochi-88M: A sweet, compact 88.1M parameter LLM built completely from scratch in PyTorch — trained on a single RTX 3060 laptop GPU.**

No Hugging Face `transformers` modeling wrappers. No pre-packaged pipelines. Just raw PyTorch tensors, linear algebra, and CUDA kernels.

Here is what went into the architecture, the hard engineering lessons learned along the way, and the benchmarks:

---

### 🧠 The Modern Architecture (LLaMA-Style)

Rather than building an older GPT-2 replica, I implemented modern 2024 architectural primitives from the ground up:

1. **Rotary Position Embeddings (RoPE):** Replaced learned position embeddings with 2D complex plane rotations ($e^{i m \theta}$), providing seamless relative position awareness and long-context generalization.
2. **RMSNorm (Pre-normalization):** Replaced LayerNorm with Root Mean Square normalization. Stripping mean-centering yielded a ~30% computational speedup and prevented gradient drift.
3. **SwiGLU Gated Feed-Forward Networks:** Implemented bilinear SiLU-gated linear projections (expanding hidden dimension to 2,048), significantly improving parameter expressiveness.
4. **Key-Value (KV) Caching:** Built an autoregressive KV-cache that updates only the newest token at each step ($O(1)$ latency per token rather than recomputing past sequence attention).
5. **Custom BPE Tokenizer:** Trained a 4,096-token SentencePiece BPE model tailored for modern English narrative flow.

---

### 🛠️ The Hard Lessons & Failure Modes (What Broke & How I Fixed It)

Building an LLM from scratch teaches you things you never learn from high-level APIs:

* **Failure #1: The Prompt Echo Loop**  
  Early fine-tuning attempts resulted in the model repeating the user's prompt in an infinite loop.  
  *Fix:* Implemented **Prompt Loss Masking** (`ignore_index = -100` on all prompt tokens in the PyTorch cross-entropy loss). Forcing gradient backprop exclusively onto continuation tokens completely eliminated prompt echoes.

* **Failure #2: Attention Head Saturation (The 15-Character Overload)**  
  When testing complex prompts with 15 characters and multi-generational family trees, the 88.1M model suffered attention confusion and hallucinated words.  
  *Fix:* Mathematical discipline. An 88M model has real attention budget boundaries. Focusing on tight 2–3 character dynamics in 300-word scenes transformed generation into coherent, literary prose.

* **Failure #3: The Inference Bottleneck**  
  Naive generation without KV-caching crawled at ~11 tokens/sec on consumer silicon.  
  *Fix:* Vectorized KV-caching combined with top-p (0.88), top-k (40), and dynamic repetition penalties boosted throughput to **56.9 tokens/sec** at just **910 MB VRAM**!

---

### 💻 Native Desktop GUI: "Mochi Story Studio"

To make the model accessible without touching a terminal, I built **Mochi Story Studio**—a standalone dark-mode desktop application (Tkinter + PyTorch backend):
* ⚡ Zero-terminal 1-click startup
* 📊 Real-time token streaming & telemetry (tok/s counter, word progress)
* 📖 Multi-scene memory and chapter exporting

---

### 📊 Benchmark Numbers at a Glance:
* **Model Name:** **Mochi-88M (🍡)**
* **Parameters:** 88,141,824 (88.1M)
* **Architecture:** 12 Layers | 12 Heads | 768 Dim | SwiGLU 2048 | RoPE | RMSNorm
* **Inference Speed:** **56.9 tokens/sec** on NVIDIA RTX 3060 Laptop (6GB VRAM)
* **Memory Footprint:** **~910 MB** (Inference) / **4.2 GB** (Training with AMP fp16)
* **Final SFT Loss:** **3.07 (Perplexity: 21.55)**

---

📂 **The entire codebase is open-source on GitHub:**  
👉 https://github.com/shashankswaroop789-eng/mochi-88m

*(Includes architecture implementation, unit tests, pre-training scripts, SFT packers, and the Mochi Story Studio GUI).*

Would love to hear thoughts, feedback, and questions from fellow ML engineers and researchers!

#MachineLearning #DeepLearning #PyTorch #LLM #GenerativeAI #ArtificialIntelligence #NLP #OpenSource #AI #SoftwareEngineering #Mochi88M
