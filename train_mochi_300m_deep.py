"""
🍡 Mochi-Novel-300M: Deep Pre-Training & Entity-Binding Engine
Trains the 295.1M parameter model on the massive 100M+ token literary corpus.
Hardware target: NVIDIA GeForce RTX 3060 (12.00 GB GDDR6 VRAM)
"""

import os
import sys
import time
import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modern_transformer import ModernGPT

# ==============================================================================
# HYPERPARAMETERS (Calibrated for 12GB RTX 3060)
# ==============================================================================
BLOCK_SIZE = 1024
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8             # Effective batch = 16 blocks * 1024 = 16,384 tokens/step
MAX_STEPS = 3000                 # Progressive deep training pass (~50M tokens processed)
EVAL_INTERVAL = 100
EVAL_ITERS = 15
LEARNING_RATE = 1.6e-4           # Optimal LR for 300M with AdamW
MIN_LR = 1.5e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

CHECKPOINT_PATH = os.path.join(BASE_DIR, "mochi_novel_300m.pt")
DATASET_PATH = os.path.join(BASE_DIR, "data", "massive_novel_100m_tokens.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "spm_novel_16k.model")

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

def main():
    print("=" * 80)
    print(" 🍡 MOCHI-NOVEL-300M DEEP PRE-TRAINING ENGINE (100M+ TOKEN CORPUS)")
    print(f" Compute Device       : {device.upper()} ({torch.cuda.get_device_name(0)})")
    print(f" Dataset Path         : {os.path.basename(DATASET_PATH)}")
    print(f" Context Length       : {BLOCK_SIZE} tokens (~750-800 words per scene)")
    print(f" Effective Batch Size : {BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE:,} tokens/step")
    print("=" * 80, flush=True)

    # 1. Load Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    # 2. Load Dataset into CPU RAM (Fast int16 tensors)
    print("\n[1/4] Loading 100M Token Corpus into Memory...", flush=True)
    t0_data = time.time()
    d = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    train_x = d["train_x"]
    train_y = d["train_y"]
    val_x = d["val_x"]
    val_y = d["val_y"]
    n_train = len(train_x)
    n_val = len(val_x)
    print(f"      Train Context Blocks : {n_train:,} ({n_train * BLOCK_SIZE:,} tokens)")
    print(f"      Val Context Blocks   : {n_val:,} ({n_val * BLOCK_SIZE:,} tokens)")
    print(f"      Loaded in {time.time()-t0_data:.2f}s", flush=True)

    def get_batch(split="train"):
        x_data = train_x if split == "train" else val_x
        y_data = train_y if split == "train" else val_y
        idxs = torch.randint(0, len(x_data), (BATCH_SIZE,))
        bx = x_data[idxs].to(device, dtype=torch.long)
        by = y_data[idxs].to(device, dtype=torch.long)
        return bx, by

    # 3. Load Model
    print("\n[2/4] Loading Mochi-Novel-300M Architecture & Weights...", flush=True)
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
    else:
        config = {}

    model = ModernGPT(
        vocab_size=config.get("vocab_size", 16384),
        dim=config.get("dim", 1024),
        n_layers=config.get("n_layers", 22),
        n_heads=config.get("n_heads", 16),
        max_seq_len=config.get("max_seq_len", 2048),
        dropout=0.05
    ).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"      Initialized from base weights: {os.path.basename(CHECKPOINT_PATH)}")
    else:
        print("      Initialized from scratch.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total Parameters: {total_params:,} ({total_params / 1e6:.1f} Million)", flush=True)

    # 4. Optimizer & Scaler
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Evaluation Helper
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch("val")
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                _, loss, _ = model(X, targets=Y)
            losses[k] = loss.item()
        val_loss = losses.mean().item()
        val_ppl = math.exp(min(val_loss, 20.0))
        model.train()
        return val_loss, val_ppl

    # Initial Validation
    print("\n[3/4] Running Initial Validation on 100M Corpus...", flush=True)
    init_val_loss, init_val_ppl = estimate_loss()
    print(f"      Initial Corpus Loss: {init_val_loss:.4f} | Perplexity: {init_val_ppl:.2f}", flush=True)

    # Training Loop
    print("\n[4/4] Starting Deep Training on 100,000,000+ Tokens...", flush=True)
    print("-" * 90)
    print("  Step | Train Loss |  Val Loss |  Val PPL |        LR |     Speed |   VRAM")
    print("-" * 90, flush=True)

    best_loss = init_val_loss
    model.train()
    t0 = time.time()

    for step in range(1, MAX_STEPS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for _ in range(GRAD_ACCUM_STEPS):
            X, Y = get_batch("train")
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                _, loss, _ = model(X, targets=Y)
                loss = loss / GRAD_ACCUM_STEPS
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Step Heartbeat every 20 steps
        if step % 20 == 0 and step % EVAL_INTERVAL != 0:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            print(f"Step {step:4d}/{MAX_STEPS} | Train Loss: {accum_loss:6.4f} | LR: {lr:8.2e} | Speed: {steps_per_sec:5.2f} st/s", flush=True)

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            val_loss, val_ppl = estimate_loss()
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"{step:6d} | {accum_loss:10.4f} | {val_loss:9.4f} | {val_ppl:8.2f} | {lr:9.2e} | {steps_per_sec:7.2f} s/s | {vram_gb:5.1f}GB", flush=True)

            if val_loss < best_loss or step == MAX_STEPS:
                best_loss = val_loss
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": 16384,
                        "dim": 1024,
                        "n_layers": 22,
                        "n_heads": 16,
                        "max_seq_len": 2048,
                        "dropout": 0.05
                    },
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "step": step,
                    "tokenizer": "spm_novel_16k.model",
                    "model_name": "Mochi-Novel-300M-Deep"
                }
                torch.save(save_dict, CHECKPOINT_PATH)
                print(f"       >>> Saved best checkpoint ({val_loss:.4f}) to {os.path.basename(CHECKPOINT_PATH)}", flush=True)

            # Live Generation Sample every 200 steps
            if step % 200 == 0:
                print("\n" + "-" * 70)
                print(f" [Step {step}] Live Coherence Check (Shashank & Samira):")
                p = "### Prompt:\nWrite a story about Shashank, a mechanic in a highway shed, and Samira, whose luxury car breaks down.\n\n### Story:\n"
                pt = sp.encode_as_ids(p)
                px = torch.tensor([pt], dtype=torch.long, device=device)
                with torch.no_grad():
                    gen_ids = model.fast_generate(px, max_new_tokens=150, temperature=0.72, top_k=40, top_p=0.88, repetition_penalty=1.12, no_repeat_ngram_size=0, prompt_len=len(pt))
                sample_text = sp.decode(gen_ids[0].tolist()[len(pt):])
                print(sample_text[:400] + "...")
                print("-" * 70 + "\n", flush=True)

    print("=" * 80)
    print(f" Deep Training Complete! Best Val Loss: {best_loss:.4f} | Saved: {CHECKPOINT_PATH}")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
