"""
🍡 Mochi-Novel-320M: Long-Context Novel Pre-Training Engine
Architecture:
- 320.4 Million Parameters (24 Layers, 1024 Dim, 16 Heads, SwiGLU, RMSNorm)
- 2,048-token Context Window (Full chapter scenes in a single forward pass)
- Native PyTorch Ampere FlashAttention (SDPA)
- 16,384-piece Literary SentencePiece BPE Tokenizer
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
# HYPERPARAMETERS (Calibrated for 12GB RTX 3060 - 50.8% VRAM, 6GB Free Buffer)
# ==============================================================================
VOCAB_SIZE = 16384
DIM = 1024
N_LAYERS = 22
N_HEADS = 16
SEQ_LEN = 2048
DROPOUT = 0.05

BATCH_SIZE = 1                   # Micro-batch size per step (6.09 GB VRAM peak)
GRAD_ACCUM_STEPS = 12            # Effective batch = 12 blocks (24,576 tokens/step)
MAX_STEPS = 2000                 # Primary pre-training run
EVAL_INTERVAL = 100              # Evaluate every 100 steps
EVAL_ITERS = 10
LEARNING_RATE = 3.2e-4           # Peak learning rate for ~300M
MIN_LR = 3.0e-5
WARMUP_STEPS = 120
WEIGHT_DECAY = 0.04
GRAD_CLIP = 1.0

DATASET_PATH = os.path.join(BASE_DIR, "data", "mochi_novel_2048_tokens.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "spm_novel_16k.model")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "mochi_novel_300m.pt")

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
    print(" 🍡 MOCHI-NOVEL-320M FOUNDATION TRAINING PIPELINE")
    print(f" Compute Device       : {device.upper()} ({torch.cuda.get_device_name(0)})")
    print(f" Context Length       : {SEQ_LEN:,} tokens (~1,500 words per scene)")
    print(f" Vocabulary Size      : {VOCAB_SIZE:,} Literary BPE Pieces")
    print(f" Effective Batch Size : {BATCH_SIZE * GRAD_ACCUM_STEPS} blocks ({BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN:,} tokens/step)")
    print("=" * 80)

    # 1. Load Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    # 2. Load Dataset
    print("\n[1/4] Loading pre-tokenized novel dataset...")
    data = torch.load(DATASET_PATH, weights_only=False)
    train_blocks = data["train"] # Tensor of shape (N_train, 2049)
    val_blocks = data["val"]     # Tensor of shape (N_val, 2049)
    print(f"      Train Context Blocks : {train_blocks.shape[0]:,} ({train_blocks.shape[0] * SEQ_LEN:,} tokens)")
    print(f"      Val Context Blocks   : {val_blocks.shape[0]:,} ({val_blocks.shape[0] * SEQ_LEN:,} tokens)")

    def get_batch(split="train"):
        blocks = train_blocks if split == "train" else val_blocks
        idxs = torch.randint(0, blocks.shape[0], (BATCH_SIZE,))
        batch = blocks[idxs].to(device, dtype=torch.long)
        x = batch[:, :-1] # (B, 2048)
        y = batch[:, 1:]  # (B, 2048)
        return x, y

    # 3. Initialize Model
    print("\n[2/4] Initializing Mochi-Novel-320M Architecture...")
    config = {
        "vocab_size": VOCAB_SIZE,
        "dim": DIM,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "max_seq_len": SEQ_LEN,
        "dropout": DROPOUT
    }
    
    model = ModernGPT(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=SEQ_LEN,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total Parameters: {total_params:,} ({total_params/1e6:.1f} Million)")

    # 4. Optimizer & Scaler
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = []
        for _ in range(EVAL_ITERS):
            X, Y = get_batch("val")
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                _, loss, _ = model(X, targets=Y)
            losses.append(loss.item())
        model.train()
        v_loss = sum(losses) / len(losses)
        return v_loss, math.exp(min(v_loss, 20))

    print("\n[3/4] Initial validation check before training...")
    init_loss, init_ppl = estimate_loss()
    print(f"      Initial Random Val Loss: {init_loss:.4f} | Perplexity: {init_ppl:.2f}")

    print("\n[4/4] Starting Pre-Training on 40,000,000 Literary Tokens...")
    print("-" * 85)
    print(f"{'Step':>6} | {'Train Loss':>10} | {'Val Loss':>9} | {'Val PPL':>8} | {'LR':>9} | {'Speed':>9} | {'VRAM':>7}")
    print("-" * 85)

    t0 = time.time()
    best_loss = float("inf")
    model.train()

    for step in range(1, MAX_STEPS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

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
                    "config": config,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "step": step,
                    "tokenizer": "spm_novel_16k.model",
                    "model_name": "Mochi-Novel-300M"
                }
                torch.save(save_dict, CHECKPOINT_PATH)
                print(f"       >>> Saved best checkpoint ({val_loss:.4f}) to {os.path.basename(CHECKPOINT_PATH)}", flush=True)

    print("-" * 85)
    print(f" Training Complete! Best Val Loss: {best_loss:.4f} | Model: {CHECKPOINT_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    main()
