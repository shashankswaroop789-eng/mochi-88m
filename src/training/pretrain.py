"""
Trains ModernGPT 88.1M on the HuggingFace Master Story Corpus (113 MB, ~30M tokens).
Features:
- 88.1 Million Parameters (12 layers, 12 heads, 768 dim, SwiGLU 2048, RoPE, RMSNorm)
- Regularized to prevent memorization (Weight Decay 0.1, Dropout 0.1)
- Cosine Learning Rate Decay with Warmup
- FP16 Mixed Precision for 2x speedup on RTX 3060
- Robust Checkpointing with resume capability
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm

# Force utf-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DATA_DIR)

from modern_transformer import ModernGPT

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 8                  # Micro-batch size (keeps VRAM ~5-6 GB on RTX 3060)
GRAD_ACCUM_STEPS = 2            # Effective batch size = 16 (8,192 tokens/step)
BLOCK_SIZE = 512                # Sequence context window
MAX_STEPS = 5000                # ~45-50 minutes on RTX 3060
EVAL_INTERVAL = 250             # Evaluate train/val loss every 250 steps
EVAL_ITERS = 30                 # Batches to average during evaluation
LEARNING_RATE = 3.5e-4          # Peak learning rate
MIN_LR = 3.5e-5                 # Minimum learning rate after cosine decay
WARMUP_STEPS = 300              # Linear warmup steps
WEIGHT_DECAY = 0.1              # AdamW weight decay (crucial against overfitting)
GRAD_CLIP = 1.0                 # Gradient norm clipping

# 88.1 Million Parameters
DIM = 768
N_LAYERS = 12
N_HEADS = 12
DROPOUT = 0.1
VOCAB_SIZE = 4096

TOKENIZER_MODEL = os.path.join(DATA_DIR, "spm_hf_master.model")
TOKEN_CACHE = os.path.join(DATA_DIR, "hf_master_tokens.pt")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "hf_master_gpt88m.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

def prepare_data():
    if not os.path.exists(TOKEN_CACHE):
        raise FileNotFoundError(f"Tokens cache not found at {TOKEN_CACHE}! Please run tokenize_master_corpus.py first.")
    print(f"[Data] Loading pre-tokenized cache: {TOKEN_CACHE}...")
    tokens = torch.load(TOKEN_CACHE, weights_only=True)
    n = int(0.95 * len(tokens))
    train_data = tokens[:n]
    val_data = tokens[n:]
    print(f"[Data] Total tokens: {len(tokens):,} | Train: {len(train_data):,} | Val: {len(val_data):,}")
    return train_data, val_data

def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split, d in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(d)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss, _ = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

def train(resume=False, start_step=0, target_steps=MAX_STEPS):
    print("=" * 80)
    print(" MODERN GPT 88.1M - HUGGINGFACE MASTER STORY PRE-TRAINING")
    print(f" Target Checkpoint: {CHECKPOINT_PATH}")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    train_data, val_data = prepare_data()

    model = ModernGPT(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=BLOCK_SIZE,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Scaled Architecture: {total_params:,} parameters ({total_params/1e6:.1f}M)")

    # Separate decay / no-decay parameters
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler('cuda')

    current_step = 0
    if resume and os.path.exists(CHECKPOINT_PATH):
        print(f"[Resume] Loading checkpoint from {CHECKPOINT_PATH}...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        current_step = start_step if start_step > 0 else ckpt.get('step', 0)
        print(f"[Resume] Successfully resumed at step {current_step}")

    model.train()
    start_time = time.time()
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE

    print(f"[Training] Beginning training from step {current_step} to {target_steps}...")
    print(f"           Tokens per step: {tokens_per_step:,} | Effective Batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    while current_step < target_steps:
        t0 = time.time()
        lr = get_lr(current_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            xb, yb = get_batch(train_data)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits, loss, _ = model(xb, targets=yb)
                loss = loss / GRAD_ACCUM_STEPS
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        step_time = time.time() - t0
        current_step += 1

        if current_step % 25 == 0 or current_step == 1:
            tokens_sec = tokens_per_step / max(step_time, 1e-5)
            elapsed = time.time() - start_time
            steps_left = target_steps - current_step
            eta_mins = (steps_left * step_time) / 60.0
            print(f"Step {current_step:4d}/{target_steps} | Loss: {accum_loss:6.4f} | LR: {lr:.2e} | "
                  f"{tokens_sec:6.0f} tok/s | Elapsed: {elapsed/60:4.1f}m | ETA: {eta_mins:4.1f}m", flush=True)

        if current_step % EVAL_INTERVAL == 0 or current_step == target_steps:
            losses = estimate_loss(model, train_data, val_data)
            print("-" * 80, flush=True)
            print(f" >>> EVAL @ STEP {current_step:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}", flush=True)
            print("-" * 80, flush=True)

            checkpoint = {
                'step': current_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': losses['train'],
                'val_loss': losses['val'],
                'config': {
                    'dim': DIM,
                    'n_layers': N_LAYERS,
                    'n_heads': N_HEADS,
                    'vocab_size': VOCAB_SIZE,
                    'max_seq_len': BLOCK_SIZE,
                    'dropout': DROPOUT
                }
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f" [Checkpoint] Saved to {CHECKPOINT_PATH}", flush=True)

    total_time = time.time() - start_time
    print("=" * 80)
    print(f" TRAINING COMPLETE in {total_time/60:.1f} minutes!")
    print(f" Final Model Saved: {CHECKPOINT_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--start_step", type=int, default=0, help="Step to resume from")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="Target total steps")
    args = parser.parse_args()

    train(resume=args.resume, start_step=args.start_step, target_steps=args.steps)
