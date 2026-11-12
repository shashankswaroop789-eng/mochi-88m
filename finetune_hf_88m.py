"""
Stage 2: Fine-Tuning ModernGPT 88.1M on Pure Modern Romance
Takes the pre-trained foundation weights from hf_master_gpt88m.pt and fine-tunes
on 3.12M tokens of pure contemporary romance (Stella, Christian, Gideon, Eva).
Purges Smashwords indie artifacts while retaining full grammatical fluency.
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
# FINE-TUNING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 8                  # Micro-batch size
GRAD_ACCUM_STEPS = 2            # Effective batch size = 16 (8,192 tokens/step)
BLOCK_SIZE = 512                # Sequence context window
MAX_STEPS = 800                 # ~8-9 minutes on RTX 3060
EVAL_INTERVAL = 100             # Evaluate every 100 steps
EVAL_ITERS = 25                 # Batches to average
LEARNING_RATE = 1.0e-4          # Conservative fine-tuning LR (vs 3.5e-4 pretrain)
MIN_LR = 1.0e-5                 # Cosine decay floor
WARMUP_STEPS = 50               # Gentle warmup
WEIGHT_DECAY = 0.1              # Weight decay to prevent forgetting
GRAD_CLIP = 1.0                 # Gradient clipping

# Architecture matching pre-training (88.1M)
DIM = 768
N_LAYERS = 12
N_HEADS = 12
DROPOUT = 0.1
VOCAB_SIZE = 4096

PRETRAINED_CHECKPOINT = os.path.join(DATA_DIR, "hf_master_gpt88m.pt")
OUTPUT_CHECKPOINT = os.path.join(DATA_DIR, "hf_finetuned_romance_gpt88m.pt")
TOKEN_CACHE = os.path.join(DATA_DIR, "finetune_romance_tokens.pt")
TOKENIZER_MODEL = os.path.join(DATA_DIR, "spm_hf_master.model")

device = "cuda" if torch.cuda.is_available() else "cpu"

def prepare_data():
    if not os.path.exists(TOKEN_CACHE):
        raise FileNotFoundError(f"Token cache {TOKEN_CACHE} not found!")
    print(f"[Data] Loading fine-tuning tokens: {TOKEN_CACHE}...")
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

def finetune():
    print("=" * 80)
    print(" STAGE 2: FINE-TUNING MODERN GPT 88.1M ON PURE CONTEMPORARY ROMANCE")
    print(f" Source Weights: {PRETRAINED_CHECKPOINT}")
    print(f" Target Checkpoint: {OUTPUT_CHECKPOINT}")
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

    # Load pre-trained foundation weights
    print(f"[Init] Loading foundation weights from {PRETRAINED_CHECKPOINT}...")
    ckpt = torch.load(PRETRAINED_CHECKPOINT, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"[Init] Successfully loaded 88.1M foundation parameters!")

    # Setup AdamW optimizer
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler('cuda')

    # Initial evaluation before fine-tuning
    print("\n[Baseline Evaluation] Measuring loss on romance test set before fine-tuning...")
    initial_losses = estimate_loss(model, train_data, val_data)
    print(f" Baseline Train Loss: {initial_losses['train']:.4f} | Baseline Val Loss: {initial_losses['val']:.4f}\n")

    model.train()
    start_time = time.time()
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE
    current_step = 0

    print(f"[Fine-Tuning] Running {MAX_STEPS} steps...")
    print(f"              Tokens per step: {tokens_per_step:,} | Effective Batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    while current_step < MAX_STEPS:
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

        if current_step % 20 == 0 or current_step == 1:
            tokens_sec = tokens_per_step / max(step_time, 1e-5)
            elapsed = time.time() - start_time
            steps_left = MAX_STEPS - current_step
            eta_mins = (steps_left * step_time) / 60.0
            print(f"Step {current_step:4d}/{MAX_STEPS} | Loss: {accum_loss:6.4f} | LR: {lr:.2e} | "
                  f"{tokens_sec:6.0f} tok/s | Elapsed: {elapsed/60:4.1f}m | ETA: {eta_mins:4.1f}m", flush=True)

        if current_step % EVAL_INTERVAL == 0 or current_step == MAX_STEPS:
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
            torch.save(checkpoint, OUTPUT_CHECKPOINT)
            print(f" [Checkpoint] Saved to {OUTPUT_CHECKPOINT}", flush=True)

    total_time = time.time() - start_time
    print("=" * 80)
    print(f" FINE-TUNING COMPLETE in {total_time/60:.1f} minutes!")
    print(f" Final Fine-Tuned Model Saved: {OUTPUT_CHECKPOINT}")
    print("=" * 80)

if __name__ == "__main__":
    finetune()
