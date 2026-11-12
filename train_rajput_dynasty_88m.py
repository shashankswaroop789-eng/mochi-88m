"""
Train Rajput Dynasty 88.1M - Domain & SFT Alignment
Fine-tunes the 88.1M ModernGPT on the scrubbed & rich Rajput Dynasty dataset.
Eliminates book title leaks, random chapter interruptions, and teaches the model:
- The Rajput family hierarchy (Savita, Vikram, Viraj, Avinash, Aman, Priya, Vijay, Riya, Karthik, Shashank)
- The Khanna Empire (Rajesh Khanna, Samira Khanna)
- The dramatic roadside breakdown and the shock marriage proposal
- Smooth, natural story continuation without meta-leaks.
"""

import os
import sys
import time
import math
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
BATCH_SIZE = 8                  # Micro-batch size
GRAD_ACCUM_STEPS = 2            # Effective batch size = 16 (8,192 tokens/step)
BLOCK_SIZE = 512                # Sequence context window
MAX_STEPS = 800                 # ~6-8 minutes on RTX 3060
EVAL_INTERVAL = 100             # Evaluate every 100 steps
EVAL_ITERS = 25                 # Batches to average for validation
LEARNING_RATE = 1.2e-4          # Fine-tuning LR
MIN_LR = 1.5e-5                 # Cosine decay floor
WARMUP_STEPS = 40               # Gentle warmup
WEIGHT_DECAY = 0.1              # Weight decay to prevent overfitting
GRAD_CLIP = 1.0                 # Gradient clipping

# Architecture matching pre-training (88.1M)
DIM = 768
N_LAYERS = 12
N_HEADS = 12
DROPOUT = 0.1
VOCAB_SIZE = 4096

PRETRAINED_CHECKPOINT = os.path.join(DATA_DIR, "hf_finetuned_romance_gpt88m.pt")
OUTPUT_CHECKPOINT = os.path.join(DATA_DIR, "hf_rajput_dynasty_gpt88m.pt")
TOKEN_CACHE = os.path.join(DATA_DIR, "finetune_rajput_tokens.pt")
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
    print(" TRAINING MODERN GPT 88.1M ON RAJPUT DYNASTY & CLEAN PROSE")
    print(f" Source Weights   : {PRETRAINED_CHECKPOINT}")
    print(f" Target Checkpoint: {OUTPUT_CHECKPOINT}")
    print(f" Compute Device   : {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print("=" * 80)

    train_data, val_data = prepare_data()

    # Load Foundation Weights
    print(f"\n[Model] Loading checkpoint: {PRETRAINED_CHECKPOINT}...")
    ckpt = torch.load(PRETRAINED_CHECKPOINT, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = ModernGPT(
        vocab_size=config.get("vocab_size", VOCAB_SIZE),
        dim=config.get("dim", DIM),
        n_layers=config.get("n_layers", N_LAYERS),
        n_heads=config.get("n_heads", N_HEADS),
        max_seq_len=config.get("max_seq_len", BLOCK_SIZE),
        dropout=DROPOUT
    ).to(device)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] Active Parameters: {param_count/1e6:.1f}M")

    # Optimizer with weight decay
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    # Initial loss evaluation
    print("\n[Baseline] Calculating initial loss on Rajput dataset...")
    init_loss = estimate_loss(model, train_data, val_data)
    print(f"  Step 0 | Train Loss: {init_loss['train']:.4f} (ppl: {math.exp(init_loss['train']):.2f}) | Val Loss: {init_loss['val']:.4f} (ppl: {math.exp(init_loss['val']):.2f})")

    print("\n" + "-" * 80)
    print(" STARTING FINE-TUNING RUN")
    print("-" * 80)

    t0 = time.time()
    best_val_loss = float('inf')

    for step in range(1, MAX_STEPS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            X, Y = get_batch(train_data)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss, _ = model(X, targets=Y)
                loss = loss / GRAD_ACCUM_STEPS
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if step % 20 == 0 or step == MAX_STEPS:
            elapsed = time.time() - t0
            tok_per_sec = (step * BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE) / elapsed
            print(f"  Step {step:4d}/{MAX_STEPS} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:,.0f} tok/s | Elapsed: {elapsed:.1f}s")

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            losses = estimate_loss(model, train_data, val_data)
            val_ppl = math.exp(losses['val'])
            print(f"\n>>> [EVAL Step {step:4d}] Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} (ppl: {val_ppl:.2f})")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f"    * New best validation loss! Saving to {OUTPUT_CHECKPOINT}...")
                save_dict = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': {
                        'vocab_size': VOCAB_SIZE,
                        'dim': DIM,
                        'n_layers': N_LAYERS,
                        'n_heads': N_HEADS,
                        'max_seq_len': BLOCK_SIZE,
                        'dropout': 0.0
                    },
                    'val_loss': best_val_loss
                }
                torch.save(save_dict, OUTPUT_CHECKPOINT)
            print("-" * 80 + "\n")

    print("\n" + "=" * 80)
    print(" FINE-TUNING COMPLETE!")
    print(f" Best Validation Loss: {best_val_loss:.4f} (ppl: {math.exp(best_val_loss):.2f})")
    print(f" Final Model Saved to : {OUTPUT_CHECKPOINT}")
    print("=" * 80)

if __name__ == "__main__":
    finetune()
