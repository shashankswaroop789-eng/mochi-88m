"""
Train Coherent Short Story Model
Fine-tunes the clean foundation weights hf_master_gpt88m.pt into a dedicated,
coherent short story generator.
- Prompt loss masking (targets = -100 on prompt)
- 2-3 characters per story focus
- Zero word-salad, zero multi-character confusion
- Saves to hf_coherent_story_gpt88m.pt
"""

import os
import sys
import time
import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DATA_DIR)

from modern_transformer import ModernGPT

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
MAX_STEPS = 200                 # ~1.5 minutes on RTX 3060
EVAL_INTERVAL = 40
EVAL_ITERS = 15
LEARNING_RATE = 8.0e-5
MIN_LR = 1.0e-5
WARMUP_STEPS = 25
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

PRETRAINED_CHECKPOINT = os.path.join(DATA_DIR, "hf_master_gpt88m.pt")
OUTPUT_CHECKPOINT = os.path.join(DATA_DIR, "hf_coherent_story_gpt88m.pt")
DATASET_PATH = os.path.join(DATA_DIR, "coherent_sft_tokens.pt")

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
    print(" TRAINING COHERENT SHORT STORY GENERATOR (ModernGPT 88.1M)")
    print(f" Source Weights   : {PRETRAINED_CHECKPOINT}")
    print(f" Target Checkpoint: {OUTPUT_CHECKPOINT}")
    print(f" Compute Device   : {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print("=" * 80)

    # 1. Load Data
    data_dict = torch.load(DATASET_PATH, weights_only=True)
    train_data = data_dict["train"]
    val_data = data_dict["val"]
    print(f"Loaded {len(train_data):,} train samples, {len(val_data):,} val samples.")

    def get_batch(split_data):
        indices = [random.randint(0, len(split_data) - 1) for _ in range(BATCH_SIZE)]
        batch_x = torch.stack([split_data[i][0] for i in indices])
        batch_y = torch.stack([split_data[i][1] for i in indices])
        return batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)

    # 2. Load Model
    print(f"Loading clean master checkpoint: {PRETRAINED_CHECKPOINT}...")
    ckpt = torch.load(PRETRAINED_CHECKPOINT, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = ModernGPT(
        vocab_size=config.get("vocab_size", 4096),
        dim=config.get("dim", 768),
        n_layers=config.get("n_layers", 12),
        n_heads=config.get("n_heads", 12),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=0.1
    ).to(device)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    print(f"Active Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 3. Optimizer & Scaler
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, d in [('train', train_data), ('val', val_data)]:
            losses = []
            for _ in range(EVAL_ITERS):
                X, Y = get_batch(d)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits, _, _ = model(X)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-100)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    print("\nCalculating baseline loss...")
    init_loss = estimate_loss()
    print(f"Step 0 | Train Loss: {init_loss['train']:.4f} (ppl: {math.exp(init_loss['train']):.2f}) | Val Loss: {init_loss['val']:.4f} (ppl: {math.exp(init_loss['val']):.2f})")

    print("\n" + "-" * 80)
    print(" STARTING TRAINING")
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
                logits, _, _ = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-100)
                loss = loss / GRAD_ACCUM_STEPS
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if step % 20 == 0 or step == MAX_STEPS:
            elapsed = time.time() - t0
            tok_per_sec = (step * BATCH_SIZE * GRAD_ACCUM_STEPS * 512) / elapsed
            print(f"  Step {step:3d}/{MAX_STEPS} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:,.0f} tok/s | Elapsed: {elapsed:.1f}s")

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            losses = estimate_loss()
            val_ppl = math.exp(losses['val'])
            print(f"\n>>> [EVAL Step {step:3d}] Train: {losses['train']:.4f} | Val: {losses['val']:.4f} (ppl: {val_ppl:.2f})")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f"    * Saving best model to {OUTPUT_CHECKPOINT}...")
                save_payload = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': {
                        'vocab_size': 4096,
                        'dim': 768,
                        'n_layers': 12,
                        'n_heads': 12,
                        'max_seq_len': 512,
                        'dropout': 0.0
                    },
                    'val_loss': best_val_loss
                }
                torch.save(save_payload, OUTPUT_CHECKPOINT)
            print("-" * 80 + "\n")

    print("=" * 80)
    print(" TRAINING COMPLETE!")
    print(f" Best Validation Loss: {best_val_loss:.4f} (Perplexity: {math.exp(best_val_loss):.2f})")
    print(f" Checkpoint Saved to: {OUTPUT_CHECKPOINT}")
    print("=" * 80)

if __name__ == "__main__":
    main()
