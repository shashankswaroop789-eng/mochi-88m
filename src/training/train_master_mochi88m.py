"""
Train Master Mochi-88M on 30,000 Stories
Fine-tunes the clean foundation model (hf_master_gpt88m.pt) on the 30,000-story
master dataset with prompt loss masking.
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

# Force utf-8 for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modern_transformer import ModernGPT

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 8                  # Micro-batch size
GRAD_ACCUM_STEPS = 2            # Effective batch size = 16 (8,192 tokens/step)
MAX_STEPS = 1200                # ~12-14 minutes on RTX 3060
EVAL_INTERVAL = 150             # Evaluate every 150 steps
EVAL_ITERS = 20
LEARNING_RATE = 1.2e-4          # SFT peak learning rate
MIN_LR = 1.5e-5
WARMUP_STEPS = 80
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

PRETRAINED_CHECKPOINT = os.path.join(BASE_DIR, "hf_master_gpt88m.pt")
OUTPUT_CHECKPOINT = os.path.join(BASE_DIR, "hf_mochi88m_master.pt")
DATASET_PATH = os.path.join(DATA_DIR, "large_story_sft_tokens.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "spm_hf_master.model")

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
    print(" TRAINING MASTER MOCHI-88M ON 30,000 STORIES")
    print(f" Base Weights     : {PRETRAINED_CHECKPOINT}")
    print(f" Target Checkpoint: {OUTPUT_CHECKPOINT}")
    print(f" Dataset Cache    : {DATASET_PATH}")
    print(f" Compute Device   : {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print("=" * 80)

    # 1. Load Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    # 2. Load Dataset
    print("Loading pre-tokenized SFT dataset cache...")
    data_dict = torch.load(DATASET_PATH, weights_only=True)
    train_data = data_dict["train"]
    val_data = data_dict["val"]
    print(f"Loaded {len(train_data):,} training samples, {len(val_data):,} validation samples.")

    def get_batch(split_data):
        indices = [random.randint(0, len(split_data) - 1) for _ in range(BATCH_SIZE)]
        batch_x = torch.stack([split_data[i][0] for i in indices])
        batch_y = torch.stack([split_data[i][1] for i in indices])
        return batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)

    # 3. Load Foundation Model
    print(f"Loading foundation checkpoint: {PRETRAINED_CHECKPOINT}...")
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

    # 4. Optimizer & Scaler
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # 5. Validation Function
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split_name, split_data in [("train", train_data), ("val", val_data)]:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split_data)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    _, loss, _ = model(X, Y)
                losses[k] = loss.item()
            out[split_name] = losses.mean().item()
        model.train()
        return out

    # 6. Sample Generation Test
    def test_sample_generation():
        model.eval()
        test_prompt = "Write a short story about an old clockmaker who finds an unusual brass key inside an antique watch."
        fmt = f"### Prompt:\n{test_prompt}\n\n### Story:\n"
        token_ids = sp.encode_as_ids(fmt)
        idx = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.fast_generate(idx, max_new_tokens=100, temperature=0.7, top_k=40)
        gen_tokens = out[0].tolist()[len(token_ids):]
        print(f"\n[Generation Preview]:\n{sp.decode(gen_tokens)}\n")
        model.train()

    print("\nStarting Training Loop...")
    t0 = time.time()
    best_val_loss = float("inf")

    for step in range(1, MAX_STEPS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            X, Y = get_batch(train_data)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                _, loss, _ = model(X, Y)
                loss = loss / GRAD_ACCUM_STEPS
            step_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % 25 == 0 or step == 1:
            tok_per_sec = (step * BATCH_SIZE * GRAD_ACCUM_STEPS * 512) / max(0.1, time.time() - t0)
            elapsed_m = (time.time() - t0) / 60
            print(f"Step {step:4d}/{MAX_STEPS} | Loss: {step_loss:.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:,.0f} tok/s | Elapsed: {elapsed_m:.1f}m")

        # Evaluation
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            eval_start = time.time()
            losses = estimate_loss()
            ppl = math.exp(min(20, losses['val']))
            print(f"\n>>> [EVAL @ Step {step:4d}] Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} (PPL: {ppl:.2f})")
            test_sample_generation()

            # Save best checkpoint
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f">>> [SAVING BEST] Val loss improved to {best_val_loss:.4f}! Saving checkpoint...")
                save_payload = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "train_loss": losses['train'],
                    "val_loss": losses['val'],
                    "config": {
                        "vocab_size": config.get("vocab_size", 4096),
                        "dim": config.get("dim", 768),
                        "n_layers": config.get("n_layers", 12),
                        "n_heads": config.get("n_heads", 12),
                        "max_seq_len": config.get("max_seq_len", 512),
                        "dropout": 0.0
                    }
                }
                torch.save(save_payload, OUTPUT_CHECKPOINT)
                print(f">>> Saved to: {OUTPUT_CHECKPOINT}\n")

    total_time = (time.time() - t0) / 60
    print("=" * 80)
    print(f" TRAINING COMPLETE in {total_time:.1f} minutes!")
    print(f" Best Val Loss    : {best_val_loss:.4f}")
    print(f" Final Model Saved: {OUTPUT_CHECKPOINT}")
    print("=" * 80)

if __name__ == "__main__":
    main()
