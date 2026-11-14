"""
🍡 Mochi-Novel-300M: Long-Form Multi-Character Alignment Training
Trains the model on real 600-1,000 word stories so it masters sustained novel pacing,
rich environmental description, organic dialogue, and multi-character consistency.
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
# SFT HYPERPARAMETERS (Calibrated for 12GB RTX 3060)
# ==============================================================================
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8             # Effective batch = 8 * 1536 = 12,288 tokens/step
MAX_STEPS = 250                  # ~4-5 minutes on RTX 3060
EVAL_INTERVAL = 50
EVAL_ITERS = 10
LEARNING_RATE = 3.5e-5           # Gentle alignment rate
MIN_LR = 5.0e-6
WARMUP_STEPS = 30
WEIGHT_DECAY = 0.02
GRAD_CLIP = 1.0

CHECKPOINT_PATH = os.path.join(BASE_DIR, "mochi_novel_300m.pt")
DATASET_PATH = os.path.join(BASE_DIR, "data", "longform_novel_sft_tokens.pt")
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
    print(" 🍡 MOCHI-NOVEL-300M LONG-FORM CHAPTER PACING TRAINING")
    print(f" Compute Device       : {device.upper()} ({torch.cuda.get_device_name(0)})")
    print(f" Long-Form Dataset    : {os.path.basename(DATASET_PATH)}")
    print(f" Base Weights         : {os.path.basename(CHECKPOINT_PATH)}")
    print(f" Target Sequence Len  : 1,536 tokens (~1,100 words per scene)")
    print("=" * 80)

    # 1. Load Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    # 2. Load Dataset
    print("\n[1/4] Loading Long-Form SFT dataset...")
    d = torch.load(DATASET_PATH, weights_only=False)
    train_data = d["train"]
    val_data = d["val"]
    print(f"      Train Samples: {len(train_data):,} | Val Samples: {len(val_data):,}")

    def get_batch(split="train"):
        data = train_data if split == "train" else val_data
        idxs = random.sample(range(len(data)), BATCH_SIZE)
        x = torch.stack([data[i][0] for i in idxs]).to(device)
        y = torch.stack([data[i][1] for i in idxs]).to(device)
        return x, y

    # 3. Load 300M Model
    print("\n[2/4] Loading Mochi-Novel-300M Architecture...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = ModernGPT(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Loaded {total_params / 1e6:.1f} Million Parameters successfully.")

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

    # Initial Validation Check
    print("\n[3/4] Running Initial Validation on Long-Form Stories...")
    init_val_loss, init_val_ppl = estimate_loss()
    print(f"      Initial Long-Form Loss: {init_val_loss:.4f} | Perplexity: {init_val_ppl:.2f}")

    # Training Loop
    print("\n[4/4] Starting Long-Form Chapter Alignment...")
    print("-" * 80)
    print("  Step | Train Loss |  Val Loss |  Val PPL |        LR |     Speed")
    print("-" * 80)

    best_loss = float("inf")
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

        if step % 25 == 0 and step % EVAL_INTERVAL != 0:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            print(f"Step {step:4d}/{MAX_STEPS} | Train Loss: {accum_loss:6.4f} | LR: {lr:8.2e} | Speed: {steps_per_sec:5.2f} st/s", flush=True)

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            val_loss, val_ppl = estimate_loss()
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            print(f"{step:6d} | {accum_loss:10.4f} | {val_loss:9.4f} | {val_ppl:8.2f} | {lr:9.2e} | {steps_per_sec:7.2f} s/s", flush=True)

            if val_loss < best_loss or step == MAX_STEPS:
                best_loss = val_loss
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "step": step,
                    "tokenizer": "spm_novel_16k.model",
                    "model_name": "Mochi-Novel-300M-LongForm"
                }
                torch.save(save_dict, CHECKPOINT_PATH)
                print(f"       >>> Saved aligned Long-Form checkpoint ({val_loss:.4f}) to {os.path.basename(CHECKPOINT_PATH)}", flush=True)

    print("-" * 80)
    print(f" Long-Form Training Complete! Best Val Loss: {best_loss:.4f} | Saved: {CHECKPOINT_PATH}")
    print("=" * 80)

    # Live Verification Test: 500 Words on Shashank & Samira
    print("\n" + "=" * 80)
    print(" 🍡 VERIFICATION: 500-Word Scene Generation (Shashank & Samira)")
    print("=" * 80)
    p1 = "### Prompt:\nWrite a story about Shashank, a proud young mechanic working in a highway tin shed, and Samira Khanna, whose luxury car breaks down on the desert road.\n\n### Story:\n"
    t1 = sp.encode_as_ids(p1)
    idx1 = torch.tensor([t1], dtype=torch.long, device=device)
    with torch.no_grad():
        out1 = model.fast_generate(
            idx1,
            max_new_tokens=450,
            temperature=0.72,
            top_k=40,
            top_p=0.88,
            repetition_penalty=1.12,
            no_repeat_ngram_size=0,
            prompt_len=len(t1)
        )
    gen_text = sp.decode(out1[0].tolist()[len(t1):])
    print(f"Generated ({len(gen_text.split())} words):\n")
    print(gen_text)
    print("=" * 80)

    # Live Verification Test 2: Arbitrary Unseen Characters (Marcus & Chloe)
    print("\n" + "=" * 80)
    print(" 🍡 VERIFICATION 2: Arbitrary Unseen Characters (Marcus & Chloe)")
    print("=" * 80)
    p2 = "### Prompt:\nWrite a story about Marcus, an old detective, and Chloe, an undercover journalist, investigating a dark warehouse near the docks.\n\n### Story:\n"
    t2 = sp.encode_as_ids(p2)
    idx2 = torch.tensor([t2], dtype=torch.long, device=device)
    with torch.no_grad():
        out2 = model.fast_generate(
            idx2,
            max_new_tokens=400,
            temperature=0.72,
            top_k=40,
            top_p=0.88,
            repetition_penalty=1.12,
            no_repeat_ngram_size=0,
            prompt_len=len(t2)
        )
    gen_text2 = sp.decode(out2[0].tolist()[len(t2):])
    print(f"Generated ({len(gen_text2.split())} words):\n")
    print(gen_text2)
    print("=" * 80)

if __name__ == "__main__":
    main()
