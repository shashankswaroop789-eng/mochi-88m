"""
🍡 Train Mochi-Novel-300M Multi-Scene Continuity Alignment
Trains the 295.1M model on scenes 1 through 6, and scene transitions (1->2, 2->3, 3->4, 4->5, 5->6):
- Resolves character confusion (Rajesh Khanna is the tycoon, Savita Rajput is the matriarch).
- Enables accurate follow-up generation for 'Generate Scene 2 with Instructions'.
- Runs live verification test with the user's exact instruction.
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

# Hyperparameters
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8             # Effective batch = 8 * 1536 = 12,288 tokens/step
MAX_STEPS = 150                  # ~2.5-3 minutes on RTX 3060
EVAL_INTERVAL = 25
EVAL_ITERS = 10
LEARNING_RATE = 4.0e-5           # Gentle alignment rate
MIN_LR = 4.0e-6
WARMUP_STEPS = 15
WEIGHT_DECAY = 0.02
GRAD_CLIP = 1.0

CHECKPOINT_PATH = os.path.join(BASE_DIR, "mochi_novel_300m.pt")
DATASET_PATH = os.path.join(BASE_DIR, "data", "multiscene_novel_sft_tokens.pt")
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
    print(" 🍡 MOCHI-NOVEL-300M MULTI-SCENE NOVEL CONTINUITY TRAINING")
    print(f" Compute Device       : {device.upper()} ({torch.cuda.get_device_name(0)})")
    print(f" Dataset              : {os.path.basename(DATASET_PATH)}")
    print(f" Target Sequence Len  : 1,536 tokens (~1,100 words per scene)")
    print(f" Total Steps          : {MAX_STEPS} (Eff. Batch: {GRAD_ACCUM_STEPS * 1536:,} tokens/step)")
    print("=" * 80)

    # 1. Load Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    # 2. Load Dataset
    print("\n[1/4] Loading Multi-Scene SFT dataset...")
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

    # 3. Load 300M Architecture
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
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = []
        for _ in range(EVAL_ITERS):
            X, Y = get_batch("val")
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss, _ = model(X, Y)
            if not torch.isnan(loss):
                losses.append(loss.item())
        model.train()
        mean_loss = sum(losses) / len(losses) if losses else 999.0
        ppl = math.exp(min(mean_loss, 20.0))
        return mean_loss, ppl

    print("\n[3/4] Estimating initial validation loss...")
    init_val_loss, init_val_ppl = estimate_loss()
    print(f"      Initial Validation Loss: {init_val_loss:.4f} | Perplexity: {init_val_ppl:.2f}", flush=True)

    # 5. Training Loop
    print("\n[4/4] Starting Multi-Scene Alignment Training...")
    best_val_loss = init_val_loss
    model.train()
    start_time = time.time()
    tokens_seen = 0

    for step in range(1, MAX_STEPS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            X, Y = get_batch("train")
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss, _ = model(X, Y)
                loss_scaled = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss_scaled).backward()
            step_loss += loss.item()
            tokens_seen += X.shape[0] * X.shape[1]

        step_loss /= GRAD_ACCUM_STEPS

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - start_time
            tok_s = tokens_seen / max(0.1, elapsed)
            vram_gb = torch.cuda.memory_allocated() / (1024**3)
            print(f"Step {step:3d}/{MAX_STEPS} | Train Loss: {step_loss:.4f} | LR: {lr:.2e} | VRAM: {vram_gb:.2f} GB | Speed: {tok_s:5.0f} tok/s", flush=True)

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            val_loss, val_ppl = estimate_loss()
            print(f" >>> [EVAL Step {step:3d}] Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"     *** New Best Validation Loss: {val_loss:.4f}! Saving checkpoint...", flush=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "step": step,
                    "model_name": "Mochi-Novel-300M Multi-Scene Masterpiece"
                }, CHECKPOINT_PATH)

    # Save final weights to guarantee latest state
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "val_loss": best_val_loss,
        "val_ppl": math.exp(min(best_val_loss, 20.0)),
        "step": MAX_STEPS,
        "model_name": "Mochi-Novel-300M Multi-Scene Masterpiece"
    }, CHECKPOINT_PATH)

    print("\n" + "=" * 80)
    print(" 🎉 MULTI-SCENE CONTINUITY TRAINING COMPLETE!")
    print(f" Final Best Val Loss: {best_val_loss:.4f}")
    print("=" * 80, flush=True)

    # 6. Live Verification of Scene 2 Instruction
    print("\n--- LIVE VERIFICATION OF USER'S EXACT SCENE 2 INSTRUCTION ---", flush=True)
    model.eval()

    prev_tail = 'Rajesh Khanna stood rooted to the gravel, staring at the boy\'s retreating back. A slow, calculating smile touched the tycoon\'s lips. He turned to his driver. "Call my personal investigator the moment we reach signal range. I want that boy\'s name, his parentage, and every breath he has taken since the day he was born."'
    instruction = "Rajesh Khanna's personal investigator uncovers that Shashank is the outcast grandson of Savita Rajput, and Rajesh decides to visit the Rajput mansion."
    
    test_prompt = f"### Prompt:\nPrevious scene ending:\n{prev_tail}\n\nNext scene:\n{instruction}\n\n### Story:\n"
    
    ids = sp.encode_as_ids(test_prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.fast_generate(
            idx,
            max_new_tokens=450,
            temperature=0.7,
            repetition_penalty=1.12,
            no_repeat_ngram_size=0,
            stop_on_eos=False
        )
    gen_tokens = out[0].tolist()[len(ids):]
    gen_text = sp.decode(gen_tokens)
    words = len(gen_text.split())
    print(f"Generated Words: {words}", flush=True)
    print("Generated Scene 2 Output:\n" + gen_text[:850] + "\n...", flush=True)

if __name__ == "__main__":
    main()
