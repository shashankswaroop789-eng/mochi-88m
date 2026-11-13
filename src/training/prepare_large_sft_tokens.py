"""
Tokenize and Pack Master Story Dataset into Tensor Cache
Features:
- SentencePiece BPE Tokenizer (spm_hf_master.model)
- Strict Prompt Loss Masking (targets[prompt] = -100)
- Sequence length = 512
- Saves train/val splits to data/large_story_sft_tokens.pt
"""

import os
import sys
import json
import time
import torch
import sentencepiece as spm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "master_story_sft_dataset.jsonl")
TOKENIZER_PATH = os.path.join(BASE_DIR, "spm_hf_master.model")
OUTPUT_PT = os.path.join(DATA_DIR, "large_story_sft_tokens.pt")

SEQ_LEN = 512
PAD_ID = 0

def main():
    print("=" * 70)
    print(" Tokenizing & Packing 30,000 Stories for Mochi-88M SFT")
    print(f" Source: {INPUT_FILE}")
    print(f" Output: {OUTPUT_PT}")
    print("=" * 70)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    print(f"Loaded tokenizer with vocab size: {sp.get_piece_size():,}")

    t0 = time.time()
    all_samples = []
    skipped = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 5000 == 0 and idx > 0:
                print(f"Processed {idx:,} stories... ({len(all_samples):,} packed, {time.time()-t0:.1f}s)")
            try:
                item = json.loads(line)
                prompt = item.get("prompt", "").strip()
                story = item.get("story", "").strip()
                if not prompt or not story:
                    continue

                prompt_str = f"### Prompt:\n{prompt}\n\n### Story:\n"
                prompt_ids = sp.encode_as_ids(prompt_str)
                story_ids = sp.encode_as_ids(story)

                # Total tokens
                total_ids = prompt_ids + story_ids
                if len(total_ids) < 60:
                    skipped += 1
                    continue

                # Truncate to SEQ_LEN if too long
                if len(total_ids) > SEQ_LEN:
                    total_ids = total_ids[:SEQ_LEN]

                # Create input_ids and targets
                input_ids = total_ids[:-1]
                targets = total_ids[1:]

                # Mask prompt tokens in targets with -100 (ignore_index)
                prompt_len = min(len(prompt_ids) - 1, len(targets))
                targets_tensor = torch.tensor(targets, dtype=torch.long)
                targets_tensor[:prompt_len] = -100

                # Pad to SEQ_LEN - 1 if needed
                cur_len = len(input_ids)
                if cur_len < SEQ_LEN - 1:
                    pad_len = (SEQ_LEN - 1) - cur_len
                    input_tensor = torch.cat([torch.tensor(input_ids, dtype=torch.long), torch.full((pad_len,), PAD_ID, dtype=torch.long)])
                    target_tensor = torch.cat([targets_tensor, torch.full((pad_len,), -100, dtype=torch.long)])
                else:
                    input_tensor = torch.tensor(input_ids, dtype=torch.long)
                    target_tensor = targets_tensor

                all_samples.append((input_tensor, target_tensor))
            except Exception:
                skipped += 1
                continue

    print(f"\nFinished packing {len(all_samples):,} samples in {time.time()-t0:.1f}s (Skipped: {skipped:,}).")

    # Split train/val (95% / 5%)
    val_size = max(500, int(len(all_samples) * 0.05))
    train_samples = all_samples[:-val_size]
    val_samples = all_samples[-val_size:]

    print(f"Train samples: {len(train_samples):,} | Val samples: {len(val_samples):,}")

    payload = {
        "train": train_samples,
        "val": val_samples,
        "seq_len": SEQ_LEN,
        "vocab_size": sp.get_piece_size(),
        "total_samples": len(all_samples)
    }

    print(f"Saving to {OUTPUT_PT}...")
    torch.save(payload, OUTPUT_PT)
    print(f"SUCCESS! File size: {os.path.getsize(OUTPUT_PT)/(1024*1024):.1f} MB")

if __name__ == "__main__":
    main()
