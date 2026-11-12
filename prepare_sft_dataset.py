"""
Prepare SFT Dataset with Prompt Loss Masking
Tokenizes both SFT instruction pairs and clean story prose.
Applies target = -100 to all prompt tokens so the model NEVER memorizes or outputs the prompt.
"""

import os
import sys
import json
import torch
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_FILE = os.path.join(DATA_DIR, "sft_story_dataset.jsonl")
SCRUBBED_ROMANCE = os.path.join(DATA_DIR, "scrubbed_clean_romance_corpus.txt")
TOKENIZER_MODEL = os.path.join(DATA_DIR, "spm_hf_master.model")
OUTPUT_FILE = os.path.join(DATA_DIR, "sft_packed_dataset.pt")

BLOCK_SIZE = 512

def main():
    print("=" * 80)
    print(" PREPARING SFT DATASET WITH PROMPT LOSS MASKING")
    print("=" * 80)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)

    samples = []  # list of (input_tensor_512, target_tensor_512)

    # 1. Process SFT Pairs
    print(f"Reading SFT examples from {JSONL_FILE}...")
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    print(f"Found {len(pairs)} primary SFT examples.")

    story_marker = "\n\n### Story:\n"
    story_marker_ids = sp.encode_as_ids(story_marker)
    sft_samples = []

    for p in pairs:
        prompt_str = f"### Prompt:\n{p['prompt']}{story_marker}"
        story_str = p['story']

        prompt_ids = sp.encode_as_ids(prompt_str)
        story_ids = sp.encode_as_ids(story_str)

        # Full sequence
        full_ids = prompt_ids + story_ids

        # Chunk into windows of BLOCK_SIZE
        # If longer than BLOCK_SIZE, chunk with overlap; if shorter, pad with 0
        prompt_len = len(prompt_ids)

        # We create multiple windows across the story
        for start_idx in range(0, max(1, len(full_ids) - 50), 256):
            chunk = full_ids[start_idx:start_idx + BLOCK_SIZE + 1]
            if len(chunk) < 64:
                continue

            # Pad if needed
            if len(chunk) < BLOCK_SIZE + 1:
                chunk = chunk + [0] * (BLOCK_SIZE + 1 - len(chunk))

            input_ids = chunk[:BLOCK_SIZE]
            target_ids = list(chunk[1:BLOCK_SIZE + 1])

            # Apply prompt masking: any target position that corresponds to prompt tokens gets -100
            for i in range(BLOCK_SIZE):
                global_pos = start_idx + i + 1
                if global_pos <= prompt_len:
                    target_ids[i] = -100  # PyTorch ignore_index

            x_t = torch.tensor(input_ids, dtype=torch.long)
            y_t = torch.tensor(target_ids, dtype=torch.long)
            # Add to SFT pool
            sft_samples.append((x_t, y_t))

    # Weight SFT samples 10x so prompt-to-story alignment is deeply internalized
    for _ in range(10):
        samples.extend(sft_samples)

    print(f"Created {len(sft_samples)} unique SFT sequences (expanded to {len(sft_samples)*10} weighted samples).")

    # 2. Add Clean Narrative Flow from Scrubbed Corpus (General Language Foundation)
    print(f"Reading clean narrative from {SCRUBBED_ROMANCE}...")
    with open(SCRUBBED_ROMANCE, "r", encoding="utf-8", errors="ignore") as f:
        romance_text = f.read()

    # Take first ~1.5M characters of pure clean prose
    prose_sample = romance_text[:1500000]
    prose_ids = sp.encode_as_ids(prose_sample)

    prose_count = 0
    for i in range(0, len(prose_ids) - BLOCK_SIZE - 1, BLOCK_SIZE):
        chunk = prose_ids[i:i + BLOCK_SIZE + 1]
        x_t = torch.tensor(chunk[:BLOCK_SIZE], dtype=torch.long)
        y_t = torch.tensor(chunk[1:BLOCK_SIZE + 1], dtype=torch.long)
        samples.append((x_t, y_t))
        prose_count += 1

    print(f"Added {prose_count} clean prose sequences.")
    print(f"Total training sequences: {len(samples):,} (all packed to {BLOCK_SIZE} tokens).")

    # Split into train / val
    split_idx = int(0.95 * len(samples))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    data_payload = {
        "train": train_samples,
        "val": val_samples,
        "block_size": BLOCK_SIZE
    }

    torch.save(data_payload, OUTPUT_FILE)
    sz_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Saved packed dataset to: {OUTPUT_FILE} ({sz_mb:.2f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    main()
