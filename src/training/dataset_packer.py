"""
Prepare Coherent SFT Dataset for Short Stories
Packs high-quality short story prompt-story pairs with prompt masking (target = -100).
Integrates clean narrative prose for language stability.
"""

import os
import sys
import json
import torch
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_FILE = os.path.join(DATA_DIR, "coherent_short_stories.jsonl")
SCRUBBED_PROSE = os.path.join(DATA_DIR, "scrubbed_clean_romance_corpus.txt")
TOKENIZER_MODEL = os.path.join(DATA_DIR, "spm_hf_master.model")
OUTPUT_FILE = os.path.join(DATA_DIR, "coherent_sft_tokens.pt")

BLOCK_SIZE = 512

def main():
    print("=" * 80)
    print(" PREPARING COHERENT SHORT STORY SFT DATASET")
    print("=" * 80)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)

    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(pairs)} curated short story pairs.")

    sft_samples = []

    for p in pairs:
        prompt_str = f"### Prompt:\n{p['prompt']}\n\n### Story:\n"
        story_str = f"{p['story']}\n"

        prompt_ids = sp.encode_as_ids(prompt_str)
        story_ids = sp.encode_as_ids(story_str)

        full_ids = prompt_ids + story_ids
        prompt_len = len(prompt_ids)

        # Pad or truncate to BLOCK_SIZE + 1
        if len(full_ids) < BLOCK_SIZE + 1:
            full_ids = full_ids + [0] * (BLOCK_SIZE + 1 - len(full_ids))
        else:
            full_ids = full_ids[:BLOCK_SIZE + 1]

        input_ids = full_ids[:BLOCK_SIZE]
        target_ids = list(full_ids[1:BLOCK_SIZE + 1])

        # Prompt masking: ignore loss on prompt tokens
        for i in range(BLOCK_SIZE):
            if i + 1 <= prompt_len:
                target_ids[i] = -100

        x_t = torch.tensor(input_ids, dtype=torch.long)
        y_t = torch.tensor(target_ids, dtype=torch.long)
        sft_samples.append((x_t, y_t))

    # Weight SFT samples 20x so the model thoroughly learns the format
    all_samples = []
    for _ in range(20):
        all_samples.extend(sft_samples)

    print(f"Expanded SFT samples to {len(all_samples)} sequences.")

    # Add general clean language sequences
    with open(SCRUBBED_PROSE, "r", encoding="utf-8", errors="ignore") as f:
        prose_text = f.read()

    prose_ids = sp.encode_as_ids(prose_text[:600000])
    prose_count = 0
    for i in range(0, len(prose_ids) - BLOCK_SIZE - 1, BLOCK_SIZE):
        chunk = prose_ids[i:i + BLOCK_SIZE + 1]
        x_t = torch.tensor(chunk[:BLOCK_SIZE], dtype=torch.long)
        y_t = torch.tensor(chunk[1:BLOCK_SIZE + 1], dtype=torch.long)
        all_samples.append((x_t, y_t))
        prose_count += 1

    print(f"Added {prose_count} clean prose sequences.")
    print(f"Total dataset: {len(all_samples):,} sequences of {BLOCK_SIZE} tokens.")

    split_idx = int(0.95 * len(all_samples))
    payload = {
        "train": all_samples[:split_idx],
        "val": all_samples[split_idx:],
        "block_size": BLOCK_SIZE
    }

    torch.save(payload, OUTPUT_FILE)
    print(f"Saved token dataset to: {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/(1024*1024):.2f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    main()
