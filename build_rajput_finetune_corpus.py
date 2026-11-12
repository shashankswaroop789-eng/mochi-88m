"""
Build Master Rajput Fine-Tuning Dataset
Merges the scrubbed clean romance corpus with the high-priority Rajput Dynasty corpus,
tokenizes with spm_hf_master.model, and saves finetune_rajput_tokens.pt.
"""

import os
import sys
import time
import re
import torch
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SCRUBBED_ROMANCE = os.path.join(DATA_DIR, "scrubbed_clean_romance_corpus.txt")
RAJPUT_CORPUS = os.path.join(DATA_DIR, "rajput_dynasty_corpus.txt")
TOKENIZER_MODEL = os.path.join(DATA_DIR, "spm_hf_master.model")
OUTPUT_TOKENS = os.path.join(DATA_DIR, "finetune_rajput_tokens.pt")

def main():
    print("=" * 80)
    print(" PREPARING CLEAN RAJPUT DYNASTY FINE-TUNING DATASET")
    print("=" * 80)

    # 1. Load scrubbed romance text (fluency & grammar foundation)
    print(f"Reading scrubbed romance text from {SCRUBBED_ROMANCE}...")
    with open(SCRUBBED_ROMANCE, "r", encoding="utf-8", errors="ignore") as f:
        romance_text = f.read()

    # 2. Load Rajput dynasty text (domain knowledge & character names)
    print(f"Reading Rajput dynasty text from {RAJPUT_CORPUS}...")
    with open(RAJPUT_CORPUS, "r", encoding="utf-8", errors="ignore") as f:
        rajput_text = f.read()

    # Repeat Rajput text 6x so it forms a substantial portion (~40-50%) of fine-tuning gradients
    weighted_rajput_text = "\n\n".join([rajput_text] * 6)

    # Final combined text
    combined_text = weighted_rajput_text + "\n\n" + romance_text + "\n\n" + weighted_rajput_text

    # Safety check for leaked artifacts
    print("\nRunning final safety checks for leaked book titles and chapter headers...")
    titles_leak = len(re.findall(r'(?i)(matcha maker|bared to you|ugly love|kiss thief)', combined_text))
    chapters_leak = len(re.findall(r'(?i)\bchapter\s+\d+\b', combined_text))
    rajput_mentions = len(re.findall(r'(?i)rajput', combined_text))
    shashank_mentions = len(re.findall(r'(?i)shashank', combined_text))
    khanna_mentions = len(re.findall(r'(?i)khanna', combined_text))

    print(f"  - Leaked Book Titles       : {titles_leak} (Goal: 0)")
    print(f"  - Leaked 'Chapter X' markers: {chapters_leak} (Goal: 0)")
    print(f"  - 'Rajput' Mentions        : {rajput_mentions:,}")
    print(f"  - 'Shashank' Mentions      : {shashank_mentions:,}")
    print(f"  - 'Khanna' Mentions        : {khanna_mentions:,}")

    # 3. Tokenize with SentencePiece
    print(f"\nLoading SentencePiece tokenizer: {TOKENIZER_MODEL}...")
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)

    print("Tokenizing combined dataset into token tensor...")
    t0 = time.time()
    token_ids = sp.encode_as_ids(combined_text)
    elapsed = time.time() - t0
    print(f"Tokenization complete: {len(token_ids):,} tokens generated in {elapsed:.2f}s ({len(token_ids)/elapsed:,.0f} tok/s)")

    # 4. Save token cache
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    torch.save(token_tensor, OUTPUT_TOKENS)
    file_mb = os.path.getsize(OUTPUT_TOKENS) / (1024 * 1024)
    print(f"\nSuccessfully saved token cache to: {OUTPUT_TOKENS} ({file_mb:.2f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    main()
