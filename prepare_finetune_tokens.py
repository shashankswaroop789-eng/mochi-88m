"""
Tokenizes pure_modern_english_corpus.txt using spm_hf_master.model
to prepare the exact token cache for fine-tuning hf_master_gpt88m.pt.
"""

import os
import sys
import time
import torch
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(DATA_DIR, "pure_modern_english_corpus.txt")
TOKENIZER_MODEL = os.path.join(DATA_DIR, "spm_hf_master.model")
OUTPUT_CACHE = os.path.join(DATA_DIR, "finetune_romance_tokens.pt")

def main():
    print(f"Loading tokenizer: {TOKENIZER_MODEL}...")
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)

    print(f"Reading {CORPUS_FILE}...")
    with open(CORPUS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    print(f"Read {len(text):,} characters.")

    print("Tokenizing pure contemporary romance text...")
    t0 = time.time()
    token_ids = sp.encode_as_ids(text)
    print(f"Generated {len(token_ids):,} tokens in {time.time()-t0:.2f}s")

    tokens_tensor = torch.tensor(token_ids, dtype=torch.long)
    torch.save(tokens_tensor, OUTPUT_CACHE)
    file_sz = os.path.getsize(OUTPUT_CACHE) / (1024 * 1024)
    print(f"Saved {len(tokens_tensor):,} tokens to {OUTPUT_CACHE} ({file_sz:.2f} MB)")

if __name__ == "__main__":
    main()
