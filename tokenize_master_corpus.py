"""
Tokenizes master_story_corpus.txt with spm_hf_master.model and saves to hf_master_tokens.pt.
"""

import os
import sys
import time
import torch
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(DATA_DIR, "master_story_corpus.txt")
TOKENIZER_MODEL = os.path.join(DATA_DIR, "spm_hf_master.model")
TOKEN_CACHE = os.path.join(DATA_DIR, "hf_master_tokens.pt")

def main():
    print(f"Loading tokenizer from {TOKENIZER_MODEL}...")
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    
    print(f"Reading corpus {CORPUS_FILE}...")
    start_time = time.time()
    with open(CORPUS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    print(f"Read {len(text):,} characters in {time.time()-start_time:.2f}s")
    
    print("Tokenizing entire corpus...")
    t0 = time.time()
    token_ids = sp.encode_as_ids(text)
    t1 = time.time()
    print(f"Generated {len(token_ids):,} tokens in {t1-t0:.2f}s (Speed: {len(token_ids)/(t1-t0):,.0f} tokens/s)")
    
    print(f"Saving to {TOKEN_CACHE}...")
    tokens_tensor = torch.tensor(token_ids, dtype=torch.long)
    torch.save(tokens_tensor, TOKEN_CACHE)
    file_sz = os.path.getsize(TOKEN_CACHE) / (1024 * 1024)
    print(f"Successfully cached {len(tokens_tensor):,} tokens to {TOKEN_CACHE} ({file_sz:.2f} MB)")

if __name__ == "__main__":
    main()
