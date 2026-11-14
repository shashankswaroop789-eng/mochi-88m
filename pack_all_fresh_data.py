"""
🍡 Master Corpus Packer: 100M+ Tokens of Fresh & Classic Literature
Sources:
1. Modern Fiction Novels (fiction_books.parquet - 4,737 books)
2. Gutenberg Classic Novels Shard 0 (gutenberg_shard0.parquet - 1,305 books)
3. Gutenberg Classic Novels Shard 1 (gutenberg_shard1.parquet - 1,300+ books) [NEW FROM HF]
4. Trending Creative Writing ShareGPT (creative_writing_sharegpt.jsonl) [NEW FROM HF]
5. WritingPrompts Human Stories (writingprompts_shard0.parquet - 136,300 stories)
Packs 100,000 blocks of 1,024 tokens = 102,400,000 tokens (int16 format).
"""

import os
import sys
import json
import time
import random
import re
import pandas as pd
import sentencepiece as spm
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TOKENIZER_PATH = os.path.join(BASE_DIR, "spm_novel_16k.model")
OUTPUT_PT = os.path.join(DATA_DIR, "massive_novel_100m_tokens.pt")

print("=" * 80)
print(" 🍡 PACKING 100,000,000+ TOKEN NOVEL TRAINING CORPUS")
print("=" * 80, flush=True)

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)
BOS = 1
EOS = 2
PAD = 0
BLOCK_SIZE = 1024

NAMES_MALE = [
    "Shashank", "Rajesh", "Marcus", "Lucas", "Julian", "Alexander", "Arjun", "Rohan",
    "Dev", "Kabir", "Aarav", "Vikram", "Ethan", "Nathan", "Gabriel", "Arthur", "Leo",
    "Damian", "Elias", "Mateo", "Carlos", "Kenji", "Jin", "Tariq", "Zayn", "Kwame",
    "Kaelen", "Theron", "Vesper", "Corin", "Dante", "Gideon", "Felix", "Vincent",
    "Dominic", "Sebastian", "Adrian", "Caspian", "Silas", "Tobias", "August", "Karthik", "Avinash"
]

NAMES_FEMALE = [
    "Samira", "Savita", "Elena", "Clara", "Sophie", "Helena", "Priya", "Ananya", "Maya",
    "Tara", "Meera", "Aisha", "Chloe", "Lyra", "Camille", "Nadia", "Isla", "Gwendolyn",
    "Vera", "Sofia", "Isabella", "Yuna", "Mei", "Layla", "Amara", "Nyx", "Selene",
    "Aurelia", "Seraphina", "Freya", "Genevieve", "Valerie", "Cora", "Astrid", "Rowan",
    "Vivienne", "Iris", "Evelyn", "Cassandra", "Miriam", "Noor"
]

all_x = []
all_y = []

# ==============================================================================
# 1. CLASSIC NOVELS (Gutenberg Shards 0 & 1 - 2,600+ books) -> 25,000 blocks (25.6M tokens)
# ==============================================================================
print("\n[1/4] Processing Classic Literature (Gutenberg Shards 0 & 1)...", flush=True)
t0 = time.time()
gut_blocks = 0
target_gut = 25000

for shard_file in ["gutenberg_shard0.parquet", "gutenberg_shard1.parquet"]:
    shard_path = os.path.join(DATA_DIR, shard_file)
    if os.path.exists(shard_path) and gut_blocks < target_gut:
        df_gut = pd.read_parquet(shard_path, columns=["TEXT"])
        for idx, row in df_gut.iterrows():
            text = str(row["TEXT"]).strip()
            if len(text) < 3000:
                continue
            toks = sp.encode_as_ids(text[:60000])  # up to ~15k tokens per book
            n_blocks = len(toks) // BLOCK_SIZE
            for b in range(n_blocks):
                blk = toks[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
                if len(blk) == BLOCK_SIZE:
                    all_x.append(torch.tensor(blk[:-1], dtype=torch.int16))
                    all_y.append(torch.tensor(blk[1:], dtype=torch.int16))
                    gut_blocks += 1
                    if gut_blocks >= target_gut:
                        break
            if gut_blocks >= target_gut:
                break

print(f"      Packed {gut_blocks:,} classic novel blocks ({gut_blocks * BLOCK_SIZE:,} tokens) in {time.time()-t0:.1f}s", flush=True)

# ==============================================================================
# 2. MODERN FICTION NOVELS (4,737 books) -> 25,000 blocks (25.6M tokens)
# ==============================================================================
print("\n[2/4] Processing Modern Fiction Novels (4,737 books)...", flush=True)
t0 = time.time()
fic_blocks = 0
target_fic = 25000
fic_path = os.path.join(DATA_DIR, "fiction_books.parquet")

if os.path.exists(fic_path):
    df_fic = pd.read_parquet(fic_path, columns=["text"])
    for idx, row in df_fic.iterrows():
        text = str(row["text"]).strip()
        if len(text) < 3000:
            continue
        toks = sp.encode_as_ids(text[:50000])
        n_blocks = len(toks) // BLOCK_SIZE
        for b in range(n_blocks):
            blk = toks[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
            if len(blk) == BLOCK_SIZE:
                all_x.append(torch.tensor(blk[:-1], dtype=torch.int16))
                all_y.append(torch.tensor(blk[1:], dtype=torch.int16))
                fic_blocks += 1
                if fic_blocks >= target_fic:
                    break
        if fic_blocks >= target_fic:
            break

print(f"      Packed {fic_blocks:,} modern fiction blocks ({fic_blocks * BLOCK_SIZE:,} tokens) in {time.time()-t0:.1f}s", flush=True)

# ==============================================================================
# 3. FRESH TRENDING HF CREATIVE WRITING (ShareGPT & Multiturn) -> 20,000 blocks (20.5M tokens)
# ==============================================================================
print("\n[3/4] Processing Fresh Trending Creative Writing from Hugging Face...", flush=True)
t0 = time.time()
hf_blocks = 0
target_hf = 20000

sharegpt_path = os.path.join(DATA_DIR, "creative_writing_sharegpt.jsonl")
if os.path.exists(sharegpt_path):
    with open(sharegpt_path, "r", encoding="utf-8") as f:
        for line in f:
            if hf_blocks >= target_hf:
                break
            try:
                obj = json.loads(line)
                convs = obj.get("conversations", [])
                if len(convs) >= 2:
                    prompt = convs[0].get("value", "").strip()
                    story = convs[1].get("value", "").strip()
                    if len(story.split()) < 150:
                        continue
                    formatted = f"### Prompt:\n{prompt}\n\n### Story:\n"
                    p_toks = sp.encode_as_ids(formatted)
                    s_toks = sp.encode_as_ids(story) + [EOS]
                    combined = p_toks + s_toks
                    if len(combined) > BLOCK_SIZE:
                        combined = combined[:BLOCK_SIZE]
                    x_ids = combined[:-1]
                    y_ids = combined[1:]
                    p_len = min(len(p_toks) - 1, len(y_ids))
                    masked_y = [-100] * p_len + y_ids[p_len:]
                    pad_len = (BLOCK_SIZE - 1) - len(x_ids)
                    if pad_len > 0:
                        x_ids = x_ids + [PAD] * pad_len
                        masked_y = masked_y + [-100] * pad_len
                    all_x.append(torch.tensor(x_ids, dtype=torch.int16))
                    all_y.append(torch.tensor(masked_y, dtype=torch.int16))
                    hf_blocks += 1
            except Exception:
                continue

print(f"      Packed {hf_blocks:,} fresh trending creative writing blocks ({hf_blocks * BLOCK_SIZE:,} tokens) in {time.time()-t0:.1f}s", flush=True)

# ==============================================================================
# 4. WRITINGPROMPTS & MULTI-CHARACTER GROUNDED STORIES -> 30,000 blocks (30.7M tokens)
# ==============================================================================
print("\n[4/4] Processing WritingPrompts & Universal Multi-Character Grounded Stories...", flush=True)
t0 = time.time()
wp_blocks = 0
target_wp = 30000
wp_path = os.path.join(DATA_DIR, "writingprompts_shard0.parquet")

if os.path.exists(wp_path):
    df_wp = pd.read_parquet(wp_path)
    for idx, row in df_wp.iterrows():
        story = str(row["story"]).strip()
        words = len(story.split())
        if words < 250:
            continue
        prompt = str(row["prompt"]).strip()
        prompt = re.sub(r'\[\s*WP\s*\]', '', prompt, flags=re.IGNORECASE).strip()
        prompt = re.sub(r'^\s*[\,\.\-\:]+\s*', '', prompt).strip()
        if len(prompt) < 15:
            continue

        # In 50% of stories: Inject universal character names!
        if wp_blocks % 2 == 0:
            m_name = random.choice(NAMES_MALE)
            f_name = random.choice(NAMES_FEMALE)
            story = re.sub(r'\bJohn\b', m_name, story)
            story = re.sub(r'\bMary\b', f_name, story)
            story = re.sub(r'\bJack\b', m_name, story)
            story = re.sub(r'\bSarah\b', f_name, story)
            formatted = f"### Prompt:\nWrite a story about {m_name} and {f_name}. {prompt}\n\n### Story:\n"
        else:
            formatted = f"### Prompt:\n{prompt}\n\n### Story:\n"

        p_toks = sp.encode_as_ids(formatted)
        s_toks = sp.encode_as_ids(story) + [EOS]
        combined = p_toks + s_toks
        if len(combined) > BLOCK_SIZE:
            combined = combined[:BLOCK_SIZE]
        x_ids = combined[:-1]
        y_ids = combined[1:]
        p_len = min(len(p_toks) - 1, len(y_ids))
        masked_y = [-100] * p_len + y_ids[p_len:]
        pad_len = (BLOCK_SIZE - 1) - len(x_ids)
        if pad_len > 0:
            x_ids = x_ids + [PAD] * pad_len
            masked_y = masked_y + [-100] * pad_len
        all_x.append(torch.tensor(x_ids, dtype=torch.int16))
        all_y.append(torch.tensor(masked_y, dtype=torch.int16))
        wp_blocks += 1
        if wp_blocks >= target_wp:
            break

print(f"      Packed {wp_blocks:,} WritingPrompts & multi-character blocks in {time.time()-t0:.1f}s", flush=True)

# ==============================================================================
# COMBINE, SHUFFLE, SAVE
# ==============================================================================
total_blocks = len(all_x)
total_tokens = total_blocks * BLOCK_SIZE

print("\n" + "=" * 80)
print(f" TOTAL BLOCKS PACKED   : {total_blocks:,}")
print(f" TOTAL TOKENS IN CORPUS : {total_tokens:,} ({total_tokens / 1e6:.1f} Million Tokens)")
print("=" * 80, flush=True)

# Shuffle
print("Shuffling corpus...", flush=True)
indices = list(range(total_blocks))
random.seed(42)
random.shuffle(indices)

split_val = int(total_blocks * 0.96)
train_indices = indices[:split_val]
val_indices = indices[split_val:]

X_tensor = torch.stack(all_x)
Y_tensor = torch.stack(all_y)

print(f"Train Blocks: {len(train_indices):,} ({len(train_indices)*BLOCK_SIZE:,} tokens)", flush=True)
print(f"Val Blocks  : {len(val_indices):,} ({len(val_indices)*BLOCK_SIZE:,} tokens)", flush=True)

dataset_dict = {
    "train_x": X_tensor[train_indices],
    "train_y": Y_tensor[train_indices],
    "val_x": X_tensor[val_indices],
    "val_y": Y_tensor[val_indices],
    "block_size": BLOCK_SIZE,
    "vocab_size": 16384,
    "tokenizer": "spm_novel_16k.model"
}

print(f"Saving to {OUTPUT_PT}...", flush=True)
torch.save(dataset_dict, OUTPUT_PT)
file_size_mb = os.path.getsize(OUTPUT_PT) / (1024 * 1024)
print(f" >>> SUCCESS! Saved {total_tokens / 1e6:.1f}M token corpus to {OUTPUT_PT} ({file_size_mb:.1f} MB)\n", flush=True)
