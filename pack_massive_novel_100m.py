"""
🍡 Massive Novel 100M Token Corpus Builder (102,400,000 Tokens)
Packs 100,000 blocks of 1,024 tokens across:
1. Modern Fiction Books (4,737 books -> 35,000 blocks / 35.8M tokens)
2. Classic Gutenberg Novels (1,305 books -> 15,000 blocks / 15.4M tokens)
3. Human WritingPrompts Stories (136,300 stories -> 35,000 blocks / 35.8M tokens with prompt masking)
4. Multi-Character Grounded Chapters (150+ names -> 15,000 blocks / 15.4M tokens with prompt masking)
Total = 102,400,000 tokens packed into int16 tensors (~205 MB).
"""

import os
import sys
import re
import time
import random
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
print(" 🍡 PACKING MASSIVE 100,000,000+ TOKEN NOVEL TRAINING CORPUS")
print("=" * 80)

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
# 1. MODERN FICTION BOOKS (35,000 blocks = 35.8M tokens)
# ==============================================================================
fic_path = os.path.join(DATA_DIR, "fiction_books.parquet")
if os.path.exists(fic_path):
    print("\n[1/4] Processing Modern Fiction Books (4,737 full novels)...")
    t0 = time.time()
    df_fic = pd.read_parquet(fic_path, columns=["text"])
    fic_blocks = 0
    target_fic = 35000

    for idx, row in df_fic.iterrows():
        text = str(row["text"]).strip()
        if len(text) < 4000:
            continue
        tokens = sp.encode_as_ids(text)
        # Slice into contiguous 1024-token blocks
        n_blocks = len(tokens) // BLOCK_SIZE
        for b in range(min(n_blocks, 15)):  # Take up to 15 blocks per book for broad coverage
            block = tokens[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
            if len(block) == BLOCK_SIZE:
                all_x.append(torch.tensor(block[:-1], dtype=torch.int16))
                all_y.append(torch.tensor(block[1:], dtype=torch.int16))
                fic_blocks += 1
                if fic_blocks >= target_fic:
                    break
        if fic_blocks >= target_fic:
            break
    print(f"      Packed {fic_blocks:,} blocks ({fic_blocks * BLOCK_SIZE:,} tokens) in {time.time()-t0:.1f}s")

# ==============================================================================
# 2. CLASSIC GUTENBERG NOVELS (15,000 blocks = 15.4M tokens)
# ==============================================================================
gut_path = os.path.join(DATA_DIR, "gutenberg_shard0.parquet")
if os.path.exists(gut_path):
    print("\n[2/4] Processing Classic Gutenberg Novels (1,305 full books)...")
    t0 = time.time()
    df_gut = pd.read_parquet(gut_path, columns=["TEXT"])
    gut_blocks = 0
    target_gut = 15000

    for idx, row in df_gut.iterrows():
        text = str(row["TEXT"]).strip()
        if len(text) < 4000:
            continue
        tokens = sp.encode_as_ids(text)
        n_blocks = len(tokens) // BLOCK_SIZE
        for b in range(min(n_blocks, 15)):
            block = tokens[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
            if len(block) == BLOCK_SIZE:
                all_x.append(torch.tensor(block[:-1], dtype=torch.int16))
                all_y.append(torch.tensor(block[1:], dtype=torch.int16))
                gut_blocks += 1
                if gut_blocks >= target_gut:
                    break
        if gut_blocks >= target_gut:
            break
    print(f"      Packed {gut_blocks:,} blocks ({gut_blocks * BLOCK_SIZE:,} tokens) in {time.time()-t0:.1f}s")

# ==============================================================================
# 3. WRITING PROMPTS HUMAN STORIES (35,000 blocks = 35.8M tokens)
# ==============================================================================
wp_path = os.path.join(DATA_DIR, "writingprompts_shard0.parquet")
if os.path.exists(wp_path):
    print("\n[3/4] Processing Human Creative Stories (136,300 stories)...")
    t0 = time.time()
    df_wp = pd.read_parquet(wp_path)
    wp_blocks = 0
    target_wp = 35000

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

        formatted_prompt = f"### Prompt:\n{prompt}\n\n### Story:\n"
        p_tokens = sp.encode_as_ids(formatted_prompt)
        s_tokens = sp.encode_as_ids(story) + [EOS]

        combined = p_tokens + s_tokens
        if len(combined) > BLOCK_SIZE:
            combined = combined[:BLOCK_SIZE]

        x_ids = combined[:-1]
        y_ids = combined[1:]
        p_len = min(len(p_tokens) - 1, len(y_ids))

        # Prompt-loss masking: set prompt tokens to -100 in targets
        masked_y = [-100] * p_len + y_ids[p_len:]

        # Pad up to BLOCK_SIZE - 1
        pad_needed = (BLOCK_SIZE - 1) - len(x_ids)
        if pad_needed > 0:
            x_ids = x_ids + [PAD] * pad_needed
            masked_y = masked_y + [-100] * pad_needed

        all_x.append(torch.tensor(x_ids, dtype=torch.int16))
        all_y.append(torch.tensor(masked_y, dtype=torch.int16))
        wp_blocks += 1
        if wp_blocks >= target_wp:
            break
    print(f"      Packed {wp_blocks:,} blocks ({wp_blocks * BLOCK_SIZE:,} tokens) in {time.time()-t0:.1f}s")

# ==============================================================================
# 4. MULTI-CHARACTER GROUNDED STORIES (15,000 blocks = 15.4M tokens)
# ==============================================================================
print("\n[4/4] Packing Universal Multi-Character Grounded Stories...")
t0 = time.time()
char_blocks = 0
target_char = 15000

# Sample from WritingPrompts and inject character name pairs
if os.path.exists(wp_path):
    for idx, row in df_wp.iloc[35000:].iterrows():
        story = str(row["story"]).strip()
        words = len(story.split())
        if words < 350:
            continue
        prompt = str(row["prompt"]).strip()
        prompt = re.sub(r'\[\s*WP\s*\]', '', prompt, flags=re.IGNORECASE).strip()
        prompt = re.sub(r'^\s*[\,\.\-\:]+\s*', '', prompt).strip()
        if len(prompt) < 15:
            continue

        m_name = random.choice(NAMES_MALE)
        f_name = random.choice(NAMES_FEMALE)

        # Ground story by replacing generic names
        grounded_story = re.sub(r'\bJohn\b', m_name, story)
        grounded_story = re.sub(r'\bMary\b', f_name, grounded_story)
        grounded_story = re.sub(r'\bJack\b', m_name, grounded_story)
        grounded_story = re.sub(r'\bSarah\b', f_name, grounded_story)

        formatted_prompt = f"### Prompt:\nWrite a story about {m_name} and {f_name}. {prompt}\n\n### Story:\n"
        p_tokens = sp.encode_as_ids(formatted_prompt)
        s_tokens = sp.encode_as_ids(grounded_story) + [EOS]

        combined = p_tokens + s_tokens
        if len(combined) > BLOCK_SIZE:
            combined = combined[:BLOCK_SIZE]

        x_ids = combined[:-1]
        y_ids = combined[1:]
        p_len = min(len(p_tokens) - 1, len(y_ids))
        masked_y = [-100] * p_len + y_ids[p_len:]

        pad_needed = (BLOCK_SIZE - 1) - len(x_ids)
        if pad_needed > 0:
            x_ids = x_ids + [PAD] * pad_needed
            masked_y = masked_y + [-100] * pad_needed

        all_x.append(torch.tensor(x_ids, dtype=torch.int16))
        all_y.append(torch.tensor(masked_y, dtype=torch.int16))
        char_blocks += 1
        if char_blocks >= target_char:
            break
print(f"      Packed {char_blocks:,} blocks ({char_blocks * BLOCK_SIZE:,} tokens) in {time.time()-t0:.1f}s")

# ==============================================================================
# COMBINE, SHUFFLE, AND SAVE
# ==============================================================================
total_samples = len(all_x)
total_tokens = total_samples * BLOCK_SIZE
print("\n" + "=" * 80)
print(f" TOTAL BLOCKS PACKED  : {total_samples:,}")
print(f" TOTAL TOKENS IN CORPUS: {total_tokens:,} ({total_tokens / 1e6:.1f} Million Tokens)")
print("=" * 80)

# Shuffle
print("Shuffling corpus...")
indices = list(range(total_samples))
random.seed(42)
random.shuffle(indices)

split_val = int(total_samples * 0.96)
train_indices = indices[:split_val]
val_indices = indices[split_val:]

X_tensor = torch.stack(all_x)
Y_tensor = torch.stack(all_y)

print(f"Train Blocks: {len(train_indices):,} ({len(train_indices)*BLOCK_SIZE:,} tokens)")
print(f"Val Blocks  : {len(val_indices):,} ({len(val_indices)*BLOCK_SIZE:,} tokens)")

dataset_dict = {
    "train_x": X_tensor[train_indices],
    "train_y": Y_tensor[train_indices],
    "val_x": X_tensor[val_indices],
    "val_y": Y_tensor[val_indices],
    "block_size": BLOCK_SIZE,
    "vocab_size": 16384,
    "tokenizer": "spm_novel_16k.model"
}

print(f"Saving to {OUTPUT_PT}...")
torch.save(dataset_dict, OUTPUT_PT)
file_size_mb = os.path.getsize(OUTPUT_PT) / (1024 * 1024)
print(f" >>> SUCCESS! Saved {total_tokens / 1e6:.1f}M token corpus to {OUTPUT_PT} ({file_size_mb:.1f} MB)\n")
