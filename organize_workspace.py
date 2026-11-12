"""
Organizes gpt_from_scratch workspace by archiving historical/superseded files
into a dedicated 'archive/' directory while keeping the active 88.1M engine clean.
"""

import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")

# Files to move into archive
FILES_TO_ARCHIVE = [
    # 1. Early Tutorial Steps (Toy Models)
    "bigram_model.py",
    "self_attention.py",
    "multi_head_model.py",
    "train_gpt.py",
    "data_loader.py",
    "chat.py",
    "sherlock_gpt.pt",
    "input.txt",
    "spm_sherlock.model",
    "spm_sherlock.vocab",
    "spm_shakespeare.model",
    "spm_shakespeare.vocab",
    "spm_shakespeare_exact.model",
    "spm_shakespeare_exact.vocab",
    
    # 2. Intermediate 13M & 27.8M Models
    "train_master_gpt.py",
    "master_gpt.pt",
    "master_input.txt",
    "spm_master.model",
    "spm_master.vocab",
    "train_modern_gpt.py",
    "modern_gpt.pt",
    "grand_corpus.txt",
    "prepare_grand_corpus.py",
    "spm_grand.model",
    "spm_grand.vocab",
    "train_tokenizer.py",

    # 3. Old 88M Runs (Overfitted on 6 books / Victorian mix)
    "train_romance_gpt88m.py",
    "romance_gpt88m.pt",
    "romance_corpus.txt",
    "romance_tokens.pt",
    "spm_romance.model",
    "spm_romance.vocab",
    "prepare_romance_corpus.py",
    "train_pure_modern_88m.py",
    "modern_romance_gpt88m.pt",
    "modern_romance_corpus.txt",
    "prepare_modern_romance_corpus.py",
    "pure_modern_tokens.pt",
    "spm_modern.model",
    "spm_modern.vocab",
    "spm_modern_romance.model",
    "spm_modern_romance.vocab",

    # 4. Old One-off Testing/Audit Scripts
    "audit_bookcorpus.py",
    "test_cleaner.py",

    # 5. 1800s Public Domain Gutenberg Texts
    "dracula.txt",
    "frankenstein.txt",
    "jekyll_hyde.txt",
    "jane_eyre.txt",
    "pride_and_prejudice.txt",
    "sense_and_sensibility.txt",
    "emma.txt",
    "persuasion.txt",
    "northanger_abbey.txt",
    "wuthering_heights.txt",
    "little_women.txt",
    "house_of_mirth.txt",
    "age_of_innocence.txt"
]

def organize():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    moved_count = 0
    total_bytes = 0

    print("=== Cleaning and Organizing Workspace ===")
    print(f"Archive Directory: {ARCHIVE_DIR}\n")

    for fname in FILES_TO_ARCHIVE:
        src = os.path.join(BASE_DIR, fname)
        dst = os.path.join(ARCHIVE_DIR, fname)

        if os.path.exists(src):
            size = os.path.getsize(src)
            shutil.move(src, dst)
            moved_count += 1
            total_bytes += size
            print(f"  [Archived] {fname} ({size / (1024*1024):.2f} MB)")

    print("-" * 60)
    print(f"Successfully archived {moved_count} files ({total_bytes / (1024*1024):.2f} MB) into 'archive/'")

if __name__ == "__main__":
    organize()
