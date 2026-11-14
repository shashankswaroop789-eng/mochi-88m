"""
Download Fresh Trending Story & Novel Datasets from Hugging Face:
1. Dampfinchen/Creative_Writing_Multiturn (114 MB JSON)
2. ChaoticNeutrals/Creative_Writing-ShareGPT (26 MB JSONL)
3. sedthh/gutenberg_english Shard 1 (330 MB Parquet - 1,300+ full classic books)
"""

import os
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = [
    {
        "name": "Creative_Writing_ShareGPT.jsonl",
        "url": "https://huggingface.co/datasets/ChaoticNeutrals/Creative_Writing-ShareGPT/resolve/main/Creative_Writing-ShareGPT.jsonl",
        "target": os.path.join(DATA_DIR, "creative_writing_sharegpt.jsonl"),
        "size_mb": 26.2
    },
    {
        "name": "Creative_Writing_Multiturn.json",
        "url": "https://huggingface.co/datasets/Dampfinchen/Creative_Writing_Multiturn/resolve/main/Creative-Writing-MultiturnV1.1.json",
        "target": os.path.join(DATA_DIR, "creative_writing_multiturn.json"),
        "size_mb": 114.3
    },
    {
        "name": "Gutenberg_English_Shard1.parquet",
        "url": "https://huggingface.co/datasets/sedthh/gutenberg_english/resolve/main/data/train-00001-of-00037-9f227d74fc154ce9.parquet",
        "target": os.path.join(DATA_DIR, "gutenberg_shard1.parquet"),
        "size_mb": 330.0
    }
]

print("=" * 80)
print(" 🚀 DOWNLOADING FRESH TRENDING LITERARY DATASETS FROM HUGGING FACE")
print("=" * 80)

def download_file(url, target_path, name, expected_mb):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024 * 1024:
        curr_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[ALREADY DOWNLOADED] {name} ({curr_mb:.1f} MB) at {target_path}")
        return True

    print(f"\n[DOWNLOADING] {name} (~{expected_mb:.1f} MB)...")
    print(f"  URL: {url}")
    t0 = time.time()
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_file:
        downloaded = 0
        total_size = int(resp.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1 MB chunks
        last_print = time.time()

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_print > 2.0:
                mb = downloaded / (1024 * 1024)
                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                speed = mb / max(0.01, time.time() - t0)
                print(f"  --> {mb:.1f} MB / {expected_mb:.1f} MB ({pct:.1f}%) | {speed:.1f} MB/s", flush=True)
                last_print = time.time()

    elapsed = time.time() - t0
    final_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"  >>> Finished {name}: {final_mb:.1f} MB in {elapsed:.1f}s ({final_mb/elapsed:.1f} MB/s)")
    return True

for d in DATASETS:
    try:
        download_file(d["url"], d["target"], d["name"], d["size_mb"])
    except Exception as e:
        print(f"  FAILED {d['name']}: {e}")

print("\n" + "=" * 80)
print(" ALL FRESH HUGGING FACE DATASETS DOWNLOADED SUCCESSFULLY!")
print("=" * 80)
