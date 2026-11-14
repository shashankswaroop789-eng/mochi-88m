"""
Gather 5 Billion Tokens of High-Quality Pre-Training Data
Downloads:
1. HuggingFaceFW/fineweb-edu (sample/100BT/000_00000.parquet - 2.05 GB, ~2.0 Billion Tokens)
2. HuggingFaceFW/fineweb-edu (sample/100BT/000_00001.parquet - 2.05 GB, ~2.0 Billion Tokens)
Combined with local literary archive (Gutenberg novels, Fiction books, WritingPrompts ~1.0 Billion Tokens)
Total Pre-Training Reservoir = 5,000,000,000+ Tokens.
"""

import os
import sys
import time
import shutil
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "fineweb_edu")
os.makedirs(DATA_DIR, exist_ok=True)

FILES = [
    {
        "name": "fineweb_edu_shard0.parquet",
        "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100BT/000_00000.parquet",
        "target": os.path.join(DATA_DIR, "fineweb_edu_shard0.parquet"),
        "expected_mb": 2053.7,
        "token_estimate": "2.0 Billion Tokens"
    },
    {
        "name": "fineweb_edu_shard1.parquet",
        "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100BT/000_00001.parquet",
        "target": os.path.join(DATA_DIR, "fineweb_edu_shard1.parquet"),
        "expected_mb": 2053.7,
        "token_estimate": "2.0 Billion Tokens"
    }
]

print("=" * 80)
print(" 🚀 GATHERING 5 BILLION TOKEN HIGH-QUALITY PRE-TRAINING RESERVOIR")
print("=" * 80)

total_gb, used_gb, free_gb = shutil.disk_usage(BASE_DIR)
print(f" Disk Space Check: {free_gb / (1024**3):.2f} GB Free on Drive")
if free_gb / (1024**3) < 8.0:
    print(" [WARNING] Less than 8 GB free disk space available. Halting for safety.")
    sys.exit(1)

def download_shard(url, target_path, name, expected_mb, token_est):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024 * 1024 * 100:
        curr_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[ALREADY EXISTS] {name} ({curr_mb:.1f} MB) -> {token_est}")
        return True

    print(f"\n[DOWNLOADING] {name} ({expected_mb:.1f} MB | {token_est})...")
    print(f"  URL: {url}")
    t0 = time.time()

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_file:
        downloaded = 0
        total_size = int(resp.headers.get("Content-Length", 0))
        chunk_size = 2 * 1024 * 1024  # 2 MB chunks
        last_print = time.time()

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_print > 3.0:
                mb = downloaded / (1024 * 1024)
                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                speed = mb / max(0.01, time.time() - t0)
                print(f"  --> {mb:.1f} MB / {expected_mb:.1f} MB ({pct:.1f}%) | Speed: {speed:.1f} MB/s", flush=True)
                last_print = time.time()

    elapsed = time.time() - t0
    final_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"  >>> Finished {name}: {final_mb:.1f} MB in {elapsed:.1f}s ({final_mb/elapsed:.1f} MB/s)")
    return True

for f in FILES:
    try:
        download_shard(f["url"], f["target"], f["name"], f["expected_mb"], f["token_estimate"])
    except Exception as e:
        print(f"  FAILED {f['name']}: {e}")

print("\n" + "=" * 80)
print(" 5 BILLION TOKEN HIGH-QUALITY DATASET ACQUISITION COMPLETE!")
print(f" Final Free Space: {shutil.disk_usage(BASE_DIR)[2] / (1024**3):.2f} GB Free")
print("=" * 80)
