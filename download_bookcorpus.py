"""
Downloads shard 0 of lucadiliello/bookcorpusopen from Hugging Face.
Shard 0 is ~298 MB and contains ~1,276 full-length self-published novels across romance and fiction.
"""

import os
import sys
import time
import requests

DATASET_URL = "https://huggingface.co/datasets/lucadiliello/bookcorpusopen/resolve/main/data/train-00000-of-00014-e40347a4a9a752dd.parquet"
OUTPUT_FILE = "bookcorpus_shard0.parquet"

def download_file(url, target_path):
    print(f"Starting download from: {url}")
    print(f"Saving to: {target_path}")
    
    start_time = time.time()
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    total_mb = total_size / (1024 * 1024)
    print(f"Total file size: {total_mb:.2f} MB")
    
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB chunks
    last_print = time.time()
    
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                # Print progress every 2 seconds
                now = time.time()
                if now - last_print >= 2.0 or downloaded == total_size:
                    percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                    speed = (downloaded / (1024 * 1024)) / max(1e-5, now - start_time)
                    print(f"  [{percent:5.1f}%] {downloaded/(1024*1024):.1f}/{total_mb:.1f} MB | {speed:.2f} MB/s")
                    last_print = now

    elapsed = time.time() - start_time
    print(f"Download complete in {elapsed:.1f}s ({target_path})")

if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 290 * 1024 * 1024:
        print(f"File {OUTPUT_FILE} already exists and is complete ({os.path.getsize(OUTPUT_FILE)/(1024*1024):.1f} MB).")
    else:
        download_file(DATASET_URL, OUTPUT_FILE)
