import os
import sys
import time
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DOWNLOADS = [
    {
        "name": "Project Gutenberg Classics (1,500+ Complete Novels)",
        "url": "https://huggingface.co/datasets/sedthh/gutenberg_english/resolve/main/data/train-00000-of-00037-f5fce855b93d2d02.parquet",
        "output": os.path.join(DATA_DIR, "gutenberg_shard0.parquet")
    },
    {
        "name": "Curated Fiction Books Corpus (Modern Novels)",
        "url": "https://huggingface.co/datasets/AlekseyKorshuk/fiction-books/resolve/main/data/train-00000-of-00001.parquet",
        "output": os.path.join(DATA_DIR, "fiction_books.parquet")
    }
]

def download_file(item):
    name = item["name"]
    url = item["url"]
    out_path = item["output"]
    
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"✓ {name} already exists ({size_mb:.1f} MB). Skipping download.")
        return
        
    print(f"\n[Downloading] {name}")
    print(f"URL: {url}")
    print(f"Dest: {out_path}")
    
    t0 = time.time()
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total_size = int(resp.headers.get("content-length", 0))
    total_mb = total_size / (1024 * 1024)
    
    downloaded = 0
    chunk_size = 1024 * 1024 # 1 MB chunks
    
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                done_mb = downloaded / (1024 * 1024)
                elapsed = max(0.1, time.time() - t0)
                speed = done_mb / elapsed
                percent = (downloaded / total_size) * 100 if total_size else 0
                print(f"\r  Progress: {done_mb:6.1f} / {total_mb:6.1f} MB ({percent:5.1f}%) | Speed: {speed:5.1f} MB/s", end="", flush=True)
                
    print(f"\n✓ Finished {name} in {time.time() - t0:.1f}s ({os.path.getsize(out_path)/(1024*1024):.1f} MB)")

def main():
    print("=" * 80)
    print(" 📚 DOWNLOADING MASTER NOVEL CORPUS FOR MOCHI-NOVEL-350M")
    print("=" * 80)
    
    for item in DOWNLOADS:
        download_file(item)
        
    print("\n" + "=" * 80)
    print(" All primary novel corpora successfully acquired!")
    print("=" * 80)

if __name__ == "__main__":
    main()
