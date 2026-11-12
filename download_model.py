"""
Download Mochi-88M Model Weights and Tokenizer from Hugging Face Hub
Repository: https://huggingface.co/Shashank1227/mochi-88m
"""

import os
import sys
import urllib.request

# Enable UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "hf_coherent_story_gpt88m.pt": "https://huggingface.co/Shashank1227/mochi-88m/resolve/main/hf_coherent_story_gpt88m.pt",
    "spm_hf_master.model": "https://huggingface.co/Shashank1227/mochi-88m/resolve/main/spm_hf_master.model"
}

def download_with_progress(url, destination):
    filename = os.path.basename(destination)
    print(f"\n[Mochi-88M] Downloading {filename} from Hugging Face...")
    print(f"            Source: {url}")
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r            Progress: [{percent:5.1f}%] {mb_downloaded:.1f} MB / {mb_total:.1f} MB")
            sys.stdout.flush()
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r            Downloaded: {mb_downloaded:.1f} MB")
            sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mochi-88M-Downloader")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, destination, reporthook=reporthook)
    print(f"\n            SUCCESS! Saved to {destination}")

def main():
    print("=" * 70)
    print(" 🍡 Mochi-88M - Hugging Face Weights Downloader")
    print(" Hugging Face Hub : https://huggingface.co/Shashank1227/mochi-88m")
    print("=" * 70)

    for filename, url in FILES.items():
        target = os.path.join(BASE_DIR, filename)
        if os.path.exists(target):
            size_mb = os.path.getsize(target) / (1024 * 1024)
            print(f"[Notice] {filename} already exists locally ({size_mb:.1f} MB). Skipping download.")
        else:
            download_with_progress(url, target)

    print("\n" + "=" * 70)
    print(" All Mochi-88M assets verified and ready to run!")
    print(" Launch GUI:   python story_studio_gui.py")
    print(" CLI Generate: python generate.py --prompt \"Your prompt here\"")
    print("=" * 70)

if __name__ == "__main__":
    main()
