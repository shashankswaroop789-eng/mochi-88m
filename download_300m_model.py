"""
Download Mochi-Novel-300M Weights and Tokenizer from Hugging Face Hub
====================================================================
Repository: https://huggingface.co/Shashank1227/mochi-novel-300m
"""

import os
import sys
import urllib.request
from pathlib import Path

# Enable UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent

REPO_URL = "https://huggingface.co/Shashank1227/mochi-novel-300m/resolve/main"

FILES = {
    "mochi_novel_300m_golden_rajput.pt": f"{REPO_URL}/mochi_novel_300m_golden_rajput.pt",
    "mochi_novel_300m.pt": f"{REPO_URL}/mochi_novel_300m.pt",
    "spm_novel_16k.model": f"{REPO_URL}/spm_novel_16k.model",
    "spm_novel_16k.vocab": f"{REPO_URL}/spm_novel_16k.vocab"
}


def download_with_progress(url, destination):
    filename = destination.name
    print(f"\n[Mochi-Novel-300M] Downloading {filename} from Hugging Face...")
    print(f"                  Source: {url}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r                  Progress: [{percent:5.1f}%] {mb_downloaded:.1f} MB / {mb_total:.1f} MB")
            sys.stdout.flush()
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r                  Downloaded: {mb_downloaded:.1f} MB")
            sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mochi-Novel-300M-Downloader")]
    urllib.request.install_opener(opener)
    try:
        urllib.request.urlretrieve(url, str(destination), reporthook=reporthook)
        print(f"\n                  SUCCESS! Saved to {destination}")
    except Exception as e:
        print(f"\n[Error] Failed to download {filename}: {e}")
        print("Please check that the file is uploaded to the Hugging Face repository.")


def main():
    print("=" * 75)
    print(" 🍡 Mochi-Novel-300M - Hugging Face Weights Downloader")
    print(f" Repository: https://huggingface.co/Shashank1227/mochi-novel-300m")
    print("=" * 75)

    for filename, url in FILES.items():
        target = BASE_DIR / filename
        if target.exists():
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"[Notice] {filename} already exists locally ({size_mb:.1f} MB). Skipping download.")
        else:
            download_with_progress(url, target)

    print("\n" + "=" * 75)
    print(" Ready to launch Story Studio GUI:")
    print("   python story_studio_gui.py")
    print("=" * 75)


if __name__ == "__main__":
    main()
