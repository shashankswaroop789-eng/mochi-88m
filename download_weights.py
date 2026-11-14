"""
Unified Model Weights Downloader for Mochi LLM Family
=====================================================
Downloads pre-trained weights and tokenizers from Hugging Face Hub:
- Mochi-Novel-300M (Flagship): https://huggingface.co/Shashank1227/mochi-novel-300m
- Mochi-88M (Compact): https://huggingface.co/Shashank1227/mochi-88m

Usage:
    python download_weights.py                # Downloads 300M weights (default)
    python download_weights.py --model 300m   # Downloads 300M weights
    python download_weights.py --model 88m    # Downloads 88M weights
    python download_weights.py --model all    # Downloads both models
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

# Enable UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent

REPOS = {
    "300m": {
        "name": "Mochi-Novel-300M (Flagship)",
        "hub": "https://huggingface.co/Shashank1227/moochi-300m",
        "files": {
            "mochi_novel_300m_golden_rajput.pt": "https://huggingface.co/Shashank1227/moochi-300m/resolve/main/mochi_novel_300m_golden_rajput.pt",
            "mochi_novel_300m.pt": "https://huggingface.co/Shashank1227/moochi-300m/resolve/main/mochi_novel_300m.pt",
            "spm_novel_16k.model": "https://huggingface.co/Shashank1227/moochi-300m/resolve/main/spm_novel_16k.model",
            "spm_novel_16k.vocab": "https://huggingface.co/Shashank1227/moochi-300m/resolve/main/spm_novel_16k.vocab"
        }
    },
    "88m": {
        "name": "Mochi-88M (Compact)",
        "hub": "https://huggingface.co/Shashank1227/mochi-88m",
        "files": {
            "hf_coherent_story_gpt88m.pt": "https://huggingface.co/Shashank1227/mochi-88m/resolve/main/hf_coherent_story_gpt88m.pt",
            "spm_hf_master.model": "https://huggingface.co/Shashank1227/mochi-88m/resolve/main/spm_hf_master.model"
        }
    }
}


def download_file(url, destination):
    filename = destination.name
    print(f"\n[Downloading] {filename}...")
    print(f"              Source: {url}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r              Progress: [{percent:5.1f}%] {mb_downloaded:.1f} MB / {mb_total:.1f} MB")
            sys.stdout.flush()
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r              Downloaded: {mb_downloaded:.1f} MB")
            sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mochi-Weights-Downloader")]
    urllib.request.install_opener(opener)
    try:
        urllib.request.urlretrieve(url, str(destination), reporthook=reporthook)
        print(f"\n              SUCCESS! Saved to {destination}")
    except Exception as e:
        print(f"\n[Notice] Could not download {filename} ({e}).")
        print(f"         If the file has not yet been uploaded to Hugging Face, run upload_300m_to_hf.py first.")


def download_model_group(group_key):
    info = REPOS[group_key]
    print(f"\n>>> Checking {info['name']} assets...")
    print(f"    Hugging Face Hub: {info['hub']}")
    for filename, url in info["files"].items():
        dest = BASE_DIR / filename
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"    [OK] {filename} already exists locally ({size_mb:.1f} MB). Skipping.")
        else:
            download_file(url, dest)


def main():
    parser = argparse.ArgumentParser(description="Download Mochi model weights from Hugging Face Hub")
    parser.add_argument("--model", type=str, choices=["300m", "88m", "all"], default="300m",
                        help="Model family to download: '300m' (default), '88m', or 'all'")
    args = parser.parse_args()

    print("=" * 75)
    print(" 🍡 Mochi LLM Family - Hugging Face Weights Downloader")
    print("=" * 75)

    if args.model in ("300m", "all"):
        download_model_group("300m")
    if args.model in ("88m", "all"):
        download_model_group("88m")

    print("\n" + "=" * 75)
    print(" Setup verified! Launch Story Studio GUI with:")
    print("   python story_studio_gui.py")
    print("=" * 75)


if __name__ == "__main__":
    main()
