"""
🍡 Mochi-88M - CLI Story Generator
Inference script featuring KV-Caching, Nucleus (top-p), Top-k, and Repetition Penalty.
"""

import os
import sys

# Enable UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import time
import torch
import sentencepiece as spm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modern_transformer import ModernGPT

def main():
    parser = argparse.ArgumentParser(description="🍡 Mochi-88M Inference Engine")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt text")
    parser.add_argument("--model", type=str, default="hf_coherent_story_gpt88m.pt", help="Checkpoint filename")
    parser.add_argument("--tokenizer", type=str, default="spm_hf_master.model", help="Tokenizer model file")
    parser.add_argument("--max_tokens", type=int, default=350, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.65, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.88, help="Nucleus sampling threshold")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling limit")
    parser.add_argument("--repetition_penalty", type=float, default=1.25, help="Penalty against repetition loops")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.join(BASE_DIR, args.model)
    tokenizer_path = os.path.join(BASE_DIR, args.tokenizer)

    if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
        print("[Mochi-88M] Missing model weights or tokenizer. Downloading from Hugging Face...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "download_model.py")], check=True)

    print("=" * 80)
    print(" 🍡 Mochi-88M - High Performance Story Inference Engine")
    print(f" Checkpoint : {os.path.basename(model_path)}")
    print(f" Device     : {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print("=" * 80)

    # 1. Load Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    # 2. Load Model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = ModernGPT(
        vocab_size=config.get("vocab_size", 4096),
        dim=config.get("dim", 768),
        n_layers=config.get("n_layers", 12),
        n_heads=config.get("n_heads", 12),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=0.0
    ).to(device)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Determine Prompt
    if args.prompt:
        user_prompt = args.prompt
    else:
        print("\nEnter your story prompt (or press Enter for default):")
        inp = input("> ").strip()
        if inp:
            user_prompt = inp
        else:
            user_prompt = (
                "Write a short story about Shashank, a proud young mechanic working in a highway tin shed, "
                "and Rajesh Khanna, a wealthy businessman whose luxury car breaks down on the desert road."
            )

    formatted_prompt = f"### Prompt:\n{user_prompt}\n\n### Story:\n"
    token_ids = sp.encode_as_ids(formatted_prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    print("\n" + "-" * 80)
    print(" GENERATING STORY (with KV-Cache & Nucleus Sampling)...")
    print("-" * 80)

    t0 = time.time()
    with torch.no_grad():
        output_tokens = model.fast_generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
    elapsed = time.time() - t0

    generated_ids = output_tokens[0].tolist()[len(token_ids):]
    story_text = sp.decode(generated_ids)
    word_count = len(story_text.split())
    speed = len(generated_ids) / max(0.01, elapsed)

    print(story_text)
    print("\n" + "=" * 80)
    print(f" Metrics: {len(generated_ids)} tokens ({word_count} words) in {elapsed:.2f}s | Speed: {speed:.1f} tok/s")
    print("=" * 80)

if __name__ == "__main__":
    main()
