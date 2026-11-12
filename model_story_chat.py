"""
Model Story Chat: Interactive, Non-Hardcoded Novel Generator using your Scratch Model
Features:
- Powered 100% by your local ModernGPT 88.1M model (no APIs, no hardcoded templates)
- Generates Chapter 1 from your custom prompt
- Asks for instructions for Chapter 2 (or automatically continues if Enter is pressed)
- Rolling context window with KV-caching and vectorized repetition penalty
- Sentence boundary completion
"""

import os
import sys
import re
import time
import argparse
import torch
import sentencepiece as spm

# Force utf-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DATA_DIR)

from modern_transformer import ModernGPT

CHECKPOINT_PATH = os.path.join(DATA_DIR, "hf_universal_story_gpt88m.pt")
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = os.path.join(DATA_DIR, "hf_master_gpt88m.pt")
TOKENIZER_PATH = os.path.join(DATA_DIR, "spm_hf_master.model")
OUTPUT_DIR = os.path.join(DATA_DIR, "novel_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_engine():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint {CHECKPOINT_PATH} not found!")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer {TOKENIZER_PATH} not found!")

    print(f"[ModelEngine] Loading Tokenizer from {TOKENIZER_PATH}...")
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    print(f"[ModelEngine] Loading 88.1M Model Checkpoint from {CHECKPOINT_PATH} onto {device}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    
    model = ModernGPT(
        vocab_size=config.get('vocab_size', 4096),
        dim=config.get('dim', 768),
        n_layers=config.get('n_layers', 12),
        n_heads=config.get('n_heads', 12),
        max_seq_len=config.get('max_seq_len', 512),
        dropout=0.0
    ).to(device)

    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[ModelEngine] 88.1M Model Ready on {device}!\n")
    return model, sp

def complete_to_sentence_end(text):
    """Trims trailing unfinished sentences."""
    match = re.search(r'[\.\!\?]["\']?\s*$', text)
    if match:
        return text.strip()
    last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_punct != -1:
        return text[:last_punct + 1].strip()
    return text.strip()

def generate_chapter_from_scratch_model(
    model, 
    sp, 
    prompt_text, 
    target_words=1200, 
    temperature=0.62, 
    top_k=40, 
    top_p=0.88, 
    repetition_penalty=1.25
):
    """
    Generates a full chapter purely from the model starting from prompt_text.
    Maintains a rolling context window so it stays within max_seq_len (512 tokens).
    """
    token_ids = sp.encode_as_ids(prompt_text)
    if not token_ids:
        token_ids = [0]

    all_generated_tokens = list(token_ids)
    word_count = len(sp.decode(all_generated_tokens).split())

    print(f"Generating ~{target_words:,} words using your 88.1M scratch model...")
    t0 = time.time()
    burst_count = 0

    while word_count < target_words:
        # Context window: keep the last 384 tokens
        context_ids = all_generated_tokens[-384:]
        idx = torch.tensor([context_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output_tokens = model.fast_generate(
                idx,
                max_new_tokens=50,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        new_ids = output_tokens[0].tolist()[len(context_ids):]
        all_generated_tokens.extend(new_ids)
        burst_count += 1
        
        current_text = sp.decode(all_generated_tokens)
        word_count = len(current_text.split())

        if burst_count % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [Progress] {word_count:,} / {target_words:,} words generated ({elapsed:.1f}s)...")

    full_text = sp.decode(all_generated_tokens)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = complete_to_sentence_end(full_text)
    final_words = len(full_text.split())
    print(f"[Done] Generated {final_words:,} words in {time.time()-t0:.1f}s!\n")
    return full_text

def run_chat_session(initial_prompt, target_words=1200, auto=False):
    model, sp = load_engine()
    
    chapter_num = 1
    current_prompt = initial_prompt
    cumulative_story = ""

    while True:
        print("=" * 80)
        print(f" GENERATING CHAPTER {chapter_num} (SCRATCH MODEL: ModernGPT 88.1M)")
        print("=" * 80)

        chapter_text = generate_chapter_from_scratch_model(
            model,
            sp,
            current_prompt,
            target_words=target_words
        )

        filename = f"rajput_scratch_chapter_{chapter_num:02d}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"THE RAJPUT LEGACY\nCHAPTER {chapter_num}\n\n")
            f.write(chapter_text)

        print(f">>> Chapter {chapter_num} saved to: {filepath}\n")
        print("-" * 80)
        print("PREVIEW OF MODEL GENERATION (FIRST 600 WORDS):")
        words_preview = chapter_text.split()[:600]
        print(" ".join(words_preview))
        print("...\n" + "-" * 80)

        cumulative_story += f"\n\n--- CHAPTER {chapter_num} ---\n\n" + chapter_text

        next_ch = chapter_num + 1
        print(f"\n================================================================================")
        print(f"  CHAPTER {chapter_num} IS COMPLETE!")
        print(f"  What should happen in CHAPTER {next_ch}?")
        print(f"  Type your instructions for Chapter {next_ch}, OR")
        print(f"  Press [Enter] to let the model decide what is best:")
        print(f"================================================================================\n")

        if auto:
            print(f"[Auto Mode] Generating Chapter {next_ch} continuation automatically...")
            current_prompt = chapter_text[-600:] + f"\n\nChapter {next_ch} continues the story:\n"
        else:
            try:
                user_instruction = input(f"Chapter {next_ch} Instructions > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended by user.")
                break

            if user_instruction:
                print(f"\n>> Incorporating your instructions into Chapter {next_ch}...")
                current_prompt = chapter_text[-400:] + f"\n\nIn the next scene: {user_instruction}\n"
            else:
                print(f"\n>> [Enter] pressed. Model will continue the story naturally...")
                current_prompt = chapter_text[-600:] + f"\n\nMeanwhile, the tension escalated as:\n"

        chapter_num = next_ch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Story Chat")
    parser.add_argument("--prompt", type=str, default=None, help="Initial story prompt")
    parser.add_argument("--words", type=int, default=1200, help="Target words per chapter")
    parser.add_argument("--auto", action="store_true", help="Auto proceed without interactive pause")
    args = parser.parse_args()

    default_prompt = (
        "The grand Rajput family gathered inside the opulent Rajput Sadan in Jaipur. "
        "Savita Rajput, the matriarch, sat at the head of the table surrounded by her sons Viraj and Avinash, "
        "and her grandsons Aman, Vijay, and Karthik. The entire family was filled with excitement and happiness, "
        "believing that the marriage between Karthik Rajput and Samira Khanna, the billionaire heiress, was destined to happen. "
        "They celebrated the golden alliance, deeply caring for one another. "
        "Yet, far away on the dusty bypass, lived Shashank Rajput, the unacknowledged son born in the shadows, "
        "working as a frail mechanic in a rusted tin shed. The Rajput family despised him with a unified, merciless hatred."
    )

    prompt = args.prompt or default_prompt
    run_chat_session(prompt, target_words=args.words, auto=args.auto)
