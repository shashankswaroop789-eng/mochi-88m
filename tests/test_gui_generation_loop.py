import sys
import os
import re
import time
import torch
import sentencepiece as spm

sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from modern_transformer import ModernGPT

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(ROOT_DIR, "spm_novel_16k.model"))

ckpt = torch.load(os.path.join(ROOT_DIR, "mochi_novel_300m.pt"), map_location="cuda", weights_only=False)
cfg = ckpt["config"]
model = ModernGPT(
    vocab_size=cfg["vocab_size"],
    dim=cfg["dim"],
    n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"],
    max_seq_len=cfg["max_seq_len"],
    dropout=0.0
).cuda()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

prompt_text = "### Prompt:\nWrite a short story about Shashank, a proud young mechanic working in a highway tin shed, and Samira Khanna a daughter of wealthy businessman whose luxury car breaks down on the desert road.\n\n### Story:\n"
target_words = 800
temperature = 0.70
repetition_penalty = 1.12

token_ids = sp.encode_as_ids(prompt_text)
all_tokens = list(token_ids)
initial_len = len(all_tokens)
last_text_len = 0

t0 = time.time()
total_tokens_generated = 0

print("--- Simulating GUI Streaming Generation Loop (Target = 800 words) ---")

while True:
    context_ids = all_tokens[-1536:]
    idx = torch.tensor([context_ids], dtype=torch.long, device="cuda")
    prompt_in_ctx = max(0, initial_len - (len(all_tokens) - len(context_ids)))

    gen_story_tokens = all_tokens[initial_len:]
    current_text = sp.decode(gen_story_tokens)
    current_words = len(current_text.split())

    allow_eos = (current_words >= min(450, int(target_words * 0.70)))
    banned_tokens = [] if allow_eos else [2]

    with torch.no_grad():
        output_tokens = model.fast_generate(
            idx,
            max_new_tokens=25,
            temperature=temperature,
            top_k=40,
            top_p=0.88,
            repetition_penalty=repetition_penalty,
            prompt_len=prompt_in_ctx,
            no_repeat_ngram_size=0,
            stop_on_eos=allow_eos,
            banned_token_ids=banned_tokens
        )

    new_ids = output_tokens[0].tolist()[len(context_ids):]
    if not new_ids:
        print(">> No new ids, breaking loop.")
        break

    eos_hit = (2 in new_ids)
    if eos_hit:
        eos_pos = new_ids.index(2)
        new_ids = new_ids[:eos_pos]

    all_tokens.extend(new_ids)
    total_tokens_generated += len(new_ids)

    gen_story_tokens = all_tokens[initial_len:]
    current_text = sp.decode(gen_story_tokens)
    current_words = len(current_text.split())

    if "###" in current_text and current_words > 200:
        print(">> '###' detected, breaking loop.")
        break

    if eos_hit and allow_eos:
        print(f">> Natural EOS termination at {current_words} words!")
        break

    if current_words >= target_words:
        print(f">> Target word count reached ({current_words} >= {target_words})!")
        break

# Trimming
full_gen_text = sp.decode(all_tokens[initial_len:])
if "###" in full_gen_text:
    full_gen_text = full_gen_text.split("###")[0].strip()
full_gen_text = re.sub(r"\n{3,}", "\n\n", full_gen_text)

match = re.search(r'[\.\!\?]["\']?\s*$', full_gen_text)
if not match:
    last_punct = max(full_gen_text.rfind('.'), full_gen_text.rfind('!'), full_gen_text.rfind('?'))
    if last_punct != -1:
        full_gen_text = full_gen_text[:last_punct + 1].strip()

final_words = len(full_gen_text.split())
print("\n" + "=" * 80)
print(f"FINAL GENERATED STORY ({final_words} words):")
print("=" * 80)
print(full_gen_text)
print("=" * 80)
print(f"Last 150 characters: {repr(full_gen_text[-150:])}")
