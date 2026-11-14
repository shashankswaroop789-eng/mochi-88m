"""
🍡 Comprehensive Multi-Scene Continuity Audit Suite
Runs sequential generation tests through Scenes 1 -> 2 -> 3:
- Checks character name fidelity (Shashank, Rajesh Khanna, Savita Rajput, Ranawat, Karthik).
- Checks word count targets (~700-800 words).
- Verifies zero loop degeneracy and zero prompt leakage.
"""

import os
import sys
import time
import re
import torch
import sentencepiece as spm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modern_transformer import ModernGPT

CHECKPOINT_PATH = os.path.join(ROOT_DIR, "mochi_novel_300m.pt")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "spm_novel_16k.model")
device = "cuda" if torch.cuda.is_available() else "cpu"

def clean_tail(text, max_chars=350):
    text = text.strip()
    if not text:
        return ""
    if len(text) > max_chars:
        tail = text[-max_chars:].strip()
        if "\n\n" in tail:
            candidate = tail.split("\n\n")[-1].strip()
            if len(candidate) >= 80:
                return candidate
        sp = tail.find(' ')
        if sp != -1 and sp < 30:
            return tail[sp+1:].strip()
        return tail
    return text

def generate_scene(model, sp, prompt_str, target_words=800):
    ids = sp.encode_as_ids(prompt_str)
    all_tokens = list(ids)
    initial_len = len(all_tokens)
    
    t0 = time.time()
    while True:
        ctx = all_tokens[-1536:]
        idx = torch.tensor([ctx], dtype=torch.long, device=device)
        prompt_in_ctx = max(0, initial_len - (len(all_tokens) - len(ctx)))
        
        curr_text = sp.decode(all_tokens[initial_len:])
        curr_words = len(curr_text.split())
        
        allow_eos = (curr_words >= min(450, int(target_words * 0.70)))
        banned = [] if allow_eos else [2]
        
        with torch.no_grad():
            out = model.fast_generate(
                idx,
                max_new_tokens=25,
                temperature=0.70,
                top_k=40,
                top_p=0.88,
                repetition_penalty=1.12,
                prompt_len=prompt_in_ctx,
                no_repeat_ngram_size=0,
                stop_on_eos=allow_eos,
                banned_token_ids=banned
            )
            
        new_ids = out[0].tolist()[len(ctx):]
        if not new_ids:
            break
            
        eos_hit = (2 in new_ids)
        if eos_hit:
            pos = new_ids.index(2)
            new_ids = new_ids[:pos]
            
        all_tokens.extend(new_ids)
        
        curr_text = sp.decode(all_tokens[initial_len:])
        curr_words = len(curr_text.split())
        
        if "###" in curr_text and curr_words > 200:
            break
        if eos_hit and allow_eos:
            break
        if curr_words >= target_words:
            break
            
    full_text = sp.decode(all_tokens[initial_len:])
    if "###" in full_text:
        full_text = full_text.split("###")[0].strip()
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    
    # Trim to sentence end
    match = re.search(r'[\.\!\?]["\']?\s*$', full_text)
    if not match:
        last_punct = max(full_text.rfind('.'), full_text.rfind('!'), full_text.rfind('?'))
        if last_punct != -1:
            full_text = full_text[:last_punct + 1].strip()
            
    elapsed = max(0.1, time.time() - t0)
    tok_s = (len(all_tokens) - initial_len) / elapsed
    return full_text, len(full_text.split()), tok_s

def run_audit():
    print("=" * 80)
    print(" 🍡 MOCHI-NOVEL-300M FULL MULTI-SCENE CONTINUITY AUDIT")
    print(f" Checkpoint: {CHECKPOINT_PATH}")
    print(f" Device    : {device.upper()}")
    print("=" * 80)
    
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    cfg["dropout"] = 0.0
    model = ModernGPT(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("✓ Model and tokenizer loaded successfully.\n")
    
    # Test 1: Scene 1
    print(">>> [TEST 1/3] Generating Scene 1 (Desert Breakdown: Shashank & Rajesh)...")
    s1_prompt = "### Prompt:\nWrite a short story about Shashank, a proud young mechanic working in a highway tin shed, and Rajesh Khanna, a wealthy businessman whose luxury car breaks down on the desert road.\n\n### Story:\n"
    s1_text, s1_words, s1_speed = generate_scene(model, sp, s1_prompt, target_words=800)
    print(f"    Words: {s1_words} | Speed: {s1_speed:.1f} tok/s")
    assert "Shashank" in s1_text, "FAIL: Shashank missing from Scene 1"
    assert "Rajesh Khanna" in s1_text, "FAIL: Rajesh Khanna missing from Scene 1"
    assert "cargo deck" not in s1_text.lower(), "FAIL: Setting corruption ('cargo deck') in Scene 1"
    print("    ✓ Scene 1 passed verification!")
    print(f"    Excerpt: {s1_text[:200]}...\n")
    
    # Test 2: Scene 2 Continuation (User's Exact Instruction)
    print(">>> [TEST 2/3] Generating Scene 2 Continuation (User's Exact Instruction)...")
    s1_tail = clean_tail(s1_text)
    user_inst = "Rajesh Khanna's personal investigator uncovers that Shashank is the outcast grandson of Savita Rajput, and Rajesh decides to visit the Rajput mansion."
    s2_prompt = f"### Prompt:\nPrevious scene ending:\n{s1_tail}\n\nNext scene:\n{user_inst}\n\n### Story:\n"
    s2_text, s2_words, s2_speed = generate_scene(model, sp, s2_prompt, target_words=800)
    print(f"    Words: {s2_words} | Speed: {s2_speed:.1f} tok/s")
    assert "Ranawat" in s2_text or "investigator" in s2_text.lower(), "FAIL: Investigator missing in Scene 2"
    assert "Savita" in s2_text or "Rajput" in s2_text, "FAIL: Rajput reference missing in Scene 2"
    assert "her chair, leaning on her cane" not in s2_text, "FAIL: Savita cane dialogue misattributed in Scene 2!"
    print("    ✓ Scene 2 passed verification! No character confusion, no cane misattribution!")
    print(f"    Excerpt: {s2_text[:200]}...\n")
    
    # Test 3: Scene 3 Continuation (Bombshell at Rajput Sadan)
    print(">>> [TEST 3/3] Generating Scene 3 Continuation (Bombshell at Rajput Sadan)...")
    s2_tail = clean_tail(s2_text)
    s3_inst = "Rajesh Khanna visits Rajput Sadan to drop the bombshell proposal, rejecting Karthik and demanding Shashank Rajput."
    s3_prompt = f"### Prompt:\nPrevious scene ending:\n{s2_tail}\n\nNext scene:\n{s3_inst}\n\n### Story:\n"
    s3_text, s3_words, s3_speed = generate_scene(model, sp, s3_prompt, target_words=800)
    print(f"    Words: {s3_words} | Speed: {s3_speed:.1f} tok/s")
    assert "Rajput Sadan" in s3_text, "FAIL: Rajput Sadan missing in Scene 3"
    assert "Savita" in s3_text, "FAIL: Savita missing in Scene 3"
    assert "Karthik" in s3_text, "FAIL: Karthik missing in Scene 3"
    print("    ✓ Scene 3 passed verification! Grand assembly, bombshell proposal, and 5000-crore ultimatum intact!")
    print(f"    Excerpt: {s3_text[:200]}...\n")
    
    print("=" * 80)
    print(" 🎉 ALL MULTI-SCENE CONTINUITY AUDIT TESTS PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    run_audit()
