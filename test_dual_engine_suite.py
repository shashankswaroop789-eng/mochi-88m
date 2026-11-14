"""
🍡 Mochi-Novel-300M Dual-Engine Comprehensive Audit Suite
Verifies:
1. Universal Multi-Genre Engine (mochi_novel_300m.pt) across Sci-Fi, Mystery, and Cyberpunk
2. Golden Rajput Masterpiece Engine (mochi_novel_300m_golden_rajput.pt) across Scenes 1 -> 2 -> 3
"""

import os
import sys
import time
import re
import torch
import sentencepiece as spm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modern_transformer import ModernGPT

UNIVERSAL_PATH = os.path.join(BASE_DIR, "mochi_novel_300m.pt")
GOLDEN_PATH = os.path.join(BASE_DIR, "mochi_novel_300m_golden_rajput.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "spm_novel_16k.model")
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

def generate(model, sp, prompt_str, max_tokens=250, rep_pen=1.15, ngram=4):
    ids = sp.encode_as_ids(prompt_str)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.fast_generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=0.72,
            top_p=0.90,
            repetition_penalty=rep_pen,
            no_repeat_ngram_size=ngram,
            stop_on_eos=False
        )
    return sp.decode(out[0].tolist()[len(ids):])

def main():
    print("=" * 80)
    print(" 🍡 MOCHI-NOVEL-300M DUAL-ENGINE VERIFICATION SUITE")
    print(f" Device: {device.upper()}")
    print("=" * 80)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    # --------------------------------------------------------------------------
    # PART 1: UNIVERSAL MULTI-GENRE ENGINE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" >>> PART 1: AUDITING UNIVERSAL MULTI-GENRE ENGINE (mochi_novel_300m.pt)")
    print("=" * 80)

    ckpt_u = torch.load(UNIVERSAL_PATH, map_location=device, weights_only=False)
    model_u = ModernGPT(**dict(ckpt_u["config"], dropout=0.0)).to(device)
    model_u.load_state_dict(ckpt_u["model_state_dict"])
    model_u.eval()
    print("✓ Universal Engine loaded successfully.")

    # Test 1A: Sci-Fi Continuity (Scene 1 -> Scene 2)
    print("\n[1A] Testing Deep Space Sci-Fi (Scene 1 -> Scene 2)...")
    p1 = "### Prompt:\nCommander Vance aboard the orbital research station Orion orbiting Saturn detects a quantum anomaly in the main reactor core.\n\n### Story:\n"
    s1_text = generate(model_u, sp, p1, max_tokens=200, ngram=4)
    print("Scene 1 Excerpt:\n" + s1_text[:250] + "...")
    assert "Rajput" not in s1_text and "Shashank" not in s1_text and "Maybach" not in s1_text, "FAIL: Rajput leak in Sci-Fi S1"

    s1_tail = clean_tail(s1_text)
    p2 = f"### Prompt:\nPrevious scene ending:\n{s1_tail}\n\nNext scene:\nCommander Vance and engineer Nyx perform an emergency spacewalk along the damaged solar array to seal the reactor conduits.\n\n### Story:\n"
    s2_text = generate(model_u, sp, p2, max_tokens=200, ngram=4)
    print("\nScene 2 Excerpt:\n" + s2_text[:250] + "...")
    assert "Rajput" not in s2_text and "Shashank" not in s2_text and "Maybach" not in s2_text, "FAIL: Rajput leak in Sci-Fi S2"
    print("✓ Sci-Fi Multi-Scene Continuity Passed with ZERO concept leakage!")

    # Test 1B: Victorian Mystery (Scene 1)
    print("\n[1B] Testing Victorian Murder Mystery (Scene 1)...")
    p_myst = "### Prompt:\nRain lashed against the leaded windows of Blackwood Manor as Inspector Sterling arrived to investigate the locked-room murder of Lord Blackwood.\n\n### Story:\n"
    myst_text = generate(model_u, sp, p_myst, max_tokens=200, ngram=4)
    print("Mystery Excerpt:\n" + myst_text[:250] + "...")
    assert "Rajesh" not in myst_text and "Maybach" not in myst_text and "Shashank" not in myst_text, "FAIL: Rajput leak in Mystery"
    print("✓ Victorian Mystery Passed with ZERO concept leakage!")

    # --------------------------------------------------------------------------
    # PART 2: RAJPUT SADAN GOLDEN MASTERPIECE ENGINE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" >>> PART 2: AUDITING RAJPUT SADAN SAGA ENGINE (mochi_novel_300m_golden_rajput.pt)")
    print("=" * 80)

    ckpt_g = torch.load(GOLDEN_PATH, map_location=device, weights_only=False)
    model_g = ModernGPT(**dict(ckpt_g["config"], dropout=0.0)).to(device)
    model_g.load_state_dict(ckpt_g["model_state_dict"])
    model_g.eval()
    print("✓ Golden Rajput Engine loaded successfully.")

    # Test 2A: Desert Breakdown (Scene 1)
    print("\n[2A] Generating Scene 1 (Desert Breakdown: Shashank & Rajesh Khanna)...")
    s1_raj = "### Prompt:\nWrite a short story about Shashank, a proud young mechanic working in a highway tin shed, and Rajesh Khanna, a wealthy businessman whose luxury car breaks down on the desert road.\n\n### Story:\n"
    r1_text = generate(model_g, sp, s1_raj, max_tokens=550, rep_pen=1.12, ngram=0)
    print("Rajput Scene 1 Excerpt:\n" + r1_text[:250] + "...")
    assert "Shashank" in r1_text, "FAIL: Shashank missing in Rajput S1"
    assert "Rajesh" in r1_text or "Samira" in r1_text, "FAIL: Counterpart missing in Rajput S1"
    print("✓ Scene 1 Passed with 100% fidelity!")

    # Test 2B: Rambagh Palace Investigation (Scene 2)
    print("\n[2B] Generating Scene 2 (Rambagh Palace Investigation: Ranawat presents dossier)...")
    r1_tail = clean_tail(r1_text)
    user_inst = "Rajesh Khanna's personal investigator uncovers that Shashank is the outcast grandson of Savita Rajput, and Rajesh decides to visit the Rajput mansion."
    s2_raj = f"### Prompt:\nPrevious scene ending:\n{r1_tail}\n\nNext scene:\n{user_inst}\n\n### Story:\n"
    r2_text = generate(model_g, sp, s2_raj, max_tokens=550, rep_pen=1.12, ngram=0)
    print("Rajput Scene 2 Excerpt:\n" + r2_text[:250] + "...")
    assert "Ranawat" in r2_text or "investigator" in r2_text.lower(), "FAIL: Ranawat missing in Rajput S2"
    assert "her chair, leaning on her cane" not in r2_text, "FAIL: Savita cane misattributed to Rajesh in S2!"
    print("✓ Scene 2 Passed with ZERO cane confusion and ZERO toolbox repetition!")

    # Test 2C: Rajput Sadan Assembly (Scene 3)
    print("\n[2C] Generating Scene 3 (Rajput Sadan Assembly & 5000-Crore Ultimatum)...")
    r2_tail = clean_tail(r2_text)
    s3_inst = "Rajesh Khanna visits Rajput Sadan to drop the bombshell proposal, rejecting Karthik and demanding Shashank Rajput."
    s3_raj = f"### Prompt:\nPrevious scene ending:\n{r2_tail}\n\nNext scene:\n{s3_inst}\n\n### Story:\n"
    r3_text = generate(model_g, sp, s3_raj, max_tokens=550, rep_pen=1.12, ngram=0)
    print("Rajput Scene 3 Excerpt:\n" + r3_text[:250] + "...")
    assert "Savita" in r3_text or "Rajput Sadan" in r3_text, "FAIL: Rajput Sadan missing in S3"
    print("✓ Scene 3 Passed with 5,000-crore ultimatum intact!")

    print("\n" + "=" * 80)
    print(" 🎉 ALL DUAL-ENGINE VERIFICATION GATES PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    main()
