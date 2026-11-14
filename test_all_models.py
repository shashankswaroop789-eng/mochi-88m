import os
import sys
import torch
import sentencepiece as spm

sys.path.insert(0, r"C:\Users\user\.gemini\antigravity\scratch\gpt_from_scratch")
from modern_transformer import ModernGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

models_to_test = [
    ("hf_rajput_dynasty_gpt88m.pt", "spm_hf_master.model"),
    ("hf_coherent_story_gpt88m.pt", "spm_hf_master.model"),
    ("hf_universal_story_gpt88m.pt", "spm_hf_master.model"),
    ("hf_mochi88m_master.pt", "spm_hf_master.model"),
]

prompt = "### Prompt:\nWrite a short story about Shashank, a proud young mechanic working in a highway tin shed, and Rajesh Khanna, a wealthy businessman whose luxury car breaks down on the desert road.\n\n### Story:\n"

for model_file, spm_file in models_to_test:
    model_path = os.path.join(r"C:\Users\user\.gemini\antigravity\scratch\gpt_from_scratch", model_file)
    spm_path = os.path.join(r"C:\Users\user\.gemini\antigravity\scratch\gpt_from_scratch", spm_file)
    if not os.path.exists(model_path) or not os.path.exists(spm_path):
        continue

    print(f"\n{'='*25} Testing {model_file} {'='*25}")
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(spm_path)

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = ModernGPT(
            vocab_size=cfg.get("vocab_size", 4096),
            dim=cfg.get("dim", 768),
            n_layers=cfg.get("n_layers", 12),
            n_heads=cfg.get("n_heads", 12),
            max_seq_len=cfg.get("max_seq_len", 512),
            dropout=0.0
        ).to(device)

        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        model.eval()

        ids = sp.encode_as_ids(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.fast_generate(
                idx,
                max_new_tokens=300,
                temperature=0.7,
                repetition_penalty=1.12,
                no_repeat_ngram_size=0,
                stop_on_eos=False
            )

        gen_tokens = out[0].tolist()[len(ids):]
        text = sp.decode(gen_tokens)
        print(f"Generated ({len(text.split())} words):\n{text[:600]}\n...")
    except Exception as e:
        print(f"Error testing {model_file}: {e}")
