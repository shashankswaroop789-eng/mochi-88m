"""
Tests generation quality of hf_master_gpt88m.pt using spm_hf_master.model.
"""

import os
import sys
import torch
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DATA_DIR)

from modern_transformer import ModernGPT

CHECKPOINT_PATH = os.path.join(DATA_DIR, "hf_finetuned_romance_gpt88m.pt")
TOKENIZER_MODEL = os.path.join(DATA_DIR, "spm_hf_master.model")

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_samples():
    print(f"Loading tokenizer: {TOKENIZER_MODEL}")
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = ckpt['config']
    
    model = ModernGPT(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len'],
        dropout=0.0
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
    print(f"Step: {ckpt['step']} | Train Loss: {ckpt['train_loss']:.4f} | Val Loss: {ckpt['val_loss']:.4f}\n")

    prompts = [
        "Stella sat at her desk in the sleek high-rise office, staring at the contract. When the door opened, Christian walked in, his dark gaze locking onto hers.",
        "\"You shouldn't have come here,\" she whispered. Christian took a step closer, his voice low and steady.",
        "The penthouse was quiet as the rain tapped against the floor-to-ceiling windows. Eva wrapped her cardigan tighter around herself as Gideon approached."
    ]

    for idx, prompt in enumerate(prompts):
        print("=" * 80)
        print(f" PROMPT {idx+1}:")
        print(prompt)
        print("-" * 80)
        print(" GENERATION:")

        token_ids = sp.encode_as_ids(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output_tokens = model.fast_generate(
                input_ids,
                max_new_tokens=250,
                temperature=0.72,
                top_k=40
            )

        generated_ids = output_tokens[0].tolist()
        generated_text = sp.decode(generated_ids)
        print(generated_text)
        print("=" * 80 + "\n")

if __name__ == "__main__":
    generate_samples()
