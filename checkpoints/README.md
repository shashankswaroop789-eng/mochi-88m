# 🍡 Mochi-88M Checkpoints

The official model checkpoints for **Mochi-88M** are hosted on the **Hugging Face Hub**:
🔗 **https://huggingface.co/Shashank1227/mochi-88m**

Because weights exceed GitHub's 100MB limit, `.pt` files are ignored in git and can be downloaded with 1 command:
```bash
python download_model.py
```

## Available Weights on Hugging Face
1. **`hf_coherent_story_gpt88m.pt` (Recommended)**: 
   - 88.1M parameter model fine-tuned with prompt loss masking on focused narrative short stories.
   - Ideal for creative writing, zero word salad, high coherence.
   - Download link: `https://huggingface.co/Shashank1227/mochi-88m/resolve/main/hf_coherent_story_gpt88m.pt`
2. **`spm_hf_master.model`**:
   - 4,096-token SentencePiece BPE tokenizer.
   - Download link: `https://huggingface.co/Shashank1227/mochi-88m/resolve/main/spm_hf_master.model`

## Checkpoint Format
```python
{
    "step": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "train_loss": float,
    "val_loss": float,
    "config": {
        "vocab_size": 4096,
        "dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "max_seq_len": 512,
        "dropout": 0.0
    }
}
```

## Loading Weights
```python
import torch
from modern_transformer import ModernGPT

ckpt = torch.load("hf_coherent_story_gpt88m.pt", map_location="cuda")
model = ModernGPT(**ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```
