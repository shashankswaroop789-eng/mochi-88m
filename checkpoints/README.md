# 🍡 Mochi-88M Checkpoints

This directory is used to store trained PyTorch model checkpoints (`.pt` files).
Because model weights exceed GitHub's 100MB file limit, they are tracked via `.gitignore`.

## Available Weights
1. **`hf_coherent_story_gpt88m.pt` (Mochi-88M Recommended)**: 
   - 88.1M parameter model fine-tuned with prompt loss masking on focused narrative short stories.
   - Ideal for creative writing, zero word salad, high coherence.
2. **`hf_master_gpt88m.pt` (Mochi-88M Foundation)**:
   - 88.1M parameter foundation model pre-trained on modern English narrative corpora (~30M tokens).

## Checkpoint Format
Each checkpoint contains:
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

To load a checkpoint:
```python
import torch
from modern_transformer import ModernGPT

ckpt = torch.load("hf_coherent_story_gpt88m.pt", map_location="cuda")
model = ModernGPT(**ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
```
