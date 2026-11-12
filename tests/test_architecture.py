"""
Unit & Integration Tests for ModernGPT Architecture
Verifies RoPE, RMSNorm, SwiGLU, and KV-Cache token generation.
"""

import sys
import os

# Enable UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modern_transformer import (
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    SwiGLU,
    ModernGPT
)

def test_rmsnorm():
    print("[1/5] Testing RMSNorm...")
    norm = RMSNorm(dim=256)
    x = torch.randn(2, 16, 256)
    out = norm(x)
    assert out.shape == x.shape, f"RMSNorm output mismatch: {out.shape}"
    # Verify variance is normalized to ~1.0
    var = out.pow(2).mean(dim=-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2), "RMSNorm failed variance normalization."
    print("      PASSED")

def test_rope():
    print("[2/5] Testing Rotary Position Embeddings (RoPE)...")
    freqs = precompute_freqs_cis(dim=64, max_seq_len=128)
    q = torch.randn(2, 32, 4, 64)
    k = torch.randn(2, 32, 4, 64)
    q_rot, k_rot = apply_rotary_emb(q, k, freqs)
    assert q_rot.shape == q.shape, f"RoPE Q shape mismatch: {q_rot.shape}"
    assert k_rot.shape == k.shape, f"RoPE K shape mismatch: {k_rot.shape}"
    print("      PASSED")

def test_swiglu():
    print("[3/5] Testing SwiGLU Gated Feed-Forward...")
    dim = 256
    hidden_dim = 684
    swiglu = SwiGLU(dim=dim, hidden_dim=hidden_dim, dropout=0.0)
    x = torch.randn(2, 16, dim)
    out = swiglu(x)
    assert out.shape == x.shape, f"SwiGLU shape mismatch: {out.shape}"
    print("      PASSED")

def test_model_forward():
    print("[4/5] Testing ModernGPT Forward Pass & Cross-Entropy Loss...")
    model = ModernGPT(vocab_size=1000, dim=128, n_layers=2, n_heads=4, max_seq_len=64, dropout=0.0)
    tokens = torch.randint(0, 1000, (2, 32))
    targets = torch.randint(0, 1000, (2, 32))
    logits, loss, _ = model(tokens, targets)
    assert logits.shape == (2, 32, 1000), f"Logits shape mismatch: {logits.shape}"
    assert loss is not None and not torch.isnan(loss), "Loss computation failed or produced NaN"
    print("      PASSED")

def test_kv_cache_generation():
    print("[5/5] Testing KV-Cache Fast Generation...")
    model = ModernGPT(vocab_size=1000, dim=128, n_layers=2, n_heads=4, max_seq_len=64, dropout=0.0)
    prompt = torch.randint(0, 1000, (1, 10))
    gen = model.fast_generate(prompt, max_new_tokens=15, temperature=0.7, top_k=20)
    assert gen.shape == (1, 25), f"Generated sequence length mismatch: {gen.shape}"
    print("      PASSED")

if __name__ == "__main__":
    print("=" * 60)
    print(" Running 🍡 Mochi-88M Architectural Self-Verification Suite")
    print("=" * 60)
    test_rmsnorm()
    test_rope()
    test_swiglu()
    test_model_forward()
    test_kv_cache_generation()
    print("=" * 60)
    print(" ALL 🍡 Mochi-88M ARCHITECTURAL TESTS PASSED (100% OPERATIONAL)!")
    print("=" * 60)
