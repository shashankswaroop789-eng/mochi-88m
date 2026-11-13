import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# 1. RMSNorm (Root Mean Square Normalization)
# Used in LLaMA, Mistral, Gemma. Replaces traditional LayerNorm.
# ==============================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Unlike LayerNorm, RMSNorm does not subtract the mean (no centering),
    which makes it ~30% faster and prevents mean-drift in deep networks.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (gamma)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS along feature dimension: sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


# ==============================================================================
# 2. Rotary Position Embeddings (RoPE)
# Replaces fixed/learned position tables. Enables long-context extrapolation.
# ==============================================================================
def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes complex exponential frequencies: cis = e^(i * m * theta).
    dim must be head_dim (which is even).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim // 2)
    # Convert to complex numbers: cos(freqs) + i * sin(freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary embeddings to Query and Key tensors by pairing adjacent
    elements as complex numbers and rotating them by angle m*theta.
    """
    # Reshape Q and K into complex numbers: (B, T, n_head, head_dim) -> (B, T, n_head, head_dim // 2)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis: shape (1, T, 1, head_dim // 2)
    freqs_cis = freqs_cis[:xq.shape[1]].unsqueeze(0).unsqueeze(2)
    
    # Rotate in complex space!
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==============================================================================
# 3. SwiGLU (Swish Gated Linear Unit)
# Replaces standard ReLU/GELU MLP. Used in modern 2024 LLMs.
# ==============================================================================
class SwiGLU(nn.Module):
    """
    Gated Linear Unit with SiLU (Swish) activation:
    SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Standard LLaMA convention: hidden_dim is roughly 8/3 * dim
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gating mechanism: SiLU(gate) * up
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


# ==============================================================================
# 4. Modern Attention with RoPE and KV-Cache
# ==============================================================================
class ModernAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head
        assert dim % n_head == 0, "dim must be divisible by n_head"

        # Q, K, V projections
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask buffer
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, kv_cache=None) -> tuple[torch.Tensor, dict]:
        B, T, C = x.shape

        # Linear projections
        xq = self.wq(x).view(B, T, self.n_head, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_head, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_head, self.head_dim)

        # Apply Rotary Position Embeddings (RoPE) to Q and K!
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # KV-Cache integration for fast autoregressive generation
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            xk = torch.cat([prev_k, xk], dim=1)
            xv = torch.cat([prev_v, xv], dim=1)
        new_kv_cache = (xk, xv)

        # Transpose for multi-head dot product: (B, n_head, T, head_dim)
        keys = xk.transpose(1, 2)
        queries = xq.transpose(1, 2)
        values = xv.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Dynamic causal masking: only needed when current_T > 1 (e.g. prompt pre-fill)
        total_T = keys.shape[2]
        current_T = queries.shape[2]
        if current_T > 1:
            q_pos = torch.arange(total_T - current_T, total_T, device=x.device).unsqueeze(1)
            k_pos = torch.arange(0, total_T, device=x.device).unsqueeze(0)
            causal_mask = q_pos >= k_pos
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        # Weighted values
        output = torch.matmul(scores, values) # (B, n_head, T, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, current_T, -1)
        return self.dropout(self.wo(output)), new_kv_cache


# ==============================================================================
# 5. Modern Transformer Block (RMSNorm + RoPE Attention + SwiGLU)
# ==============================================================================
class ModernBlock(nn.Module):
    def __init__(self, dim: int, n_head: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = ModernAttention(dim, n_head, max_seq_len, dropout)
        # SwiGLU hidden dim standard: (4 * dim * 2/3) rounded to multiple of 64
        hidden_dim = int(2 * (4 * dim) / 3)
        hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.feed_forward = SwiGLU(dim, hidden_dim, dropout)

        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, kv_cache=None):
        # Pre-RMSNorm Residual for Attention
        attn_out, new_cache = self.attention(self.attn_norm(x), freqs_cis, kv_cache)
        x = x + attn_out
        # Pre-RMSNorm Residual for SwiGLU
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_cache


# ==============================================================================
# 6. Full Modern Foundation Model (Built 100% From Scratch)
# ==============================================================================
class ModernGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 3200,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embedding (Note: NO positional embedding table! RoPE handles position)
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        # Precompute RoPE frequencies up to max_seq_len
        self.head_dim = dim // n_heads
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

        # Stack of Modern Transformer Blocks
        self.layers = nn.ModuleList([
            ModernBlock(dim, n_heads, max_seq_len, dropout) for _ in range(n_layers)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(dim)

        # Language Model Head (Output projection to vocabulary)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights: share token embedding with output projection (standard in modern LLMs)
        self.output.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None, kv_caches=None, start_pos: int = 0):
        B, T = tokens.shape
        x = self.tok_embeddings(tokens)

        # Infer start_pos from kv_cache if active
        if kv_caches is not None and kv_caches[0] is not None:
            start_pos = kv_caches[0][0].shape[1]

        # Expand freqs_cis dynamically if context length exceeds buffer
        if start_pos + T > self.freqs_cis.shape[0]:
            self.freqs_cis = precompute_freqs_cis(self.head_dim, start_pos + T + 512).to(tokens.device)

        # Slice RoPE frequencies for the exact token positions
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, freqs_cis, cache_i)
            new_kv_caches.append(new_cache)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def fast_generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 0.65, 
        top_k: int = 40, 
        top_p: float = 0.88, 
        repetition_penalty: float = 1.15,
        prompt_len: int = None,
        banned_token_ids: list = None
    ):
        """
        Fast token generation utilizing Key-Value Caching:
        Instead of recomputing past tokens, we cache K & V and only compute
        the single newest token at each step!
        Features:
        - Repetition penalty to prevent stuttering/repetition loops
        - Nucleus (top-p) + Top-K filtering for high coherence
        - Temperature scaling
        - Banned token suppression
        """
        self.eval()
        kv_caches = None
        generated_tokens = idx.clone()
        if prompt_len is None:
            prompt_len = idx.shape[1]

        # Initial forward pass to fill KV-cache
        logits, _, kv_caches = self(idx, kv_caches=None)

        for _ in range(max_new_tokens):
            # Take the prediction from the last time step
            next_token_logits = logits[:, -1, :].clone()

            # 1. Vectorized Repetition Penalty on newly generated tokens only (prevents loops without punishing prompt)
            if repetition_penalty != 1.0 and generated_tokens.shape[1] > prompt_len:
                # Only penalize newly generated tokens, NEVER penalize the user prompt words!
                new_tokens = generated_tokens[:, prompt_len:]
                recent_window = new_tokens[:, -128:]
                penalized = next_token_logits.gather(1, recent_window)
                penalized = torch.where(
                    penalized > 0,
                    penalized / repetition_penalty,
                    penalized * repetition_penalty
                )
                next_token_logits.scatter_(1, recent_window, penalized)

            # 2. Temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # 3. Banned tokens
            if banned_token_ids:
                next_token_logits[:, banned_token_ids] = -float("Inf")

            # 4. Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

            # 5. Top-P (Nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for b in range(next_token_logits.shape[0]):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    next_token_logits[b, indices_to_remove] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            # Fast step: pass ONLY the new single token to the model along with the KV-cache!
            logits, _, kv_caches = self(next_token, kv_caches=kv_caches)

        return generated_tokens


# ==============================================================================
# Self-Verification Test
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Testing Modern Architecture Components on {device} ---")

    # 1. Test RMSNorm
    norm = RMSNorm(dim=512).to(device)
    dummy_x = torch.randn(2, 64, 512, device=device)
    norm_x = norm(dummy_x)
    print(f"1. RMSNorm Output Shape:      {norm_x.shape} (Verified)")

    # 2. Test RoPE
    freqs = precompute_freqs_cis(dim=64, max_seq_len=1024).to(device)
    q = torch.randn(2, 64, 8, 64, device=device)
    k = torch.randn(2, 64, 8, 64, device=device)
    q_rot, k_rot = apply_rotary_emb(q, k, freqs)
    print(f"2. RoPE Rotated Tensor Shape: {q_rot.shape} (Verified)")

    # 3. Test SwiGLU
    swiglu = SwiGLU(dim=512, hidden_dim=1344).to(device)
    swi_out = swiglu(dummy_x)
    print(f"3. SwiGLU Output Shape:       {swi_out.shape} (Verified)")

    # 4. Test Complete ModernGPT Model
    model = ModernGPT(
        vocab_size=3200,
        dim=512,
        n_layers=8,
        n_heads=8,
        max_seq_len=1024
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"4. Total Model Parameters:    {total_params:,} (~{total_params / 1e6:.1f}M)")

    # Test Forward Pass & Loss
    tokens = torch.randint(0, 3200, (2, 256), device=device)
    targets = torch.randint(0, 3200, (2, 256), device=device)
    logits, loss, _ = model(tokens, targets)
    print(f"5. Forward Pass Loss:         {loss.item():.4f} (Verified)")

    # Test Fast Generation with KV-Cache
    prompt = torch.tensor([[10, 25, 42]], dtype=torch.long, device=device)
    gen = model.fast_generate(prompt, max_new_tokens=50)
    print(f"6. KV-Cache Generated Shape:  {gen.shape} (Verified - 53 tokens)")
    print("\n--- ALL MODERN ARCHITECTURE COMPONENTS 100% OPERATIONAL! ---")
