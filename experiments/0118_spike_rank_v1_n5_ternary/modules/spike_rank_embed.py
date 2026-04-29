"""Spike-rank embedding v1 (variant A: sparse-only).

Each token's embedding row has K nonzero values. Top-K STE for trainability.
Tied lm_head reuses the same sparse matrix.

References:
- Math: scratch/2026-04-29_spike_rank_embedding_derivation.md
- Toys: scratch/2026-04-29_spike_rank_v1_tiny.py
- Brief option (c).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSTE(torch.autograd.Function):
    """Forward: keep top-K |w| per row, zero rest. Backward: STE.

    Identical mechanism to dendrocentric v1's TopKSTE — could share if 0117 is also
    in this experiment's modules/ tree, but for safety we duplicate (subdir-isolated).
    """
    @staticmethod
    def forward(ctx, w, k):
        abs_w = w.abs()
        _, topk_idx = abs_w.topk(k, dim=-1)
        mask = torch.zeros_like(w)
        mask.scatter_(1, topk_idx, 1.0)
        return w * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def topk_ste(w, k):
    return TopKSTE.apply(w, k)


class SpikeRankEmbed(nn.Module):
    """Sparse token embedding. Top-K nonzero values per row via STE.

    Drop-in replacement for nn.Embedding with tied lm_head. Use:
        embed = SpikeRankEmbed(V, d, K)
        e = embed(token_ids)         # (..., d) -- replaces nn.Embedding lookup
        logits = embed.lm_head(x)    # (..., V) -- replaces tied-weight matmul
    """
    def __init__(self, vocab_size: int, d_model: int, k: int):
        super().__init__()
        self.V = vocab_size
        self.d = d_model
        self.K = k
        # Continuous full param. Init scale matches existing TIED_EMBED_INIT_STD for
        # warm-start parity; the dense init is rendered K-sparse via top-K STE.
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.05)

    def _sparse_weight(self):
        return topk_ste(self.weight, self.K)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed lookup. Equivalent to F.embedding(token_ids, sparse(weight))."""
        return F.embedding(token_ids, self._sparse_weight())

    def lm_head(self, x: torch.Tensor) -> torch.Tensor:
        """Tied lm_head. Equivalent to F.linear(x, sparse(weight))."""
        return F.linear(x, self._sparse_weight())
