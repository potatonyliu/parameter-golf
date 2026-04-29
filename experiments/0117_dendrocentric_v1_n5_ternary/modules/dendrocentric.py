"""Dendrocentric layer v1: sparse-selection MLP replacement (no order sensitivity).

References:
- Boahen 2022 (Nature 612), Box 2 + "The dendrite of a pyramidal neuron"
- Math: scratch/2026-04-29_dendrocentric_v1_derivation.md
- Toys: scratch/2026-04-29_dendrocentric_v1_tiny.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSTE(torch.autograd.Function):
    """Forward: keep top-K |w| per row, zero rest. Backward: STE (gradient flows to all positions).

    Trainability bridge for emergent sparse pattern. Mirrors BitLinear's STE for
    ternary quantization but applied to sparsity instead of value-quantization.
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
        # Bypass top-K mask — gradient flows to all positions.
        return grad_output, None


def topk_ste(w, k):
    return TopKSTE.apply(w, k)


class DendrocentricLayer(nn.Module):
    """Replace MLP block with a dendrite bank.

    Each dendrite j reads from K-sparse input channels (via top-K STE on dendro_W_d),
    sums weighted contributions, applies sigmoidal NMDA-like nonlinearity, and
    contributes to a ternary BitLinear output projection.
    """
    def __init__(self, d_model: int, n_dendrites: int, k_per_dendrite: int, alpha: float = 1.0):
        super().__init__()
        from modules.bitlinear import BitLinear
        self.d_model = d_model
        self.M = n_dendrites
        self.K = k_per_dendrite
        self.alpha = alpha
        # Continuous full-rank parameter; rendered K-sparse via top-K STE in forward.
        # Substring 'dendro_' in CONTROL_TENSOR_NAME_PATTERNS keeps these fp32-protected.
        self.dendro_W_d = nn.Parameter(torch.randn(n_dendrites, d_model) * 0.02)
        # Per-dendrite threshold (1D, auto-fp32 since 1D).
        self.dendro_theta = nn.Parameter(torch.zeros(n_dendrites))
        # Output projection M -> d. Use BitLinear (ternary) to match the rest of the body.
        self.dendro_W_out = BitLinear(n_dendrites, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        W_d_sparse = topk_ste(self.dendro_W_d, self.K)  # (M, d), K nonzeros per row
        a = F.linear(x, W_d_sparse) - self.dendro_theta  # (..., M)
        h = torch.sigmoid(self.alpha * a)
        return self.dendro_W_out(h)
