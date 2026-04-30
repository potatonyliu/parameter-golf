"""Dendrocentric layer v2: K-sparse selection + DFSM stored ordering + soft-rank correlation.

v2 changes vs v1:
- Adds dendro_L (M, K) trainable per-dendrite preferred ordering of K selected channels.
- Forward: gather K weighted values per dendrite, soft-rank both observed values and stored L,
  centered Pearson correlation, sigmoid NMDA activation, BitLinear output projection.

References:
- Boahen 2022 (Nature 612), Box 2 (NMDA-gated dendrite, ordered arrival)
- Mordvintsev 2022 DFSM (softmax-then-harden trainability bridge)
- Math: scratch/2026-04-30_dendrocentric_v2_derivation.md
- Toys: scratch/2026-04-30_dendrocentric_v2_tiny.py (all 4 pass)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSTE(torch.autograd.Function):
    """Forward: keep top-K |w| per row, zero rest. Backward: STE."""
    @staticmethod
    def forward(ctx, w, k):
        abs_w = w.abs()
        _, topk_idx = abs_w.topk(k, dim=-1)
        mask = torch.zeros_like(w)
        mask.scatter_(-1, topk_idx, 1.0)
        return w * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def topk_ste(w, k):
    return TopKSTE.apply(w, k)


class DendrocentricLayer(nn.Module):
    """v2: K-sparse selection + DFSM stored ordering + soft-rank correlation.

    Per token, per dendrite j:
    1. Gather K values from selected input channels (top-K of |dendro_W_d[j, :]|).
    2. Soft-rank these K values: r_k = sum_{l != k} sigmoid(tau_x * (z_k - z_l)).
    3. Soft-rank stored L_j: rho_k = sum_{l != k} sigmoid(tau_l * (L_{j,k} - L_{j,l})).
    4. Centered Pearson correlation: s_j = (12 / (K(K^2 - 1))) * sum (r_k - (K-1)/2)(rho_k - (K-1)/2).
    5. NMDA activation: h_j = sigmoid(alpha * s_j - theta_j).
    6. Output: o = dendro_W_out @ h (BitLinear ternary).

    s_j ∈ [-1, +1] (Pearson on rank vectors). alpha=4 maps this to sigmoid range with good dynamic.
    """
    def __init__(
        self,
        d_model: int,
        n_dendrites: int,
        k_per_dendrite: int,
        alpha: float = 4.0,
        tau_x: float = 1.0,
        tau_l: float = 1.0,
    ):
        super().__init__()
        from modules.bitlinear import BitLinear
        self.d_model = d_model
        self.M = n_dendrites
        self.K = k_per_dendrite
        self.alpha = alpha
        self.tau_x = tau_x
        self.tau_l = tau_l
        # Continuous full-rank parameter; rendered K-sparse via top-K STE in forward.
        # Substring 'dendro_' in CONTROL_TENSOR_NAME_PATTERNS keeps these fp32-protected.
        self.dendro_W_d = nn.Parameter(torch.randn(n_dendrites, d_model) * 0.02)
        # Per-dendrite preferred ordering (DFSM trainable). Init std=1.0 → pairwise diffs ~√2,
        # sigmoid args in non-saturated regime.
        self.dendro_L = nn.Parameter(torch.randn(n_dendrites, k_per_dendrite) * 1.0)
        # Per-dendrite NMDA threshold (1D, auto-fp32).
        self.dendro_theta = nn.Parameter(torch.zeros(n_dendrites))
        # Output projection M -> d. Ternary BitLinear to match the rest of the body.
        self.dendro_W_out = BitLinear(n_dendrites, d_model, bias=False)

        # Pearson normalization constant: K * (K^2 - 1) / 12 for K-rank vectors.
        self._pearson_norm = float(k_per_dendrite * (k_per_dendrite * k_per_dendrite - 1)) / 12.0
        self._rank_center = (k_per_dendrite - 1) / 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])  # (N, d)

        # Apply TopKSTE so gradients flow through the sparse-selected positions
        # to dendro_W_d (and STE bypasses to all positions).
        W_d_sparse = topk_ste(self.dendro_W_d, self.K)  # (M, d), K nonzeros per row

        # Get top-K column indices for gathering x.
        abs_w = self.dendro_W_d.abs()
        _, topk_idx = abs_w.topk(self.K, dim=-1)  # (M, K)

        # Gather x at top-K positions per dendrite.
        # x_flat: (N, d), topk_idx: (M, K) → x_gathered: (N, M, K)
        gathered = x_flat[:, topk_idx]  # (N, M, K)

        # Weight by W_d_sparse at top-K positions (carries STE gradient to W_d).
        w_at_topk = torch.gather(W_d_sparse, dim=-1, index=topk_idx)  # (M, K)
        weighted = gathered * w_at_topk.unsqueeze(0)  # (N, M, K)

        # Soft-rank of stored L first (small: shape (M, K, K) fp32). Reused across all tokens.
        L = self.dendro_L.to(torch.float32)
        L_diff = L.unsqueeze(-1) - L.unsqueeze(-2)  # (M, K, K)
        L_sig = torch.sigmoid(self.tau_l * L_diff)
        eye_L = torch.eye(self.K, device=L.device, dtype=L_sig.dtype)
        L_sig = L_sig * (1.0 - eye_L)
        rho = L_sig.sum(dim=-1)  # (M, K)
        rho_c = rho - self._rank_center  # (M, K) fp32

        # Soft-rank of K weighted values per dendrite. UNFUSED single-tensor version
        # (faster than chunking over K because it avoids 8 sequential GPU kernel
        # launches per layer; torch.compile can't fuse a Python `for` loop).
        # Memory: (N, M, K, K) × 2B fp16 ≈ 2 GB at M=1024 N=16384 K=8 — fits.
        # If OOM: drop M, drop batch, or reintroduce chunking.
        weighted_f = weighted.to(torch.float32)  # (N, M, K)
        diff = weighted_f.unsqueeze(-1) - weighted_f.unsqueeze(-2)  # (N, M, K, K)
        sig = torch.sigmoid(self.tau_x * diff)
        eye_x = torch.eye(self.K, device=x.device, dtype=sig.dtype)
        sig = sig * (1.0 - eye_x)
        r = sig.sum(dim=-1)  # (N, M, K)
        r_c = r - self._rank_center  # (N, M, K)

        # Centered Pearson correlation s ∈ [-1, +1].
        dot = (r_c * rho_c.unsqueeze(0)).sum(dim=-1)  # (N, M)
        s = dot / self._pearson_norm  # (N, M)

        # NMDA-like sigmoid activation with per-dendrite theta bias.
        h = torch.sigmoid(self.alpha * s - self.dendro_theta.unsqueeze(0).to(torch.float32))
        h = h.to(weighted.dtype)  # back to original dtype

        # Output projection (ternary BitLinear).
        out = self.dendro_W_out(h)  # (N, d_model)
        return out.reshape(*orig_shape[:-1], -1)
