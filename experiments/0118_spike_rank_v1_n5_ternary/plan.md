# Experiment 0118_spike_rank_v1_n5_ternary

Parent: 0107_bigger_ternary_ssm_prod (n=5 ternary, val 1.5256 at 1k steps, 9.08 MB)

## Question

**Replace token embedding with a K-sparse representation (variant A: sparse-only, no rank).** Each token's embedding row stores K=8 nonzero values via top-K STE. Tied lm_head reuses the same sparse matrix. Tests whether sparse token embedding composes with our SSM stack.

This is brief option (c) "spike-rank embedding" v1. Variant B (rank-only / no learned values) is deferred to v2 — that variant is more brief-aligned but riskier (removes per-token value freedom).

References:
- Brief: `scratch/2026-04-29_session_planning.md` option (c).
- Math derivation: `scratch/2026-04-29_spike_rank_embedding_derivation.md` (cap math: 488 KB savings vs current int8 embed).
- Toy verification: `scratch/2026-04-29_spike_rank_v1_tiny.py` (5 progressive toys all pass: by-hand forward, shape flow, STE backward, tied lm_head, synthetic next-token training converges to loss 0.0).
- Cousin: dendrocentric v1 (0117) uses identical TopKSTE mechanism applied to MLP layer; this experiment applies it to embedding.

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.50, 1.58] at 1k steps**, with three plausible regimes:
- **25%**: v1 lands within +/- 0.005 of parent. Sparse embed matches dense at our regime.
- **40%**: v1 lands neutral-to-slightly-worse (1.53-1.55). Cap savings real, training-duration-bound.
- **35%**: v1 lands materially worse (>=1.55). Per-token expressiveness reduced too far at K=8. Try K=16 or K=32 in v1.5.

The 1k-step 5090 smoke is most likely UNINFORMATIVE about the absolute magnitude — like dendrocentric, the real test is H100 long-train.

This experiment's primary value: (a) verify the layer trains stably without NaN, (b) confirm cap math (predicted ~8.6 MB = 9.08 - 0.488 freed), (c) confirm gradient flows correctly through the tied sparse embed/lm_head (toys say yes; verify on real model).

Predicted artifact: ~8.6 MB.
Predicted step_avg: similar to 0107 (1300-1400 ms) — embed lookup is cheap; matmul through sparse W_e in lm_head is similar cost to dense.

## Change

env.sh inherits 0107, ADD:
- `SPIKE_RANK_EMBED=1` (default 0 = byte-identical to 0107, uses standard nn.Embedding)
- `SPIKE_RANK_K=8` (sparsity per token row)
- `MAX_WALLCLOCK_SECONDS=1800`

Append `spike_` to `CONTROL_TENSOR_NAME_PATTERNS` if the implementation needs any fp32-protected scalars (probably not — only the W_e sparse pattern is trained).

train_gpt.py code change:

### 1. Add a new module file: `experiments/0118_spike_rank_v1_n5_ternary/modules/spike_rank_embed.py`

```python
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
```

### 2. Add Hyperparameters fields (mirror dendrocentric/EMA pattern)

```python
spike_rank_embed = int(os.environ.get("SPIKE_RANK_EMBED", "0"))
spike_rank_k = int(os.environ.get("SPIKE_RANK_K", "8"))
```

### 3. Find and modify the embed + lm_head construction sites

**Find the existing patterns** in train_gpt.py:
- The token embedding (likely `self.embed = nn.Embedding(vocab_size, d_model)` in GPT.__init__ or similar)
- The lm_head (likely `self.lm_head = nn.Linear(d_model, vocab_size, bias=False)` then a weight-tying line `self.lm_head.weight = self.embed.weight`)
- The forward pass usage (`x = self.embed(token_ids)` and `logits = self.lm_head(x)` or similar)

**Branch on env-var**:
```python
if Hyperparameters.spike_rank_embed == 1:
    from modules.spike_rank_embed import SpikeRankEmbed
    self._spike_embed = SpikeRankEmbed(vocab_size, d_model, Hyperparameters.spike_rank_k)
    self.embed = None  # not used; route through _spike_embed
    self.lm_head = None  # not used
else:
    self.embed = nn.Embedding(vocab_size, d_model)
    self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    self.lm_head.weight = self.embed.weight  # existing tie
```

**At forward call sites**:
```python
# Embedding lookup
if Hyperparameters.spike_rank_embed == 1:
    x = self._spike_embed(token_ids)
else:
    x = self.embed(token_ids)

# LM head (after the body)
if Hyperparameters.spike_rank_embed == 1:
    logits = self._spike_embed.lm_head(x)
else:
    logits = self.lm_head(x)
```

The subagent should INSPECT the existing patterns carefully — embed/lm_head construction may use specific helpers (CastedLinear, etc.). Match the existing style. The TIED_EMBED_INIT_STD env var handling for our existing model should be preserved when SPIKE_RANK_EMBED=0 (default).

### 4. Verify integration with `quantize_state_dict_int8` at line ~2132

The standard quant path serializes `self.embed.weight` as int8. Our sparse weight is also a regular nn.Parameter — it should serialize the same way. The K-sparse structure is preserved in float values (zeros are explicitly stored). At runtime, the sparse pattern is recomputed via top-K STE on dequantized weights.

For more cap savings, the v1.5 follow-up could store ONLY the K nonzero indices + values (separately) instead of the full V x d matrix at int8. Defer to v1.5 — v1 just confirms the mechanism.

## Disconfirming

- Crash / NaN: probably from F.embedding receiving a sparse tensor where it doesn't expect one — verify F.embedding accepts dense weight tensors (it does; we're just passing a tensor with zeros). Or from gradient explosion if STE flows through unprotected scaling.
- val_bpb >= 1.60: K=8 too restrictive. Try K=32 in v1.5.
- val_bpb in [1.55, 1.60]: training-duration-bound; keep for H100 deploy with caveat.
- val_bpb in [1.52, 1.55]: marginal. Cap savings might justify keeping.
- val_bpb < 1.52: positive. Direct contribution to brief's spike-rank thesis.
- artifact_mb similar to 0107 (no savings): the cap savings claim is wrong because the int8-stored sparse weight doesn't compress better than the dense one. Future v1.5 stores only indices+values.
- step_avg drift > 1.5x parent: F.linear through K-sparse weight is slower than dense matmul. Investigate.

## Notes from execution
