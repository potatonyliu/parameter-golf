"""Dendritic memory module (0092).

A bank of M "dendrites", each storing:
  * a frozen K-token pattern (warm-started from top-frequent K-grams in the
    training stream — pattern_keys is a non-trainable buffer),
  * a learnable low-rank content vector (shape (M, d_content), zero-init).

At each token position t >= K-1, exact-match the trailing K-gram against the
pattern bank (vectorized searchsorted on encoded keys). For matched positions,
gather the matched content vectors, project up via a shared `proj` head
(zero-init -> module is a no-op at training start), and scatter into the
output tensor (B, L, d_model). The output is intended to be added to the
residual stream BEFORE the first block.

Design notes:
  * The pattern bank is built once at GPT.__init__ time from training tokens.
    This keeps high-frequency K-grams firing on EVERY batch, addressing the
    sparse-gradient failure mode (0073/0080) that prevents content vectors
    from converging.
  * Key encoding for K=4: key = p3 * V^3 + p2 * V^2 + p1 * V + p0
    (the WHOLE K-gram, not K-1 context + 1 next). Max key value at V=1024,
    K=4 is V^4 = 2^40 — comfortably fits in int64.
  * MPS perf: searchsorted against a 32K-element sorted buffer is fine on MPS.
    If profiling shows it slow, fall back to a CPU detour (see
    `experiments/0089_fuzzy_kgram_softdp/modules/trigram_side_memory.py:678+`
    `_apply_fuzzy_kgram_fallback` for the pattern).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class DendriticMemory(nn.Module):
    """M-dendrite frozen-pattern + learnable-content bank, K-gram exact match."""

    def __init__(
        self,
        vocab_size: int,
        K: int,
        M: int,
        d_content: int,
        d_model: int,
    ) -> None:
        super().__init__()
        if K != 4:
            raise ValueError(f"DendriticMemory currently supports K=4 only, got K={K}")
        self.vocab_size = int(vocab_size)
        self.K = int(K)
        self.M = int(M)
        self.d_content = int(d_content)
        self.d_model = int(d_model)

        # Pattern keys: sorted ascending int64. Filled by populate_patterns().
        # Sentinel value -1 means "unfilled" (sorted at the front so any positive
        # query key never collides). Once populate_patterns runs, all entries
        # are valid pattern hashes.
        self.register_buffer(
            "pattern_keys",
            torch.full((self.M,), -1, dtype=torch.int64),
            persistent=True,
        )
        # Track whether populate_patterns() has been called.
        self.register_buffer(
            "_populated",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )

        # Trainable content bank: (M, d_content) fp32, zero init.
        self.content = nn.Parameter(
            torch.zeros(self.M, self.d_content, dtype=torch.float32)
        )

        # Shared projection head from d_content -> d_model. Zero-init so the
        # whole dendritic add starts as a no-op (residual untouched at step 0).
        self.proj = nn.Linear(self.d_content, self.d_model, bias=False)
        nn.init.zeros_(self.proj.weight)
        # Tag for GPT._init_weights() so it preserves zero-init across the
        # generic linear-init pass.
        self.proj._zero_init = True

    # --------------------------------------------------------------------- #
    # Pattern bank construction                                             #
    # --------------------------------------------------------------------- #

    def populate_patterns(self, train_tokens: np.ndarray) -> dict[str, float]:
        """Build top-M most-frequent K-grams from `train_tokens` and store the
        sorted-key buffer in self.pattern_keys.

        Args:
            train_tokens: 1-D numpy array of int (uint16/int32/int64) token ids.

        Returns: dict with diagnostics (n_unique, fill_rate, build_seconds, ...).
        """
        t0 = time.time()
        train = np.asarray(train_tokens).astype(np.int64, copy=False)
        V = self.vocab_size
        K = self.K
        N = train.size
        if N < K:
            raise ValueError(f"train_tokens too short ({N}) for K={K}")

        # Encode K-grams: key = p3*V^3 + p2*V^2 + p1*V + p0 (for K=4).
        # All four positions of the K-gram, NOT context+next.
        # max key = V^K - 1 = 2^40 - 1 at V=1024, K=4 -> int64.
        VK_pow = [V ** i for i in range(K)]  # [1, V, V^2, V^3]
        # Slice out K parallel views; offset i takes train[i : N-K+1+i].
        # Then weight by V^(K-1-i) so that the LAST token gets weight 1 (units).
        keys = np.zeros(N - K + 1, dtype=np.int64)
        for i in range(K):
            keys += train[i : N - K + 1 + i] * VK_pow[K - 1 - i]
        n_kgrams = keys.size

        # Count frequency. With V=1024, K=4 the keyspace is 2^40 — too big for
        # bincount directly. Use np.unique to get exact frequencies of present
        # K-grams (memory ~ #distinct K-grams * 16 bytes; for 50M tokens the
        # number of distinct 4-grams is bounded by 50M and in practice much less).
        unique_keys, counts = np.unique(keys, return_counts=True)
        n_unique = unique_keys.size

        # Pick top-M by count. argpartition is O(n); then sort the picked slice
        # ascending by KEY (not by count) so searchsorted works.
        if n_unique <= self.M:
            topk_idx = np.arange(n_unique)
            n_picked = n_unique
        else:
            # argpartition for top-M by count (descending). The last M entries
            # in the partitioned order are the M largest counts.
            topk_idx = np.argpartition(-counts, self.M - 1)[: self.M]
            n_picked = self.M

        topk_keys = unique_keys[topk_idx]
        # Sort ascending by key for searchsorted lookup.
        topk_keys.sort()

        # Fill the buffer. If n_picked < M, leave the remaining entries at
        # sentinel (-1) — they sort to the front and never match a real query.
        new_buf = torch.full((self.M,), -1, dtype=torch.int64)
        new_buf[: n_picked] = torch.from_numpy(topk_keys.astype(np.int64))
        new_buf, _ = torch.sort(new_buf)  # ensure overall sorted ascending
        self.pattern_keys.copy_(new_buf.to(self.pattern_keys.device))
        self._populated.fill_(1)

        # Diagnostics
        total_count_top = int(counts[topk_idx].sum())
        coverage = float(total_count_top) / float(n_kgrams) if n_kgrams > 0 else 0.0
        elapsed = time.time() - t0
        diag = {
            "n_kgrams_total": float(n_kgrams),
            "n_unique_kgrams": float(n_unique),
            "n_picked": float(n_picked),
            "fill_rate": float(n_picked) / float(self.M),
            "training_coverage": coverage,
            "build_seconds": float(elapsed),
        }
        print(
            f"[dendritic] populated M={self.M} from {N:,} tokens in {elapsed:.2f}s | "
            f"unique 4-grams={n_unique:,} | picked={n_picked:,} "
            f"(fill={diag['fill_rate']:.3f}) | "
            f"top-M coverage of train K-grams={coverage:.4f}",
            flush=True,
        )
        return diag

    # --------------------------------------------------------------------- #
    # Forward                                                               #
    # --------------------------------------------------------------------- #

    @torch.compiler.disable
    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute (B, L, d_model) additive residual from K-gram pattern firings.

        For t < K-1 (not enough context yet) the output row is zero.
        For t >= K-1, encode the trailing K-gram, searchsorted into pattern_keys,
        and on exact match gather content[idx], project, and scatter into the
        output tensor.

        @torch.compiler.disable: this forward uses boolean indexing (idx[mask])
        which produces a dynamic-shape output — incompatible with the parent
        train_gpt.py's torch.compile(fullgraph=True). The decorator carves a
        graph break around this module so the rest of the model still compiles.
        Cost: dendritic forward runs in eager mode; small overhead for a small
        module.
        """
        B, L = input_ids.shape
        K = self.K
        V = self.vocab_size
        device = input_ids.device

        # The output residual contribution. Always (B, L, d_model). Returned
        # in fp32; the caller adds it to a (possibly bf16) residual stream and
        # the addition handles the cast.
        out = torch.zeros(B, L, self.d_model, dtype=torch.float32, device=device)

        if L < K:
            return out
        # The original check `if int(self._populated.item()) == 0: return out`
        # was a guard for the dry-run-during-compile case, but `.item()` breaks
        # torch.compile(fullgraph=True) on CUDA (Tensor.item() with
        # capture_scalar_outputs=False — TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
        # env var should fix it but doesn't reliably). Since populate_patterns()
        # is called once before training/eval starts in this codebase, the
        # buffer is always populated by the time forward runs in production —
        # the guard is unreachable. Removing it lets dynamo capture the graph.

        # Encode K-grams at every t in [K-1, L). Resulting shape: (B, L - K + 1).
        # Position-t key uses input_ids[:, t-(K-1) : t+1].
        # For K=4: key[b, t] = ids[b, t-3]*V^3 + ids[b, t-2]*V^2 + ids[b, t-1]*V + ids[b, t]
        ids64 = input_ids.to(torch.int64)
        L_keys = L - K + 1
        keys = torch.zeros(B, L_keys, dtype=torch.int64, device=device)
        for i in range(K):
            # token at relative offset i (0 = oldest, K-1 = current) is multiplied
            # by V^(K-1-i) so the LAST token gets weight 1.
            keys += ids64[:, i : i + L_keys] * (V ** (K - 1 - i))

        # searchsorted in pattern_keys. Result is the insert index in [0, M].
        # Clip to [0, M-1] so gather is safe; then exact-match check.
        # MPS supports int64 searchsorted on small buffers; for 32K it's fast.
        flat_keys = keys.reshape(-1)
        idx = torch.searchsorted(self.pattern_keys, flat_keys)
        idx = idx.clamp(max=self.M - 1)
        matched_keys = self.pattern_keys[idx]
        match_mask = matched_keys == flat_keys  # (B*L_keys,) bool

        # Original early-return `if n_matched == 0: return out` used .item()
        # which breaks torch.compile(fullgraph=True). The downstream gather/
        # scatter ops handle the empty-mask case correctly (empty index tensors
        # → zero-element gathers → no-op scatters), so removing the guard is
        # numerically equivalent for non-empty mask and a tiny cost for empty.
        # Gather content for matched positions, project, scatter into out.
        # `idx[match_mask]` gives the dendrite indices.
        match_dendrite_idx = idx[match_mask]                    # (n_matched,) long
        gathered = self.content.index_select(0, match_dendrite_idx)  # (n_matched, d_content)
        # Project up to d_model. Cast proj.weight to gathered.dtype to keep
        # the matmul in fp32 (proj.weight is fp32 because it's a Linear).
        projected = self.proj(gathered)                         # (n_matched, d_model) fp32

        # Scatter into out[b, t, :] where (b, t) corresponds to flat index
        # f = b * L_keys + (t - (K-1)). Inverse: out_t = K - 1 + (f % L_keys),
        # out_b = f // L_keys.
        match_flat_idx = torch.nonzero(match_mask, as_tuple=False).squeeze(-1)  # (n_matched,) long
        out_b = match_flat_idx // L_keys
        out_t = (match_flat_idx % L_keys) + (K - 1)
        out[out_b, out_t, :] = projected.to(out.dtype)

        return out
