# Experiment 0106_full_selectivity_prod

Parent: 0103_ssm_frontier_cuda_2k (SSM frontier at production batch, no ternary; pre-quant 1.4587, cap-busts 21.4 MB)

## Question

**Does the prior session's "kill > full" finding (verified at MPS 200 steps via 0035-0039 2-seed sentinel) invert at production batch + 1000 steps?** The original justification was that input-dependent selectivity (dt, B, C from in_proj) is undertrained at 5M tokens. At production batch (131072 × 1000 = 131M tokens), that premise no longer holds. The kill-vs-full result at scale is **genuinely new territory** — the reviewer's mid-session "re-litigating already 4-seed sentinel'd" caution applied to MPS, not production batch.

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.42, 1.50]** (single-seed). Reasoning:
- 0103 (kill at production batch, 1k steps, no ternary): pre-quant 1.4587, post 1.4591.
- 0035/0036 (full at MPS 200, 2-seed): mean ~2.16; 0038/0039 (kill at MPS 200, 2-seed): mean ~2.04. Δ MPS = +0.12 (kill better).
- At production scale, full's selectivity is no longer training-starved; expected: kill > full Δ shrinks toward 0 OR flips.

Outcome buckets:
- val ≤ 1.45 (full beats kill at production): "kill > full" inverts at scale. **Big finding** — prior session's headline mechanism doesn't transfer. Implications for the writeup: the MPS-200 ablation conclusions need a CUDA-production caveat.
- val ∈ [1.45, 1.48]: Roughly tied. Kill's MPS advantage is gone, full is competitive. Either is fine to use; probably keep kill for inference-time efficiency (no input-dependent matmuls).
- val ≥ 1.50: Kill > full holds at production. Confirms mechanism scales. Use kill in writeup with confidence.

## Change

env.sh inherits 0103 verbatim, override `MAMBA2_KILL_SELECTIVITY=0`. Production batch 131072, canonical LR 0.045, 1000 steps. No code change.

Predicted run: ~800 ms/step × 1000 = ~13 min wall + ~30s eval = ~14 min total.

## Disconfirming

- Crash mid-run (NaN, late instability): full-selectivity at production batch hits a different stability regime than at MPS. Investigate.
- val ≥ 1.55: full is dramatically worse than kill at production scale, despite the undertraining premise being gone. Suggests "kill > full" is structural (not training-related). Worth a single-seed re-confirm at a different config.
- val << 1.40: implausibly low; check artifact size and quant integrity.

## Notes from execution
