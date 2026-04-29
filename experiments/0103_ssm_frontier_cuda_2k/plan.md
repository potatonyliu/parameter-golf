# Experiment 0103_ssm_frontier_cuda_2k

Parent: 0098_ssm_frontier_cuda_200 (SSM frontier kill-Mamba-2 triple-parallel; CUDA 200-step val 2.0028)

## Question

What's the SSM frontier's val_bpb at 10× more training (2000 CUDA steps), with NO ternary, NO dendritic, NO side memory? This is the **clean SSM baseline at extended training** — needed for two decompositions:
- vs 0102 (SSM + ternary): does ternary add or subtract on SSM at 2000 steps?
- vs 0101 (transformer alone at 2000 steps): is the MPS-200 finding `−0.085 BPB SSM contribution` preserved at 2000 CUDA steps? (The cross-architecture answer to "does SSM still beat attention with more training".)

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb at 2000 steps in [1.55, 1.75]** (single-seed). Reasoning:
- 0098 SSM @ 200 steps pre-quant 2.00. The training-duration win going from 200→2000 should be ~−0.30 to −0.40 BPB (matches what 0099 showed for transformer + ternary going same factor).
- Predicted: 1.65 ± 0.10.
- vs 0101 transformer baseline (TBD ~1.55–1.75): SSM contribution at extended training likely [-0.10, +0.05]. The MPS 200-step finding was -0.085; at 10× training this could shrink (selectivity-related advantages stabilize) or stay constant (kill-LTI is structural).

## Change

env.sh inherits 0098 (SSM frontier config) verbatim, then overrides:
- `ITERATIONS=2000`

WARMDOWN_ITERS=300 stays (gives 0–30 warmup, 30–1700 const, 1700–2000 warmdown). MAX_WALLCLOCK_SECONDS=0 + ALLOW_NO_WALLCLOCK_CAP=1.

No code changes. Same train_gpt.py as 0098 (which is canonical), no BitLinear, no dendritic.

Predicted run: ~205ms/step (compile-friendly, no troublesome modules) × 2000 steps = ~7 min wall. Plus eval ~30s. Total ≤8 min.

## Disconfirming

- val_bpb > 1.85: SSM stack doesn't benefit much from 10× training, contradicting the transformer findings. Suggests the MPS-200 SSM optima was already near saturation at the small training-data regime.
- val_bpb < 1.40: would beat the H100 SP1024 record at 1.10 by a wide margin, implausible given we're on 5090 with smaller batch — re-check measurement.
- crash: shouldn't happen since 0098 ran clean.

## Notes from execution
