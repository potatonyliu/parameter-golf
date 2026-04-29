# Experiment 0114_n5_ternary_long_5k

Parent: 0107_bigger_ternary_ssm_prod (n=5 ternary at production batch, 1k steps, val 1.5252)

## Question

**Long-train on the cap-frontier setpoint.** After 0111 confirmed n=7 under-trains worse at 1k steps and 0112 confirmed the SSD scan is load-bearing (-0.046 BPB), the right long-training target is the n=5 ternary stack at 5x more steps.

Two questions answered at once:
1. **BitNet 25x training-closure**: ternary penalty narrowed from MPS-200 (+0.10 BPB) to CUDA-2k (+0.076) to CUDA-1k production (+0.082). Does it close further at 5k steps?
2. **Submittable scaling slope**: 0103->0104 (no-ternary, n=3, 1k->5k) showed -0.115 BPB slope. Does the same slope hold for n=5 + ternary at production batch?

This is the SUBMITTABLE long-train (artifact ~9 MB, well under 16 MB cap), not the cap-bust scaling test 0104 represents. The result here is what we'd quote in a writeup as "best submittable val_bpb at 1x training budget."

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.39, 1.46]** (single-seed). Mental models:
- **Slope holds**: 1.5232 (1k mean) - 0.115 (5x slope) = 1.408. Predicted 1.40-1.43.
- **Slope flatter** (typical of larger/ternary models that already train well): 1.43-1.46.
- **BitNet closure helps**: ternary penalty narrows further than slope alone, lands 1.38-1.41.

Predicted artifact: ~9.0 MB (training duration doesn't change weight size; same as 0107).
Predicted step_avg: ~1345 ms (same as 0107).
Total wallclock: 5000 * 1345 ms = 6725 s = 112 min ≈ 1.87h.
Cost: ~$1.87 at $1/hr 5090.

## Change

env.sh inherits 0107 verbatim, override:
- `ITERATIONS=5000` (was 1000)
- `WARMDOWN_ITERS=750` (15% of total — schedule scales with iterations; was 300 for 1k)
- `MAX_WALLCLOCK_SECONDS=9000` (2.5 hour cap; preflight rejects 0)

All other env vars (ternary, n=5, batch 131072, MATRIX_LR=0.045, SEED=1337) unchanged.

Single-seed for triage. If result is interesting (val ≤ 1.43), cross-seed confirm with SEED=42 fork (~1.87h additional) becomes a writeup-quality decision — but only if we're committing this number to the writeup. Otherwise screening single-seed is fine.

## Disconfirming

- val_bpb >= 1.46: slope flatter than expected; bigger model + ternary plateaus. Pivot to architectural improvements (parallel residuals).
- val_bpb in [1.43, 1.46]: typical scaling, no surprises. Continue with parallel-residuals next.
- val_bpb in [1.40, 1.43]: matches slope prediction. Strong submittable result. Plan for parallel-residuals on top.
- val_bpb <= 1.39: BitNet closure faster than expected. Strong direction. Cross-seed confirm becomes worthwhile.
- Crash / NaN: ternary at long-train may be less stable than at 1k; possibly LR schedule too aggressive. Re-run with WARMDOWN_ITERS=1500 (30%).
- step_avg drift > 1.5x predicted: kernel cache pressure, OOM precursor. Investigate.

## Notes from execution
