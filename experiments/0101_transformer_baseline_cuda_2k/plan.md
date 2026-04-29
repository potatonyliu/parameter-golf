# Experiment 0101_transformer_baseline_cuda_2k

Parent: 0076_confidence_gated_blend (current best transformer + static side memory + per-context α + confidence gate; MPS 2-seed mean val 1.95141)

## Question

What's the val_bpb of the **non-ternary transformer baseline at CUDA 2000 steps**, exactly the architecture 0099 modifies? Without this comparator, 0099's pre-quant val_bpb only tells us the absolute number; with this comparator, we can decompose `0099 - 0101 = ternary penalty at 2000 steps`. **Also gives us a CUDA training-duration calibration**: how much val_bpb improves going from 200 → 2000 steps on the transformer best stack (not the SSM frontier — that'd need yet another run forking 0064).

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb at 2000 steps in [1.40, 1.60]**. The 0076 MPS 200-step pre-quant was ~1.96; at 10× more training the val should drop substantially. Per the records archive (`records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md` and similar), CUDA SP1024 transformer baselines at 6000+ steps land around 1.20-1.40 BPB; our 2000-step run should be in that ballpark trajectory.

**Post-quant val_bpb in [1.42, 1.62]** — quant_tax for non-ternary fp/int8 is typically +0.005 to +0.010, much smaller than the regularization-effect negative tax 0093 showed.

This sets up a clean decomposition:
- `0099 - 0101` = ternary cost (with v2 packed serialization) at 2000 CUDA steps
- If `0099 - 0101 < 0.020` → ternary direction is highly viable for H100 push
- If `0099 - 0101 ∈ [0.020, 0.040]` → marginal closure; H100 gain uncertain
- If `0099 - 0101 > 0.040` → BitNet 25× claim doesn't transfer at our regime; pivot

## Change

env.sh inherits 0076 verbatim, then overrides:
- `ITERATIONS=2000` (vs MPS 200)
- `TERNARY_BODY=0` (explicit, in case parent inheritance has it set; though 0076 should already have it 0)
- `MATRIX_LR=0.045` (canonical, since we're not in ternary-LR-rescue regime)

WARMDOWN_ITERS=300 inherited gives the same schedule shape as 0099 (0–30 warmup, 30–1700 const, 1700–2000 warmdown). MAX_WALLCLOCK_SECONDS=0 + ALLOW_NO_WALLCLOCK_CAP=1 launch.

No code changes.

## Disconfirming

- val_bpb >= 1.80 at 2000 steps: the transformer baseline doesn't benefit much from 10× more training at this batch/architecture, suggesting the regime is bottlenecked elsewhere (architecture? batch? learning rate?). Either way, calibration result.
- val_bpb < 1.30: the transformer baseline at 2000 steps is already competitive with H100 records, which would be surprising given we're on a 5090 with smaller batch. Triple-check (likely a measurement error).

## Notes from execution
