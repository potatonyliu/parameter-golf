# Experiment 0109_pure_attn_prod

Parent: 0098_ssm_frontier_cuda_200 (canonical SSM frontier topology)

## Question

**Does the SSM contribution observed at MPS-200 steps (-0.085 BPB vs pure-attn) preserve at production batch + 1000 steps on CUDA?** The previous session's writeup-anchor finding was 0058/0059 pure-attn 2-seed mean **2.0876** vs 0051 SSM frontier 2.0017 = SSM contribution -0.085 BPB. At production scale, this contribution might:
- Preserve (-0.05 to -0.10 BPB SSM advantage): SSM frontier is the right writeup direction.
- Shrink (-0.01 to -0.05 BPB): SSM contribution diminishes with scale. Mixed signal.
- Disappear (≥ 0): pure attn matches or beats SSM at production scale. **Bad finding for the SSM thesis.**

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb at production batch 131072 × 1000 steps in [1.46, 1.60]** (single-seed). 0103 (SSM frontier no-ternary same config) hit 1.4587. If SSM contribution at scale is similar to MPS-200 (-0.085), pure-attn lands ~1.55. If contribution shrinks toward zero, pure-attn lands ~1.47.

Bucket interpretation:
- pure-attn ≤ 1.48: SSM contribution disappears at production scale. Negative finding for the writeup.
- pure-attn ∈ [1.50, 1.55]: SSM contribution preserved. Strong writeup story.
- pure-attn > 1.55: SSM contribution larger at scale than at MPS-200. Even better writeup story.

## Change

env.sh inherits 0098 verbatim, then OVERRIDES topology to pure-attn:
- `ATTN_LAYER_POSITIONS=0,1,2`
- `PARALLEL_LAYER_POSITIONS=` (empty)
- `MAMBA2_LAYER_POSITIONS=` (empty)
- `TRAIN_BATCH_TOKENS=131072` (match 0103 production batch)
- `ITERATIONS=1000` (match 0103)
- `MATRIX_LR=0.045` (canonical, no LR×3)

## Disconfirming

- val_bpb ≤ 1.46: pure-attn already matches the SSM frontier at scale; SSM contribution gone.
- val_bpb ≥ 1.60: pure-attn dramatically worse — implausible without a config bug.
- Crash: pure-attn at production should be the most stable config; if it crashes, something fundamental.

## Notes from execution
