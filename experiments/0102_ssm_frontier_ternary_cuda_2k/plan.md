# Experiment 0102_ssm_frontier_ternary_cuda_2k

Parent: 0099_ternary_extended_cuda_2k (carries BitLinear + LR×3 + train_gpt.py with both SSM and ternary support)

## Question

**Does the BitNet ternary win (0099 pre-quant 1.6499 at 2k CUDA, transformer + side-mem) compose with the SSM frontier win (0098 val 2.0028 at 200 CUDA, kill-Mamba-2 triple-parallel)?** This is the FIRST attempt at the program.md deliverable: an SSM contribution + standard-stack composition. If the compound holds, the H100 deliverable is "SSM frontier + ternary body + (port the rest of the standard stack)."

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb at 2000 steps in [1.50, 1.75]** (single-seed). Reasoning:
- 0098 SSM frontier @ 200 steps: pre-quant 2.00. Extending to 2000 steps should drop to ~1.65–1.75 (similar -0.3 to -0.4 BPB benefit from 10× training that 0099 showed).
- BitNet ternary penalty at 2000 CUDA steps: needs 0099-vs-0101 to know exactly, but expected to be SMALL (the BitNet 25× claim says monotonically narrowing).
- Compound estimate: SSM-2k baseline 1.70 ± 0.05 + ternary penalty 0.02 ± 0.02 = 1.72 ± 0.07.

Outcome buckets:
- val_bpb < 1.55: STRONG compound win — ternary on SSM beats transformer-baseline at same training; prepares the H100 push with a clear story.
- val_bpb in [1.55, 1.75]: matches expectation; the win composes additively, advance to longer training.
- val_bpb > 1.80: ternary disrupts the SSM stack at this scale — possibly the kill-Mamba-2 LTI dynamics interact poorly with ternary weights. Investigate before scaling.
- crash on CUDA: BitLinear + Mamba-2 conv1d combo has a path issue. Debug.

## Change

env.sh inherits 0099 (BitLinear hooks + ternary configs + LR×3) verbatim, then OVERRIDES:
- `ATTN_LAYER_POSITIONS=`, `MAMBA2_LAYER_POSITIONS=`, `PARALLEL_LAYER_POSITIONS=0,1,2`, `PARALLEL_SSM_TYPE=mamba2_kill`, `MAMBA2_KILL_SELECTIVITY=1`, `BIGRAM_VOCAB_SIZE=0` (the SSM frontier topology from 0098)
- `CONTROL_TENSOR_NAME_PATTERNS=` to the SSM-frontier set (keeps A_log, dt_bias, conv1d etc. fp32 alongside ternary body)
- `TRIGRAM_SIDE_MEMORY=0` (skip the build + dynamo bug)
- `ITERATIONS=2000`

No code changes — train_gpt.py from 0099 supports both stacks (BitLinear hooks AND PARALLEL_SSM_TYPE).

Predicted run: ~205ms/step × 2000 = ~7 min training (compile re-enabled because no dendritic) + brief eval = ~8 min wall.

## Disconfirming

- val_bpb > 1.85 at 2000 steps: SSM + ternary does NOT compose; ternary penalty appears amplified by SSM stack. Either Mamba-2 conv1d has weight-distribution that ternary breaks, or the LR×3 is too high for SSM (SSM may need different LR rescue). Pivot to investigating SSM-specific ternary recipe.
- val_bpb between 0099's 1.65 and the 200-step SSM frontier 2.00: SSM stack benefit didn't fully transfer to longer training in compound, suggests something in CONTROL_TENSOR_NAME_PATTERNS or MAMBA2_KILL_SELECTIVITY differs subtly under ternary.
- Crash: CUDA-specific path issue with BitLinear in the parallel block forward. Debug.

## Notes from execution
