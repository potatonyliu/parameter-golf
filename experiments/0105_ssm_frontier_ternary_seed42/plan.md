# Experiment 0105_ssm_frontier_ternary_seed42

Parent: 0102_ssm_frontier_ternary_cuda_2k (SSM frontier + ternary + production batch + canonical LR; single-seed pre-quant 1.5414)

## Question

Cross-seed confirmation of 0102's val_bpb 1.5414. With SEED=42 (vs 0102's default 1337), we get the **2-seed mean** for the SSM × ternary × production-batch compound win. If σ_pair is small relative to the Δ vs comparators (0101 transformer baseline at 1.5739; 0099 transformer + ternary at 1.6499), the result is promote-quality.

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb at 1000 steps batch 131072, SEED=42 in [1.51, 1.58]** (single-seed, expected near 0102's 1.5414).

Calibration from prior cross-seed observations in this codebase:
- Mamba-2 / kill-Mamba-2 family σ_pair at MPS 200-step: 0.001 to 0.003 (per journal Current threads).
- Transformer + side-mem family σ_pair at MPS 200-step: 0.0036 typically.
- CUDA 2k regime: unknown but expected similar order — we have no prior cross-seed CUDA data for SSM stack.

If σ_pair ≤ 0.005, the 0102 result is robust and the compound is real. Δ to nearest comparator (0101 at 1.5739) is 0.032, ~6× a typical σ — should hold up cleanly.

## Change

env.sh inherits 0102 verbatim, override `SEED=42`. No code changes.

Predicted run: ~700-800 ms/step × 1000 = 12-13 min wall.

## Disconfirming

- |0105 - 0102| ≥ 0.030: σ_pair too wide for the Δ to be considered real. Either 0102 was a freak or the family σ at production batch is genuinely larger. Investigate before promoting.
- val_bpb ≥ 1.60 at SEED=42: suggests 0102 was a lucky single-seed and the compound isn't robust at this regime. Falsifies promote.
- val_bpb ≤ 1.50 at SEED=42: even better than 0102. Genuine promote candidate; mean would be ≤1.52, well below 0101.

## Notes from execution
