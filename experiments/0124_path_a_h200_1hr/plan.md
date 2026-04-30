# Experiment 0124_path_a_h200_1hr

Parent: 0121_path_a_h100_5k

## Question

Does the SSM frontier stack (kill-Mamba-2 triple-parallel + n=5 + ternary + EMA β=0.999) scale to records-class val_bpb when given the dominant-gap closer (training tokens)? Per `scratch/2026-04-29_record_recipe_analysis.md`, training duration is "the dominant gap to records (-0.40 to -0.60 BPB)" — this experiment closes that gap.

## Hypothesis [LIKELY]

At 3600s wallclock × 4×H200 → ~9000-14000 steps × 524288 batch = 4.7-7.3B tokens (1.5-2.2× records' 3.27B). Slope from 0103→0104 (-0.115 per 5× tokens) extrapolated to 6.3B tokens predicts:

- 0107 (5090 1k × 131k batch = 131M tokens): 1.5232
- 0104 (5090 5k × 131k = 655M, no-ternary, cap-bust): 1.3442
- Records (8×H100 6240 × 524k = 3.27B, full polish): 1.1063
- Our 4×H200 long-train (~6.3B tokens, our SSM stack): predicted **[1.15, 1.30]**

If lands ≤1.20 → strong demonstration of SSM at scale, competitive with non-records peer "1 Bit Quantization" 1.1239.

## Change

Forked from 0121 (Round 1 de-risk). Differences:
- MAX_WALLCLOCK_SECONDS=600 → 3600 (1hr)
- ITERATIONS=10000 → 20000 (upper bound)
- WARMDOWN_ITERS=300 → 1800 (15% of expected steps)

Same: kill-Mamba-2 triple-parallel + n=5 + ternary + brotli + EMA β=0.999, batch 524288, full eval.

## Disconfirming

- val_bpb > 1.40 at 1hr → slope didn't transfer; SSM stack is fundamentally limited
- val_bpb < 1.10 → too good, suspect bug (records' best is 1.1063 with way more polish)
- artifact_mb > 16.0 → cap-bust unexpected at this stack
- NaN at any step → LR cliff; reduce MATRIX_LR
- Step time > 600ms/step → step time prediction was way off; we trained far fewer tokens than expected

## Notes from execution

(Filled by run.)

## Launch (4×H200 SXM, after 0121 Round 1 confirms infrastructure clean)

```
ssh runpod-tcp 'cd /workspace/parameter-golf-ssm && source .venv/bin/activate && \
                tmux new -d -s exp0124 \
                "bash scripts/runpod/launch_h100.sh experiments/0124_path_a_h200_1hr --nproc 4"'
```
