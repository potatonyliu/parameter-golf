# Experiment 0104_ssm_frontier_long_5k

Parent: 0098_ssm_frontier_cuda_200 (kill-Mamba-2 triple-parallel + brotli; CUDA 200-step val 2.0028)

## Question

**Where does the SSM frontier land at extended training?** This is the user's question made concrete: "are we certain we cannot hit leaderboard records?". The 8×H100 record is 1.1063 BPB at 6240 steps × batch 524288 × 8 GPUs = 3.3B tokens. We'll run on a single 5090 at batch 131072 × 5000 steps = 655M tokens (~20% of the record's training, ~6.5% of the dataset). The extrapolation tells us whether the gap to 1.10 is mostly **training-duration-bound** (reachable by running longer) or **mechanism-bound** (need different architecture).

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb at 5000 steps in [1.20, 1.45]** (single-seed). Reasoning:
- 0098 SSM @ 200 steps batch 24576 (49M tokens): pre-quant 2.00
- 0099 transformer + ternary @ 2000 steps batch 24576 (49M tokens): pre-quant 1.65 (10× more steps × same batch = 10× tokens, dropped 0.35 BPB)
- 0103 (queued) SSM @ 1000 steps batch 131072 (131M tokens): TBD, est [1.50, 1.65]
- 0104 SSM @ 5000 steps batch 131072 (655M tokens): expected [1.20, 1.45] — 5× more tokens than 0103, log-scaling suggests ~0.3 BPB drop.

If this prediction holds, records are likely reachable by going to 30k+ steps at production batch (~2 hours additional).

Outcome buckets:
- val ≤ 1.30: SSM frontier scales well; records reachable. Plan: run a bigger one (10k+ steps) as the writeup deliverable.
- val ∈ [1.30, 1.45]: scaling, but slower than predicted. Plan: try longer training first, then mechanism work.
- val ≥ 1.45 (plateau near 0103): mechanism is bottlenecked. **The bold body-axis swing becomes the right move.**

## Change

env.sh inherits 0098 (canonical SSM frontier) verbatim, then overrides:
- `TRAIN_BATCH_TOKENS=131072` (5.3× MPS, fits 5090)
- `ITERATIONS=5000`
- `WARMDOWN_ITERS=750` (= 15% of iterations, matches the records' standard schedule shape)

`MAX_WALLCLOCK_SECONDS=0` + `ALLOW_NO_WALLCLOCK_CAP=1` for step-based lr_mul.

No code changes — the 0098 train_gpt.py is canonical (no BitLinear, no dendritic, no trigram).

Predicted run: ~600ms/step (with compile enabled, larger batch is slower per-token but better-amortized) × 5000 = ~50 min wall.

## Disconfirming

- val ≥ 1.50 at 5000 steps: SSM frontier doesn't scale with training duration; records are NOT reachable just by training longer. Mechanism work becomes essential.
- val ≥ 1.30 (within +0.20 of 0103 1k-step value): substantial training-duration gain but not enough to project to records. Need to combine with mechanism improvements.
- Crash mid-run (NaN, late instability): SSM-specific instability beyond ~1500 steps that we never saw at MPS 200. Investigate.

## Notes from execution
