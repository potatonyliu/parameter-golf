# Experiment 0123_transformer_anchor_h100

Parent: 0119_ema_n5_ternary_beta99

## Question

At H100 600s wallclock, does our codebase's TRANSFORMER variant (pure-attention, no SSM) beat or trail the SSM variants (Path A 0121 kill-Mamba-2, v2 0122 dendrocentric)? Calibration anchor: tells us whether our gap to records' 1.10 BPB is **architectural** (SSM structurally worse than transformer at our scale) or **stack-port-debt** (we're missing parallel-residuals, GPTQ, sliding-window eval).

## Hypothesis [CONJECTURE]

At H100 600s:
- 0121 Path A (SSM) predicted in [1.20, 1.40]
- 0123 Transformer predicted in [1.18, 1.35] (slightly better since transformer is well-tuned in our codebase, lower per-step time → more steps)

Predicted Δ vs 0121 Path A: [-0.05, +0.05] (uncertain).

## Change

Forked from 0119. Differences:
- Architecture: kill-Mamba-2 triple-parallel → pure-attention 5-of-5
  (`ATTN_LAYER_POSITIONS=0,1,2,3,4`, `PARALLEL_LAYER_POSITIONS=`, `MAMBA2_KILL_SELECTIVITY=0`)
- EMA_BETA=0.99 → 0.999
- ITERATIONS=1000 → 10000 (wallclock fires first)
- TRAIN_BATCH_TOKENS=131072 → 524288
- WARMDOWN_ITERS=300 → 600
- VAL_TOKENS=16384 → 0 (full eval)
- MAX_WALLCLOCK_SECONDS=1800 → 600

## Disconfirming

- val_bpb > 1.50 → debug; transformer should at least match SSM
- val_bpb < 1.10 → too good, suspect bug
- artifact_mb > 16.0 → cap-bust; transformer with ternary should fit n=5
- NaN → LR cliff (try MATRIX_LR=0.020)

## Notes from execution

(Filled by run.)

## Launch (Round-3 contingency, after 0121 + 0122 succeed)

```
ssh runpod-tcp 'cd /workspace/parameter-golf-ssm && source .venv/bin/activate && \
                tmux new -d -s exp0123 \
                "bash scripts/runpod/launch_h100.sh experiments/0123_transformer_anchor_h100 --nproc 8"'
```

## Decision-tree role

After 0121 (SSM Path A) and 0122 (v2 dendrocentric) land, 0123 calibrates the writeup:
- 0121 ≤ 0123: SSM thesis supported at H100 scale.
- 0121 ≈ 0123: stack-port-debt is the gap; SSM-architecture-effect small.
- 0121 > 0123: our SSM stack underperforms; consider transformer for final submission.
