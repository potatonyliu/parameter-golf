# Experiment 0121_path_a_h100_5k

Parent: 0119_ema_n5_ternary_beta99

## Question

ROUND 1 DE-RISK: does the SSM frontier stack run cleanly on actual 4×H200 SXM hardware at 600s wallclock, with predictable step time and clean eval? Goal: verify infrastructure + measure step time before committing $16 to long-train (0124).

## Hypothesis [LIKELY]

At 4×H200 600s × 524288 batch:
- Predicted step time 250-400ms → 1500-2400 steps × 524288 = 0.79-1.26B tokens
- Predicted post-quant val_bpb in [1.30, 1.50] (~2.4× records' tokens, but our stack lacks polish)

Goes / no-go for Round 2 (0124 long-train):
- GO if val_bpb ≤ 1.50, no NaN, step time ≤ 600ms, artifact_mb ≤ 16.0
- NO-GO if NaN or val_bpb > 1.60 → debug before scaling

## Change

Forked from 0119 (5090 1k EMA β=0.99 verification). Differences:
- EMA_BETA=0.99 → 0.999 (correct for ≥5k training; ~20% capture window at our scale)
- ITERATIONS=1000 → 10000 (upper bound; wallclock fires first)
- TRAIN_BATCH_TOKENS=131072 → 524288 (production batch on H200)
- WARMDOWN_ITERS=300 (15% of expected ~2000 steps)
- VAL_TOKENS=16384 → 0 (full eval)
- MAX_WALLCLOCK_SECONDS=600 (de-risk Round 1 budget; ~$2.66 on 4×H200)

NO novel mechanism. DENDROCENTRIC=0. NO PARALLEL_RESIDUAL (defer; not in 0119 base).

## Disconfirming

- val_bpb > 1.60 → stack broken at H200; debug before Round 2
- NaN / training divergence → LR cliff; reduce MATRIX_LR
- step_avg_ms > 600 → step time worse than predicted; tokens trained way fewer than expected
- artifact_mb > 16.0 → ternary infrastructure failed at H200

## Notes from execution

(Filled by run.)

## Launch (4×H200 SXM, ROUND 1)

```
ssh runpod-tcp 'cd /workspace/parameter-golf-ssm && source .venv/bin/activate && \
                tmux new -d -s exp0121 \
                "bash scripts/runpod/launch_h100.sh experiments/0121_path_a_h100_5k --nproc 4"'
```

## After Round 1: decision tree

- val_bpb ≤ 1.40 + healthy infrastructure → launch 0124 (Path A long-train 1hr) immediately
- val_bpb 1.40-1.60 + healthy infrastructure → launch 0124 anyway; results valid
- val_bpb > 1.60 OR NaN OR step_avg > 600ms → DEBUG. Do NOT launch 0124 until fixed.
- artifact_mb > 16.0 → infrastructure issue; debug
