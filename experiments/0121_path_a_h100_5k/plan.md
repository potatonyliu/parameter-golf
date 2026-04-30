# Experiment 0121_path_a_h100_5k

Parent: 0119_ema_n5_ternary_beta99

## Question

Does our best-confirmed stack (kill-Mamba-2 triple-parallel + n=5 + ternary + EMA β=0.999 + parallel-residuals + brotli + production batch) at H100 5000 steps land at a competitive submittable val_bpb? This is the INSURANCE submission — must produce a valid, sub-cap result regardless of whether v2 works.

## Hypothesis [LIKELY]

Stack composes at 5k+ steps. EMA β=0.999 (correct hyperparameter for ≥5k training, math-verified at 0119) + parallel-residuals (training-duration-bound at 1k smoke, records show -0.005 at H100 6k+) + ternary body + brotli all expected to compose.

Predicted post-quant val_bpb ∈ [1.20, 1.40]. Anchor: 0107 mean 1.5232 at 5090-1k; 0104 (no ternary, 5k steps, cap-busting) hit 1.3442; H100 records 1.10 at ≥10k steps.

## Change

Forked from 0119 (5090 1k EMA β=0.99 verification run). Differences:
- EMA_BETA=0.99 → 0.999 (window=1000, captures last 20% of 5k training)
- ADD PARALLEL_RESIDUAL=1 (records' tier-1 offset-merge)
- ITERATIONS=1000 → 5000
- TRAIN_BATCH_TOKENS=131072 → 524288 (8×H100 fits)
- WARMDOWN_ITERS=750 (15% of training, records' standard)
- VAL_TOKENS=0 (full eval, called twice for pre+post quant)
- MAX_WALLCLOCK_SECONDS=2400 (40 min budget on 8×H100)

NO novel mechanism. DENDROCENTRIC=0 (default).

## Disconfirming

- Δ vs 5090-1k anchor (0107 mean 1.5232) > +0.10 at H100 5k → stack does NOT compose at H100; investigate before v2 deploy
- val_bpb > 1.50 → far from records' floor; suggests training-duration insufficient or major bug
- artifact_mb > 16.0 → ternary infrastructure failed at H100; debug
- NaN / training divergence → LR cliff at production batch; reduce MATRIX_LR

## Notes from execution

(Filled by run.)

## Launch (8×H100, when pod up)

```
ssh runpod-tcp 'cd /workspace/parameter-golf-ssm && source .venv/bin/activate && \
                tmux new -d -s exp0121 \
                "bash scripts/runpod/launch_h100.sh experiments/0121_path_a_h100_5k --nproc 8"'
```
