# Experiment 0107_bigger_ternary_ssm_prod

Parent: 0102_ssm_frontier_ternary_cuda_2k (SSM frontier + ternary at production batch; pre-quant 1.5414 / post-quant 1.5417, artifact 5.6 MB)

## Question

**Does the cap headroom freed by ternary (10+ MB unused in 0102) translate into actual val_bpb gain when spent on a bigger model?** This is the **headline question for SP1024 ternary work**: ternary trades +0.082 val for -16 MB cap (per 0102 vs 0103), so the right test is "does spending the freed cap on more model recover and surpass the val cost?"

If a 67%-bigger SSM with ternary lands at val ≤ 1.46 (= 0103's no-ternary smaller-model number, but cap-busted), then **ternary + bigger model is a strict win at fixed cap budget** at SP1024. That's a real writeup contribution — no record at SP1024 has explored this direction.

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.42, 1.55]** (single-seed). Reasoning:
- 0102 (NUM_UNIQUE_LAYERS=3, ternary, production batch, 1k steps): pre-quant 1.5414.
- 0103 (NUM_UNIQUE_LAYERS=3, no-ternary, production batch, 1k steps): pre-quant 1.4587 (cap-bust 21.4 MB).
- Bigger model at our scale: scaling laws suggest ~0.05-0.15 BPB improvement per 67% more params, depending on data efficiency.
- Cap math: 5.6 MB × (5/3) ≈ 9.3 MB → well under 16 MB cap, leaves ~6 MB further headroom.

Outcome buckets:
- val ≤ 1.46: bigger ternary BEATS smaller no-ternary at fixed cap. **Strong headline finding for SP1024.**
- val ∈ [1.46, 1.50]: marginal — bigger ternary is closer to 0103's 1.46 than smaller ternary's 1.54. Likely a real direction; multi-seed needed.
- val ∈ [1.50, 1.55]: bigger model didn't help much; ternary cost still binding.
- val > 1.55: bigger model HURTS in this regime. Surprising — would suggest data-efficiency / undertraining at 1k steps for bigger model.

## Change

env.sh inherits 0102 verbatim, override:
- `NUM_UNIQUE_LAYERS=5` (was 3) — bumps unique blocks per loop. With NUM_LOOPS=3, total effective layers = 15 (was 9).

No code change. Same TRAIN_BATCH_TOKENS=131072, MATRIX_LR=0.045, ITERATIONS=1000, TERNARY_BODY=1, PARALLEL_LAYER_POSITIONS=0,1,2 (ALL 5 positions parallel? Need to check default). Actually PARALLEL_LAYER_POSITIONS=0,1,2 explicitly only sets first 3; positions 3,4 will get default block (TransformerBlock). That changes the architecture. **TODO: extend PARALLEL_LAYER_POSITIONS=0,1,2,3,4 so all 5 unique blocks are parallel attn||kill-Mamba-2.**

Predicted run: step time scales ~linearly with params; expect ~1300 ms/step × 1000 = ~22 min wall.

## Disconfirming

- val ≥ 1.55: bigger model doesn't help at our scale — could be data-efficiency limited at 1k steps. Re-test at 5k steps if budget allows.
- Cap-bust at >16 MB: my 9.3 MB estimate is off. Check.
- Crash: more layers may interact poorly with kill-Mamba-2 in unanticipated ways. Investigate.

## Notes from execution
