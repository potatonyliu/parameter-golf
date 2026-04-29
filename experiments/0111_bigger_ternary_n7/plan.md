# Experiment 0111_bigger_ternary_n7

Parent: 0107_bigger_ternary_ssm_prod (NUM_UNIQUE_LAYERS=5 + ternary + production batch, val 1.5256, 9.08 MB)

## Question

**Does depth keep paying back at production scale, beyond NUM_UNIQUE_LAYERS=5?** The cap-frontier push.

The 2-seed mean for NUM_UNIQUE_LAYERS=5 + ternary at production-1k is 1.5232 (0107/0108) at 9.0 MB submittable. The ternary recipe frees enough cap that we have ~7 MB of headroom under 16 MB cap. If depth keeps paying back, the actual cap-frontier submittable is somewhere around NUM_UNIQUE_LAYERS=7-8, not 5.

Cap math (linear extrapolation from 0102 NUM_UNIQUE_LAYERS=3 -> 5.63 MB and 0107 NUM_UNIQUE_LAYERS=5 -> 9.08 MB):
- per-layer marginal: 1.725 MB
- floor (heads + embed + control tensors + brotli): 0.46 MB
- NUM_UNIQUE_LAYERS=7 -> ~12.5 MB (3.5 MB headroom)
- NUM_UNIQUE_LAYERS=8 -> ~14.3 MB (1.75 MB headroom)
- NUM_UNIQUE_LAYERS=9 -> ~16.0 MB (borderline cap-bust)

NUM_UNIQUE_LAYERS=7 is the safe cap-frontier — a clear positive delta here would justify NUM_UNIQUE_LAYERS=8 next; a flat delta would say "we're at the depth ceiling at production-1k."

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.49, 1.52]** (single-seed). 0102 -> 0107 paid -0.023 for +2 unique layers. Linearly: 0107 -> n=7 should pay another -0.023 -> ~1.502. With diminishing returns: -0.012 -> ~1.513.

If linear: clear cap-frontier signal, push to n=8 next.
If diminishing: depth ceiling at production-1k. Pivot to other axes (training duration or parallel residuals).

Predicted artifact: 12.5 +/- 1 MB (cap-safe).
Predicted step_avg: ~1880 ms; total wallclock ~31 min.

## Change

env.sh inherits 0107 verbatim, override:
- `NUM_UNIQUE_LAYERS=7` (was 5)
- `PARALLEL_LAYER_POSITIONS=0,1,2,3,4,5,6` (extends to cover all 7 unique blocks)
- `MAX_WALLCLOCK_SECONDS=2400` (40 min - buffer above predicted 31 min)
- All other env vars (ternary, batch 131072, MATRIX_LR=0.045, ITERATIONS=1000, SEED=1337) unchanged.

Single-seed for triage. If delta vs 0107 >= 0.010 (clear cap-frontier signal), follow up with SEED=42 cross-seed confirm and possibly NUM_UNIQUE_LAYERS=8.

## Disconfirming

- val_bpb >= 1.520: depth ceiling reached. 0107's NUM_UNIQUE_LAYERS=5 is the right setpoint. Pivot.
- val_bpb < 1.495: bigger-than-linear improvement. Strong cap-frontier signal — sweep to n=8 fast.
- Crash / NaN: bigger model + canonical LR + production batch may be unstable. If so, lower MATRIX_LR to 0.030 and rerun.
- Artifact >= 16.0 MB: cap math was off; back off to n=6.

## Notes from execution
