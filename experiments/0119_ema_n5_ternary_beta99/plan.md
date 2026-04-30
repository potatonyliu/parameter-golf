# Experiment 0119_ema_n5_ternary_beta99

Parent: 0116_ema_n5_ternary (inherits EMA code + env.sh; only override is EMA_BETA)

## Question

**Disambiguate 0116's catastrophic val (2.2022) — was it correct EMA + wrong hyperparameter, or broken implementation?** My own derivation (`scratch/2026-04-29_ema_derivation.md`) predicted: at beta=0.999 with 1k training steps, shadow effective window = 1/(1-beta) = 1000 = entire training, so shadow ≈ value at training start = near-init weights. But that prediction is consistent with EITHER (a) correct EMA + lag = full window OR (b) broken EMA that doesn't update.

Re-running at beta=0.99 (window=100, captures last ~10% of training) tests this: if implementation is CORRECT, shadow at beta=0.99 should land NEAR 0107's val (1.5252) — possibly slight gain from late-training stability, possibly tiny regression if last-100-steps weights are noisier than final. If implementation is BROKEN (shadow doesn't update), val will land near 2.20 again.

This is the cheapest informative experiment from this session's wrap blueprint. ~22 min, $0.40.

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.510, 1.535] at 1k steps** if EMA implementation is correct (most likely outcome):
- Lower bound: -0.015 if late-training shadow captures step-700-to-1000 stability gain.
- Upper bound: +0.010 if last-100-step shadow is noisier than final weights.
- Most likely: ~0107's 1.5252 ± 0.005.

If val ≈ 2.20 (matches 0116): EMA implementation is BROKEN — shadow not actually updating.
If val < 1.49: implausibly large gain at 1k smoke.
If val > 1.55: still hurts at beta=0.99 — investigate why (warmup-gating wrong, BitLinear interaction).

## Change

env.sh inherits 0116, override only `EMA_BETA=0.99` (was 0.999). All other env vars (n=5 ternary + production batch + canonical LR + EMA_WARMUP_OFFSET=auto + MAX_WALLCLOCK_SECONDS=1800) unchanged.

train_gpt.py is byte-identical to 0116 (same code path; only env-var differs).

## Disconfirming

- val_bpb >= 2.0: EMA broken — shadow not updating. CRITICAL bug; investigate before H100 deploy.
- val_bpb in [1.55, 2.0]: hurts but not catastrophic. EMA partially working; investigate warmup-gating or shadow init.
- val_bpb in [1.510, 1.535]: implementation correct, hyperparameter only. CONFIRMS infrastructure for H100. Add EMA at beta=0.999 to H100 deploy stack with confidence.
- val_bpb < 1.50: implausibly good at 1k. Re-check.
- Crash / NaN: should match 0116 (didn't crash). If new crash, something changed in the env.sh.

## Notes from execution
