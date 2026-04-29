# Experiment 0110_bigger_ternary_seed2024

Parent: 0107_bigger_ternary_ssm_prod

Status: NOISE-FLOOR-SENTINEL (third seed of bigger-ternary-SSM family)

## Question

Third-seed run of the 0107/0108 config (NUM_UNIQUE_LAYERS=5 + ternary + parallel ATTN||kill-Mamba-2 + production batch 131072 + canonical LR 0.045 + 1000 steps). Completes the **noise-floor-sentinel** for the bigger-ternary-SSM family, characterizing cross-seed σ for THIS architecture class. Per the `noise-floor-sentinel` skill hard-rule, no SSM-family experiment is promoted before this skill completes; 0107/0108 are currently informational-only at 2-seed mean 1.5232 σ_pair=0.0034. With three seeds we get a calibrated σ to anchor promote thresholds against, and the 3-seed mean unblocks invoking the `promote` skill.

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.50, 1.55]** (single-seed). 0107 SEED=1337 = 1.5252 pre-quant, 0108 SEED=42 = 1.5200 pre-quant; their σ_pair was 0.0034 (post-quant). 3-seed σ likely tracks σ_pair within ~50%.

Post-quant prediction: in [1.51, 1.55]. Most-likely 3-seed mean: ≈ 1.523, σ ≈ 0.003–0.005.

If σ is in this range, the family advance threshold ≈ 3σ ≈ 0.010–0.015 — close to the transformer's, so program.md's standard thresholds approximately apply (no big recalibration needed for this family).

## Change

env.sh inherits 0107 verbatim, override `SEED=2024`. `MAX_WALLCLOCK_SECONDS=1800` (preflight requires non-zero on pod). All other env vars unchanged from 0107: NUM_UNIQUE_LAYERS=5, PARALLEL_LAYER_POSITIONS=0,1,2,3,4, ternary, batch 131072, MATRIX_LR=0.045, 1000 steps.

SEED=2024 is fresh in this codebase (0107=1337, 0108=42, 0102=1337, 0105=42 — no prior 2024 run). Step-1 train_loss should differ across all three sentinel seeds (cheap sanity check that we're not accidentally re-running a duplicate).

## Disconfirming

- |0110 - 0108| > 0.030 OR |0110 - 0107| > 0.030: cross-seed σ is wider than expected. Bimodal-LR-cliff regime (per primer §4.2). Run a 4th seed before concluding (per `noise-floor-sentinel` step 5).
- 0110 outside [1.49, 1.57]: outlier. Either freak-seed or a real instability we haven't characterized. Inspect step trajectory; possibly run 4th.
- 0110 ≥ 1.56: bigger-model + ternary is unstable. Promote candidate falls to 0102/0105 path (NUM_UNIQUE_LAYERS=3 + ternary).

## Notes from execution
