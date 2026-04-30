# Experiment 0122_dendrocentric_v2_h100_5k

Parent: 0120_dendrocentric_v2

## Question

Does v2 dendrocentric (DFSM-style stored ordering) at H100 5000 steps produce a competitive val_bpb and prove the brief's ordering claim? This is the BRIEF-PROVING submission attempt; Path A (0121) is the insurance.

## Hypothesis [CONJECTURE]

DFSM-trained per-dendrite permutations + soft-rank correlation gives access to ~6.6× temporal-rank density per K=8 selection (verified in `temporal_rank_capacity_sim.py`). At H100 5k+ steps, gradient signal through soft-rank is strong enough to fix discriminative permutations.

Predicted post-quant val_bpb ∈ [Path A − 0.05, Path A + 0.10]. Honest priors: ~25-30% v2 wins outright at H100 5k; ~40% lands neutral; ~30% lands worse (training-duration-bound or mechanism-wrong); ~5% numerical issue.

## Change

Forked from 0120 (5090 v2 smoke). H100 deploy adjustments:
- TRAIN_BATCH_TOKENS=32768 → 524288 (H100 has 80GB; v2 buffer fits)
- ITERATIONS=1000 → 5000
- ADD EMA_BETA=0.999 + EMA_WARMUP_OFFSET (matching 0121 stack)
- ADD PARALLEL_RESIDUAL=1 (matching 0121 stack)
- WARMDOWN_ITERS=750
- VAL_TOKENS=0 (full eval)
- MAX_WALLCLOCK_SECONDS=2400

Architecture identical to 0120 (same v2 dendrocentric module, M=2048, K=8, α=4.0, τ_x=τ_L=1.0).

## Disconfirming

- Δ vs 0121 Path A H100 baseline > +0.05 → ordering doesn't help at fair conditions; brief DISPROVEN. Submit 0121.
- std(L) per dendrite stays at init 1.0 throughout → DFSM trainability bridge doesn't fire in our regime; null finding.
- artifact_mb > 16.0 → v2 cap bloat at H100; ineligible.
- NaN at any step → numerical issue; debug or drop.

## Notes from execution

(Filled by run.)

## Launch (8×H100, when 0121 Path A finishes successfully)

```
ssh runpod-tcp 'cd /workspace/parameter-golf-ssm && source .venv/bin/activate && \
                tmux new -d -s exp0122 \
                "bash scripts/runpod/launch_h100.sh experiments/0122_dendrocentric_v2_h100_5k --nproc 8"'
```
