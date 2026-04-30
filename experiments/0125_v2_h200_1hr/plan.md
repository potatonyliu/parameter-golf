# Experiment 0125_v2_h200_1hr

Parent: 0122_dendrocentric_v2_h100_5k

## Question

At fair token budget (matched to 0124 Path A long-train), does v2 dendrocentric (DFSM stored ordering) beat Path A baseline? This is the brief-aligned ordering test at adequate training duration.

## Hypothesis [CONJECTURE]

At 1hr × 4×H200 → ~9000-14000 steps × 524288 batch = 4.7-7.3B tokens (same as 0124 Path A).
Honest priors:
- 25-30% v2 beats Path A by ≥0.020 BPB (ordering helps)
- 40-45% v2 within ±0.020 of Path A (ordering neutral)
- 25-30% v2 worse by ≥0.020 (mechanism imperfect or still token-bound)

Predicted v2 val_bpb in [Path A − 0.05, Path A + 0.10].

## Change

Forked from 0122. Differences:
- MAX_WALLCLOCK_SECONDS=600 → 3600 (1hr; matches 0124)
- ITERATIONS=10000 → 20000
- WARMDOWN_ITERS=600 → 1800

Same: M=1024 K=8 v2 dendrocentric, batch 524288, EMA β=0.999, full eval.

## Disconfirming

- v2 > Path A by ≥0.05 → ordering claim FAILS at fair conditions
- std(L) stays at init 1.0 throughout → DFSM didn't fire
- artifact_mb > 16.0 → unexpected cap-bust
- NaN at any step → numerical issue

## Notes from execution

(Filled by run.)

## Launch (after 0124 Round 2 lands; ROUND 3)

```
ssh runpod-tcp 'cd /workspace/parameter-golf-ssm && source .venv/bin/activate && \
                tmux new -d -s exp0125 \
                "bash scripts/runpod/launch_h100.sh experiments/0125_v2_h200_1hr --nproc 4"'
```

## Decision tree

- v2 ≪ Path A (≥0.05 better): SUBMIT v2 (ordering supported at scale)
- v2 ≈ Path A (±0.020): SUBMIT lower; ordering neutral; cap-density (ternary) carries
- v2 ≫ Path A (≥0.020 worse): SUBMIT Path A; ordering not supported (clean null)
