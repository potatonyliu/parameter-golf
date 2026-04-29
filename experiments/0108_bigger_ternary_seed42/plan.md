# Experiment 0108_bigger_ternary_seed42

Parent: 0107_bigger_ternary_ssm_prod (NUM_UNIQUE_LAYERS=5 + ternary at production batch; SEED=1337 single-seed val 1.5256, artifact 9.08 MB)

## Question

SEED=42 cross-seed confirm of 0107 to get a 2-seed mean for the bigger-ternary-SSM promote candidate. If 0107/0108 mean ≤ 1.53 at submittable cap (~9 MB), this is the new best **submittable 2-seed** result of session — beats 0102/0105 mean 1.5462 by -0.015+ BPB AND has substantially more cap headroom for further architectural additions next session.

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.51, 1.55]** (single-seed). 0107 SEED=1337 was 1.5252 pre-quant. σ_pair from 0102/0105 was ~0.006-0.009; expect similar magnitude here. 2-seed mean predicted ~1.52-1.53.

## Change

env.sh inherits 0107 verbatim, override `SEED=42`. Same NUM_UNIQUE_LAYERS=5, PARALLEL_LAYER_POSITIONS=0,1,2,3,4, ternary, batch 131072, LR 0.045, 1000 steps.

## Disconfirming

- |0108 - 0107| > 0.030: cross-seed σ at bigger-model regime is wider than expected. Either freak seed or wider σ at higher capacity. Re-evaluate σ.
- 0108 ≥ 1.56: bigger model is unstable; smaller model + ternary (0102/0105 path) is the more stable promote candidate.

## Notes from execution
