# Experiment 0002_regression_check_cuda_v2

Parent: canonical

## Question
Does this fresh CUDA pod bit-reproduce 0001_baseline_repro (MPS val_bpb 2.5212, 6.907 MB) within the CUDA-vs-MPS bf16 drift tolerance? Cheap insurance before novel SSM work.

## Hypothesis [LIKELY]
val_bpb falls in 2.5212 ± 0.05 (i.e. roughly 2.47 - 2.57). MPS and CUDA differ in bf16 reduction order, but on a 200-step canonical baseline the drift is typically <0.02 BPB. Artifact size should match within ±0.1 MB (no architecture changes — only numeric drift in stored weights, which int8-quantizes identically up to LSB noise).

## Change
None. Canonical env.sh from new_experiment.sh, including MAX_WALLCLOCK_SECONDS=0 (deliberate — selects the step-based lr_mul branch, matching the MPS anchor's schedule). Preflight is bypassed for this run; standard launches set MAX_WALLCLOCK_SECONDS to a real number.

## Disconfirming
- val_bpb drifts > 0.05 from 2.5212 → CUDA path has a regression OR our MPS→CUDA drift assumption is wrong; investigate before novel work.
- Crash, OOM, NaN → CUDA-only failure mode (e.g. flash-SDP path); investigate.
- Artifact size differs by > 0.1 MB → quant export changed; harness drift.

## Notes from execution
Filled by regression_sentinel.sh after run.
