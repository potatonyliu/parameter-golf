# Experiment 0098_ssm_frontier_cuda_200

Parent: 0064_brotli_swap (= 0051 triple-parallel + brotli swap; MPS val 2.0030 single-seed, artifact 13.44 MB, step 8.64s)

## Question

Does the kill-Mamba-2 triple-parallel SSM frontier reproduce on CUDA (RTX 5090) at the same 200-step config? Establishes the **CUDA SSM anchor** before any longer-training or mechanism-flip work in this session, and verifies MPS→CUDA numerics on the SSM stack (transformer-only anchor 0002 already showed 0.0097 drift; the SSM stack adds Mamba-2 conv1d/recurrence which exercises different kernel paths).

## Hypothesis [CONJECTURE]

val_bpb_post_quant in [1.99, 2.02] — within ±0.01 of MPS 0064's 2.0030. CUDA bf16 reduction order on the recurrence may shift the result by ~0.005-0.010 (similar magnitude to the transformer anchor's 0.0097). Step time ~700ms (Mamba-2 sequential scan slower than transformer's 134ms; ~10× speedup vs MPS 8.64s).

Outcome buckets:
- val ∈ [1.99, 2.02], step <1500ms → CLEAN TRANSFER. Use 2.00 as CUDA SSM anchor; advance to 0099 long-training.
- val outside [1.97, 2.04] → real numerics drift. Investigate before extending to 2k steps.
- crash / NaN / step1 ≠ ln(vocab) → kernel path issue, debug.

## Change

env.sh inherits 0064 verbatim. The MPS-tuned config is the apples-to-apples comparison:
- `ITERATIONS=200`, `WARMDOWN_ITERS=300`, `MAX_WALLCLOCK_SECONDS=0` (step-based lr_mul branch — preflight needs `ALLOW_NO_WALLCLOCK_CAP=1`).
- `TRAIN_BATCH_TOKENS=24576` (MPS-memory-tuned; not yet bumped to CUDA canonical 524288 — that's a different question).
- `PARALLEL_LAYER_POSITIONS=0,1,2`, `PARALLEL_SSM_TYPE=mamba2_kill`, `MAMBA2_KILL_SELECTIVITY=1`, `BIGRAM_VOCAB_SIZE=0`.
- No code changes.

Launch with `ALLOW_NO_WALLCLOCK_CAP=1` (sentinel-style canonical-repro on a foreign device).

## Disconfirming

- val ≥ 2.04 (drift > 0.04 vs MPS): CUDA numerics or kernel path materially shifts SSM training; not just bf16 reduction order. Block before extending.
- val ≤ 1.97: implausibly large gain on the same code/config — investigate (data path? eval bug?).
- step time > 3× prediction (>2100ms): kernel path wrong; flag.
- step1 not ≈ ln(vocab) ≈ 6.93 or step2 > 2× step1: trajectory broken.

## Notes from execution
