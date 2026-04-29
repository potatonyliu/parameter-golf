# Experiment 0100_dendritic_extended_cuda_2k

Parent: 0092_dendritic_memory_v1 (M=32K dendrites + warm-started K=4 patterns from 50M tokens + learnable d_content=32 + zero-init proj head; MPS 200-step val 2.020, parent 0086 was 2.013, Δ +0.007 = NEUTRAL)

## Question

Do dendritic content vectors converge with 10× more training? The journal notes a four-experiment chain — 0073 hash-HSM, 0080 dense-attn HSM, 0092 dendritic v1, 0094 dendritic+LR×3 — that all came out NEUTRAL at 200 MPS steps. The previous session interpreted this as "200 steps too short for any learnable on-top," but didn't actually test the interpretation by extending training. **This is the test.** If 0100 shows val_bpb significantly below 0092's parent baseline at 2000 steps, the training-duration-ceiling interpretation is supported and dendritic likely unlocks at H100 20k. If 0100 still tracks the no-dendritic baseline, the interpretation is wrong and dendritic has a real ceiling — closing the entire learnable-side-content thread cleanly.

This pairs with 0099 (ternary at 2000 steps): the two previous-session headline interpretations get tested together, in ~50 min total CUDA wall.

Also CUDA-validates `modules/dendritic_memory.py` (sorted-key buffer for exact K-gram match, gather/scatter ops). Prior MPS launch crashed once with a dtype mismatch; CUDA may surface different latent issues.

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb at 2000 steps:** I expect dendritic content vectors to converge meaningfully if the prior "training-duration-bound" interpretation is correct, since the dendritic side path goes from "20% fire rate, content vectors random" at step 200 to "20% fire rate, content vectors learned" at step 2000. If the content vectors carry signal proportional to ln(updates) (typical for slowly-converging tail params), at 10× more steps we should see Δ vs no-dendritic baseline materialize.

Predicted ranges (single-seed, low confidence):
- val_bpb in [1.50, 1.62]: dendritic content vectors converge, mechanism unlocks. Δ vs no-dendritic baseline at same 2000 steps in [-0.01, -0.04].
- val_bpb in [1.62, 1.70]: partial convergence, mixed signal.
- val_bpb in [1.70, 1.85]: ceiling is real, training-duration interpretation falsified.

These bands are wide because: (a) we don't have a 0086-equivalent (no dendritic) at CUDA 2000 steps as a clean comparator; (b) the 0099 ternary run will give us one indirect comparator (same parent 0086 ancestry, sans dendritic); (c) the BitNet ternary penalty's behavior is itself a confounder if 0099 doesn't close cleanly.

## Change

env.sh appends `ITERATIONS=2000` to inherited 0092 config. All else inherited verbatim — TERNARY_BODY=1, DENDRITIC_MEMORY=1, M=32K, K=4, d_content=32, BUILD_TOKENS=50M, MATRIX_LR default 0.045 (NOT the LR×3 from 0094 — 0092 was the original, 0094 was the LR-rescue follow-up that was also neutral).

No code changes. The `modules/dendritic_memory.py` and modules/bitlinear.py are inherited verbatim from 0092.

Predicted run time: ~700ms/step × 2000 = ~24 min wall plus dendritic overhead at forward time. Plus eval ~30s. Total ≤30 min.

## Disconfirming

- Pre-quant val_bpb > 1.85 at 2000 steps with dendritic ON: dendritic mechanism does NOT unlock with 10× more training. Closes the learnable-side-content thread.
- Pre-quant val_bpb tracks 0092's 200-step result (>1.95): training-duration interpretation falsified. Need code-level intervention (architecture change) not training-length change.
- Crash on CUDA: dendritic gather/scatter has a CUDA-specific bug; debug.
- Step1 train_loss > 25: BitLinear+dendritic stack has init-time difference vs 0098/0099 — flag and inspect.

## Notes from execution
