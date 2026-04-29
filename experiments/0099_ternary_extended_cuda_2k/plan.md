# Experiment 0099_ternary_extended_cuda_2k

Parent: 0093_ternary_lr_rescue_packed (BitLinear ternary body + MATRIX_LR=0.135 + v2 packed-ternary serialization on the recur+SwiGLU+mlp=8 transformer baseline + static side memory; MPS 200-step val 1.993 at 8.21 MB)

## Question

Does the +0.045 BPB ternary penalty (0093 vs 0076 fp/int8 baseline 1.948) close with extended training, as BitNet b1.58 predicts? At 200 MPS steps we are ~100× short of BitNet's "needs ~25× more steps to match fp16" benchmark; 2000 CUDA steps gives 10× more training, an interpolation point that should show *substantial* closure if the recipe transfers, but not full convergence. **Load-bearing for the entire ternary direction** — if the penalty is flat at 10× more training, the H100 cascade is not warranted; if it shrinks meaningfully, the H100 ternary push is the right next move.

This also tests CUDA validation of the BitLinear primitive (`modules/bitlinear.py`: absmean STE forward + packed-ternary export) which has only run on MPS to date. Bug surface: STE backward via autograd, packed serialization at quant export, dtype paths.

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb at 2000 steps in [1.45, 1.65]** (vs 0093 MPS 200-step pre-quant 2.0563). The 2000-step transformer-best run on CUDA (without ternary) should land somewhere in the same band; the ternary penalty narrows from +0.05 (200 steps) toward +0.02 (2000 steps) per BitNet's monotonic-in-steps claim. Single-seed.

**Post-quant val_bpb in [1.46, 1.66]** (lossless 2-bit packing → quant tax stays small ~0.01).

Outcome buckets:
- pre-quant val_bpb ≤ 1.55 AND penalty narrows: BitNet claim transfers; advance to H100 push.
- val_bpb 1.55–1.70: partial closure, mixed signal; need a non-ternary 2000-step baseline at same config to decompose.
- val_bpb > 1.70 OR penalty stays at +0.045: ternary recipe fundamentally undertrained at this regime; investigate before H100.
- crash on CUDA at any step: BitLinear has a CUDA-specific dtype/STE bug; debug.

## Change

env.sh appends `ITERATIONS=2000` to inherited 0093 config. Schedule shape: 0–30 LR warmup, 30–1700 constant LR, 1700–2000 linear warmdown (step-based lr_mul branch via `MAX_WALLCLOCK_SECONDS=0` + `ALLOW_NO_WALLCLOCK_CAP=1`).

No code changes. The BitLinear and trigram_side_memory modules are inherited verbatim from 0093.

Predicted run time: ~700ms/step × 2000 = ~24 min wall, plus ~30s eval. Total under 30 min.

## Disconfirming

- Pre-quant val_bpb > 1.70 at 2000 steps: BitNet's "more steps closes the gap" prediction does not transfer to our SP1024 / 16 MB / 5M-parameter regime; the ternary direction needs different intervention (init? optimizer? LR schedule?), not just more steps.
- Pre-quant val_bpb tracks 0093's 200-step result (>1.95): the recipe is not learning at the predicted rate; recall the journal's "0093 stack at MPS 200 steps shows the penalty was only HALF rescued by LR×3" — extended training may be a no-op if the bottleneck is recipe-fundamental.
- Crash on CUDA: invalidates the cheap-test premise; need a port pass before drawing any conclusion.
- step1 train_loss > 25 (significantly larger than 0098's 20.6): different from the SSM-frontier stack, possibly a BitLinear-specific init quirk to flag.

## Notes from execution
