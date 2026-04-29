# Experiment 0117_dendrocentric_v1_n5_ternary

Parent: 0107_bigger_ternary_ssm_prod (n=5 ternary, val 1.5256 at 1k steps, 9.08 MB)

## Question

**Replace the SwiGLU MLP in each block with a Boahen-inspired dendrocentric layer (sparse-selection version, v1).** Tests whether a 3x-cap-efficient sparse-compute MLP-replacement can match or beat the dense MLP at our regime. This is the simplest possible test of the brief's option (f) — full Boahen-style ordering is deferred to v2.

Brief: `scratch/2026-04-29_session_planning.md` option (f).
Boahen: `references/Boahen_2022_dendrocentric_learning_Nature612.pdf` Box 2 + "The dendrite of a pyramidal neuron" — sparse synaptic inputs + NMDA sigmoidal nonlinearity.
Math derivation: `scratch/2026-04-29_dendrocentric_v1_derivation.md` (cap math: 3x saving vs MLP at this M, K).
Toy verification: `scratch/2026-04-29_dendrocentric_v1_tiny.py` (4 progressive toys all pass: by-hand, shape flow, top-K STE backward, sparse-target recovery).
Design choices: `scratch/2026-04-29_dendrocentric_design_choices.md` (Option A: replace MLP, all 5 layers; M=2048, K=8; sigmoid NMDA; ternary output projection).

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.51, 1.58] at 1k steps**, with three plausible regimes:
- **20%**: v1 lands within +/- 0.005 of parent (1.5252). Compute density matches MLP — strong direction.
- **40%**: v1 lands neutral-to-slightly-worse (1.53-1.55). Cap-efficient but training-duration-bound.
- **40%**: v1 lands materially worse (>=1.55). Either mechanism wrong or training-duration-bound. Distinguished by H100 long-train.

**The 1k-step 5090 smoke is most likely UNINFORMATIVE about the dendrocentric thesis** — prior dendritic attempts (0073, 0080, 0092, 0094, 0100) were all neutral at short training. Real test is H100 long-train.

This experiment's primary value is: (a) verify the layer trains stably without NaN, (b) confirm step-time cost, (c) confirm cap math (predicted ~7.6 MB for n=5 dendritic + ternary BitLinear output projection — saves ~1.5 MB vs 0107's 9.08 MB).

Predicted artifact: ~7.6 MB (saved cap from MLP replacement).
Predicted step_avg: 1100-1400 ms (sparse compute is faster than dense MLP).

## Change

env.sh inherits 0107, ADD:
- `DENDROCENTRIC=1` (default 0 = byte-identical to 0107, uses standard SwiGLU MLP)
- `DENDRO_M=2048` (number of dendrites per layer)
- `DENDRO_K=8` (sparsity per dendrite — STE keeps top-K |w| in each row)
- `DENDRO_ALPHA=1.0` (NMDA sigmoidal slope; fixed in v1)
- `MAX_WALLCLOCK_SECONDS=1800`

Append `dendro_` to `CONTROL_TENSOR_NAME_PATTERNS` so per-dendrite theta scalars stay fp32 in optimizer routing AND at int8 quant export.

train_gpt.py code change:

### 1. Add a new module file: `experiments/0117_dendrocentric_v1_n5_ternary/modules/dendrocentric.py`

```python
"""Dendrocentric layer v1: sparse-selection MLP replacement (no order sensitivity).

References:
- Boahen 2022 (Nature 612), Box 2 + "The dendrite of a pyramidal neuron"
- Math: scratch/2026-04-29_dendrocentric_v1_derivation.md
- Toys: scratch/2026-04-29_dendrocentric_v1_tiny.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSTE(torch.autograd.Function):
    """Forward: keep top-K |w| per row, zero rest. Backward: STE (gradient flows to all positions).

    Trainability bridge for emergent sparse pattern. Mirrors BitLinear's STE for
    ternary quantization but applied to sparsity instead of value-quantization.
    """
    @staticmethod
    def forward(ctx, w, k):
        abs_w = w.abs()
        _, topk_idx = abs_w.topk(k, dim=-1)
        mask = torch.zeros_like(w)
        mask.scatter_(1, topk_idx, 1.0)
        return w * mask

    @staticmethod
    def backward(ctx, grad_output):
        # Bypass top-K mask — gradient flows to all positions.
        return grad_output, None


def topk_ste(w, k):
    return TopKSTE.apply(w, k)


class DendrocentricLayer(nn.Module):
    """Replace MLP block with a dendrite bank.

    Each dendrite j reads from K-sparse input channels (via top-K STE on dendro_W_d),
    sums weighted contributions, applies sigmoidal NMDA-like nonlinearity, and
    contributes to a ternary BitLinear output projection.
    """
    def __init__(self, d_model: int, n_dendrites: int, k_per_dendrite: int, alpha: float = 1.0):
        super().__init__()
        from modules.bitlinear import BitLinear
        self.d_model = d_model
        self.M = n_dendrites
        self.K = k_per_dendrite
        self.alpha = alpha
        # Continuous full-rank parameter; rendered K-sparse via top-K STE in forward.
        # Substring 'dendro_' in CONTROL_TENSOR_NAME_PATTERNS keeps these fp32-protected.
        self.dendro_W_d = nn.Parameter(torch.randn(n_dendrites, d_model) * 0.02)
        # Per-dendrite threshold (1D, auto-fp32 since 1D).
        self.dendro_theta = nn.Parameter(torch.zeros(n_dendrites))
        # Output projection M -> d. Use BitLinear (ternary) to match the rest of the body.
        self.dendro_W_out = BitLinear(n_dendrites, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        W_d_sparse = topk_ste(self.dendro_W_d, self.K)  # (M, d), K nonzeros per row
        a = F.linear(x, W_d_sparse) - self.dendro_theta  # (..., M)
        h = torch.sigmoid(self.alpha * a)
        return self.dendro_W_out(h)
```

### 2. Add Hyperparameters fields (around the existing env-var reads at lines ~80-130)

```python
dendrocentric: int = int(os.environ.get("DENDROCENTRIC", "0"))
dendro_m: int = int(os.environ.get("DENDRO_M", "2048"))
dendro_k: int = int(os.environ.get("DENDRO_K", "8"))
dendro_alpha: float = float(os.environ.get("DENDRO_ALPHA", "1.0"))
```

Match the existing class style (Hyperparameters is a plain class — see how `ema_beta` was added in 0116; mirror that pattern).

### 3. In the Block class wherever MLP/SwiGLU is instantiated, branch on the env var

You'll need to FIND the existing MLP construction in train_gpt.py. The field name might be `self.mlp`, `self.feed_forward`, `self.ffn`, etc. Match the existing pattern.

Replace the construction line(s) with:
```python
if args.dendrocentric == 1:
    from modules.dendrocentric import DendrocentricLayer
    self.mlp = DendrocentricLayer(
        d_model=d_model,
        n_dendrites=args.dendro_m,
        k_per_dendrite=args.dendro_k,
        alpha=args.dendro_alpha,
    )
else:
    self.mlp = SwiGLU(d_model, mlp_mult=args.mlp_mult)  # existing path; verify exact signature
```

The downstream call site `out = self.mlp(x)` must work for both branches. DendrocentricLayer's forward returns shape `(..., d_model)` — same as SwiGLU. No call-site change needed.

### 4. CONTROL_TENSOR_NAME_PATTERNS update in env.sh

The existing line in 0107:
```
export CONTROL_TENSOR_NAME_PATTERNS="attn_scale,attn_scales,...,delta_bias,conv1d"
```

Append `dendro_` (substring match catches `dendro_W_d`, `dendro_theta`, `dendro_W_out` — though `dendro_W_out`'s weight is the BitLinear weight, which gets routed via the `bitlinear_weight_names` set at quant export, separate from CONTROL_TENSOR_NAME_PATTERNS):
```
export CONTROL_TENSOR_NAME_PATTERNS="...,delta_bias,conv1d,dendro_"
```

## Disconfirming

- Crash / NaN: implementation bug. Check (a) BitLinear handles n_dendrites=2048 -> d_model=512, (b) top-K STE backward is registered correctly (test on a single forward+backward in scratch), (c) sigmoid doesn't saturate (check `a` magnitudes < 5).
- val_bpb > 1.60: dendritic compute is materially worse than MLP — likely needs ordering (v2). Document clearly; defer v2 to next session.
- val_bpb in [1.55, 1.60]: training-duration-bound (matches prior dendritic neutral pattern). Keep for H100 deploy candidate list with explicit "needs longer training" tag.
- val_bpb in [1.52, 1.55]: marginal. Probably worth keeping.
- val_bpb < 1.52: surprising positive. Add to H100 deploy. Run cross-seed if writeup-quality.
- artifact_mb > 9.0 (worse than 0107): cap math wrong. Investigate.
- step_avg > 0107's 1345 ms: sparse compute slower than dense — STE overhead. Accept if val gain justifies.

## Notes from execution

- Created `experiments/0117_dendrocentric_v1_n5_ternary/modules/dendrocentric.py` (66 lines) verbatim from plan §1: TopKSTE autograd Function + DendrocentricLayer class. Imports BitLinear lazily inside __init__ to avoid circulars.
- Added 4 fields to Hyperparameters at lines 207-216 (immediately after `conf_gate_threshold` on line 206, before the `# MUON OPTIMIZER` separator). Plain class style with `name = int(os.environ.get(...))` (no type annotations) — matches the rest of Hyperparameters; the type-hinted form in the plan would have been the only annotated field in the class.
- MLP construction site: `Block.__init__`, formerly line 1270 `self.mlp = MLP(dim, mlp_mult)` — now lines 1280-1292, branched on `Hyperparameters.dendrocentric == 1`. Variable name in Block is `dim` (not `d_model`); passed as `d_model=dim` to DendrocentricLayer. Block does not have `args` in its constructor signature, so the branch reads class-level Hyperparameters attributes directly (consistent with the existing MLP class, which also reads `os.environ` directly at line 1196). This preserves the no-plumbing-change requirement.
- Default DENDROCENTRIC=0 path verified byte-identical to 0107 via `diff`: only the additive Hyperparameters block and the if/else around the MLP line differ; the `else` branch contains the unchanged `self.mlp = MLP(dim, mlp_mult)` line.
- env.sh verified: CONTROL_TENSOR_NAME_PATTERNS appended with `dendro_`, DENDROCENTRIC=1, DENDRO_M=2048, DENDRO_K=8, DENDRO_ALPHA=1.0, MAX_WALLCLOCK_SECONDS=1800 all present (lines 39, 138, 184-188).
