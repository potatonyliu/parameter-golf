# Experiment 0120_dendrocentric_v2

Parent: 0117_dendrocentric_v1_n5_ternary

## Question

Does dendrocentric ordering (DFSM-style soft-rank with stored permutation) help val_bpb at H100 5k+ steps where v1 (no ordering) failed at 1k MPS smoke (+0.156)?

This is the brief-aligned test of "Can ordering of binary spikes carry more usable information than bits alone?"

5090 1k smoke is **code-verify only** per `feedback_5090_explore_h100_writeup.md` — not a Δ-decision. Real test is H100 5k+.

## Hypothesis [CONJECTURE]

Adding stored ordering (DFSM-trainable permutation per dendrite) gives access to temporal-rank coding (~6.6× density at K=8 from `temporal_rank_capacity_sim.py`). At H100 5k+ steps with enough training to fix DFSM permutations, predicted val_bpb in [Path A − 0.05, Path A + 0.05].

5090 1k smoke prediction (code-verify): trains stably (no NaN, monotonic descent), val_bpb in [1.55, 1.85]; step time within 2× v1 (1051ms → 1100-2100ms); std(L) grows from init 1.0.

## Change

**Module**: `experiments/0120_dendrocentric_v2/modules/dendrocentric.py` — replace v1 forward with v2 forward (math: `scratch/2026-04-30_dendrocentric_v2_derivation.md`, toys: `scratch/2026-04-30_dendrocentric_v2_tiny.py` (all 4 pass)).

v2 changes vs v1:
1. Add `dendro_L: nn.Parameter(torch.randn(M, K) * 1.0)` — stored preferred ordering per dendrite.
2. Constructor adds `tau_x: float = 1.0, tau_l: float = 1.0` args.
3. Forward replaced with:
   - Re-compute top-K indices per dendrite (independent of TopKSTE).
   - Gather x at top-K indices: `gathered = x_flat[:, topk_idx]` shape `(N, M, K)`.
   - Weight by W_d at top-K positions (STE-aware via dendro_W_d): `weighted = gathered * w_at_topk`.
   - Soft-rank of K weighted values per dendrite (using tau_x): `r ∈ R^{N,M,K}`.
   - Soft-rank of stored L (using tau_l): `rho ∈ R^{M,K}`.
   - Centered Pearson correlation: `s = (12/(K(K^2-1))) · sum_k (r_k - (K-1)/2)(rho_k - (K-1)/2)`.
   - NMDA activation: `h = sigmoid(alpha · s - dendro_theta)` where alpha=4.0 by default.
   - Output: `o = dendro_W_out(h)` (BitLinear unchanged).

**train_gpt.py** — minimal edits to `Hyperparameters` (~lines 207-216) and DendrocentricLayer instantiation (~line 1283-1290):
1. Add to Hyperparameters:
   - `dendro_tau_x = float(os.environ.get("DENDRO_TAU_X", "1.0"))`
   - `dendro_tau_l = float(os.environ.get("DENDRO_TAU_L", "1.0"))`
2. Update `dendro_alpha` default to `4.0` (was 1.0 for v1; v2 needs higher α since s_j ∈ [-1,+1]).
3. Update DendrocentricLayer instantiation to pass `tau_x=Hyperparameters.dendro_tau_x, tau_l=Hyperparameters.dendro_tau_l`.

**env.sh** — overrides for v2:
- `DENDRO_ALPHA=4.0` (was 1.0; v2 score range needs sharper sigmoid)
- `DENDRO_TAU_X=1.0`
- `DENDRO_TAU_L=1.0`
- `ITERATIONS=1000` (5090 1k smoke)
- `MAX_WALLCLOCK_SECONDS=2700` (45 min cap; v2 expected ~50% slower than v1's 18min/1k)
- All other inherited from 0117 (n=5 ternary stack)

## Disconfirming

**5090 1k smoke** disconfirms v2 IF:
- NaN before step 100 → numerical issue in soft-rank (sigmoid arg overflow on large pairwise diffs)
- step_avg_ms > 3000ms (3× v1) → forward computation cost prohibitive even at H100 scale
- val_bpb > 1.95 (worse than 0118 spike-rank +0.359) → mechanism worse than no-ordering simpler version

**H100 5k+** disconfirms v2 IF:
- Δ vs Path A baseline > +0.05 → ordering doesn't help at fair conditions, brief DISPROVEN
- std(L) per dendrite stays at init 1.0 throughout → DFSM trainability bridge doesn't fire in our regime

## Notes from execution

(Filled by subagent / during run)
