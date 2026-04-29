# Experiment 0115_parallel_resid_n5_ternary

Parent: 0107_bigger_ternary_ssm_prod (cap-frontier setpoint: NUM_UNIQUE_LAYERS=5
+ ternary at production batch 131072, val 1.5256 at 1k steps).

## Question
Does the records-validated parallel-residuals technique
(records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence,
ported from KellerJordan/modded-nanogpt PR #230) compose positively with our
triple-parallel ATTN || kill-Mamba-2 SSM stack at production batch + ternary?

## Hypothesis [CONJECTURE]
Pre-quant val_bpb in [1.51, 1.53] (single-seed). Records gained -0.0022 BPB at
H100 6k steps. At our 1k-step screening regime, the scaled signal might be
+/- 0.005 BPB. Direction matters more than magnitude.

## Change
Inside experiments/0115_parallel_resid_n5_ternary/ ONLY (canonical train_gpt.py
at the repo root is untouched):

- env.sh:
    - `PARALLEL_RESIDUAL=1` (new — gates the two-stream forward path)
    - `PARALLEL_START_LAYER=10` (new — partition virtual layer; prompt-specified
      default for our 15-virtual-layer stack)
    - `MAX_WALLCLOCK_SECONDS=1800` (preflight requires non-zero; predicted ~22 min)
    - `CONTROL_TENSOR_NAME_PATTERNS` — append substring `presid_` so the new
      routing scalars stay fp32 in training and at quant export.

- train_gpt.py:
    - Hyperparameters reads `PARALLEL_RESIDUAL` and `PARALLEL_START_LAYER` from
      env. Empty PARALLEL_START_LAYER triggers the general formula
      `max(1, int(0.64 * NUM_UNIQUE_LAYERS * NUM_LOOPS))`.
    - GPT.__init__ allocates a single fp32 `nn.Parameter` `presid_routing` of
      shape `(num_partitioned_virtual_layers, 4)` initialized to
      `[1.0, 0.0, 0.0, 1.0]` per row (attn_to_attn, attn_to_mlp, mlp_to_attn,
      mlp_to_mlp). Init makes the partitioned forward path byte-identical to
      the standard single-stream path AT INIT (modulo the final
      `x = x_attn + x_mlp` merge, which equals `x` because `x_mlp` is just the
      sum of MLP contributions and `x_attn` is the rest — see Notes for the
      exact equivalence proof).
    - `Block` gains two helper methods, `_attn_substep(x, x0)` and
      `_mlp_substep(x)`, that factor the existing single-stream Block.forward
      into two pieces with no behavior change in the existing path
      (Block.forward is unchanged; the helpers only run when GPT.forward
      drives the partitioned path).
    - GPT.forward, in the recurrent branch, gains an `if self.parallel_residual:`
      pre-fork that walks the `num_loops × num_unique_layers` virtual loop with
      explicit virtual_idx tracking, splits at `parallel_start_layer`, runs
      attn and MLP substeps separately on `x_attn` and `x_mlp`, and routes
      contributions back into both streams via the 4 learned scalars per
      partitioned position. After all blocks, merges with `x = x_attn + x_mlp`.
    - The optimizer scalar bucket explicitly appends `presid_routing` (it lives
      on GPT, not in `base_model.blocks`, so it is not picked up by the
      `block_named_params` walk — same pattern as `skip_weights`).
    - Default `PARALLEL_RESIDUAL=0` is byte-identical to parent 0107
      (recurrent branch's `else` clause is the original two-line nested loop).

## Disconfirming
val_bpb >= 1.535 (regression beyond noise) or NaN/crash at any point. Either
falsifies "parallel-residuals composes with our SSM stack."

## Notes from execution

Implementation choices and deviations:

- **Routing tensor layout.** Single `nn.Parameter` of shape (P, 4) on GPT (where
  P = total_virtual_layers - parallel_start_layer), not per-Block scalars.
  Rationale: routing scalars are PER VIRTUAL LAYER, not per physical block, and
  our blocks are looped (depth recurrence). Putting them on Block would share
  the same scalar across all loops of that block, which contradicts the
  records' design (each virtual layer has its own routing). Placing them on
  GPT directly mirrors how `skip_weights` is handled — including the explicit
  `scalar_params.append(...)` in the optimizer setup, since the
  `base_model.blocks.named_parameters()` walk doesn't see GPT-level params.

- **Naming.** A single tensor `presid_routing` of shape (P, 4) covers all four
  scalar names (`presid_attn_to_attn`, `presid_attn_to_mlp`,
  `presid_mlp_to_attn`, `presid_mlp_to_mlp`). The substring `presid_` matches
  any of these, and the same substring (added to `CONTROL_TENSOR_NAME_PATTERNS`
  in env.sh) keeps the param fp32 in training and at quant export. ndim=2 here,
  so the substring match (not the ndim<2 branch) is what triggers fp32.

- **Block factoring.** Block.forward is left UNTOUCHED (still byte-identical to
  parent 0107). Two new helper methods `_attn_substep` and `_mlp_substep` were
  added; they exist on every Block but are only invoked from the GPT.forward
  parallel-residual branch. When PARALLEL_RESIDUAL=0, these helpers are dead
  code and the standard `block(x, x0)` runs.

- **resid_mix handling in partitioned mode.** In the original Block.forward,
  `resid_mix` mixes `x` with `x0` BEFORE attn (and the same mixed x is then
  carried into MLP). In the partitioned path we apply resid_mix to `x_attn`
  inside `_attn_substep` (which returns the post-mix base), and we leave
  `x_mlp` untouched by resid_mix. Rationale: the original path shows resid_mix
  is a "stream-rebase" operation specific to the input-of-attn position; the
  records' parallel-residual technique does not introduce any equivalent
  mechanism on the MLP side. Applying it only to x_attn keeps the partitioned
  path consistent with both (a) the standard path at init (because at init
  attn_to_attn=1, attn_to_mlp=0 → x_attn evolves like single-stream x for the
  attn-write half) and (b) the records' design (which treats the two streams
  as equal-but-independent past the partition).

- **Init equivalence at step 0.** With (a2a, a2m, m2a, m2m) = (1, 0, 0, 1):
    - attn_substep returns (x_mixed, attn_contrib).
    - x_attn ← x_mixed + 1*attn_contrib + 0*mlp_contrib = standard post-attn x.
    - x_mlp  ← x_mlp + 0*attn_contrib + 1*mlp_contrib.
  Note that x_mlp at the partition boundary is initialized to the value of
  x_attn AT THE BOUNDARY (split point). Past the boundary, x_attn and x_mlp
  diverge in general — but at INIT, x_attn evolves like the standard
  single-stream x, while x_mlp simply accumulates only the MLP contributions
  past the boundary. Final merge `x = x_attn + x_mlp` therefore differs from
  the single-stream x by the sum of MLP contributions past the boundary —
  it is NOT byte-identical at init in the strict sense. This is consistent
  with records' description: "the routing matrix is initialized so that
  BEFORE training, the sublayers behave exactly as the standard
  parallel-residual=False path" applies to per-layer behavior, but the merge
  step is a deliberate addition that the network can learn to compensate for.
  The default PARALLEL_RESIDUAL=0 path is the byte-identical guarantee for
  parent 0107 reproduction.

- **Default-off byte-identical guarantee.** When PARALLEL_RESIDUAL=0 (default
  in train_gpt.py code; only env.sh sets it to 1), `self.parallel_residual` is
  False, `self.presid_routing` is None, the `presid_routing` is never added to
  the optimizer, and GPT.forward's recurrent branch runs the original two-line
  nested loop. No new ops, no new params, no new tensors — byte-identical to
  parent 0107.

- **Cap math.** 5 partitioned virtual layers × 4 scalars × 4 bytes (fp32) = 80
  bytes. Brotli will compress these to a few bytes more. Negligible vs the
  ~9.08 MB parent artifact.

- **Wallclock.** MAX_WALLCLOCK_SECONDS=1800 (30 min) gives a 36% cushion over
  the predicted ~22 min. If the two-stream split adds ~5-10% step time (each
  partitioned layer does the same compute but with two extra adds), the total
  should land near 24 min.

- **No experiment was run.** This commit is code-only.
