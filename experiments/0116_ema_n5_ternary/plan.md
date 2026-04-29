# Experiment 0116_ema_n5_ternary

Parent: 0107_bigger_ternary_ssm_prod (n=5 ternary, val 1.5256 at 1k steps, 9.08 MB)

## Question

Port the records-validated EMA-of-weights technique to our SSM stack. Records use β=0.999 with ~5-6k step training; gain ~-0.005 BPB. We test at 1k screening to verify composition with our triple-parallel ATTN||kill-Mamba-2 + ternary stack. The actual gain at our regime is expected to be small (-0.001 to -0.003) but the technique compounds with longer training and is required infrastructure for the H100 deployment recipe.

Math derivation in `scratch/2026-04-29_ema_derivation.md`. Toy verification in `scratch/2026-04-29_ema_toy.py` (passes — shadow lag formula `1/(T*(1-beta))` confirmed).

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.520, 1.532]** (single-seed). 0107 SEED=1337 = 1.5252 pre-quant; expect EMA at beta=0.999 starting after warmup to land roughly at parent or slightly better. Direction matters more than magnitude at 1k.

If val <= 1.524: EMA composes positively with our SSM stack. Add to H100 deploy stack.
If val in [1.525, 1.532]: neutral at 1k. Expected — EMA is a long-train technique. Keep for H100 deploy anyway (records confirm gain at 5k+).
If val > 1.532: EMA hurts at our regime. Investigate; likely the warmup-gating or BitLinear interaction.

Predicted artifact: 9.08 MB (shadow is eval-time-only; not in artifact).
Predicted step_avg: ~1345-1400 ms (small overhead from shadow update each step).

## Change

env.sh inherits 0107 verbatim, ADD:
- `EMA_BETA=0.999` (default 0.0 = disabled, byte-identical to 0107)
- `EMA_WARMUP_OFFSET=` (empty -> defaults to LR_WARMUP_STEPS = 30; before this step, shadow = current weights with no decay)
- `MAX_WALLCLOCK_SECONDS=1800` (preflight requires non-zero; predicted ~22 min)

`CONTROL_TENSOR_NAME_PATTERNS` is UNCHANGED. Shadow buffers are NOT model parameters; they're stored separately in a dict and never quantized.

train_gpt.py code change (subagent's job, all in `experiments/0116_ema_n5_ternary/train_gpt.py`):

### 1. Hyperparameters dataclass (around line 80-130 area where other env-vars are read)
Read the new env var:
```python
ema_beta: float = float(os.environ.get("EMA_BETA", "0.0"))
ema_warmup_offset_raw: str = os.environ.get("EMA_WARMUP_OFFSET", "")
```
In `__post_init__` (or wherever else other env vars get post-processed), if `ema_warmup_offset_raw` is empty/unset, set `self.ema_warmup_offset = self.lr_warmup_steps`. Else set to `int(self.ema_warmup_offset_raw)`.

### 2. EMA shadow init (in main(), AFTER model creation, BEFORE training loop)
Right after `model = ...` is constructed and before `step = 0` (around line 1844):
```python
ema_shadow: dict[str, torch.Tensor] | None = None
if args.ema_beta > 0.0:
    ema_shadow = {
        name: p.detach().float().clone()
        for name, p in base_model.named_parameters()
        if p.requires_grad
    }
    log0(f"[ema] enabled beta={args.ema_beta} warmup_offset={args.ema_warmup_offset} "
         f"shadowing {len(ema_shadow)} parameters")
```

Note: `base_model` is the unwrapped model (without DDP wrapper). Use `base_model.named_parameters()` so the names match for the swap later.

### 3. Shadow update (in training loop, AFTER the `for opt in optimizers: opt.step()` for-loop, BEFORE `step += 1`)
Around line 1906-1909, insert:
```python
if ema_shadow is not None:
    if step >= args.ema_warmup_offset:
        # EMA update: w_ema = beta*w_ema + (1-beta)*w_current. fp32 throughout.
        beta = args.ema_beta
        one_minus_beta = 1.0 - beta
        with torch.no_grad():
            for name, p in base_model.named_parameters():
                if name in ema_shadow:
                    ema_shadow[name].mul_(beta).add_(p.detach().float(), alpha=one_minus_beta)
    else:
        # Pre-warmup: shadow tracks current (no decay; avoids averaging random init).
        with torch.no_grad():
            for name, p in base_model.named_parameters():
                if name in ema_shadow:
                    ema_shadow[name].copy_(p.detach().float())
```

Note: `step` here refers to the variable BEFORE the `step += 1` line. So at "step 30" step variable is 30 from previous iteration's increment? Be careful — read context to determine if `step` is 0-indexed-pre-increment or 1-indexed-post-increment at this point in the loop. The loop's `last_step = step == args.iterations` check (line 1846) suggests `step` IS the count of completed steps before this iteration's opt.step. So inserting BEFORE `step += 1` (line 1909) means we update shadow BEFORE incrementing — at this moment, this iteration's update has just been applied. Use the post-increment value: condition `(step + 1) >= args.ema_warmup_offset`. Verify by reading the surrounding loop context.

### 4. Swap shadow into model BEFORE final eval (at last_step==True, just before line 1852 eval_val call)
Around line 1849-1852, insert before the `val_loss, val_bpb = eval_val(...)`:
```python
# 0116 EMA: swap shadow into model parameters for the final eval.
# Pre-quant eval, quant export, and post-quant eval all see the shadow weights.
# We don't restore — training is done at this point.
if ema_shadow is not None and last_step:
    log0(f"[ema] swapping shadow into model for final eval (beta={args.ema_beta})")
    with torch.no_grad():
        for name, p in base_model.named_parameters():
            if name in ema_shadow:
                p.data.copy_(ema_shadow[name].to(p.dtype))
```

This swap applies ONLY at last_step. For periodic mid-train val (when val_loss_every > 0), eval uses the live (non-EMA) weights. Our env has VAL_LOSS_EVERY=0 so this never fires mid-train; the swap is a one-shot at the end.

### 5. NO change needed at quant export site (line 2132)
After step 4, `base_model.state_dict()` returns the shadow-swapped weights. `quantize_state_dict_int8` sees those. Export uses shadow. Post-quant roundtrip eval also uses shadow (since we don't restore originals).

### 6. NO change needed for BitLinear
BitLinear inherits from nn.Linear; `self.weight` is the standard fp32 nn.Parameter. EMA on `self.weight` is EMA on the fp32 underlying weights — correct. The STE quantization is applied in forward and never affects the stored weight.

## Disconfirming

- val_bpb < 1.52: EMA helps even at 1k. Surprising. Add to H100 stack with high confidence.
- val_bpb in [1.52, 1.532]: typical neutral-at-1k. Keep for H100 (long-train sweet spot).
- val_bpb > 1.532: EMA hurts. Investigate — possibly warmup-gating wrong, or BitLinear's STE causes shadow drift in unexpected direction. Try beta=0.99 (window=100, captures only late training).
- Crash / NaN: shadow tensor type mismatch (fp32 vs bf16 cast issue). Verify `p.data.copy_(ema_shadow[name].to(p.dtype))` casts correctly.
- step_avg increase > 50ms vs 0107's 1345ms: shadow update is dominating step time. Profile and reduce frequency (every-N steps).

## Notes from execution

- Section 1 inserted at lines 208-216 (Hyperparameters class, after `conf_gate_threshold`). Hyperparameters is a plain class (no `@dataclass`/`__post_init__`); class-body executes at import time, so `ema_warmup_offset` is computed inline using the already-defined `lr_warmup_steps` class attribute (mirrors the existing inline `_w_sum` pattern at line ~186).
- Section 2 inserted at lines 1854-1864 in main() (after `t0 = time.perf_counter()`, before `step = 0`). `ema_shadow` lives in main()'s local scope; uses `base_model.named_parameters()` so names match the swap site.
- Section 3 inserted at lines 1940-1960 in the training loop (after `for opt in optimizers: opt.step()` + `zero_grad_all()`, before `step += 1`).
- Step-counter resolution: at the section-3 insertion point `step` is the PRE-INCREMENT count of completed updates (i.e., 0 on the very first iteration before this iteration's update). Since this iteration's `opt.step` has just been applied, the count of completed steps is `(step + 1)`. Used `(step + 1) >= args.ema_warmup_offset` per the plan's verified guidance: with `EMA_WARMUP_OFFSET=30`, the EMA decay branch first fires when step=29 (the 30th update has just been applied).
- Section 4 inserted at lines 1874-1882 (inside `if should_validate:` after `training_time_ms += ...`, before `val_loss, val_bpb = eval_val(...)`). Gated on `last_step` so mid-train val (when val_loss_every>0) uses live weights; final eval, int8 export, and post-quant roundtrip eval all see swapped (shadow) weights.
- Sections 5 and 6 confirmed untouched: quant export at line 2132 (now line 2168 after insertions) and BitLinear module are unmodified.
- Default `EMA_BETA=0.0` keeps `ema_shadow=None` and short-circuits all four if-blocks → byte-identical-in-execution to parent 0107.
