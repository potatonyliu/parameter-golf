# Experiment 0113_full_sel_rescue_lr_third

Parent: 0103_ssm_frontier_cuda_2k (kill-Mamba-2 triple-parallel + no ternary, production batch 131072 x 1k steps, val 1.4587 cap-bust 21.4 MB)

## Question

**Was the "kill > full" finding at production scale a real architectural difference, or just an LR-cliff artifact?** 0106 ran full-selectivity Mamba-2 at production batch with canonical LR=0.045 and DIVERGED (val 3.54, train_loss froze 6.05 nats). The journal entry called it "informative-by-exclusion: kill > full at production by exclusion." But we never re-tried full-sel with a lower LR — Mamba's primer §4.2 documents sharp LR cliffs, so a 3x LR drop is a routine rescue, not exotic.

This experiment runs full-selectivity Mamba-2 at MATRIX_LR=0.015 (= 0.045/3), same production batch and 1000 steps. If it trains stably and lands competitive with 0103 (1.4587), the kill-vs-full conclusion at scale weakens — it was an LR question, not a mechanism question. If it diverges OR lands >= 0.030 worse, kill > full holds genuinely.

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.42, 1.50]** if full-sel trains stably at LR/3, OR diverges (val >= 3.0).

Mental models:
- **LR was the only issue** [most likely]: full-sel @ LR/3 trains stably, lands ~1.45 (close to 0103's 1.4587). kill > full at scale was LR-binding only. Reframes the kill-Mamba-2 family as "the under-tuned-LR-friendly variant", not "the architecturally-better variant."
- **Selectivity is genuinely under-trained at 1k steps**: full-sel @ LR/3 trains stably but lands 1.50+. Per primer §4.1, input-dependent projections (dt/B/C from in_proj) need more training than LTI constants. Confirms 0035/0036 at production scale.
- **Lower LR also diverges**: production batch + selectivity is fundamentally unstable at our regime. Need LR=0.005 or lower.

Predicted artifact: ~22 MB (slightly larger than 0103 — full-sel adds in_proj output channels for dt/B/C), cap-bust expected.

## Change

env.sh inherits 0103, override:
- `MAMBA2_KILL_SELECTIVITY=0` (was 1 — turn the selectivity back ON)
- `MATRIX_LR=0.015` (was 0.045 — divide by 3 to clear the LR cliff)
- `MAX_WALLCLOCK_SECONDS=1800` (preflight)

All other env vars unchanged. Single-seed.

## Disconfirming

- val_bpb diverges (>= 3.0): selectivity at production batch needs even lower LR. Try LR=0.005 next.
- val_bpb in [1.42, 1.48]: kill-vs-full was an LR-tuning question. Major reframe — full-sel at properly-tuned LR is competitive with kill at scale. Opens the H100 deployment direction (records use full-sel-style mechanisms).
- val_bpb in [1.49, 1.55]: full-sel marginally worse than kill at LR/3. Both viable. Ambiguous.
- val_bpb >= 1.56: full-sel + LR/3 is materially worse. kill > full at scale holds. Stay on kill direction.
- step_avg >= 1.5x 0103's 803ms: selectivity costs more compute at scale. Worth tracking even if val is similar.

## Notes from execution
