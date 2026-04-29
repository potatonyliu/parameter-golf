# Experiment 0112_kill_scan_ablation

Parent: 0103_ssm_frontier_cuda_2k (kill-Mamba-2 triple-parallel + no ternary, production batch 131072 × 1k steps, val 1.4587 cap-bust 21.4 MB)

## Question

**At production scale, is the SSM's recurrent scan contributing anything beyond what the conv1d already does?** The 0109 result showed SSM contribution shrinks from -0.085 BPB at MPS-200 to -0.011 BPB at production-1k vs pure-attn. This experiment isolates *which* part of the kill-Mamba-2 block is responsible for the residual -0.011: the conv1d (depthwise local mixing) or the LTI scan (long-range dynamics).

By keeping conv1d + in_proj + out_proj + D_skip + gate, but bypassing the SSD scan (set y = silu(x_conv) before the scan, instead of running the scan), we get a "conv1d-only" block that has the same parameter count (B/C/A_log/dt_bias still allocated but unused) and the same computational shape minus the recurrent dynamics.

If conv1d-only ≈ 0103: the SSM advantage at production scale is conv1d-shaped, not state-shaped. **Reframes the writeup**: the architectural contribution is depthwise convolution adjacent to attention, not state-space dynamics. Honest negative finding for the SSM thesis as articulated.

If conv1d-only > 0103 by ≥0.010: the LTI scan contributes meaningfully. SSM thesis holds at production scale; -0.011 vs pure-attn is real SSM-of-some-kind value.

## Hypothesis [CONJECTURE]

**Pre-quant val_bpb in [1.46, 1.48]** (single-seed). Two mental models:

- **Conv1d carries most of the SSM contribution at scale**: -0.011 BPB SSM-vs-attn at production = conv1d effect almost entirely. Predicted 0112 ≈ 1.46-1.47 (matches 0103's 1.4587).
- **The LTI scan contributes ~half**: 0112 lands ~1.470-1.475, +0.011 to +0.016 worse than 0103.

Either outcome is informative. The first reframes the thesis; the second confirms the current direction at scale.

Predicted artifact: ~21 MB cap-bust (same as 0103 — same params allocated, just one path ignored). step_avg likely ~600-700 ms (faster than 0103's 803ms because skipping the SSD scan).

## Change

env.sh inherits 0103 verbatim, override:
- `MAMBA2_KILL_SCAN=1` (NEW env-var, default 0 = byte-identical to 0103)
- `MAX_WALLCLOCK_SECONDS=1800` (preflight requires non-zero)

train_gpt.py code change in `Mamba2Block.forward` (current code at lines ~960-1017):

After computing `x_act = F.silu(x_conv)` (line 987), branch on the new env-var:

```python
if self._kill_scan:
    # 0112: conv1d-only ablation. Skip the SSD scan; treat conv1d output
    # (post-silu) as the "y" that would have come out of the scan.
    # Reshape to per-head and skip everything between line 990 (reshape) and
    # line 1011 (Y.reshape back to d_inner). i.e. y = x_act directly.
    y = x_act  # (b, l, d_inner)
else:
    # ... existing scan path (lines 990-1011) ...
    ...

# Continue with D_skip + gate as before (lines 1013-1017):
y = y + self.D_skip.float() * x_act.float()
y = y * F.silu(z.float())
return self.out_proj(y.to(in_dtype))
```

The B/C/dt parameters are still allocated (don't strip them — that would change param count and confound the comparison) but their values are ignored when MAMBA2_KILL_SCAN=1. This isolates "what does the scan computation contribute" from "what do the scan parameters contribute" — the latter is a follow-up, not this experiment.

Read `self._kill_scan` from `os.environ.get("MAMBA2_KILL_SCAN", "0") == "1"` in `__init__` (mirror the existing `self._kill_selectivity` pattern at line 946).

## Disconfirming

- val_bpb ≤ 1.465: scan adds nothing or harms at scale — strong reframe signal.
- val_bpb ≥ 1.480: scan contributes meaningfully (≥0.020 vs 0103). SSM thesis holds.
- val_bpb in [1.465, 1.480]: ambiguous; likely scan-adds-a-little. Mid-zone.
- Crash / NaN: code change broke the block path. Subagent rework.
- step_avg unchanged from 0103 (~800 ms): the scan computation wasn't actually skipped — verify code path with a debug print.

## Notes from execution

- Subagent edited only `experiments/0112_kill_scan_ablation/train_gpt.py` (canonical repo-root copy untouched). In `Mamba2Block.__init__`, added `self._kill_scan = os.environ.get("MAMBA2_KILL_SCAN", "0") == "1"` directly after the `_kill_selectivity` block (line 965). In `Mamba2Block.forward`, wrapped the SSD scan path (former lines 990–1011: per-head reshape, dt/A discretization, B/C broadcast, `ssd_minimal_discrete`, reshape back) in `if self._kill_scan: y = x_act.float()  else: <original scan>`; downstream `D_skip + silu(z)` gate + `out_proj` unchanged.
- B/C/A_log/dt_bias parameters and the `in_proj` split shape are untouched — unused when `MAMBA2_KILL_SCAN=1` so param count matches 0103 exactly.
- No deviations from plan.md; default path (`MAMBA2_KILL_SCAN` unset/0) executes the original lines verbatim (indentation-shifted into the `else:` branch only) so behavior is byte-identical to 0103.
