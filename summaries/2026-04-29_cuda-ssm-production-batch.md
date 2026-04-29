# Session 2026-04-29 · CUDA SSM at production batch — first paid-pod run

**Best results [n=2]**: 0107/0108 SSM + ternary at NUM_UNIQUE_LAYERS=5, batch 131072, canonical LR=0.045, 1000 steps. **2-seed mean post-quant val_bpb 1.5232** at **9.0 MB** (σ_pair 0.0034). Best CUDA-regime submittable. NOT promoted — SSM promotes require noise-floor-sentinel (3 seeds), deferred to next session.

**Best raw val [n=1, cap-bust]**: 0104 SSM frontier no-ternary at 5000 steps × batch 131072 = post-quant **1.3442** at 24.06 MB. Establishes the training-duration scaling slope.

**Span**: 2026-04-29 ~10:30 EDT → 15:30 EDT (5 hours). Cost ~$5 of pod time at $1/hr 5090.

**Experiments**: 0098-0109 (12 experiments queued, 11 ran cleanly, 1 informative diverge).

**Theme**: Validate MPS→CUDA SSM transfer, characterize SSM × ternary × production-batch space, demonstrate gap to records is training-duration-bound not mechanism-bound.

---

## Stack of confirmed wins (this session)

1. **0098 SSM CUDA anchor [n=1, sentinel]**: kill-Mamba-2 triple-parallel + brotli at 200 steps batch 24576. val_bpb post-quant 2.0028. Δ vs MPS 0064 = -0.0002 (clean transfer). Step time 12.5× MPS speedup. → `journals/2026-04-29 · session opening: CUDA pod 5090, MPS→CUDA SSM transfer, workflow patches`.
2. **0102 SSM × ternary at production batch [n=1]**: post-quant 1.5417 at 5.6 MB. → `journals/2026-04-29 · 0102 SSM × ternary × production-batch compound — best result of session [n=1]`.
3. **0103 SSM no-ternary at production batch [n=1]**: post-quant 1.4591 at 21.4 MB (cap-bust). The same model without ternary, demonstrating the cap-trade. → `journals/2026-04-29 · 0103 result recasts 0102 — ternary's value is purely cap-saving [n=1]`.
4. **0104 SSM no-ternary @ 5k steps [n=1]**: post-quant 1.3442 — best raw val of session, establishes scaling slope -0.115 BPB per 5× tokens at production batch.
5. **0107/0108 bigger ternary SSM [n=2 with σ_pair=0.0034]**: NUM_UNIQUE_LAYERS=5 with all parallel attn||kill-Mamba-2 + ternary. 2-seed mean **1.5232** at 9.0 MB. **NEW BEST SUBMITTABLE 2-SEED OF SESSION**.

## Cross-experiment lessons

1. **MPS→CUDA SSM transfer is clean** [n=1]: 0098 reproduced MPS 0064 to within 0.0002 BPB. Triple-parallel kill-Mamba-2 stack is hardware-portable.
2. **Production batch 131072 fits 5090 (32 GiB) at NUM_UNIQUE_LAYERS=3**, OOMs at canonical 524288. With NUM_UNIQUE_LAYERS=5, peak VRAM ~21 GiB at batch 131072.
3. **Ternary trade is cap-saving, NOT val-saving at our regime** [n=2 production batch]: 0102 vs 0103 = +0.082 val cost for -16 MB cap. The "compound win" framing (initially) was an artifact of the production-batch effect. Actual trade is well-defined and consistent.
4. **Ternary cap-trade pays back partially when freed cap is spent on more depth** [n=2]: 0107/0108 (NUM_UNIQUE_LAYERS=5 + ternary, 9 MB) at 1.5232 beats 0102/0105 (NUM_UNIQUE_LAYERS=3 + ternary, 5.5 MB) at 1.5462 by -0.023 BPB. But still loses to 0103 (NUM_UNIQUE_LAYERS=3 no-ternary, cap-bust 21.4 MB) at 1.46 by +0.066 BPB.
5. **SSM contribution shrinks at production scale** [n=1]: 0109 pure-attn at production batch hits 1.4699; SSM frontier 0103 hits 1.4587 → SSM advantage is **only -0.011 BPB**, vs MPS-200's -0.085. SSM advantage was regime-specific. **Meaningful negative finding** for the SSM thesis as articulated.
6. **Full-selectivity Mamba-2 at production batch + canonical LR DIVERGES** [n=1]: 0106 val 3.54 (uniform random ~4.17). LR=0.045 too aggressive once selectivity adds dt/B/C input projections at production gradient signal. Confirms kill > full at production by exclusion. Lower-LR rerun would be the rescue.
7. **Training duration is the dominant axis for closing the records gap** [n=1]: 0103→0104 = -0.115 BPB just from 5× more training. Records use 33% of fineweb10B (3.3B tokens); we used 6.5% in our longest run (655M tokens at 0104). Another 5× training (~25k steps, 3h on 5090) projects to ~1.20-1.25, within 0.1 BPB of records.
8. **BitNet 25× claim transfers QUALITATIVELY but slowly at our 27M-param SP1024 regime** [n=2 from 0099/0101 + 0102/0103]: ternary penalty narrowed from MPS-200 ~+0.10 to CUDA-2k +0.076 (small batch) to +0.082 (production batch). Closure rate ~0.005 per 5× tokens — would still leave ~+0.05 at H100 record's training. **Cap-saving is the value, not val parity.**

## Dead axes (verified this session, don't re-test without changing other levers)

- **Full-selectivity Mamba-2 at LR=0.045 + production batch**: diverges. Need lower LR to even train. (0106).
- **Ternary at our 27M-param SP1024 regime won't reach val parity by training duration alone**: BitNet claim partially transfers but closure rate too slow. Stay with ternary as cap-saver (0099/0102/0107).

## Set in stone (this session)

- 0098 anchor val_bpb 2.0028 = canonical SSM frontier MPS→CUDA reproduction baseline.
- Stop hook on RunPod must check pod-side processes via SSH (fixed in `.claude/hooks/stop-reminder.sh`).
- `await_steps.sh` Linux/macOS portability: `stat -c %Y` first (fixed).
- `await_steps.sh` `grep -c` empty-match double-zero: use `|| true` not `|| echo 0` (fixed).
- `ssh runpod` PTY blocks Bash; use `ssh runpod-tcp`. Documented in skill SKILL.md.
- Pod git remote is `origin` (Mac is `fork`).
- torch.compile fullgraph=True breaks on `Tensor.item()` and boolean indexing (`tensor[mask]`); for modules with these patterns, disable compile entirely on CUDA. Hit at 0099 trigram (post-quant only) and 0100 dendritic (training).
- `run_experiment.sh` post-script requires `parent` field in result.json — `echo "{}"` to clear stale state breaks it.

## Set in hypothesis (single-seed, needs confirmation)

- 0107/0108 mean 1.5232 at 9 MB ([n=2, σ_pair 0.0034]) vs 0102/0105 mean 1.5462 at 5.5 MB ([n=2, σ_pair 0.0064]). Bigger model + ternary trades 3.5 MB cap for -0.023 BPB. **Needs noise-floor-sentinel for proper promote.**
- 0104 SSM 5k-step extrapolation 1.3442 [n=1]. Linear-log scaling projection puts 25-30k steps near 1.20-1.25.
- SSM contribution shrinks to -0.011 BPB at production scale [n=1]. Would need cross-seed to confirm.

## Walk reflections this session

- `walks/2026-04-29_1311.md` — generated 0106 (full-selectivity at production), 0107 (bigger ternary), 0109 (pure-attn at production). All ran. Two yielded informative results, one diverged informatively.
- `walks/2026-04-29_1448.md` — generated 0108 (cross-seed of 0107) + 0109 (pure-attn anchor). Both informative.

## Follow-ups for next session, ranked by EV

1. **Noise-floor-sentinel on the SSM frontier + ternary at production batch** (3 seeds): unblocks promote of 0107/0108-like winners. ~36 min.
2. **25-30k step run on the SSM frontier no-ternary stack** at production batch (~3h on 5090): closes most of the training-duration gap to records. Predicted val_bpb 1.20-1.25.
3. **Tier-1 ports**: parallel residuals + EMA at SSM frontier (subagent specs pre-written in `scratch/2026-04-30_next_session_plan.md`).
4. **0107/0108 + longer training**: bigger ternary SSM at 5k+ steps. Fork 0108. ~50 min.
5. **Pure-attn cross-seed confirm of 0109** (1 experiment, ~14 min): solidifies the SSM-contribution-at-scale finding.
6. **Full-selectivity at production batch with LOWER LR** (~14 min): rescue 0106. If lands ≤ 1.46, full-selectivity transfers at scale with proper LR; if not, kill > full holds beyond exclusion.

Lower-priority parked: dendritic (5 neutral attempts — likely closed thread); spike-rank embedding/body (untested); GLA/Hyena fork (untested); DENDROCENTRIC layer (deferred per the walk).

## Reflections

**What went well**:
- Walk-driven experiments (0106-0109) produced more signal than the verification cascade alone.
- Outside-eyes review at session midpoint correctly flagged anchoring on cascade work and prompted thread-2 pivot.
- The chain-script pattern (`/tmp/launch_chain*.sh`) saved many minutes of manual handoff between experiments.
- Pre-committed decomposition rules (`scratch/2026-04-29_decomposition_rules.md`) made the 0102/0103 result interpretation unambiguous.

**What didn't go well**:
- 5+ launch attempts on 0100 due to cascading torch.compile + `.item()` / boolean indexing bugs. Should have tested module under `torch.compile(fullgraph=True)` BEFORE first launch on pod. Lesson: any MPS-developed module needs a CUDA compile-test smoke before production launch.
- 0099 post-quant crashed (dynamo bug) — lost the post-quant data point. Workaround: TRIGRAM_SIDE_MEMORY=0 for screening; for promote candidates, fix trigram_side_memory.py:634 .item() call.
- Initially called 0102 a "compound win" before 0103 landed. Self-correcting was good; pre-committing decomposition rules earlier would have prevented the premature claim.
- Two separate `echo "{}"` resets of result.json broke the run_experiment.sh post-script (`parent` KeyError) at 0102 chain. Should have known the harness contract.

**Anti-patterns to avoid next session**:
- Don't bump batch into OOM territory blindly. 5090 at NUM_UNIQUE_LAYERS=3 fits 131072; canonical 524288 OOMs. NUM_UNIQUE_LAYERS=5 is also fine at 131072 (~21 GiB peak).
- Don't disable torch.compile reflexively when something breaks — the cost is 2× slower training. Investigate the actual module first.
- Don't stop the chain on every restart and lose previously-completed work; check `result.json` before clearing.

## Next agent's first concrete action

**Run noise-floor-sentinel on the 0107/0108 config (NUM_UNIQUE_LAYERS=5 + ternary + production batch + 1k steps).** Three SEEDS (1337, 42, 2024). If σ stays tight (~0.003-0.005), invoke `promote` skill on the 3-seed mean. Expected ~36 min total. Then proceed to the 25-30k step long-training run.
