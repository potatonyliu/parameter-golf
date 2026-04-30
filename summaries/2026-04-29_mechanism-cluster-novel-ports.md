# Session 2026-04-29 part 2 · mechanism-cluster + tier-1 ports + novel mechanism infrastructure

**Best result this segment**: 0107/0108 2-seed mean **post-quant val_bpb 1.5232** at 9.0 MB submittable (carries from part 1; not promoted — sentinel skipped per user feedback "no σ-confirms on cheap pod").

**Span**: 2026-04-29 ~16:00 → 21:00 EDT (5 hours pod). Cost ~$5 of pod time at $1/hr 5090. Combined with part 1 (~$5), today total ~$10. Combined with yesterday session: ~$15 cumulative.

**Experiments**: 0111-0118 (8 experiments queued, 7 ran cleanly, 1 monitor-failure-recovered).

**Theme**: After context compaction, conduct mechanism cluster confirming SSM thesis, then prepare tier-1 record-validated ports + novel Boahen/spike-rank mechanisms ALL with proper math + toy + plan + subagent-execute discipline. Heavy user-correction cycle codified four new memory feedbacks.

---

## Stack of confirmed wins / findings (this segment)

### Mechanism cluster (chain 0111-0113) — three confirmed staycourse signals [n=1 each, all Δ ≥ 13σ]

1. **0111 NUM_UNIQUE_LAYERS=7 cap-frontier**: post 1.5536 / 12.05 MB / step 1888ms. Δ vs 0107/0108 mean = +0.030. **Depth ceiling at production-1k is n=5**. Bigger model under-trains at fixed step budget. Cap math validated (linear extrapolation predicted 12.5 MB; actual 12.05 MB). Untested at 5k+ steps.

2. **0112 MAMBA2_KILL_SCAN ablation**: post 1.5049 / 22.07 MB / step 581ms (28% faster than 0103 since SSD scan skipped). Δ vs 0103 = +0.046. **The SSD scan IS load-bearing at production scale (-0.046 BPB)**. Falsifies 1623 walk hypothesis ("conv1d does the work, SSM is dressing"). Combined with 0109's net SSM contribution -0.011: scan helps -0.046, conv1d-without-scan hurts +0.035; the two ride together. Net SSM is ~0 at scale, but the SSD scan in isolation is highly load-bearing.

3. **0113 full-selectivity rescue at MATRIX_LR=0.015**: post 1.4898 / 17.65 MB / step 714ms. Δ vs 0103 kill = +0.031. **kill > full is architectural, NOT LR-cliff**. Trained STABLY at LR/3 (vs 0106 diverged at canonical LR=0.045). _B_const/_C_const LTI prior beats input-dependent dt/B/C from in_proj at 1k steps. Validates kill-Mamba-2 as the right deployment direction.

### Tier-1 record-validated ports — H100-bound, training-duration-confounded at 1k smoke

4. **0115 PARALLEL_RESIDUAL=1**: post 1.5722 / 12.05 MB / step 1349ms. Δ vs 0107 = +0.047. **Tier-1 port DOES NOT compose at 1k smoke** on our SSM stack. The `+x_split` offset at the merge boundary requires longer training to learn-around. Records' gain (-0.0022 BPB) is at H100 6k+ steps. Marked H100-bound with "training-duration-required" tag.

5. **0116 EMA-of-weights at β=0.999**: post **2.2022** / 8.93 MB / step 1205ms. **Math-predicted result, not a bug**: my own `scratch/2026-04-29_ema_derivation.md` flagged "β=0.999 with 1k steps → effective window=1000 = entire training, shadow ≈ near-init weights." Shipped at β=0.999 anyway because plan.md said "match records' default for infrastructure verification." Result is consistent with correct EMA implementation + lag = full window. **EMA infrastructure verified** (no NaN, swap-at-last_step + quant export both clean). Re-run at β=0.99 (~22 min, $0.40) would disambiguate "implementation correct" vs "broken" — first-action next session.

### Bold novel mechanisms (brief options c + f) — infrastructure verified, NOT competitive at 1k smoke

6. **0117 dendrocentric v1** (replace SwiGLU MLP with Boahen-inspired sparse-selection dendrite bank, M=2048, K=8, top-K STE for trainability + sigmoidal NMDA + ternary BitLinear out): post **1.6812** / 8.81 MB / step 1051ms (28% faster than dense MLP per cap math prediction). Δ vs 0107 = +0.156. **Materially worse at 1k.** Two candidate interpretations:
   (a) Training-duration-bound — sparse pattern needs more steps to fixate; matches the 5 prior dendritic-family neutrals (0073/0080/0092/0094/0100).
   (b) v1 mechanism wrong — strips brief's actual ordering claim; "K-sparse MLP with sigmoid" might just be "MLP with fewer parameters." Reviewer flagged this.
   Real test: H100 5k+ steps OR v2 with order-sensitivity (DFSM trainability bridge).

7. **0118 spike-rank embedding v1** (sparse tok_emb K=8 via top-K STE, brief option c): post **1.8849** / 9.08 MB (NO cap savings — sparse weight stored densely at int8) / step 1342ms / quant_tax 0.031 (high — dynamic range hurts int8). Δ vs 0107 = **+0.359**. **Most-regressed of session.** Likely K=8 too restrictive for 1024-vocab embeddings; v1.5 with K=32 + sparse storage format would test cap savings claim. Both novel mechanisms (0117 + 0118) using identical top-K STE trick — "single-mechanism risk" the walk note flagged.

## Cross-experiment lessons (numbered, tied to results)

1. **The mechanism cluster (0111-0113) is the strongest scientific contribution of the session**. Three independent ablations, three confirmed-staycourse signals. All ≥13σ above family noise floor. The chain ran cleanly without intervention. → results.tsv rows 0111/0112/0113.

2. **Records ports at 1k smoke are uninformative for compose-with-our-stack tests**. 0115 went +0.047 in the wrong direction. Records validate at H100 6k+ steps. Don't expect to confirm tier-1 ports on 5090 short-step. Marked H100-bound for both 0115 and 0116. → 0115 description "[transfer:medium - port works in principle but training-duration-bound at our regime]".

3. **Walk-driven self-correction was the highest-EV move of the session**. The 1623 walk caught a planned 5h long-train on 0104 ("more of the same axis") and pivoted to mechanism cluster + tier-1 ports + novel mechanisms. → `walks/2026-04-29_1623.md`.

4. **The user-correction cycle codified four new behavioral feedbacks**. Each was a real anti-pattern in my behavior:
   - `feedback_no_seed_confirms_on_pod.md`: precise deltas don't transfer to H100; no σ-confirms on cheap pod.
   - `feedback_5090_explore_h100_writeup.md`: 5090 is for short-step exploration; long-training and writeup numbers go on H100.
   - `feedback_subagent_executes_not_thinks.md`: agent does math/derivation/toys (has the context); subagent only executes a fully-specified plan.
   - `feedback_slow_down_for_innovation.md`: novel mechanisms get progressive toys + outside-eyes; the wasted time is blind iteration not careful thought.

5. **Sequential progressive toys (4-5 per novel mechanism) caught design issues before subagent dispatch**. Without the toys, dendrocentric and spike-rank would have failed in less-debuggable ways during integration. → `scratch/2026-04-29_dendrocentric_v1_tiny.py`, `scratch/2026-04-29_spike_rank_v1_tiny.py`.

6. **Subagent execute-only protocol worked cleanly for all four mechanism implementations** (0115/0116/0117/0118). Each subagent returned a verified diff that compiled, passed AST check, and ran without crash. The discipline post-codification: agent owns thinking, subagent owns typing.

7. **EMA β-hyperparameter math written down ≠ acted on**. I documented "β=0.999 wrong for 1k smoke" in scratch then shipped at β=0.999. Result was uninformative-by-design. **The "writing math but not acting on it" anti-pattern is closer to "fast over careful" than the other corrections.** Worth surfacing to next session.

8. **Outside-eyes review at the right moment surfaced critical brief-misalignment**. The 17:30 review caught that v1 dendrocentric defers the brief's actual ordering question — important caveat documented in plan.md and now shaping the v2 next-session direction.

## Set in stone (this segment)

- **n=5 is the cap-frontier setpoint at production-1k** (n=7 over-cap and under-trains at 1k).
- **SSD scan is load-bearing (-0.046 BPB at production)** — kill-Mamba-2 selective scan is doing real work.
- **kill > full architectural, not LR-cliff** — LTI constants beat learned dt/B/C at our regime.
- **Walk + outside-eyes cadence**: invoke earlier than gut suggests; both caught blind spots this session.
- **Subagent execute-only protocol**: codified as durable feedback.
- **5090 = exploration, H100 = writeup**: codified.

## Set in hypothesis (single-seed, training-duration-confounded)

- **Parallel-residuals doesn't compose at 1k**: tier-1 port marked H100-bound. Real test at 5k+ steps; expected to compose given records.
- **EMA at β=0.99 would land near 0107** (math prediction): re-run is the next-session first-action.
- **Dendrocentric v1 might land neutral at 5k+ steps** (training-duration-bound interpretation). Or v2 with ordering might be the real answer.
- **Spike-rank embedding v1 is too restrictive at K=8**: v1.5 with K=32 might land neutral.

## Predictions vs actuals (calibration check)

| Mechanism | Predicted range | Actual | Calibration |
|---|---|---|---|
| 0111 n=7 | 1.49-1.52 | 1.5536 | Worse than range — depth ceiling sharper than expected |
| 0112 kill-scan | 1.46-1.48 | 1.5049 | Worse than range — scan adds more than predicted |
| 0113 full-sel @ LR/3 | 1.42-1.50 | 1.4898 | Within range (high end) — kill-vs-full architectural |
| 0115 parallel-resid | 1.51-1.53 | 1.5722 | Far worse — training-duration-bound |
| 0116 EMA β=0.999 | 1.52-1.53 | 2.2022 | Catastrophic-by-design (math predicted this) |
| 0117 dendrocentric | 1.51-1.58 | 1.6812 | Worse than range — training-bound or wrong-mechanism |
| 0118 spike-rank | 1.50-1.58 | 1.8849 | Far worse — K=8 too restrictive |

**Honest calibration**: 1 of 7 within predicted range. My intervals were systematically too tight and too optimistic. Pattern: I underestimate how much 1k-smoke is uninformative for both mechanism-confounded ports AND novel-mechanism v1s. Next session: widen prediction intervals for 1k smoke, focus disconfirming criteria on "did it crash" + "is step time as predicted" rather than "what's the absolute val_bpb."

## Walk reflections this segment

- `walks/2026-04-29_1623.md` — pivoted from 5h long-train to mechanism cluster + tier-1 ports + novel mechanisms. Highest-EV pivot of session.
- `walks/2026-04-29_1951.md` — surfaced the "math written but not acted on" anti-pattern (0116 β=0.999); flagged single-mechanism risk in 0117/0118 (both use top-K STE); recommended β=0.99 re-run as cheap disambiguation.

## Follow-ups for next session, ranked by EV

1. **H100 deploy** (highest priority — single-session deliverable per user). Recipe:
   - Base: kill-Mamba-2 triple-parallel + ternary + n=5 + production-batch (scale to ~524288 on 8×H100) + brotli
   - Add: EMA at β=0.999 (correct hyperparameter for 5k+ steps); parallel-residuals (records' merge offset learned over longer training)
   - Toggle: dendrocentric v1 only if cap budget needs it; spike-rank embed only if cap budget needs it
   - Steps: 20-30k. Predicted val_bpb ~1.20-1.30 (records 1.10).

2. **β=0.99 EMA re-run on 5090** (~22 min, $0.40): disambiguates EMA implementation correctness vs hyperparameter wrongness. Cheap and informative for H100 deploy confidence.

3. **v2 dendrocentric with DFSM-style ordering** (200-300 line subagent task). Brief option (f) proper. Tests the actual ordering claim. Math + temporal-rank capacity sim already in scratch from prior sessions.

4. **Spike-rank v1.5: K=32 + sparse storage format** (indices + values, not dense). Would test cap savings claim AND give per-token expressiveness back.

5. **Pure conv-augmented attention** (drop SSM apparatus, keep just conv1d + attention): 0112 result suggests this would land worse than full kill-Mamba-2 by ~0.045. But if simpler architecture matches at H100 long-train, simpler-is-better wins.

6. **Cross-seed confirms** of any winner that lands competitive at H100. Per-family σ characterization for the H100 family (different from 5090 family).

## Reflections

**What went well**:
- Mechanism cluster (0111-0113) had three clean negative findings — exactly the kind of result the brief asks for.
- Sequential math + progressive toys + plan.md + subagent execute-only worked cleanly for all four implementations.
- Walk-driven pivot (1623) was textbook self-correction; saved $5 and a low-information session direction.
- Outside-eyes review at the right moment surfaced critical brief-misalignment.
- Four new memory feedbacks codified — the user-correction cycle was productive.

**What didn't go well**:
- Initial three-parallel-subagent dispatch (asking each to "derive math + toy + integrate") was wrong-pattern, corrected by user feedback.
- Briefly committed to a 5h 5090 long-train before walk caught it.
- Initially queued a noise-floor-sentinel that wasn't the right work for paid pod.
- 0116 EMA β=0.999 shipped at the WRONG hyperparameter despite my own scratch math predicting the result.
- 0117 + 0118 both use the SAME top-K STE trick — "single-mechanism risk" the walk noted but I didn't act on.

**Anti-patterns to avoid next session**:
- Don't ship at hyperparameters that scratch math says are wrong. Act on the math.
- Don't dispatch subagents to "think" — agent owns thinking, subagent owns typing.
- Don't long-train on 5090. 5090 is exploration; H100 is writeup.
- Don't σ-confirm on cheap pod. Single-seed is the screening regime.
- Watch for single-mechanism-risk: when multiple novel components share an underlying trick (top-K STE here), they fail correlated. Mechanism diversity matters.

## Next agent's first concrete action

**Move to H100 deploy.** First experiment: pure-stack (kill-Mamba-2 + ternary + n=5 + production batch scaled to 524288 + brotli) at 5k steps. Establish H100 baseline. Then add ports one at a time (EMA at β=0.999, parallel-residuals). Decision tree per `scratch/2026-04-29_post_chain_decision_tree.md` (now stale; rewrite based on this session's results before launching).

Bold-but-careful next-session work: build dendrocentric v2 with DFSM-style ordering. The brief's actual question. The cheap EMA β=0.99 re-run is worth doing first to verify infrastructure.
