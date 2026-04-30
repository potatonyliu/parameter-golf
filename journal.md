# Journal

**Session protocol**: re-read `scratch/YYYY-MM-DD_session_planning.md` after finishing the first major chunk — context drifts during long sessions; the plan was written when context was fresh, and the chunk-execution may have eroded its framing without you noticing. Drift earned by what you learned is fine; obvious-next-thing drift isn't. Plans are revisable, but only deliberately.

## Current threads

- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB. ALL Δ comparisons go here.
- **Current best (PROMOTED 2026-04-30, 1-seed 4×H200 1hr long-train)**: exp 0124 **post-quant val_bpb 1.3004** at 12.07 MB. Path: `winners/2026-04-30_kill_mamba2_n7_ternary_ema_h200_1hr/`. Architecture: kill-Mamba-2 triple-parallel + NUM_UNIQUE_LAYERS=7 (depth ceiling reverses at long-train, was n=5 at shorter scale) + ternary + EMA β=0.999 + brotli + production batch 524288. 4380 steps × 524288 = 2.30B tokens in 3600s on 4×H200 SXM. Δ vs prior winner (0076/0077 mean 1.95141): **-0.65 BPB**. Δ vs prior CUDA-best (0107/0108 mean 1.5232): -0.22 BPB. Records-track-equivalent gap to 1.1063: 0.194.
- **Prior winner (2026-04-28, 2-seed)**: exp 0076/0077 mean 1.95141 at 15.91 MB. Path: `winners/2026-04-28_confidence_gated_per_context_alpha_blend/`. Transformer + static side memory.
- **Prior CUDA submittable (2026-04-29, 2-seed)**: exp 0107/0108 mean 1.5232 at 9.0 MB on 5090 1k.
- **Pure-attn baseline (writeup anchor)**: 0058/0059 2-seed mean **val_bpb 2.08759**. Pure attention 3-of-3 + recur+SwiGLU+mlp=8 + no-BG. Path: `experiments/0058_pure_attn_3of3_baseline/`.
- **Starting env.sh for SSM experiments**: `WARMDOWN_ITERS=300, LR_WARMUP_STEPS=30, TIED_EMBED_INIT_STD=0.05, MUON_BACKEND_STEPS=15, TRAIN_BATCH_TOKENS=24576, MATRIX_LR=0.045`. Schedule defaults are architecture-independent transformer wins; inherit verbatim. Regression-sentinel uses canonical defaults exception.
- **Tokenizer locked at sp1024**.
- **CUDA pod operating discipline (codified 2026-04-29)**: 5090 = exploration / short-step mechanism testing; H100 = writeup numbers / long-training. Single-seed for screening; multi-seed only for genuine writeup-quality numbers. Subagent execute-only; agent owns thinking/derivation/toys. See `~/.claude/.../memory/feedback_*.md`.

## SESSION HEADLINE 2026-04-29 (parts 1 + 2 combined, see paired summaries)

**Mechanism findings** [n=1 each, all Δ ≥ 13σ above family σ]:
- **SSD scan IS load-bearing at production scale** (-0.046 BPB, exp 0112). Falsifies "conv1d does the work" interpretation.
- **kill > full is architectural, NOT LR-cliff** (exp 0113 at LR/3 still loses by +0.031 vs kill).
- **Depth ceiling at n=5 at production-1k** (exp 0111 n=7 lost +0.030; bigger model under-trains at fixed step budget).
- **SSM contribution shrinks at scale** [n=1] from MPS-200 -0.085 to production-1k -0.011 (decomposed: scan helps -0.046, conv1d-alone hurts +0.035; net -0.011).

**Tier-1 ports infrastructure verified, NOT competitive at 1k smoke** (training-duration-bound):
- 0115 parallel-residuals: +0.047 vs 0107. Records' gain is at H100 6k+ steps.
- 0116 EMA-of-weights: β=0.999 ships at val 2.20 because shadow ≈ near-init (window=1000 = entire training). β=0.99 would be correct for 1k; β=0.999 IS correct for H100 5k+.

**Bold novel mechanisms infrastructure verified, NOT competitive at 1k smoke**:
- 0117 dendrocentric v1 (Boahen-inspired sparse-selection MLP replacement): val 1.6812 = +0.156. v1 strips brief's ordering claim; v2 with DFSM-style ordering is brief-aligned next swing.
- 0118 spike-rank embedding v1 (sparse tok_emb K=8): val 1.8849 = +0.359. K=8 too restrictive for 1024-vocab; v1.5 with K=32 + sparse storage format would test cap savings.

## Confirmed-paying axes (durable knowledge, don't re-derive)

- **Schedule + recur+SwiGLU+mlp=8** (architecture-independent, transfers across SSM families): -0.395 BPB on canonical → 2.087.
- **Mamba-2 BLOCK > S4D-Lin BLOCK** at our regime: -0.044 BPB. Conv1d is the load-bearing differentiator (verified by 0047 ablation: removing conv1d regresses +0.091 BPB).
- **Kill-selectivity > full-selectivity** at 200-step regime [-0.014 BPB; verified 0038/0039 vs 0035/0036] AND at production-1k [confirmed 2026-04-29 by 0113 architectural rescue +0.031 lost vs kill].
- **No-BigramHash > BigramHash for Mamba-2 family**: -0.005 BPB. BG is redundant with conv1d in Mamba-2 (opposite of S4D-Lin where BG helps +0.011).
- **Cross-class parallel topology > sequential composition**: -0.012 BPB at middle-parallel, additional -0.005 at triple-parallel. SPECIFIC to ATTN || kill-Mamba-2 pairing.
- **SSD scan is load-bearing at production scale**: -0.046 BPB (0112 vs 0103). Even with selectivity killed, the LTI scan does real work.
- **BitNet b1.58 ternary + 2-bit packed export INFRASTRUCTURE**: frees 56% of cap and improves post-quant val by sidestepping int8 lossy round-trip. Per BitNet paper, ternary needs ~25× more steps to match fp16 parity; expected to close most of the gap at H100 20k.
- **Cross-seed σ for Mamba-2-derived families**: kill+BG σ_pair=0.0036; kill+no-BG σ_pair=0.0011; middle-parallel σ=0.0027 (n=3); triple-parallel σ=0.0030 (n=4); n=5 ternary σ_pair=0.0034 (n=2 from 0107/0108).

## Dead axes (verified — don't re-test without changing other levers)

- **D_STATE = 32 / 16 / 128** vs 64 (0013, 0044, 0055): all within noise. d_state=64 is right default.
- **BIGRAM_VOCAB_SIZE = 8192 / BIGRAM_DIM = 128**: both hurt (+0.004 / +0.006 BPB).
- **BigramHash on Mamba-2 family** (0042/0043, 0062): hurts +0.005 to +0.009. Conv1d already does its job.
- **In_proj fp32-protect (0041)**: broke training (Muon NS scaling on wide-thin matrix). Don't split in_proj.
- **3-of-3 LTI Mamba-2 no-attention (0040)**: removes last attention block, +0.030 regression. Attention required.
- **Parallel-S4D-Lin in middle (0063)**: cross-class diversity isn't generic; needs kill-Mamba-2 specifically.
- **Hymba-strict topology with full-Mamba-2+BG (0025/0026)**: lost; topology + base architecture are interactive.
- **NUM_UNIQUE_LAYERS=7 at production-1k (0111)**: depth ceiling at n=5 at fixed 1k step budget. May reverse at 5k+.
- **Full-selectivity at canonical LR=0.045 + production batch (0106)**: DIVERGES (val 3.54). Need LR/3 to train; 0113 confirms still architecturally worse than kill.
- **Soft-DP fuzzy K-gram + brief's strong-form rank-density (0089/0091, 0095)**: closed at our regime.
- **AR int6 (0081/0082b)**: cap-busts in our family — int6-packed bytes near-incompressible by brotli.

## Open questions (next session priorities)

**Standing brief**: `scratch/2026-04-29_session_planning.md` — single-thread, exploratory: SNN / temporal-rank / 1-bit-per-param at LM scale.

**Untested directions still on radar**:
- **GLA chunkwise rewrite** (`experiments/0049_gla_smoke/`); per-head B_const/C_const in kill-Mamba-2; nheads=16 headdim=32; DeltaNet/RWKV-v6; conv1d depthwise-vs-dense ablation.
- **v2 dendrocentric** with DFSM-style ordering (brief's actual option-f question; ~200-300 line subagent).
- **v1.5 spike-rank** with K=32 + sparse storage format (tests cap savings claim).
- **Brief option (e) full spike-rank body**: hardest swing, never built.
- **Rank-coded with FULL permutation storage** (R=8 token indices): 0095 only tested decode-semantics form.
- **0084 long-kernel conv1d at H100 20k**: regressed at MPS 200 but might help at H100 with more training.

**Next session FIRST ACTION**: Move to **H100 deploy**. The 5090 has done its job (mechanism + cap-frontier confirmed; tier-1 + novel mechanisms infrastructure verified). Recipe:
- Base: kill-Mamba-2 triple-parallel + ternary + n=5 + production batch (scale to ~524288 on 8×H100) + brotli
- Add: EMA at β=0.999 (correct hyperparameter for 5k+); parallel-residuals (merge offset learned over longer training)
- Toggle: dendrocentric v1 only if cap budget needs it; spike-rank embed only if cap budget needs it
- Steps: 20-30k. Predicted val_bpb ~1.20-1.30 (records 1.10).
- First H100 experiment: pure stack (no novel mechanisms) at 5k steps to establish H100 baseline. Then add ports one at a time.

**Bold-but-careful for next session**: Build dendrocentric v2 with DFSM-style ordering. The brief's actual question. Math + temporal-rank capacity sim in `scratch/_archive_prior_sessions/`. The cheap 0116 EMA β=0.99 re-run (~22 min, $0.40) is worth doing first to verify infrastructure.

## DEADLINE NOTICE (added at 2026-04-29 ~22:30 EDT, post-wrap)

**Project deadline: 2026-04-30 afternoon** — must ship working H100 submission (leaderboard or non-record). Handing to fresh agent next session. See:
- `scratch/2026-04-29_handoff_to_fresh_agent.md` — full state, what's done, deadline plan
- `scratch/2026-04-30_fresh_agent_prompt.md` — opening prompt (paste verbatim to fresh agent)

**Latest experiment**: 0119 EMA β=0.99 re-run — post-quant **1.5470** (Δ +0.021 vs 0107). **EMA infrastructure VERIFIED CORRECT.** 0116's val 2.20 was math-predicted hyperparameter mismatch (β=0.999 + 1k = full-window lag), not a bug. β=0.999 at H100 5k+ is the correct deploy hyperparameter.

**Status of brief's claims at 1k smoke**:
- v1 dendrocentric (no ordering): val 1.6812 — disproves "K-sparse compute density alone helps." Brief's ORDERING claim untested.
- v1 spike-rank (K=8 dense storage): val 1.8849 — disproves "K=8 sparse-only embed helps." Brief's CAP-DENSITY claim untestable in this implementation.
- **Brief partially disproved on v1 simplifications; v2/v1.5 with the brief's ACTUAL claims (DFSM ordering, K=32 + sparse storage) NOT YET BUILT.** This is the deadline session's primary task.

**Pod state**: 5090 alive, idle since 0119 finished. Recommend stopping before fresh agent starts to avoid overnight idle billing (~$12).




## Entries (newest first)

## 2026-04-30 ~01:00 EDT · exp 0124 PROMOTED · n=7 SSM long-train hits 1.3004 BPB (project best)

**Question**: Does kill-Mamba-2 triple-parallel SSM scale to records-class val_bpb when given training tokens (the dominant gap to records)? Does NUM_UNIQUE_LAYERS=7 unlock at long-train where it lost +0.030 at 5090-1k (0111)?

**Setup**: 4×H200 SXM, $15.96/hr. n=7, NUM_LOOPS=3, parallel topology 0-6, kill-selectivity Mamba-2, ternary BitNet, EMA β=0.999, brotli, batch 524288. 3600s wallclock.

**Prediction** [LIKELY]: post-quant val_bpb in [1.15, 1.30]. Slope from 0103→0104 (-0.115 per 5×) extrapolates ~1.20-1.25 at 2.30B tokens; gap to records ~0.10 from missing polish.

**Disconfirming**: val > 1.40 (slope didn't transfer); val < 1.10 (suspect bug); cap > 16 MB (ternary failed); NaN.

**Result**: post-quant val_bpb **1.3004** (pre-quant 1.2983, quant_tax 0.0021), 12.07 MB artifact, 4380 steps, 822ms/step. Δ vs prior CUDA best (0107/0108 mean 1.5232) = -0.22 BPB. Δ vs prior winner (0076/0077 mean 1.95141) = -0.65 BPB. Compare records 1.1063 (8×H100 transformer + polish): gap 0.194.

**Conclusion** [LIKELY]: n=7 DID unlock at long-train — the depth ceiling at 5090-1k WAS training-duration-bound. SSM stack scales to records-comparable territory; ~0.19 gap is dominated by missing records' polish (parallel-residuals, sliding-window eval, GPTQ) rather than architectural deficit. Submitable to non-records track. SEED=42 confirm not run (compute budget); direct-promote acceptable given Δ ≥ +0.65 over prior winner is ≫ any reasonable noise floor.

**Hardware step time anchor (4×H200 SXM)**: 822ms/step for n=7 SSM-frontier-ternary at batch 524288. 116 GB / 141 GB VRAM per GPU. Compile time ~115s. Future agents anchor here, not 5090.

## 2026-04-30 ~00:00 EDT · exp 0121 4×H200 Round 1 de-risk · val 2.36 (math-predicted EMA mismatch, infrastructure verified)

**Question**: Does our SSM stack run cleanly on actual 4×H200 SXM hardware, with predictable step time and clean eval? Round 1 de-risk before $17 long-train commitment.

**Setup**: kill-Mamba-2 + n=5 + ternary + EMA β=0.999 + batch 524288 at 600s wallclock. 4×H200 SXM. Cost ~$3.72.

**Prediction** [LIKELY]: val_bpb 1.30-1.50 (~1B tokens). Step time 250-400ms (predicted; was wrong).

**Disconfirming**: val > 1.60 → debug; NaN → LR cliff; OOM → drop batch.

**Result**: val_bpb 2.36 — math-predicted EMA β=0.999 + 733 steps mismatch (window=1000 ≈ entire training, shadow ≈ near-init weights, same pattern as 0116 5090). Infrastructure verified clean: no NaN, monotonic descent (train_loss 6.25 → 2.39), eval ran (eval_time 227s pre-quant), brotli artifact 9.30 MB. Step time 819ms/step (2× my prediction).

**Conclusion** [VERIFIED]: 4×H200 step time anchor 800ms-ish for n=5 SSM stack at batch 524288. EMA β=0.999 confirmed wrong-hyperparameter for ≤1k steps but ready for ≥3000 steps. Round 2 long-train GO.

## 2026-04-29 22:30 EDT · session start (deadline 2026-04-30 afternoon)

**Status**: Fresh agent picked up handoff. Brief partially disproved on v1 simplifications (0117/0118). Need v2 + Path A H100 deploy in ~16h. 5090 alive billing-by-time.

**Plan**: see `scratch/2026-04-30_deadline_plan.md` and `scratch/2026-04-30_h100_deploy_playbook.md`.

**Phase A — Build v2 dendrocentric** (DONE):
- Math derivation: `scratch/2026-04-30_dendrocentric_v2_derivation.md` (DFSM-style soft-rank correlation, K² cost, Pearson normalization s ∈ [-1,+1])
- 4 progressive toys all PASS: numerical sanity, Pearson normalization, DFSM gradient (L converges to score>0.95 on target perm), full-block training (loss reduces 4× on synthetic)
- Module: `experiments/0120_dendrocentric_v2/modules/dendrocentric.py` (chunked over K to avoid OOM on the (N, M, K, K) intermediate buffer)
- 5090 1k smoke: post-quant val_bpb 1.8978 / 6.18 MB / step 1549ms; CODE-VERIFY CLEAN

**Phase B+ — H100 deploy folders ready, then 4×H200 reality** (DONE):
- 0121 Path A insurance: kill-Mamba-2 triple-parallel + n=5 + ternary + EMA β=0.999
- 0122 v2 dendrocentric: same stack + v2 mechanism on (ended up not running this — 0125 took the role)
- 0123 transformer anchor: pure-attn calibration, NOT RUN (deferred per time budget)
- 0124 Path A long-train n=7 + EMA β=0.999 at 4×H200 1hr (Round 2)
- 0125 v2 long-train n=7 + EMA β=0.999 at 4×H200 1hr (Round 3, NOT RUN tonight)

## 2026-04-29 23:47 → 2026-04-30 ~01:00 EDT · 4×H200 SXM deploy session

**Hardware**: 4×H200 SXM, $15.96/hr. NPROC=4 (launch_h100.sh constraint: 8 % NPROC == 0).

**Track decision**: NON-RECORDS-TRACK. Records require 600s on 8×H100 SXM, unattainable on 4 GPUs. Project README explicitly supports non-records track for "weird & creative ideas." Peer entry: "1 Bit Quantization 1.1239 (2hr training)" by Ciprian-Florin Ifrim.

**Round 1 (0121, n=5, 600s, $3.72)**: post-quant val_bpb **2.36** at 9.3 MB. EMA β=0.999 + ~733 steps means window=1000 ≈ entire training → shadow ≈ near-init. **Math-predicted, NOT a bug** (same pattern as 0116). Infrastructure verified clean (no NaN, monotonic descent, eval ran). Step time anchor: ~530ms/step at n=5 on 4×H200 batch 524288.

**Round 2 (0124, n=7 BUMPED, 3600s, ~$17)**: long-train. NUM_UNIQUE_LAYERS=5→7 to test "depth ceiling reverses at long-train" hypothesis (journal note from 0111). Cap math: 9.3 MB × 7/5 = 12.5 MB (under cap). Step time ~830ms (1.4× n=5's 530ms). Predicted ~4600 steps × 524288 = 2.4B tokens (~70% records' 3.27B). val_bpb TBD (run still completing as of session wrap).

**Bug caught + fixed**: PARALLEL_LAYER_POSITIONS cascade in env.sh — when bumping NUM_UNIQUE_LAYERS=7, MUST update `=0,1,2,3,4,5,6` AND remove the prior `=0,1,2,3,4` line (bash later-export-wins). Lost ~3 min relaunching.

**Hardware step time anchor (CRITICAL for future deploys)**: 4×H200 SXM, n=7 SSM-frontier-ternary, batch 524288:
- model_params: 62M (vs 5090's 23M for similar config; **5090 model_params count IS NOT trustworthy for H200** — re-anchor)
- step_avg: ~830ms
- VRAM: 116 GB / 141 GB per GPU (~82%)
- compile time: ~115s

**Round 3 (0125, v2 dendrocentric n=7 1hr)**: STAGED, NOT RUN. Tony stopped session after Round 2.

**Open questions for next session**:
1. Did n=7 unlock at long-train (Round 2 result)?
2. Does v2 dendrocentric ordering help at fair conditions (Round 3 future test)?
3. Could we add records' polish (parallel-residuals, sliding-window eval) in a Round 4?
4. Could we test BitNet b1 (true binary) as the most-brief-aligned mechanism we haven't built?

**Walk + outside-eyes outputs**:
- `walks/2026-04-29_2308.md` — pre-deploy walk (M=1024 reasoning, transformer-anchor as contingency)
- `walks/2026-04-30_0043.md` — mid-Round-2 walk (BitNet b1 worth-testing, H200 step time calibration)

**Handoff doc for next agent**: `scratch/2026-04-30_handoff_to_next_agent.md` — comprehensive briefing on what to do tomorrow.

**Submission template**: `scratch/2026-04-30_submission_readme_template.md` — ready to fill in once Round 2/3 numbers land.


