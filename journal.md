# Journal

**Session protocol**: re-read `scratch/YYYY-MM-DD_session_planning.md` after finishing the first major chunk — context drifts during long sessions; the plan was written when context was fresh, and the chunk-execution may have eroded its framing without you noticing. Drift earned by what you learned is fine; obvious-next-thing drift isn't. Plans are revisable, but only deliberately.

## Current threads

- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB. ALL Δ comparisons go here.
- **Current best (PROMOTED 2026-04-28, 2-seed)**: exp 0076/0077 **2-seed mean val_bpb 1.95141** (cross-seed σ_pair=0.0061). Path: `winners/2026-04-28_confidence_gated_per_context_alpha_blend/`. Architecture: combined K=3+K=4 static side memory + per-context α blend + model-confidence gate. Artifact 15.91 MB (88 KB safety under cap).
- **Best CUDA-regime submittable (2026-04-29, 2-seed, NOT promoted)**: exp 0107/0108 **2-seed mean val_bpb 1.5232** at 9.0 MB. Architecture: kill-Mamba-2 triple-parallel + brotli + ternary + NUM_UNIQUE_LAYERS=5 + production batch 131072 + canonical LR + 1000 steps. NOT promoted because SSM-family family σ characterization deferred per user feedback "no σ-confirms on cheap pod, precise deltas don't transfer to H100." See `summaries/2026-04-29_cuda-ssm-production-batch.md`.
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




## Entries (newest first)

