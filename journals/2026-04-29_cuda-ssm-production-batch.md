# Journal · 2026-04-29 · cuda-ssm-production-batch

Rotated from journal.md on 2026-04-29 15:22 EDT.

## Entries (newest first)

## 2026-04-29 · session opening: CUDA pod 5090, MPS→CUDA SSM transfer, workflow patches

**Setup**: First session on a fresh RTX 5090 (32 GiB) RunPod pod after the e2e workflow validation (commits a72ac8b/255e51a). 7-hour budget. CUDA anchor 0002_regression_check_cuda_v2 = val_bpb 2.5115 (vs MPS anchor 2.5212; bf16-reduction-order drift 0.0097, transformer-only baseline). Pod billing ~$1/hr.

**Workflow patches landed before research**:
- `Host runpod` in `~/.ssh/config` proxies through ssh.runpod.io with `RequestTTY yes` → fails from agent Bash with `Error: Your SSH client doesn't support PTY`. **Use `Host runpod-tcp` (direct TCP)** for every agent SSH call. Documented inline in `runpod-launch` SKILL.md; bit me on first command.
- The pod's git remote is named `origin` (setup_pod.sh's default), not `fork` like Mac. `git push fork ...` from pod fails with "fork does not appear to be a git repository". Use `git push origin autoresearch-ssm` on the pod.
- `await_steps.sh` log_mtime helper had BSD-stat-first ordering; on Linux `stat -f %m` writes filesystem-info to stdout (not redirected) and poisons the arithmetic with "File: unbound variable". Fixed: try GNU `stat -c %Y` first.

**0098_ssm_frontier_cuda_200 [n=1, sentinel]**: forked 0064 (= 0051 SSM frontier triple-parallel kill-Mamba-2 + brotli; MPS val 2.0030) → CUDA val_bpb_post_quant **2.0028** (Δ −0.0002 vs MPS, clean transfer). step_avg 692ms = ~12.5× MPS speedup. **CUDA per-step rate stabilizes at ~205ms** once warmup compile (~100s) is amortized — the 692ms reported is `(98s warmup + 199 × 205ms) / 200`. Use 205ms not 692ms for budget estimates. Step1 train_loss = 20.6 (>>ln(vocab)=6.93) is **expected for the recur+SwiGLU+mlp=8 + TIED_EMBED_INIT_STD=0.05 stack family** — the larger embed init produces peaked-on-wrong-tokens initial logits; model recovers below ln(vocab) by step 10 and converges identically to MPS. The "step1 ≈ ln(vocab)" rule from program.md is for the canonical config; this stack family has its own init regime. Per-step on the SSM stack is ~3× the transformer-only sentinel's 134ms — Mamba-2 conv1d + recurrence are compute-heavier per step but still trivial wall time.

**Implication for time budget**: 200 CUDA steps = ~3 min wall; 2000 CUDA steps = ~9 min training + (variable) eval. **Trigram-side-memory build adds ~10 min of CPU time at quant export** (it's 100M-token K-gram counting), so any experiment inheriting `TRIGRAM_SIDE_MEMORY=1` from 0076 ancestry pays this independent of training duration. For pure-pre-quant comparisons (BitNet penalty test, dendritic convergence test) the trigram build is irrelevant; consider TRIGRAM_SIDE_MEMORY=0 for screening.

**CUDA dynamo gotcha cascade — full picture after 4 fix attempts on 0100**. `torch.compile(fullgraph=True)` on CUDA (line 1738 of train_gpt.py) is incompatible with several MPS-development patterns:

1. `Tensor.item()` inside compiled forward → `Unsupported Tensor.item() call with capture_scalar_outputs=False`. Advertised env var `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` is **unreliable** (didn't take effect in 0100 even when set in env.sh). Reliable fix: remove the .item() call if its guarded path is unreachable in production.
2. Boolean indexing (`tensor[bool_mask]`) → `Dynamic shape operator: aten.nonzero.default's output shape depends on input Tensor data`. Re-write to use `torch.where(mask, x, zero)` masking which preserves shape, OR disable compile for the affected module.
3. `@torch.compiler.disable` on a sub-module: REJECTED because the parent compile is `fullgraph=True` (no graph breaks allowed). The decorator only works if the parent doesn't require fullgraph.
4. Net result: a module using boolean indexing on a fullgraph=True parent has only ONE clean fix — **disable torch.compile for the entire experiment** (set `compiled_model = base_model` unconditionally). ~10-20% slower step time (no kernel fusion), but the experiment runs.

Confirmed sites:
- `modules/trigram_side_memory.py:634, 487` (`.item()`) — fires at post-quant eval. Workaround for screening: `TRIGRAM_SIDE_MEMORY=0`. For full-stack promote runs: surgical .item() removal.
- `modules/dendritic_memory.py:196, 225, 232` (`.item()` and boolean indexing). Workaround: `compiled_model = base_model` in train_gpt.py for that experiment.

**The lesson generalizes**: any new SSM/side-memory module developed for MPS must be tested under `torch.compile(fullgraph=True, dynamic=False)` BEFORE drawing CUDA conclusions. Simplest local check: import the module, wrap a small synthetic forward with `torch.compile(fullgraph=True)`, run on a tiny tensor on CPU (or pull a 1-step CUDA test if pod is up). Catches the bug at design time, not 4 launch retries deep on the pod.

**0100 (n=1) status: training in progress with compile disabled**, expected ~30 min wall (~560ms/step uncompiled vs ~250ms compiled).

**Direction for the session**: outside-eyes review of the initial SSM-mechanism plan (kill-vs-full at extended training) flagged it as re-litigating already 4-seed-sentinel'd findings. Pivoted to the previous session's UNSETTLED interpretations: BitNet 25× claim (0099 ternary at 2k CUDA steps, parent 0093) and training-duration-ceiling claim for learnable side-content (0100 dendritic at 2k, parent 0092). 0101 = no-ternary no-dendritic 2k baseline as decomposition comparator. Each tests one falsifiable claim per ~20-min run.

## 2026-04-29 · 0099/0100/0101 results @ 2k CUDA steps batch 24576

**Observed [n=1 each, pre-quant unless noted]**:
- 0099 (transformer + ternary + LR×3 + side mem): pre-quant 1.6499. Post-quant FAILED (torch.compile dynamo bug at trigram_side_memory.py:634 .item() during quantized eval).
- 0100 (transformer + ternary + dendritic, default LR): post-quant 1.6668, quant_tax 0.001, artifact 5.28 MB.
- 0101 (transformer alone, default LR): post-quant 1.5793 first run / 1.578 second run. **Cap-busts at 22.13 MB** because no-ternary stores body weights as 4× int8 (8 bits/param) instead of packed-ternary (2 bits/param).

**Decomposition (single-seed, pre-quant unless noted)**:
- BitNet ternary penalty at CUDA 2k: `0099 - 0101 = +0.076 BPB`. MPS-200-step penalty was ~+0.10. Closure: only 0.024 BPB in 10× more training. **Slow closure.** Per BitNet's 25× claim at 700M params, our 27M-param SP1024 regime closes the gap much more slowly; at H100 20k steps would still leave ~+0.05 BPB ternary penalty.
- Dendritic at default LR + ternary: `0100 - 0099 = +0.016 BPB`. Confounded (LR×3 missing). Likely dendritic ≈ neutral, consistent with 0094 MPS finding [LIKELY at single seed].
- **Reframe of "ternary penalty"**: ternary costs +0.076 BPB but BUYS ~14 MB of cap headroom (artifact 7.85 vs 22.13 MB). The headroom is spendable on other stack additions. The strict "penalty" framing misses the cap-cost trade.

**Conclusion** [LIKELY n=1]: ternary direction is COMPLEMENTARY to the cap budget, not a free win on val_bpb at our scale. Worth keeping as the cap-saving primitive, but the 1.10 record is fp16 (not ternary) — meaning the record extracts the 14 MB headroom *without* paying the ternary penalty. That's a hint: at 8×H100 with full standard stack, the ternary trade may not pay off.

## 2026-04-29 · pivot post-decomposition + record-recipe analysis

After the chain-restart issue (initial 0102/0103 OOM at canonical batch 524288 + dynamo whack-a-mole), reset the queue:
- 0102/0103: SSM stack at production batch (131072 — fits 5090; 5.3× MPS), 1000 steps. SSM × ternary compound test + clean SSM-only baseline.
- **0104**: SSM frontier @ 5000 steps batch 131072 = 655M tokens trained = **1.5× the H100 record's per-GPU training data**. Single 5090, ~50-67 min wallclock, ~$1 of pod time. The "where can SSM frontier land?" extrapolation point.

**Record-recipe gap analysis** (`scratch/2026-04-29_record_recipe_analysis.md`): the SP1024 1.1063 record uses the standard stack (parallel residuals attn/MLP, mixed quant int6/int8, BigramHash, RECUR_LAYERS+REPEAT_UNTIE_MLP, sliding-window eval, EMA, AR self-gen GPTQ) on top of an 11-layer transformer. We have brotli + the SSM frontier topology but NONE of the other ports. Per 2026-04-28 journal, those ports are already enumerated as Tier-1 work — **journals say BigramHash hurts kill-Mamba-2 (+0.005-0.009)** so we don't need that one, but parallel residuals / EMA / sliding-window eval / mini-DR / REPEAT_UNTIE_MLP do apply. Tonight's deliverable is "characterized SSM frontier + clear path to records," not "match 1.10".

**Decision tree for 0104** (also in `scratch/2026-04-29_record_recipe_analysis.md`):
- val ≤ 1.30: scaling works; run multi-seed long version + Tier-1 ports.
- val ∈ [1.30, 1.45]: slow scaling; port one missing standard-stack piece (parallel residuals first).
- val ≥ 1.45: mechanism-bound; bold body-axis swing (SSM with sliding-window attention, OR GLA family fork, OR per-head selectivity-mixed Mamba-2).

## 2026-04-29 · 0102 SSM × ternary × production-batch compound — best result of session [n=1]

**Setup**: forked 0099 (ternary infra), overrode env to SSM frontier topology (`PARALLEL_LAYER_POSITIONS=0,1,2, PARALLEL_SSM_TYPE=mamba2_kill, MAMBA2_KILL_SELECTIVITY=1, BIGRAM_VOCAB_SIZE=0`), production batch `TRAIN_BATCH_TOKENS=131072`, canonical `MATRIX_LR=0.045` (NOT the LR×3 from 0099 — at canonical batch, canonical LR), `ITERATIONS=1000`. TRIGRAM_SIDE_MEMORY=0 (avoids dynamo bug).

**Observed**: pre-quant val_bpb **1.5414**, post-quant **1.5417** (quant_tax 0.0003 — packed-ternary makes quantization near-lossless), artifact 5.63 MB, step_avg 813.57 ms.

**Δ comparisons** [all single-seed, pre-quant unless noted]:
- vs 0098 (SSM frontier @ 200 steps small batch, 49M tokens): -0.46 BPB. Training-duration win on SSM stack.
- vs 0099 (transformer + ternary + LR×3 @ 2k small batch, 49M tokens): -0.108 BPB. SSM stack + production batch handily beats matched-tokens transformer. **The compound is real.**
- vs 0101 (transformer baseline @ 2k small batch, 49M tokens): -0.032 BPB. Even without ternary, SSM at 2.6× more tokens (production batch's 5.3× tokens-per-step × 0.5× steps = 2.6× total tokens) beats transformer at small batch.

**Conclusion** [LIKELY n=1]: At our 5090 budget, the SSM frontier stack benefits SUBSTANTIALLY from production batch + canonical LR. The +0.076 BPB ternary penalty observed at small batch with LR×3 (0099 vs 0101) **does NOT appear at production batch with canonical LR** — 0102 at production batch IS lower than 0101 at small batch despite having ternary. This is consistent with BitNet's "more tokens / canonical recipe closes gap" claim, and recasts the 0099 ternary penalty as primarily an LR-and-batch-mismatch artifact rather than a fundamental ternary cost.

**Implication for the writeup**: SSM + ternary + production batch + canonical LR is the deployment recipe to optimize. Ternary's cap-saving (5.6 MB vs 22 MB cap-bust) buys ~10 MB of headroom at no val_bpb cost in this regime. The freed cap can be spent on more layers / depth / width.

**Next**: 0103 (SSM no-ternary at production batch, decomposition control) and 0104 (SSM at 5k steps, training-duration extrapolation) — chain restarted with these two only after 0102 chain post-script crashed on a 'parent' KeyError (due to my earlier `echo "{}"` reset of result.json). Fix: restore result.json with parent field before re-running.

## 2026-04-29 · 0103 result recasts 0102 — ternary's value is purely cap-saving [n=1]

**0103 setup**: identical to 0102 EXCEPT `TERNARY_BODY=0` (the ternary is removed). Same SSM frontier topology, same canonical LR 0.045, same production batch 131072, 1000 steps.

**0103 observed**: pre-quant **1.4587**, post-quant **1.4591**, artifact **21.43 MB (CAP-BUSTS by 5.4 MB)**, step_avg 803 ms.

**Implication**: at production batch + canonical LR, the SSM frontier's TRAINING benefits from the regime, but ternary still costs +0.082 val_bpb pre-quant (1.4587 vs 1.5414). The 0102 result that I called a "compound win" was really the production-batch + canonical-LR effect alone — ternary is a NET COST on val_bpb at this regime.

**HOWEVER**: ternary's value is now precisely characterized as a **cap-saver, not a val-saver**:
- 0103 (no ternary): val 1.459, artifact 21.4 MB → INELIGIBLE for submission (>16 MB)
- 0102 (with ternary): val 1.541, artifact 5.6 MB → SUBMITTABLE with 10.4 MB headroom
- Trade: ternary buys ~16 MB of cap for +0.082 val_bpb cost.

The cleanest framing: **0103 demonstrates the model can land at val_bpb 1.46 at our regime, but at 21.4 MB. To make it submittable WITHOUT ternary we'd shrink the model — at smaller params, val_bpb would rise**. The right next experiment to characterize this: **shrunk-no-ternary at cap = SUBMIT-equivalent SSM frontier**. Likely lands somewhere between 1.46 (full size) and 1.54 (full + ternary).

**Best results so far this session**:
- Best **val_bpb**: 0103 pre-quant 1.4587 (ineligible: cap-busts 21.4 MB)
- Best **submittable**: 0102 post-quant 1.5417 at 5.6 MB (single-seed)
- Best by leaderboard standard: 0102 since cap eligibility is required.





