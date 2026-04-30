# Non-record Submission: 1.3004 BPB — kill-Mamba-2 SSM, ternary, n=7 depth, EMA, 1-hour 4×H200

**1.58-bit Ternary Quantization + 7L unique × 3 loop weight sharing + Parallel attn‖kill-Mamba-2 + LTI selectivity (B/C constants) + EMA β=0.999 + brotli**

**val_bpb: 1.3004** (post-quant, seed=1337, 1.30040229 exact) | **12.07 MB** artifact | 4×H200 SXM, 4,380 steps (3,600s wallclock)

> This is a **non-record submission** — the run uses 4×H200 SXM rather than the 8×H100 reference and trains for 1 hour rather than 10 minutes. It is also the first SSM-based entry I've seen in either track. The reason it's here, not in `track_10min_16mb`, is hardware: I couldn't get an 8×H100 SXM allocation in the time I had, so I ran on what was available (`runpod.io` 4×H200 bundle, Iceland region) and used the time budget to compensate. The run is reproducible from `train_gpt.py` + `env.sh` in this folder, and the eval/quant path is the standard root harness — no test-time training, no sliding-window eval, no GPTQ.

## Results (seed=1337, 4×H200 80GB SXM)

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.2983 |
| **Post-quant val_bpb (int8 + brotli)** | **1.3004** (1.30040229) |
| Post-quant val_loss | 2.1957 (2.19567479) |
| Quant tax | 0.0021 |
| Steps | 4,380 |
| ms/step | 821.94 |
| Train wallclock | 3,600s (cap fired) |
| Eval time pre+post | 357s (full 60M-token val) |
| Tokens trained | 4,380 × 524,288 ≈ 2.30B |
| Artifact (int8+brotli) | 12,074,422 bytes (12.07 MB) |
| Code size | 104,676 bytes |
| Model parameters | 61.66M |
| GPUs / nproc | 4 / `--nproc 4` (grad_accum=2) |

### Comparison

vs. existing `track_non_record_16mb/` entries:

| Submission | BPB | Notes |
|---|---|---|
| `2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear` | 1.1239 | 1-bit quant, 8×H100, 50k steps (~2.15h), 8192 BPE |
| `2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` | 1.2074 | Plain transformer, 8×H100, 4h |
| **This** (kill-Mamba-2 SSM) | **1.3004** | First SSM. 4×H200, 1h, sp1024 |

vs. the `track_10min_16mb/` records-track baseline `2026-03-31_ParallelResiduals_MiniDepthRecurrence` (1.1063 BPB, 8×H100, 600s): the gap is roughly 0.19 BPB. About 0.02 of that is missing eval polish (records' sliding-window-stride-64 eval is worth ~0.017 vs full-window, see `2026-03-19_SlidingWindowEval`); the rest is a mix of architecture (SSM vs transformer at sp1024 scale) and the records' standard-stack ports I haven't backported (parallel-residuals, AR self-gen GPTQ, mini depth recurrence, BigramHash, sliding-window eval).

## Why an SSM at all

Most of the records track is sp1024/sp4096/sp8192 transformers with an increasingly polished standard stack (parallel residuals, GPTQ, sliding-window eval, mini depth recurrence). The motivating question here was whether a state-space-model body — once you've actually figured out which Mamba-2 variant works at this regime — can hold its own, since none of the existing entries try one. The result here doesn't beat records, but it's the cleanest data point I have on "kill-Mamba-2 frontier at non-trivial training," and the headline number is in the same neighborhood as the records-track 1.1063 entry, just with a lot of porting still to do.

## Architecture

- 7 unique transformer-style blocks looped 3× via depth recurrence (`NUM_UNIQUE_LAYERS=7 NUM_LOOPS=3`, effective depth 21, full weight sharing across loops, no U-Net skip)
- `dim=512`, `num_heads=8`, `num_kv_heads=4` (GQA), tied input/output embeddings, sp1024 vocab
- Each block is a **PARALLEL block**: attention and a kill-selectivity Mamba-2 SSM read the same normalized input and have their outputs summed. `PARALLEL_LAYER_POSITIONS=0,1,2,3,4,5,6 PARALLEL_SSM_TYPE=mamba2_kill`.
- **kill-Mamba-2** = standard Mamba-2 SSD selective scan but with `B`, `C`, and `dt` replaced by learned per-head/per-state constants (`_B_const`, `_C_const`) instead of input-dependent projections. Same `in_proj`, `conv1d`, `out_proj`, `A_log`, `dt_bias`, `D_skip`. The intuition is that `dt`/`B`/`C` are doing a lot of input-dependent work that the much simpler LTI variant captures most of for our regime, with strictly less gradient signal to mismanage. Earlier experiments (0038/0042 at MPS-200, 0113 at H100-equivalent batch) showed `kill > full` is architectural rather than an LR-cliff artifact at this scale.
- No BigramHash — it interacts negatively with kill-Mamba-2 in our family (0042 confirmed +0.005-0.009 regression). The records' transformers use `BIGRAM_DIM=112` profitably; SSMs in this stack don't.
- Body weights are 1.58-bit ternary via BitNet-b1.58 absmean STE (`TERNARY_BODY=1`). Ternary weights are stored as 2-bit packed at quant export (`v2 packed-ternary export: 63 BitLinear weights → 2-bit packed`). 1D and small (≤65,536-element) tensors stay fp32 via `CONTROL_TENSOR_NAME_PATTERNS` (`A_log,A_im,B_proj,C_proj,dt_log,D_skip,dt_bias,delta_bias,conv1d,...`).
- **EMA-of-weights** at β=0.999 (`EMA_BETA=0.999`, `EMA_WARMUP_OFFSET=lr_warmup_steps=30`), shadow swapped into the model at last step before eval. Window ≈ 1000 steps captures roughly the last 23% of training.
- **brotli** (quality=11) for entropy coding on top of int8 + 2-bit ternary packing. Beats zlib by a small margin on this bytestream (`brotli/zlib ratio: 0.985`).

The schedule (warmdown=1800, lr_warmup=30, MATRIX_LR=0.045, batch=524288, embed_lr=0.05, scalar_lr=0.04) is a transformer-records setup that I kept verbatim — these values track @KellerJordan's modded-nanogpt numbers ported through earlier PRs and have transferred cleanly across SSM and transformer blocks in our experiments.

## What did the work

A few non-obvious things mattered:

- **The `kill > full` finding for Mamba-2 is real, not an LR artifact.** When I first tried full-selectivity Mamba-2 with the modded-nanogpt LR (`MATRIX_LR=0.045`) at production batch (524288 / 8 GPUs), it diverged. At LR/3 it trained stably but still lost +0.031 BPB to `kill`. The constants-only LTI prior wins on its own, not because the full version was just a bad LR away from working. (Background: `MAMBA2_KILL_SELECTIVITY=1` in `env.sh`.)
- **The depth ceiling at `NUM_UNIQUE_LAYERS=5` was training-duration-bound.** At 5,090 1k-step smokes I had n=7 losing +0.030 BPB to n=5, and a journal note speculating it might reverse at long-train. It does. n=7 with 4,380 long-train steps lands at 1.3004; the prior best (5,090 1k smokes, n=5) was 1.5232. About -0.22 BPB from training duration alone is consistent with the slope I'd projected from the 0103→0104 5×-tokens delta (-0.115 per 5×).
- **The artifact has 4 MB of cap headroom.** At n=7 the int8+brotli artifact is 12.07 MB — comfortably under the 16 MB cap. Going to n=9 cap-busts (16.7 MB extrapolated). There's room to spend the headroom on a wider model (`d_model=640`) or bigger MLP (`MLP_MULT=12`) but I haven't tested either at this scale.
- **EMA β=0.999 needs ≥ 3,000 steps to be the right hyperparameter.** At my Round-1 600s de-risk (733 steps) the same β gave val_bpb 2.36 because the shadow window (~1000 steps) covered the entire training run, so the post-shadow-swap model was nearly at init. Same effect that @signalrush flagged in `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (where warmdown=3500 was paired with a longer schedule). For ≤1000-step runs, β=0.99 is the right choice — verified at 0119.

## What I deliberately did not do

These are records-track ports that I'd expect to compose cleanly on top, but ran out of time to backport. Listing for the next person who wants to push this further:

- **Parallel residuals** (Marko Sisovic's 2026-03-31 record). I have a working implementation on a separate branch (exp 0115) but didn't merge it onto the EMA branch in time. ~30-60 min of code-merge.
- **Sliding-window eval** at stride=16 or stride=64 (Matthew Li's 2026-03-19 entry, used by every record since). This would close the eval polish gap of ~0.017 BPB. It's a clean port from `track_10min_16mb/2026-03-31_Scylla_FullGPTQ_XSA11_FA3_0.9485/train_gpt.py:1175 def eval_val_sliding`; I just hadn't budgeted the 1-2 hours.
- **AR self-generated GPTQ** (abaybektursun's 2026-03-25). I tested an int6 variant earlier (exp 0081) and it cap-busted — int6+brotli is incompatible in our family, and the AR self-gen path needed more debug than I had time for. Open question whether int8+GPTQ would compose on top of the 2-bit packed ternary body; my guess is yes.
- **Mini depth recurrence on layers 4,5** (the modded-nanogpt port). I'm using full weight sharing across all loops, which is a different recurrence pattern from the records' "looped middle pair." Worth checking which is better at SSM scale.
- **BitNet b1 (true 1-bit)** as in `2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear`. Brief told me to think about this; haven't built it yet. Combined with the SSM body it'd be the first 1-bit SSM submission.
- **Test-time training**, in any form. Records' top entries lean on legal score-first TTT for ~-0.01 BPB. Not in my eval path.

## Reproducibility

```bash
# 4×H200 SXM, RunPod template equivalent
SEED=1337 \
NUM_UNIQUE_LAYERS=7 NUM_LOOPS=3 \
PARALLEL_LAYER_POSITIONS=0,1,2,3,4,5,6 \
PARALLEL_SSM_TYPE=mamba2_kill MAMBA2_KILL_SELECTIVITY=1 \
BIGRAM_VOCAB_SIZE=0 \
TERNARY_BODY=1 \
EMA_BETA=0.999 EMA_WARMUP_OFFSET= \
TRAIN_BATCH_TOKENS=524288 ITERATIONS=20000 \
WARMDOWN_ITERS=1800 LR_WARMUP_STEPS=30 MATRIX_LR=0.045 \
TIED_EMBED_INIT_STD=0.05 MUON_BACKEND_STEPS=15 \
MAX_WALLCLOCK_SECONDS=3600 VAL_TOKENS=0 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

(The full `env.sh` in this folder lists every env var; the `CONTROL_TENSOR_NAME_PATTERNS` list in particular is load-bearing for SSMs and easy to get wrong.)

`brotli` and `sentencepiece` are required at quant-export time; they're in `requirements.txt`. `train_gpt.py` here is the experiment-folder copy from `experiments/0124_path_a_h200_1hr/train_gpt.py`.

The wallclock cap (`MAX_WALLCLOCK_SECONDS=3600`) is what bounds the run — `ITERATIONS=20000` is just a safe upper bound on step count. Different hardware will hit the wallclock at different step counts; on 4×H200 SXM at 822 ms/step that's 4,380.

## Acknowledgements

The schedule (warmdown=1800 / lr_warmup=30 / MATRIX_LR=0.045 / muon_backend=15 / TIED_EMBED_INIT_STD=0.05) is from earlier sp1024 transformer wins in this challenge; thanks to everyone who put those numbers on the leaderboard. The kill-Mamba-2 architecture is a simplification of the standard Mamba-2 SSD scan from Dao & Gu, with selectivity replaced by per-head/per-state constants. EMA-of-weights is the records' tier-1 port from @signalrush's 2026-03-22 entry, with the β chosen to suit the training length here. Brotli compression on top of int8+2-bit-packed-ternary is the standard records-track quant path.
