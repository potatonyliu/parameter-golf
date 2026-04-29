---
name: runpod-launch
description: Invoke at the start of any session running on a RunPod pod, and before each experiment launch on the pod. Carries the cost-aware rhythm (pod bills per second whether you train or idle, so prep N+1 while N runs), the first-time setup command, the preflight discipline, and the single-GPU vs multi-GPU launch difference. Tony tells you which GPU is up — the skill is GPU-agnostic.
---

# RunPod Launch

You are operating on a paid GPU. The pod bills per second from `Running` to `Stopped`. **You don't stop pods — Tony does.** You can suggest stopping; you must not run anything destructive against the pod's lifecycle.

The two consequences of paying-while-idle:

1. **Be ready before the GPU is.** When Tony says the pod's up, your env.sh, plan.md, and decision tree should already exist on local. Don't draft them on the pod's clock.
2. **Stack work.** While experiment N trains, prep N+1's plan.md and env.sh. The launch task notifies on completion — that's the cue to launch N+1, not the cue to start writing N+1. The `launch-and-await` skill carries the background-launch pattern; the rhythm matters more on a paid pod than on local MPS.

If the next experiment depends on N's outcome, draft 2-3 conditional next-steps (`if val_bpb < X then A else B`) so you're not re-thinking from scratch when results land.

## On a fresh pod

Tony deploys the pod and gives you the SSH command. Before anything else:

```bash
# 1. tmux. SSH disconnect kills non-tmux runs.
tmux new -s work
# (or: tmux a -t work to reattach)

# 2. Check setup. If /workspace/.pod_setup_complete is missing:
cd /workspace
bash <(curl -fsSL https://raw.githubusercontent.com/potatonyliu/parameter-golf/autoresearch-ssm/scripts/runpod/setup_pod.sh)
# Idempotent. Clones into /workspace/parameter-golf-ssm, makes a venv with
# --system-site-packages (so torch from the image is visible), installs
# requirements-cuda.txt, downloads ~2 GB of FineWeb. ~5-10 min.

# 3. Activate.
cd /workspace/parameter-golf-ssm && source .venv/bin/activate

# 4. CUDA reproducibility check (only on the FIRST run of a fresh pod):
bash scripts/runpod/regression_sentinel.sh
# Reproduces 0001_baseline_repro on CUDA, compares to MPS anchor (val_bpb
# 2.5212 ± 0.05). PASS = safe to proceed. FAIL = stop and report — likely a
# PyTorch / CUDA / image-version drift, not something to paper over.
```

**Always inside `/workspace`** — `/root` and `/tmp` are container disk and get wiped on pod stop. `/workspace` is the network volume and survives.

## Per-experiment

Same as local MPS, plus a preflight gate:

```bash
./new_experiment.sh <slug> <parent>
# edit experiments/NNNN_<slug>/env.sh — set RUN_ID, MAX_WALLCLOCK_SECONDS,
# adjust ITERATIONS / TRAIN_BATCH_TOKENS for the GPU (see below).
# fill plan.md — Question / Hypothesis / Change / Disconfirming.

bash scripts/runpod/preflight.sh experiments/NNNN_<slug>
# Refuses to launch if: cwd not under /workspace, not in tmux, GPU invisible,
# .venv inactive, data shards missing, plan.md unfilled, MAX_WALLCLOCK_SECONDS
# unset/0/>7200, /workspace <2 GiB free.
```

Launch depends on GPU count (Tony will tell you which is up):

```bash
# Single-GPU CUDA (RTX 5090 / 4090 / 1×H100):
cd experiments/NNNN_<slug> && ../../run_experiment.sh

# Multi-GPU CUDA (8×H100 SXM):
bash scripts/runpod/launch_h100.sh experiments/NNNN_<slug>
# Wraps `python -m torch.distributed.run --standalone --nproc_per_node=8`,
# mirrors run_experiment.sh's metric parsing into result.json + results.tsv.
```

Constraint from `train_gpt.py`: `WORLD_SIZE` must divide 8 (so `grad_accum_steps = 8 // world_size` stays integral). Valid: 1, 2, 4, 8.

## env.sh adjustments vs MPS-tuned parents

Existing `experiments/NNNN_*/env.sh` files were tuned for MPS. When you fork from one for a CUDA pod, override:

- `MAX_WALLCLOCK_SECONDS=1800` (30 min default — preflight rejects 0/unset).
  Adjust per experiment: shorter for smokes, longer for extended training.
  Never `0` on a pod.
- `ITERATIONS` — MPS uses 200; CUDA pods can afford 1000-20000 depending on
  what you're testing. Don't keep 200 by accident.
- `TRAIN_BATCH_TOKENS` — MPS uses 8192-24576 (small for memory + speed);
  canonical PG is 524288. On a 32 GiB GPU (5090 / H100) the canonical fits.
  On a 24 GiB GPU (4090) try 262144 first; if OOM, halve again. The 0093
  ternary stack with MLP_MULT=8 + parallel attn||mamba2 is the heaviest
  case — if it fits, anything will.
- `VAL_TOKENS=16384` is fine for ranking. For a promote candidate, set to
  `0` (full eval); H100 full-eval is ~1-2 min.

## While the run is going

Pod is billed continuously. Use the wait window:

- `./new_experiment.sh next_slug <parent>` — set up the next experiment.
- Fill its plan.md, draft env.sh edits.
- Sketch in `scratch/` if the experiment has a math step.
- When the current run notifies completion, the next is ready to launch — minimize the gap.

If you find yourself sitting and watching the log scroll, that's pure waste. The trajectory gate (first 10 steps from `await_steps.sh`) is enough; once it passes, switch to the next experiment's prep.

## Stop discipline (Tony's job, not yours)

When you finish the planned experiments:

1. `tmux ls` — anything still running?
2. `nvidia-smi` — is the GPU idle?
3. `git status` and push your sub-branch (NOT to `autoresearch-ssm` directly).
4. **Tell Tony explicitly: "Done. Please stop the pod."** Don't go silent.
   Don't terminate. Don't hibernate. Just say so and let him stop it.

If Tony is asleep and the work is genuinely complete, the pod is sitting there billing. Asking him sooner is cheaper than waiting.

## Failure modes to flag, not retry blindly

- **`import torch` fails after setup** → the venv was created without
  `--system-site-packages`. Delete `.venv` and re-run setup_pod.sh.
- **"no kernel image is available for execution"** on 5090 / Blackwell → the
  image's torch is too old (need ≥ 2.5). Report; don't workaround.
- **`torchrun` not found** → use `python -m torch.distributed.run` (already
  what `launch_h100.sh` does). If you're calling torchrun directly, switch.
- **Step time 3× the prediction** → kernel path is wrong (no flash-SDP /
  fallback to math). Stop and investigate before burning more compute.
- **Two consecutive crashes from the same root cause** → fix the cause,
  don't keep launching variants.
- **NaN / Inf in val_loss** → SSM late-instability is documented (program.md
  §SSM-specific harness facts); don't keep launching variants.
- **`artifact_mb > 16.0`** → submission-illegal; don't burn more cycles.

When in doubt about whether to keep going, tell Tony. The cost of a 5-min
pause to confirm is small compared to the cost of a wrong-direction hour.
