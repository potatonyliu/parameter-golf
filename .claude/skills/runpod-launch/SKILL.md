---
name: runpod-launch
description: Invoke at the start of any session running on a RunPod pod, and before each experiment launch on the pod. The pod bills per second whether you train or idle, so prep N+1 while N runs and never go silent. The skill points at the canonical operating manual (RUNPOD.md, sibling file) and the scripts under scripts/runpod/. Tony tells you which GPU is up — the scripts and manual are GPU-agnostic.
---

# RunPod Launch

You are operating on a paid GPU. The pod bills per second from `Running` to `Stopped`. **You do not stop pods — Tony does.** You never run anything that controls pod lifecycle (deploy / stop / terminate / billing). You suggest, he acts.

The full operating manual is at [`RUNPOD.md`](./RUNPOD.md) in this same skill folder. Read it now if you haven't this session, then return here for the launch checklist.

## What's load-bearing in RUNPOD.md

You can skim, but these sections are non-negotiable:

- **§5 — connection pattern.** You operate from Tony's Mac via SSH. One-shot: `ssh runpod 'cmd'`. Long-running: `ssh runpod 'cd ... && tmux new -d -s expNNNN "..."'`. Never run training in a foreground SSH session.
- **§6 — agent hard limits.** No pod-lifecycle commands, ≤5 experiments queued without checking in, never push to `main`, never force-push, no secrets in commits.
- **§9 — standard cloud experiment loop.** The 8-step plan-locally / pull-on-pod / preflight / launch / commit-from-pod / pull-on-Mac rhythm. Two `git push`/`git pull` events per experiment is the cost of state consistency.
- **§10 — cost guardrails.** Stop and ask Tony if any of these trip: 2h cumulative session, $10 cumulative, 20 min hung run, 30 min idle pod.

## The two consequences of paying-while-idle

1. **Be ready before the GPU is.** When Tony says the pod's up, your env.sh, plan.md, and decision tree should already exist on local. Don't draft them on the pod's clock.
2. **Stack work.** While experiment N trains, prep N+1's plan.md and env.sh. The launch task notifies on completion — that's the cue to launch N+1, not the cue to start writing N+1. The `launch-and-await` skill carries the background-launch pattern; the rhythm matters more on a paid pod than on local MPS.

If the next experiment depends on N's outcome, draft 2–3 conditional next-steps (`if val_bpb < X then A else B`) so you're not re-thinking from scratch when results land.

## Per-experiment checklist

```
[ ] new_experiment.sh on local Mac (creates experiments/NNNN_<slug>/)
[ ] Edit env.sh — set MAX_WALLCLOCK_SECONDS (default 1800; never 0/unset on a pod)
                  adjust ITERATIONS / TRAIN_BATCH_TOKENS for the GPU
[ ] Fill plan.md (Question / Hypothesis / Change / Disconfirming)
[ ] git commit + git push (so the pod can pull)
[ ] ssh runpod 'cd /workspace/parameter-golf-ssm && git pull'
[ ] ssh runpod 'cd ... && source .venv/bin/activate && bash scripts/runpod/preflight.sh experiments/NNNN_<slug>'
[ ] tmux-launch:
      single GPU:  ssh runpod 'cd .../experiments/NNNN_<slug> && tmux new -d -s expNNNN "source ../../.venv/bin/activate && ../../run_experiment.sh"'
      8×H100:      ssh runpod 'cd /workspace/parameter-golf-ssm && source .venv/bin/activate && tmux new -d -s expNNNN "bash scripts/runpod/launch_h100.sh experiments/NNNN_<slug>"'
[ ] Poll progress no more than once per ~30s: ssh runpod 'tmux capture-pane -t expNNNN -p | tail -20'
[ ] On completion: ssh runpod 'cd ... && git add ... && git commit ... && git push' (results commit FROM the pod)
[ ] On Mac: git pull
```

`scripts/runpod/preflight.sh` rejects launches if: cwd outside `/workspace`, GPU invisible, .venv inactive, data shards missing, plan.md unfilled, MAX_WALLCLOCK_SECONDS unset/0/>7200, /workspace <2 GiB free.

## env.sh adjustments vs MPS-tuned parents

When forking from an MPS parent for a CUDA pod, you almost always need to override:

- **`MAX_WALLCLOCK_SECONDS=1800`** (30 min default — preflight rejects 0/unset). Adjust per experiment; never `0` on a pod.
- **`ITERATIONS`** — MPS uses 200; CUDA pods can afford 1000–20000 depending on the question. Don't accidentally inherit 200.
- **`TRAIN_BATCH_TOKENS`** — MPS uses 8192–24576 (small for memory + speed); canonical PG is 524288. On 32 GiB GPUs (5090 / H100) the canonical fits. On 24 GiB (4090) try 262144 first; if OOM, halve again. The 0093 ternary stack with MLP_MULT=8 + parallel attn||mamba2 is the heaviest case.
- **`VAL_TOKENS=16384`** is fine for ranking. Promote candidate? Set to `0` (full eval); on H100 a full eval is ~1–2 min.

## Stop discipline

When you finish the planned experiments:

1. `ssh runpod 'tmux ls'` — anything still running?
2. `ssh runpod 'nvidia-smi'` — GPU idle?
3. `ssh runpod 'cd ... && git status && git log --oneline -3'` — everything committed and pushed?
4. **Tell Tony explicitly: "Done. Pod is idle, you can stop it."** Don't go silent. Don't terminate. Don't hibernate.

If Tony's asleep and the work is genuinely complete, the pod is sitting there billing. Asking sooner is cheaper than waiting.

## Failure modes — flag, don't retry blindly

- `import torch` fails after setup → venv was made without `--system-site-packages`. `rm -rf .venv` and re-run setup_pod.sh.
- "no kernel image is available for execution" on Blackwell GPUs (5090) → image torch is too old (need ≥ 2.5). Report; don't workaround.
- `torchrun` not found → use `python -m torch.distributed.run` (already what `launch_h100.sh` does).
- Step time 3× the prediction → kernel path is wrong (no flash-SDP / fallback to math). Stop and investigate.
- Two consecutive crashes from the same root cause → fix the cause, don't keep launching variants.
- NaN / Inf in val_loss → SSM late-instability is documented (`program.md` §SSM-specific harness facts); don't keep launching variants.
- `artifact_mb > 16.0` → submission-illegal; don't burn more cycles.
- `ssh runpod` hangs/refuses → pod was stopped or networking blip. Tell Tony before retrying.

When in doubt, tell Tony. The cost of a 5-min pause to confirm is small compared to the cost of an hour in the wrong direction on a $24/hr pod.
