---
name: runpod-launch
description: Invoke at the start of any session running on a RunPod pod, and before each experiment launch on the pod. The pod bills per second whether you train or idle, so prep N+1 while N runs and never go silent. The skill points at the canonical operating manual (RUNPOD.md, sibling file) and the scripts under scripts/runpod/. Tony tells you which GPU is up — the scripts and manual are GPU-agnostic.
---

# RunPod Launch

You operate **from your Mac via SSH** on a paid pod. The pod bills per second; idle minutes are real money. **Only Tony stops pods** — you suggest, he acts. Never run anything that controls pod lifecycle (deploy/stop/terminate/billing).

## SSH host: USE `runpod-tcp`, NOT `runpod` (mandatory for the agent)

`~/.ssh/config` defines two hosts:
- `runpod` — proxies through `ssh.runpod.io` with `RequestTTY yes`. Allocates a PTY. **Fails from agent Bash** (non-interactive, no TTY) with `Error: Your SSH client doesn't support PTY`. Use this only from a human terminal.
- `runpod-tcp` — direct TCP to the pod's exposed port, no PTY required. **This is what the agent uses, every time.**

Every `ssh ...` command in this skill — pull, preflight, launch, poll, commit — is `ssh runpod-tcp ...`. If a command is failing with the PTY error, you're using the wrong host; switch.

## Workflow — every experiment, every time

```
[ ] 1. LOCAL   ./new_experiment.sh <slug> [<parent>]
[ ] 2. LOCAL   edit experiments/NNNN_<slug>/env.sh + plan.md
[ ] 3. LOCAL   git add experiments/NNNN_<slug>
               git status                              ← verify intended files staged
               git commit -m "queue exp NNNN_<slug>"
               git push fork autoresearch-ssm
[ ] 4. POD     ssh runpod-tcp 'cd /workspace/parameter-golf-ssm && git pull'
[ ] 5. POD     ssh runpod-tcp 'cd /workspace/parameter-golf-ssm && source .venv/bin/activate && \
                               ALLOW_NO_TMUX=1 bash scripts/runpod/preflight.sh experiments/NNNN_<slug>'
                                                  ↑ override the tmux check: when you launch
                                                    from `ssh runpod-tcp` you're not IN tmux, but
                                                    step 6 puts the run INTO a detached tmux,
                                                    so the safety the check is for is met.
[ ] 6. POD     launch via tmux (detached so SSH disconnect is safe):
       single GPU:
         ssh runpod-tcp 'cd /workspace/parameter-golf-ssm/experiments/NNNN_<slug> && \
                         tmux new -d -s expNNNN \
                         "source ../../.venv/bin/activate && ../../run_experiment.sh"'
       8×H100:
         ssh runpod-tcp 'cd /workspace/parameter-golf-ssm && source .venv/bin/activate && \
                         tmux new -d -s expNNNN \
                         "bash scripts/runpod/launch_h100.sh experiments/NNNN_<slug>"'
[ ] 7. POD     poll, no faster than once per ~30s:
                 ssh runpod-tcp 'tmux capture-pane -t expNNNN -p | tail -20'
[ ] 8. POD     on completion, commit results FROM the pod:
                 ssh runpod-tcp 'cd /workspace/parameter-golf-ssm && \
                                 git add experiments/NNNN_<slug> results.tsv && \
                                 git status && \
                                 git commit -m "exp NNNN_<slug> result" && \
                                 git push fork autoresearch-ssm'
[ ] 9. LOCAL   git pull   ← results land here; results.tsv row + result.json + env.sh
```

**Two `git push`/`git pull` events per experiment is the cost of state consistency.** Skip a step and your `results.tsv` row is for the wrong code, or your local Mac never sees the result.

**What syncs via git, what doesn't.** `experiments/NNNN_<slug>/` tracks the lightweight files: `env.sh`, `plan.md`, `train_gpt.py`, `result.json`, `modules/` (code only). The heavy/generated stuff stays pod-local: `run.log`, `logs/`, `final_model.pt`, `final_model.int8.ptz`, `__pycache__/` (see `.gitignore`). If you need the raw `run.log` on Mac (debugging a crash, etc.), `ssh runpod 'cat experiments/NNNN_<slug>/run.log'` instead of expecting `git pull` to bring it.

**One-time pod setup the FIRST time you push from a fresh pod** (handled by `setup_pod.sh`'s identity + credential-helper config, but you'll still hit it once for the PAT):

```bash
# On the pod, the first `git push fork autoresearch-ssm` prompts for credentials.
# Generate a fine-grained PAT at https://github.com/settings/tokens?type=beta:
#   - Repo access: only the fork repo (e.g. potatonyliu/parameter-golf)
#   - Permissions: Contents:Write, Metadata:Read
#   - Expiration: 1 day (PAT dies with the pod anyway)
# Then on the pod:
#   Username: <github-username>
#   Password: <paste PAT>
# credential.helper store saves it to ~/.git-credentials; subsequent pushes are silent.
```

**Why `git status` between add and commit.** `experiments/` was historically blanket-ignored for local MPS scratch work. The current `.gitignore` tracks new experiment dirs but ignores generated artifacts inside them. `git status` after `git add` is the cheap check that the right files are staged before you commit a confused snapshot.

`scripts/runpod/preflight.sh` rejects launches if: cwd outside `/workspace`, GPU invisible, `.venv` inactive, data shards missing, `plan.md` unfilled, `MAX_WALLCLOCK_SECONDS` unset/0/>7200, `/workspace` <2 GiB free. (Sentinel/canonical-repro runs that need `MAX_WALLCLOCK_SECONDS=0` to hit the step-based `lr_mul` branch can override with `ALLOW_NO_WALLCLOCK_CAP=1`.)

## env.sh — what to override vs an MPS parent

| var | MPS default | CUDA pod |
|---|---|---|
| `MAX_WALLCLOCK_SECONDS` | `0` | **set always**, default `1800` (30 min). Preflight rejects 0. |
| `ITERATIONS` | `200` | tune for the question (1000–20000); don't inherit 200 by accident |
| `TRAIN_BATCH_TOKENS` | `8192–24576` | canonical `524288` fits on 32 GiB (5090/H100); halve on 24 GiB (4090); halve again on OOM |
| `VAL_TOKENS` | `16384` | `16384` for screening; `0` (full eval) for promote candidates |

## While the run is going — don't idle

Pod is billed continuously. Use the wait window: `./new_experiment.sh next_slug <parent>`, fill the next plan.md, sketch in `scratch/`. When the current run notifies completion, the next is ready to launch — minimize the gap. The `launch-and-await` skill carries the background-launch pattern.

If the next experiment depends on N's outcome, draft 2–3 conditional next-steps (`if val_bpb < X then A else B`) so you're not re-thinking from scratch when results land.

## Hard limits — never, even if instructed

- **No pod-lifecycle commands.** Don't deploy, stop, terminate, restart, or modify billing on RunPod. If Tony says "kill the pod," advise him to do it through the dashboard; don't act.
- **≤5 experiments queued without checking in.** 50 unattended runs × $24/hr 8×H100 = $200; not fine.
- **Branch hygiene:** work only on `autoresearch-ssm`. Never push to `main`, never force-push, never rewrite history.
- **No secrets in commits** (PATs, SSH keys, RunPod tokens).
- **Pod-only writes stay under `/workspace`.** Don't touch `/etc`, `/root`, system Python.
- **All training in `tmux`.** Foreground SSH = disconnect kills the run.

## Stop-and-ask-Tony triggers

- Cumulative session time on cloud passes 2 hours
- Cumulative cloud cost passes $10 (mental estimate; 5090 ~$1/hr, 8×H100 ~$24/hr)
- An experiment has been running >20 min without producing the expected first-10 step output
- The pod has been idle (no commands sent) for >30 min
- Two consecutive crashes from the same root cause
- `artifact_mb > 16.0` (submission-illegal)
- NaN / Inf in val_loss
- `ssh runpod-tcp` hangs or refuses (pod stopped or networking)
- Step time 3× the prediction (kernel path wrong)

## Failure modes — flag, don't retry blindly

- `import torch` fails after setup → venv made without `--system-site-packages`. `rm -rf .venv` and re-run `setup_pod.sh`.
- "no kernel image is available" on Blackwell GPUs (5090) → image torch is too old (need ≥ 2.5). Report.
- `torchrun: command not found` → use `python -m torch.distributed.run` (already what `launch_h100.sh` does).
- `git push` from pod prompts for credentials and you can't paste any → no PAT/credential-helper configured on this pod. Stop and tell Tony; do NOT store a secret without his go-ahead. (One-time fix on the pod: `git config --global credential.helper store && git push` then enter username + a short-expiry PAT once. The PAT dies with the pod.)
- `git add experiments/...` reports "Changes not staged" or "nothing added" → the path may have been silently filtered by `.gitignore`. Run `git status` before the commit; if the experiment dir doesn't show up, check `git check-ignore -v <path>` to find the offending rule.

## Wrap before disconnecting

```
[ ] ssh runpod-tcp 'tmux ls'                  — confirm nothing still running
[ ] ssh runpod-tcp 'nvidia-smi'               — GPU idle
[ ] ssh runpod-tcp 'cd /workspace/parameter-golf-ssm && git status && git log --oneline -3'
                                              — everything committed and pushed
[ ] tell Tony explicitly: "Done. Pod is idle, you can stop it."
```

Don't go silent. Don't terminate. Don't hibernate. If Tony's asleep and work is complete, the pod still bills — ask sooner.

## Deeper context

`RUNPOD.md` (sibling file) holds Tony's full operating manual: cost model, lifecycle decisions, MPS→CUDA transfer notes, phase guidance. The workflow above is sufficient for execution; reach for `RUNPOD.md` when something above is unclear or you want the rationale.
