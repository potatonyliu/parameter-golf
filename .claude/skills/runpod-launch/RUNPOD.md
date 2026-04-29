# RUNPOD.md

Operating manual for working on RunPod against the parameter-golf harness. Two audiences:

- **Tony** (sections marked `[H]`): pod lifecycle, cost, deploy/terminate.

- **Agent** (sections marked `[A]`): what to do once Tony has provisioned a pod and shared SSH details.

  The agent is allowed to operate on the pod autonomously — SSH in, run experiments, commit results — under hard limits in §6. The agent is **not** allowed to control pod lifecycle. That's Tony's responsibility, always.

  This doc supersedes any earlier draft. If `program.md` says "agent runs on Mac with MPS locally," treat that as the **default** mode; cloud mode is invoked only when this doc is referenced explicitly in a session start.

---

## 1. Cost model `[H][A]`

RunPod bills **per-second** while a pod is `Running`. The GPU is reserved whether you're using it or not. Money does not pause for thought.

| State        | GPU cost         | Disk cost                       | Recoverable? |
| ------------ | ---------------- | ------------------------------- | ------------ |
| `Running`    | Full hourly rate | Yes                             | —            |
| `Stopped`    | $0               | Volume disk only (~$0.10/GB/mo) | Yes          |
| `Terminated` | $0               | $0                              | No           |

Reference rates (verify before deploying — these shift):

- RTX 5090, A40, RTX 4090: $0.40–$0.99/hr

- 1× H100 SXM: ~$3/hr

- 8× H100 SXM: ~$24/hr — **this is where mistakes get expensive**

  **A forgotten 8× pod overnight = ~$200.** This is the failure mode this doc exists to prevent.

---

## 2. Pod lifecycle `[H only]`

| Decision                           | Rule                                                         |
| ---------------------------------- | ------------------------------------------------------------ |
| Stepping away >30 min on cheap pod | Stop                                                         |
| Done for the day                   | Stop                                                         |
| Project complete                   | Terminate                                                    |
| Mid-experiment, going AFK          | Run inside `tmux`, leave running, **set a phone alarm**      |
| Mid-session on 8×H100              | Don't stop mid-session — run jobs back-to-back, terminate when done |

Stop ≠ Terminate. Stop preserves the volume disk. Terminate deletes everything except network volume (which is region-locked and inventory-restricted, so we don't use it for parameter-golf).

**Before bed, always check:** https://console.runpod.io/pods — confirm nothing in `Running` state you didn't intend.

---

## 3. Provisioning a pod `[H only]`

1. **Region + GPU availability.** Check the deploy page for stock. For Phase 1 the 5090/A40/4090 tier is fine. For Phase 2 you need 8× H100 SXM specifically — verify it's available before committing to the rest of the workflow.

2. **Template:** Parameter Golf (`y5cejece4j`). Pre-installs PyTorch + CUDA, pre-clones the upstream openai/parameter-golf repo at `/workspace/parameter-golf`.

3. **Settings:** SSH terminal access ON, Jupyter notebook OFF, volume disk 50 GB.

4. **Deploy.** Wait ~30–60 sec for boot.

5. **Get SSH command** from the pod's "Connect" tab.

6. **Add to `~/.ssh/config`** (so the agent and you have a single name to use):

   ```
   # ssh.runpod.io proxy host (no PTY allocation in non-interactive mode — DO NOT use from agent Bash)
   Host runpod
       HostName ssh.runpod.io
       User <runpod-pod-id>-<8-char-suffix>
       IdentityFile ~/.ssh/id_ed25519
       RequestTTY yes

   # Direct TCP (THIS is what the agent uses from Bash — no PTY required)
   Host runpod-tcp
       HostName <pod-ip>
       Port <pod-port>
       User root
       IdentityFile ~/.ssh/id_ed25519
   ```

   **The agent must use `ssh runpod-tcp` from Bash, NOT `ssh runpod`.** Reason: `Host runpod` goes through ssh.runpod.io which forces PTY allocation; agent Bash is non-interactive (no TTY) so every command fails with "Your SSH client doesn't support PTY". `runpod-tcp` connects directly to the pod's exposed TCP port and works headless. (Throughout this manual and SKILL.md, every `ssh runpod` example is the agent-equivalent `ssh runpod-tcp`.)

7. **First connection:** `ssh runpod-tcp`, accept host key, you should land in `/root` or `/workspace`.

---

## 4. Initial pod setup (once per pod) `[H or A, whoever's first]`

The Parameter Golf template pre-clones upstream openai/parameter-golf at `/workspace/parameter-golf`. We need the fork (`autoresearch-ssm` branch) cloned at `/workspace/parameter-golf-ssm`. `scripts/runpod/setup_pod.sh` is the one-shot setup. From a fresh pod:

```bash
ssh runpod
tmux new -s setup
cd /workspace
# Bootstrap setup script (it isn't on the pod yet on first deploy):
bash <(curl -fsSL https://raw.githubusercontent.com/potatonyliu/parameter-golf/autoresearch-ssm/scripts/runpod/setup_pod.sh)
```

The script (idempotent) clones the fork into `/workspace/parameter-golf-ssm`, checks out `autoresearch-ssm`, creates `.venv` with `--system-site-packages` (so the image's preinstalled torch+CUDA is visible), `pip install -r requirements-cuda.txt`, downloads FineWeb sp1024 (~2 GB), CUDA smoke-tests, writes `/workspace/.pod_setup_complete`. Total ~5–10 min.

After setup, verify the harness reproduces the MPS anchor under CUDA numerics:

```bash
source /workspace/parameter-golf-ssm/.venv/bin/activate
cd /workspace/parameter-golf-ssm
bash scripts/runpod/regression_sentinel.sh
```

This forks canonical, runs 200 steps, compares val_bpb_post_quant to the MPS anchor 2.5212 ± 0.05 and artifact MB 6.907 ± 0.1. ~1–2 min on a 5090. PASS = the CUDA path is sound (rankings of MPS deltas should transfer; numbers will differ — see §7). FAIL = stop and investigate before burning more compute.

---

## 5. Connection pattern `[A]`

The agent operates from Tony's Mac, executing remote commands via SSH. Two patterns:

**One-shot commands** (state queries, file reads, short scripts):

```bash
ssh runpod 'cd /workspace/parameter-golf-ssm && git log --oneline -5'
ssh runpod 'nvidia-smi'
ssh runpod 'cat experiments/0064_xxx/run.log'
```

**Long-running commands** (training experiments — anything >30 sec): always inside `tmux`:

```bash
ssh runpod 'cd /workspace/parameter-golf-ssm && tmux new -d -s exp064 "./run_experiment.sh"'
# ... do other work ...
ssh runpod 'tmux capture-pane -t exp064 -p | tail -30'  # check progress
ssh runpod 'tmux ls'                                     # see what's still running
```

**Never** run training in a foreground SSH session. SSH disconnects = killed run.

---

## 6. Agent hard limits `[A]`

Things the agent must NEVER do, even if instructed in the conversation:

- **Pod lifecycle:** never run anything that deploys, stops, terminates, restarts, or modifies billing on RunPod. If Tony says "kill the pod," the agent advises him to do it through the dashboard but does not act.

- **Cost runaway:** never queue >5 experiments without checking in. Each `keep`-tier experiment consumes ~5–10 min of GPU. 5 experiments × $1/hr × 10 min ≈ $0.83 — fine. 50 unattended × $24/hr × 10 min = $200 — not fine.

- **Pod-specific writes outside `/workspace`:** never modify `/etc`, `/root`, or system Python. The pod is ephemeral.

- **Branch hygiene:** the agent works only on its assigned branch (`autoresearch-ssm` by default). Never push to `main`, never force-push, never rewrite history.

- **Long blocking commands:** never run something that prevents Tony from reclaiming the pod. All training in `tmux`. Detach immediately after launch.

- **Secrets:** never write GitHub tokens, RunPod API keys, or SSH keys into committed files. If Tony's PAT is needed for `git push`, use it from env vars or a `.netrc` Tony provides; do not commit it.

  If any of the above conflicts with a session instruction, the agent stops and asks.

---

## 7. The MPS → CUDA transfer question `[A]`

**Numbers do not transfer numerically. Rankings often do, but not always.**

Concrete differences:

- **MPS anchor (exp 0001_baseline_repro):** val_bpb 2.5212, 200 steps, VAL_TOKENS=16384. This is the local smoke baseline.

- **CUDA upstream baseline (8×H100, 10 min, full eval):** ~1.2244. Different vocab/seq config, different precision paths, different step count, different eval batch.

- A 10-min run on 1× 5090 sits between these — different again.

  What this means for the agent on cloud:

- **Don't compare CUDA val_bpb against the MPS anchor.** They're different distributions.

- **Establish a CUDA anchor first.** Run the canonical config under cloud conditions (`MAX_WALLCLOCK_SECONDS=300` on 5090, or `MAX_WALLCLOCK_SECONDS=600` on 8×H100). That's the new comparison point.

- **Validate top MPS candidates, don't blindly explore.** The point of cloud time is to confirm whether the current MPS winner (see `winners/` for the latest, or journal "Current threads" for the active candidate) actually beats canonical at full step counts.

- **Watch for regime-dependent effects.** On MPS we see ~200 steps. On 8×H100 in 10 min you'll get thousands. Some MPS wins are early-training tricks (`[transfer:low]` in `results.tsv`) that may not survive longer schedules.

---

## 8. Phase 1 vs Phase 2 `[A]`

| Phase | GPU              | Wallclock | Cost/run   | What to do                                                   |
| ----- | ---------------- | --------- | ---------- | ------------------------------------------------------------ |
| 1     | 1× 5090/A40/4090 | 5–10 min  | $0.10–0.20 | Establish CUDA baseline. Verify top 1–3 MPS winners reproduce. ~10 runs OK. |
| 2     | 8× H100 SXM      | 10 min    | ~$4        | Final submission run(s) on the best confirmed config. 1–3 runs max. |

**Phase 1 first principles:**

1. Run canonical on CUDA → get the CUDA anchor (`scripts/runpod/regression_sentinel.sh` does this).

2. Run the current MPS winner's config on CUDA → confirm Δ vs CUDA anchor. (See `winners/` for the latest promoted winner, and journal "Current threads" for the active candidate that's not yet a winner.)

3. If Δ holds, that's your candidate for Phase 2.

4. If Δ collapses, dig in (likely an early-training artifact — see §7). Try other top-of-leaderboard configs from `winners/` or `results.tsv` rows tagged `[transfer:high]`.

   **Phase 2 first principles:**

5. Don't deploy Phase 2 pod until Phase 1 has produced a clear winner.

6. Once allocated, run jobs back-to-back. Don't stop the pod between runs.

7. Each run is a vote; spend them on confirming, not exploring.

8. Save logs immediately after each run (`scp runpod:/workspace/.../run.log .`) — terminating the pod erases the volume.

---

## 9. Standard cloud experiment loop `[A]`

Adapt the MPS loop with three changes:

- `MAX_WALLCLOCK_SECONDS` becomes the time-budget knob (not `ITERATIONS`). For Phase 1 use 300; for Phase 2 use 600.
- All launches go through `tmux` over SSH (§5).
- Compare against the **CUDA anchor**, not the MPS anchor.

```bash
# 1. Plan locally — pick a slug describing the experiment
./new_experiment.sh <slug> [<parent_id>]   # creates experiments/NNNN_<slug>/

# 2. Edit env.sh (set MAX_WALLCLOCK_SECONDS) and plan.md locally, commit, push

# 3. Pull on pod
ssh runpod 'cd /workspace/parameter-golf-ssm && git pull'

# 4. Preflight (refuses to launch if env.sh / plan.md / GPU / disk look wrong)
ssh runpod 'cd /workspace/parameter-golf-ssm && source .venv/bin/activate && bash scripts/runpod/preflight.sh experiments/NNNN_<slug>'

# 5. Launch on pod via tmux
#    - Single-GPU:
ssh runpod 'cd /workspace/parameter-golf-ssm/experiments/NNNN_<slug> && tmux new -d -s expNNNN "source ../../.venv/bin/activate && ../../run_experiment.sh"'
#    - 8×H100 multi-GPU (wraps `python -m torch.distributed.run`):
ssh runpod 'cd /workspace/parameter-golf-ssm && source .venv/bin/activate && tmux new -d -s expNNNN "bash scripts/runpod/launch_h100.sh experiments/NNNN_<slug>"'

# 6. Wait/check (don't poll faster than once every 30 sec)
ssh runpod 'tmux capture-pane -t expNNNN -p | tail -20'

# 7. When complete, commit results from pod
ssh runpod 'cd /workspace/parameter-golf-ssm && git add experiments/NNNN_*/result.json results.tsv && git commit -m "exp NNNN result" && git push'

# 8. Pull on Mac to sync results.tsv
git pull
```

The harness (`run_experiment.sh`, `await_steps.sh`) works as-is — it doesn't care whether the GPU is MPS or CUDA. The only env vars to override are `MAX_WALLCLOCK_SECONDS` and possibly `ITERATIONS` / `WARMDOWN_ITERS`.

---

## 10. Cost guardrails `[A]`

The agent runs `date` at session start and at natural breaks. Track approximate cost mentally:

- 1× 5090: ~$0.99/hr → ~$0.02 per minute.

- 8× H100 SXM: ~$24/hr → ~$0.40 per minute.

  **Stop and ask Tony before continuing if:**

- Session time on cloud passes 2 hours (cumulative).

- Cumulative cloud cost passes $10 (mental estimate).

- An experiment has been running >20 min without producing the expected first-10 step output (likely hung).

- The pod has been idle (no commands sent) for >30 min — possibly Tony forgot to stop it.

  **Always log session cost estimate in journal at wrap-time.**

---

## 11. State sync between Mac and pod `[A]`

The pod has the same fork as Tony's Mac, but they drift during a session. Discipline:

- Agent makes code changes on **Mac**, commits, pushes, then `ssh runpod 'git pull'` before running.
- Agent does NOT edit code directly on the pod (drift risk; pod gets terminated, work lost).
- Experiment outputs (`experiments/NNNN/run.log`, `result.json`, `results.tsv` row) get committed **from the pod**, pushed, then Tony's Mac pulls.
- This means there are two `git push` / `git pull` events per experiment. Worth it for state consistency.

---

## 12. Wrapping a cloud session `[A]`

Before disconnecting (or before Tony goes to bed):

```bash
# 1. Confirm nothing still running
ssh runpod 'tmux ls'   # should be empty, OR explicitly note what's still running

# 2. Confirm everything is committed and pushed
ssh runpod 'cd /workspace/parameter-golf-ssm && git status && git log --oneline -5'

# 3. Pull final state to Mac
git pull

# 4. Write summary to summaries/ on Mac (the wrap-session skill)

# 5. Tell Tony the pod state explicitly: "Pod still running, X experiments queued"
#    OR "Pod is idle, you can stop it from the dashboard"
```

The agent does not stop the pod. Tony stops it after reading the wrap.

---

## 13. Pre-flight checklist for Tony before handing off to agent `[H]`

Before telling the agent "go run experiments on cloud":

- [ ] Pod is `Running` and SSH-reachable from Mac (`ssh runpod 'echo ok'`)
- [ ] `~/.ssh/config` has `Host runpod` entry
- [ ] Repo is set up on pod (§4 done)
- [ ] CUDA anchor experiment has run (manually or by agent)
- [ ] Branch is set: agent works on `autoresearch-ssm`, you stay on `main`
- [ ] Budget communicated to agent: e.g. "you have until 2 AM" or "5 experiments max"
- [ ] Phone alarm set to check on pod

---

## 14. What this doc does not cover

- **Multi-pod orchestration.** One pod at a time.

- **Network volume.** Skipped due to GPU inventory restrictions when the filter is on.

- **Distributed training across multiple pods.** Out of scope for parameter-golf.

- **Custom Docker images.** Use the Parameter Golf template as-is.

  If any of these become relevant, this doc gets a new section.
