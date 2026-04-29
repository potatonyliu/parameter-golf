# Experiment 0097_cloud_workflow_smoke

Parent: canonical

## Question
Does the documented Mac↔Pod RunPod workflow actually work end-to-end? Specifically: does `git add experiments/NNNN_<slug>` (after the gitignore fix in 12d384e) sync the env.sh / plan.md / train_gpt.py / result.json bidirectionally without silent gitignore drops? Does preflight on the pod accept the env.sh? Does run_experiment.sh launch and produce a result.json? Does the pod-side commit + push back to the fork actually work?

## Hypothesis [LIKELY]
Workflow runs to completion. ITERATIONS=50, MAX_WALLCLOCK_SECONDS=120 — should finish in <60s on a 5090 (135ms/step × 50 + ~5s eval). val_bpb will be high (no real training; this is a SYNC test) but flags should show no crash, no nan, no size violation, exit code 0. The result.json + results.tsv row appear on Mac after `git pull`.

## Change
None to architecture. env.sh only: ITERATIONS=200→50, MAX_WALLCLOCK_SECONDS=0→120 to keep cost trivial. Slug carries the workflow-test intent.

## Disconfirming
- `git add experiments/0097_cloud_workflow_smoke` on Mac silently no-ops (gitignore fix didn't take) → workflow doc still wrong.
- Pod's `git pull` doesn't see the new dir → push from Mac silently failed.
- Preflight rejects → some sanity check is wrong (e.g., the new MAX_WALLCLOCK_SECONDS handling).
- run_experiment.sh crashes / hangs → CUDA harness bug we missed.
- Pod's `git push` prompts for credentials and can't proceed → PAT/credential-helper not set up; the workflow is blocked.
- Mac's `git pull` doesn't bring the result back → pod commit didn't actually push.

## Notes from execution
Filled in real time as the workflow runs.
