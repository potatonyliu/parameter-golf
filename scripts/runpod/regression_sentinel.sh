#!/usr/bin/env bash
# regression_sentinel.sh — first run on every fresh pod.
#
# Reproduces 0001_baseline_repro on CUDA and compares val_bpb to the MPS
# anchor (val_bpb 2.5212, 6.907 MB). Tolerance is wider than MPS-vs-MPS
# because CUDA and MPS bf16 numerics diverge — we expect drift, but it
# should be small and bounded.
#
# Why this matters:
#   - Catches any device-path regression before novel SSM work.
#   - Establishes the CUDA-vs-MPS numeric baseline so future Δ comparisons
#     don't conflate "experiment effect" with "MPS→CUDA drift."
#   - Cheap on 4090 (~1-2 min for ITERATIONS=200, full eval ~30s).
#
# Usage:
#   bash scripts/runpod/regression_sentinel.sh
#
# Exit codes:
#   0 = within tolerance (good)
#   1 = drift exceeded (investigate before proceeding)
#   2 = setup error / can't run

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

# MPS anchor: 0001_baseline_repro from results.tsv (val_bpb 2.5212, 6.907 MB).
ANCHOR_BPB=2.5212
ANCHOR_MB=6.907
# CUDA-vs-MPS bf16 numeric drift tolerance. ±0.05 BPB on a 200-step canonical
# baseline is conservative — typical observed drift is ±0.01 BPB. Widen this
# only with a documented reason; tightening is fine.
TOL_BPB=0.05
TOL_MB=0.1

EXP_NAME="${RUN_ID_OVERRIDE:-regression_check_cuda}"
EXP_SLUG="${EXP_NAME}"

# Find next NNNN.
NEXT_ID=$(ls -d experiments/[0-9][0-9][0-9][0-9]_* 2>/dev/null | sort | tail -1 | sed -E 's|experiments/([0-9]{4})_.*|\1|')
NEXT_ID=$(printf "%04d" $((10#${NEXT_ID:-0} + 1)))
EXP_DIR="experiments/${NEXT_ID}_${EXP_SLUG}"

if [[ -d "$EXP_DIR" ]]; then
  echo "ERROR: $EXP_DIR already exists. Pick a different RUN_ID_OVERRIDE." >&2
  exit 2
fi

echo "Creating ${EXP_DIR} (canonical defaults)..."
./new_experiment.sh "${EXP_SLUG}" >/dev/null

# Override env.sh to set a wallclock cap (preflight requires it on RunPod).
# We KEEP canonical defaults otherwise so this is a true regression repro.
# Note: setting MAX_WALLCLOCK_SECONDS to a small but nonzero value would
# trigger the wallclock branch of lr_mul, breaking the canonical schedule.
# We use a generous cap (1800s) that won't fire during a 200-step run but
# satisfies preflight's "must be set" rule.
cat >> "${EXP_DIR}/env.sh" <<'EOF'

# regression_sentinel.sh override: cap wallclock for RunPod safety. Cap is
# generous enough not to fire during the 200-step canonical run; the
# step-based warmdown branch of lr_mul still controls LR.
export MAX_WALLCLOCK_SECONDS=1800
EOF

# Fill plan.md so run_experiment.sh's plan-check passes.
cat > "${EXP_DIR}/plan.md" <<EOF
# Experiment ${NEXT_ID}_${EXP_SLUG}

Parent: canonical

## Question
Does this fresh CUDA pod bit-reproduce 0001_baseline_repro (MPS val_bpb 2.5212, 6.907 MB) within the CUDA-vs-MPS bf16 drift tolerance? Cheap insurance before novel SSM work.

## Hypothesis [LIKELY]
val_bpb falls in 2.5212 ± 0.05 (i.e. roughly 2.47 - 2.57). MPS and CUDA differ in bf16 reduction order, but on a 200-step canonical baseline the drift is typically <0.02 BPB. Artifact size should match within ±0.1 MB (no architecture changes — only numeric drift in stored weights, which int8-quantizes identically up to LSB noise).

## Change
None. Canonical env.sh from new_experiment.sh + MAX_WALLCLOCK_SECONDS cap for RunPod safety.

## Disconfirming
- val_bpb drifts > 0.05 from 2.5212 → CUDA path has a regression OR our MPS→CUDA drift assumption is wrong; investigate before novel work.
- Crash, OOM, NaN → CUDA-only failure mode (e.g. flash-SDP path); investigate.
- Artifact size differs by > 0.1 MB → quant export changed; harness drift.

## Notes from execution
Filled by regression_sentinel.sh after run.
EOF

# -------- launch --------
echo "Launching..."
cd "${EXP_DIR}"
../../run_experiment.sh 2>&1 | tee /tmp/regression_sentinel.log
RC=$?
cd "${REPO_ROOT}"

if (( RC != 0 )); then
  echo "ERROR: run_experiment.sh exited ${RC}. See ${EXP_DIR}/run.log" >&2
  exit 1
fi

# -------- compare --------
RESULT=$(grep -oE 'val_bpb_post_quant: [0-9.]+' "${EXP_DIR}/run.log" 2>/dev/null | tail -1 | awk '{print $2}')
if [[ -z "$RESULT" ]]; then
  # Fallback: parse result.json
  RESULT=$(python3 -c "import json; print(json.load(open('${EXP_DIR}/result.json'))['metrics']['val_bpb_post_quant'])" 2>/dev/null)
fi
ARTIFACT_MB=$(python3 -c "import json; print(json.load(open('${EXP_DIR}/result.json'))['metrics']['artifact_mb'])" 2>/dev/null)

if [[ -z "$RESULT" || "$RESULT" == "None" ]]; then
  echo "ERROR: couldn't read val_bpb_post_quant from ${EXP_DIR}." >&2
  exit 1
fi

DRIFT_BPB=$(python3 -c "print(round(abs(${RESULT} - ${ANCHOR_BPB}), 6))")
DRIFT_MB=$(python3 -c "print(round(abs(${ARTIFACT_MB} - ${ANCHOR_MB}), 6))")

echo ""
echo "=== Regression Sentinel ==="
echo "  anchor:    val_bpb=${ANCHOR_BPB}  artifact_mb=${ANCHOR_MB}  (MPS, 0001_baseline_repro)"
echo "  this run:  val_bpb=${RESULT}     artifact_mb=${ARTIFACT_MB}  (CUDA, ${EXP_DIR})"
echo "  drift:     |Δbpb|=${DRIFT_BPB} (tol ${TOL_BPB})    |Δmb|=${DRIFT_MB} (tol ${TOL_MB})"

PASS=$(python3 -c "print(1 if ${DRIFT_BPB} <= ${TOL_BPB} and ${DRIFT_MB} <= ${TOL_MB} else 0)")
if [[ "$PASS" == "1" ]]; then
  echo "  RESULT: PASS — CUDA path within tolerance, safe to proceed."
  echo ""
  echo "  Append the observed val_bpb to your CUDA baseline notes:"
  echo "    journal.md: \"CUDA regression sentinel: val_bpb=${RESULT}, drift=${DRIFT_BPB} from MPS anchor.\""
  exit 0
else
  echo "  RESULT: FAIL — drift exceeded tolerance. STOP and investigate." >&2
  echo "  Possible causes:" >&2
  echo "    - Different PyTorch/CUDA version than expected (check torch.__version__)" >&2
  echo "    - Flash-SDP / cudnn-SDP path differs from MPS math" >&2
  echo "    - tokenizer or vocab size mismatch (rare; setup_pod.sh would have caught)" >&2
  echo "    - bug in train_gpt.py that only fires on CUDA" >&2
  exit 1
fi
