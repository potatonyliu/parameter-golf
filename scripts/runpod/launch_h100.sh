#!/usr/bin/env bash
# launch_h100.sh — multi-GPU launcher that wraps torchrun and reuses the
# metric-parsing/result.json/results.tsv path from run_experiment.sh.
#
# Why a separate launcher (instead of patching run_experiment.sh):
#   - run_experiment.sh's `python train_gpt.py` works for MPS/single-GPU CUDA
#     and we don't want to risk breaking the local MPS workflow.
#   - torchrun sets RANK/LOCAL_RANK/WORLD_SIZE (which train_gpt.py needs) and
#     handles process group bring-up cleanly. Easier to wrap than embed.
#
# Usage:
#   bash scripts/runpod/launch_h100.sh experiments/NNNN_<slug> [--nproc 8]
#
# This script:
#   1. Validates the experiment dir + plan.md (same checks as run_experiment.sh).
#   2. Sources env.sh.
#   3. Runs `torchrun --standalone --nproc_per_node=N train_gpt.py`.
#   4. Parses metrics from logs/${RUN_ID}.txt and writes result.json.
#   5. Appends one row to repo-root results.tsv.
#
# Defaults to --nproc 8 (8×H100). Override for 1×H100, 2×H100, etc.

set -uo pipefail

EXP_DIR="${1:-}"
NPROC=8

# Parse remaining args.
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc) NPROC="$2"; shift 2 ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${EXP_DIR}" || ! -d "${EXP_DIR}" ]]; then
  echo "Usage: $0 <experiment_dir> [--nproc N]" >&2
  exit 1
fi

# Constraint from train_gpt.py: WORLD_SIZE must divide 8 (so grad_accum_steps stays integral).
if (( 8 % NPROC != 0 )); then
  echo "ERROR: --nproc=${NPROC} does not divide 8. Use 1, 2, 4, or 8." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP_DIR_ABS="$(cd "${EXP_DIR}" && pwd)"

cd "${EXP_DIR_ABS}"

# Same plan.md / file checks as run_experiment.sh — unsafe to skip.
if [[ ! -f result.json || ! -f train_gpt.py || ! -f env.sh ]]; then
  echo "Error: ${EXP_DIR} missing result.json, train_gpt.py, or env.sh" >&2
  exit 1
fi
if [[ ! -f plan.md ]]; then
  echo "Error: plan.md missing." >&2
  exit 1
fi
UNFILLED=0
for pat in '<!-- What are you actually asking' '<!-- Predicted direction' '<!-- Exact env vars' '<!-- What outcome would falsify'; do
  grep -qF "$pat" plan.md && UNFILLED=$((UNFILLED + 1))
done
if (( UNFILLED > 0 )); then
  echo "Error: plan.md has ${UNFILLED} unfilled template section(s)." >&2
  exit 1
fi

# shellcheck source=/dev/null
source env.sh

if [[ -z "${RUN_ID:-}" ]]; then
  echo "Error: RUN_ID not set after sourcing env.sh" >&2
  exit 1
fi

EXPERIMENT_ID=$(basename "${EXP_DIR_ABS}")

STRUCTURED_LOG="logs/${RUN_ID}.txt"

echo "Launching ${EXPERIMENT_ID} via torchrun --nproc_per_node=${NPROC}..."
echo "  ITERATIONS=${ITERATIONS:-?}  MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-?}"
echo "  expected world_size=${NPROC}, grad_accum=$((8 / NPROC))"
echo ""

# Use `python -m torch.distributed.run` instead of `.venv/bin/torchrun`.
# Why: setup_pod.sh creates the venv with --system-site-packages so torch
# comes from the image's system install — but the torchrun console script
# is generated only inside the env where torch was pip-installed, so it
# typically lives at /usr/local/bin/torchrun (system PATH), NOT in
# .venv/bin/torchrun. Calling `python -m torch.distributed.run` works
# regardless and uses the venv's python (which sees torch via system-site-packages).
set +e
"${REPO_ROOT}/.venv/bin/python" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC}" \
  train_gpt.py \
  > run.log 2>&1
RUN_RC=$?
set -e

if [[ ! -f "${STRUCTURED_LOG}" ]]; then
  echo "Warning: structured log not found at ${STRUCTURED_LOG} — parsing run.log as fallback." >&2
  STRUCTURED_LOG="run.log"
fi

# -------- metric parsing (mirrors run_experiment.sh) --------
VAL_BPB_PRE=$(grep -oE 'step:[0-9]+/[0-9]+ val_loss:[0-9.a-z]+ val_bpb:[0-9.a-z]+' "$STRUCTURED_LOG" | tail -1 | grep -oE 'val_bpb:[0-9.a-z]+' | cut -d: -f2 || echo "")
VAL_LOSS_PRE=$(grep -oE 'step:[0-9]+/[0-9]+ val_loss:[0-9.a-z]+' "$STRUCTURED_LOG" | tail -1 | grep -oE 'val_loss:[0-9.a-z]+' | cut -d: -f2 || echo "")
VAL_BPB_POST=$(grep -oE 'final_int8_zlib_roundtrip_exact val_loss:[0-9.a-z]+ val_bpb:[0-9.a-z]+' "$STRUCTURED_LOG" | tail -1 | grep -oE 'val_bpb:[0-9.a-z]+' | cut -d: -f2 || echo "")
VAL_LOSS_POST=$(grep -oE 'final_int8_zlib_roundtrip_exact val_loss:[0-9.a-z]+' "$STRUCTURED_LOG" | tail -1 | grep -oE 'val_loss:[0-9.a-z]+' | cut -d: -f2 || echo "")
if [[ -z "$VAL_BPB_POST" ]]; then
  VAL_BPB_POST=$(grep -oE 'final_int8_zlib_roundtrip val_loss:[0-9.a-z]+ val_bpb:[0-9.a-z]+' "$STRUCTURED_LOG" | tail -1 | grep -oE 'val_bpb:[0-9.a-z]+' | cut -d: -f2 || echo "")
  VAL_LOSS_POST=$(grep -oE 'final_int8_zlib_roundtrip val_loss:[0-9.a-z]+' "$STRUCTURED_LOG" | tail -1 | grep -oE 'val_loss:[0-9.a-z]+' | cut -d: -f2 || echo "")
fi
STEP_AVG_MS=$(grep -oE 'step:[0-9]+/[0-9]+ train_loss:[0-9.a-z]+ train_time:[0-9]+ms step_avg:[0-9.]+ms' "$STRUCTURED_LOG" | tail -1 | grep -oE 'step_avg:[0-9.]+' | cut -d: -f2 || echo "")
NUM_STEPS=$(grep -oE '^step:[0-9]+/[0-9]+ train_loss:' "$STRUCTURED_LOG" | tail -1 | grep -oE 'step:[0-9]+' | cut -d: -f2 || echo "")
ARTIFACT_BYTES=$(grep -oE 'Total submission size int8\+zlib: [0-9]+ bytes' "$STRUCTURED_LOG" | tail -1 | sed -E 's/.*: ([0-9]+) bytes/\1/' || echo "")
CODE_BYTES=$(grep -oE 'Code size: [0-9]+ bytes' "$STRUCTURED_LOG" | tail -1 | sed -E 's/.*: ([0-9]+) bytes/\1/' || echo "")
COMPRESSION_RATIO=$(grep -oE 'payload_ratio:[0-9.]+x' "$STRUCTURED_LOG" | tail -1 | sed -E 's/.*:([0-9.]+)x/\1/' || echo "")

HAS_NAN="false"
for v in "$VAL_BPB_PRE" "$VAL_BPB_POST" "$VAL_LOSS_PRE" "$VAL_LOSS_POST"; do
  if [[ "$v" == "nan" || "$v" == "inf" || "$v" == "-inf" ]]; then HAS_NAN="true"; fi
done
CRASHED="false"
if [[ -z "$VAL_BPB_POST" || $RUN_RC -ne 0 || "$HAS_NAN" == "true" ]]; then CRASHED="true"; fi

QUANT_TAX=""
if [[ -n "$VAL_BPB_PRE" && -n "$VAL_BPB_POST" && "$HAS_NAN" == "false" ]]; then
  QUANT_TAX=$(python3 -c "print(round(${VAL_BPB_POST} - ${VAL_BPB_PRE}, 6))")
fi

SIZE_VIOLATION="false"
ARTIFACT_MB=""
if [[ -n "$ARTIFACT_BYTES" ]]; then
  ARTIFACT_MB=$(python3 -c "print(round(${ARTIFACT_BYTES} / 1_000_000, 3))")
  if (( ARTIFACT_BYTES > 16000000 )); then SIZE_VIOLATION="true"; fi
fi

VBP_PRE="$VAL_BPB_PRE" VBP_POST="$VAL_BPB_POST" VLP_PRE="$VAL_LOSS_PRE" VLP_POST="$VAL_LOSS_POST" \
  QT="$QUANT_TAX" SAM="$STEP_AVG_MS" NS="$NUM_STEPS" AB="$ARTIFACT_BYTES" AM="$ARTIFACT_MB" \
  CB="$CODE_BYTES" CR="$COMPRESSION_RATIO" CRASHED="$CRASHED" SV="$SIZE_VIOLATION" \
  HN="$HAS_NAN" RC="$RUN_RC" NPROC_STR="$NPROC" \
python3 <<'PYEOF'
import json, os
def num(s):
    if s in ("", None): return None
    if s in ("nan", "inf", "-inf"): return s
    try: return float(s)
    except ValueError: return None
with open("result.json") as f:
    r = json.load(f)
r["metrics"] = {
    "val_bpb_pre_quant":  num(os.environ["VBP_PRE"]),
    "val_bpb_post_quant": num(os.environ["VBP_POST"]),
    "val_loss_pre_quant":  num(os.environ["VLP_PRE"]),
    "val_loss_post_quant": num(os.environ["VLP_POST"]),
    "quant_tax":          num(os.environ["QT"]),
    "step_avg_ms":        num(os.environ["SAM"]),
    "num_steps":          num(os.environ["NS"]),
    "artifact_bytes":     num(os.environ["AB"]),
    "artifact_mb":        num(os.environ["AM"]),
    "code_bytes":         num(os.environ["CB"]),
    "compression_ratio":  num(os.environ["CR"]),
}
r["flags"] = {
    "crashed": os.environ["CRASHED"] == "true",
    "size_violation": os.environ["SV"] == "true",
    "has_nan": os.environ["HN"] == "true",
    "exit_code": int(os.environ["RC"]),
    "device": "cuda_h100_x" + os.environ.get("NPROC_STR", ""),
}
with open("result.json", "w") as f:
    json.dump(r, f, indent=2)
PYEOF

# Append to results.tsv.
TSV="${REPO_ROOT}/results.tsv"
if [[ ! -f "$TSV" ]]; then
  printf "id\tparent\tval_bpb\tpre_quant_bpb\tquant_tax\tartifact_mb\tstep_avg_ms\tcrashed\tsize_violation\tstatus\tdescription\n" > "$TSV"
fi
PARENT=$(python3 -c "import json; print(json.load(open('result.json'))['parent'])")
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tTODO\tH100x%d\n" \
  "$EXPERIMENT_ID" "$PARENT" "${VAL_BPB_POST:-null}" "${VAL_BPB_PRE:-null}" "${QUANT_TAX:-null}" "${ARTIFACT_MB:-null}" "${STEP_AVG_MS:-null}" "$CRASHED" "$SIZE_VIOLATION" "$NPROC" \
  >> "$TSV"

cat <<EOF

=== ${EXPERIMENT_ID} (H100 x${NPROC}) ===
  val_bpb_post_quant: ${VAL_BPB_POST:-null}
  val_bpb_pre_quant:  ${VAL_BPB_PRE:-null}
  quant_tax:          ${QUANT_TAX:-null}
  step_avg_ms:        ${STEP_AVG_MS:-null}
  num_steps:          ${NUM_STEPS:-null}
  artifact_mb:        ${ARTIFACT_MB:-null}
  crashed:            ${CRASHED}
  has_nan:            ${HAS_NAN}
  size_violation:     ${SIZE_VIOLATION}
  exit_code:          ${RUN_RC}

--- First 10 training steps ---
EOF
grep -E "^step:[0-9]+/[0-9]+ train_loss:" "${STRUCTURED_LOG}" | head -10

echo ""
echo "Structured log: ${STRUCTURED_LOG}"
echo "Raw log:        run.log"
echo ""
echo "REMINDER: STOP the pod when done with the cascade. 8×H100 ≈ \$24/hr."
