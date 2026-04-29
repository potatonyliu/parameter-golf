#!/usr/bin/env bash
# preflight.sh — refuses to launch unless the pod looks safe to run.
#
# Run this BEFORE every `run_experiment.sh`. It catches the predictable
# RunPod failure modes that waste $$/hr:
#   - working from /root (wiped on stop) instead of /workspace
#   - not in tmux (SSH disconnect kills the run)
#   - GPU not visible / wrong type
#   - data/tokenizer missing
#   - env.sh missing MAX_WALLCLOCK_SECONDS (runaway-cost guard)
#   - disk near full (saves crash mid-run)
#   - no .pod_setup_complete marker (pod may not be set up yet)
#
# Usage:
#   bash scripts/runpod/preflight.sh experiments/0097_my_exp
#   # exits 0 if all checks pass, exits 1 with explanation otherwise.
#
# Override checks (use sparingly, document why):
#   ALLOW_NO_TMUX=1     skip tmux check (e.g., short interactive smoke)
#   ALLOW_ANY_GPU=1     skip VRAM / GPU type check
#   ALLOW_LONG_RUN=1    allow MAX_WALLCLOCK_SECONDS > 7200

set -uo pipefail

EXP_DIR="${1:-}"
if [[ -z "$EXP_DIR" || ! -d "$EXP_DIR" ]]; then
  echo "Usage: $0 <experiment_dir>" >&2
  echo "  e.g.  $0 experiments/0097_4090_regression_check" >&2
  exit 2
fi

FAIL=0
fail() { echo "  ✗ $1" >&2; FAIL=$((FAIL + 1)); }
ok()   { echo "  ✓ $1"; }

echo "preflight: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  experiment: $EXP_DIR"
echo ""

# -------- 1. /workspace check --------
PWD_REAL=$(pwd -P)
if [[ "${PWD_REAL}" != /workspace/* && "${PWD_REAL}" != /workspace ]]; then
  fail "cwd is ${PWD_REAL}, not under /workspace. /root etc. get WIPED on pod stop. cd /workspace/parameter-golf-ssm and re-run."
else
  ok "cwd under /workspace"
fi

# -------- 2. tmux check --------
if [[ -z "${TMUX:-}" ]]; then
  if [[ "${ALLOW_NO_TMUX:-0}" == "1" ]]; then
    ok "not in tmux (overridden by ALLOW_NO_TMUX=1)"
  else
    fail "not inside tmux. SSH disconnect = killed run = wasted GPU \$. Run 'tmux new -s work' first, OR set ALLOW_NO_TMUX=1 if interactive smoke."
  fi
else
  ok "tmux session: ${TMUX##*,}"
fi

# -------- 3. setup marker --------
if [[ -f /workspace/.pod_setup_complete ]]; then
  ok "setup marker present"
else
  fail "/workspace/.pod_setup_complete missing — has setup_pod.sh been run on this pod?"
fi

# -------- 4. .venv exists + active --------
if [[ ! -d ".venv" ]]; then
  fail ".venv not found. Run: bash scripts/runpod/setup_pod.sh"
elif [[ -z "${VIRTUAL_ENV:-}" ]]; then
  fail ".venv not activated. Run: source .venv/bin/activate"
elif [[ "${VIRTUAL_ENV}" != *"/parameter-golf-ssm/.venv" ]]; then
  fail "Wrong venv active (${VIRTUAL_ENV}). Activate the project venv: source .venv/bin/activate"
else
  ok ".venv active: ${VIRTUAL_ENV##*/}"
fi

# -------- 5. GPU visible --------
GPU_INFO=""
if command -v nvidia-smi >/dev/null 2>&1; then
  if GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null); then
    GPU_COUNT=$(echo "$GPU_INFO" | wc -l | tr -d ' ')
    GPU_NAME=$(echo "$GPU_INFO" | head -1 | awk -F',' '{print $1}' | xargs)
    GPU_MEM_MIB=$(echo "$GPU_INFO" | head -1 | awk -F',' '{print $2}' | grep -oE '[0-9]+' | head -1)
    ok "GPU x${GPU_COUNT}: ${GPU_NAME} (${GPU_MEM_MIB} MiB)"
  else
    fail "nvidia-smi failed — driver issue?"
  fi
else
  fail "nvidia-smi not found — this pod has no GPU?"
fi

# -------- 6. python torch.cuda --------
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  if ! python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    fail "torch.cuda.is_available() == False. Driver/PyTorch mismatch."
  else
    TORCH_VER=$(python3 -c 'import torch; print(torch.__version__)')
    ok "torch ${TORCH_VER}, cuda available"
  fi
fi

# -------- 7. data + tokenizer --------
DATA_GLOB="data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"
TOKENIZER="data/tokenizers/fineweb_1024_bpe.model"
TRAIN_SHARDS=$(ls $DATA_GLOB 2>/dev/null | wc -l | tr -d ' ')
if [[ ! -f "$TOKENIZER" ]]; then
  fail "tokenizer missing: $TOKENIZER (run setup_pod.sh)"
else
  ok "tokenizer present"
fi
if (( TRAIN_SHARDS < 10 )); then
  fail "only ${TRAIN_SHARDS} train shards present (expected 80). Run setup_pod.sh."
else
  ok "${TRAIN_SHARDS} train shards present"
fi

# -------- 8. env.sh sanity --------
ENV_SH="${EXP_DIR}/env.sh"
if [[ ! -f "$ENV_SH" ]]; then
  fail "env.sh missing in experiment dir: $ENV_SH"
else
  # Source in a subshell so we don't pollute the caller. We accumulate inner
  # failures so the outer FAIL count reflects how broken the env.sh is, not
  # just "1 thing wrong." The subshell exits with the inner fail count.
  set +e
  ENV_FAIL=$(
    set +u
    # shellcheck disable=SC1090
    source "$ENV_SH"
    set -u
    inner_fail=0

    if [[ -z "${MAX_WALLCLOCK_SECONDS:-}" ]]; then
      echo "  ✗ env.sh: MAX_WALLCLOCK_SECONDS unset. ALWAYS set this on RunPod — runaway runs cost \$\$." >&2
      inner_fail=$((inner_fail + 1))
    elif [[ "${MAX_WALLCLOCK_SECONDS}" == "0" ]]; then
      echo "  ✗ env.sh: MAX_WALLCLOCK_SECONDS=0 disables the wallclock cap. NOT acceptable on RunPod — set a real number (e.g., 1800 for 30 min)." >&2
      inner_fail=$((inner_fail + 1))
    elif (( MAX_WALLCLOCK_SECONDS > 7200 )) && [[ "${ALLOW_LONG_RUN:-0}" != "1" ]]; then
      echo "  ✗ env.sh: MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS} > 7200s (2h). Set ALLOW_LONG_RUN=1 if intentional." >&2
      inner_fail=$((inner_fail + 1))
    else
      echo "  ✓ env.sh MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
    fi

    if [[ -z "${ITERATIONS:-}" ]]; then
      echo "  ✗ env.sh: ITERATIONS unset" >&2
      inner_fail=$((inner_fail + 1))
    else
      echo "  ✓ env.sh ITERATIONS=${ITERATIONS}"
    fi

    if [[ "${VAL_TOKENS:-0}" == "0" ]]; then
      echo "  ⚠ env.sh: VAL_TOKENS=0 → full validation. On a single 4090 this can take 10+ min and is called twice (pre+post quant). Set VAL_TOKENS=16384 for a 16K-token cap unless the experiment specifically wants a full eval (e.g. promote candidate)." >&2
    fi

    if [[ -z "${RUN_ID:-}" ]]; then
      echo "  ✗ env.sh: RUN_ID unset" >&2
      inner_fail=$((inner_fail + 1))
    else
      echo "  ✓ env.sh RUN_ID=${RUN_ID}"
    fi
    # Print the count to stdout so the parent can capture it.
    echo "$inner_fail"
  )
  set -e
  # The last line of ENV_FAIL is the inner_fail count.
  ENV_FAIL_COUNT=$(echo "$ENV_FAIL" | tail -1)
  # Re-emit the messages (everything except the final count) on stderr, since
  # they were captured by the command substitution above. The check messages
  # were already written to stderr in the subshell, but command substitution
  # captures stdout; we don't actually need to re-emit anything because the
  # human-readable lines went to stderr inside the subshell. The count is the
  # only stdout line we capture.
  if [[ "$ENV_FAIL_COUNT" =~ ^[0-9]+$ ]] && (( ENV_FAIL_COUNT > 0 )); then
    FAIL=$((FAIL + ENV_FAIL_COUNT))
  fi
fi

# -------- 9. plan.md filled --------
PLAN="${EXP_DIR}/plan.md"
if [[ ! -f "$PLAN" ]]; then
  fail "plan.md missing in $EXP_DIR"
else
  unfilled=0
  for pat in '<!-- What are you actually asking' '<!-- Predicted direction' '<!-- Exact env vars' '<!-- What outcome would falsify'; do
    grep -qF "$pat" "$PLAN" && unfilled=$((unfilled + 1))
  done
  if (( unfilled > 0 )); then
    fail "plan.md has ${unfilled} unfilled template sections — fill Question/Hypothesis/Change/Disconfirming."
  else
    ok "plan.md filled"
  fi
fi

# -------- 10. disk space --------
# /workspace network volume; warn if <10GiB free, fail if <2GiB.
AVAIL_KIB=$(df -P /workspace 2>/dev/null | awk 'NR==2 {print $4}')
if [[ -z "$AVAIL_KIB" ]]; then
  fail "couldn't read /workspace disk usage"
else
  AVAIL_GIB=$(( AVAIL_KIB / 1024 / 1024 ))
  if (( AVAIL_GIB < 2 )); then
    fail "/workspace has <2 GiB free (${AVAIL_GIB} GiB). Run will likely crash on log/checkpoint write. Free space first."
  elif (( AVAIL_GIB < 10 )); then
    echo "  ⚠ /workspace has only ${AVAIL_GIB} GiB free — OK for one run, prune logs soon."
  else
    ok "/workspace ${AVAIL_GIB} GiB free"
  fi
fi

echo ""
if (( FAIL > 0 )); then
  echo "preflight FAILED with ${FAIL} issue(s). Fix before launching." >&2
  exit 1
fi
echo "preflight OK. Safe to launch:"
echo "  cd ${EXP_DIR} && ../../run_experiment.sh"
exit 0
