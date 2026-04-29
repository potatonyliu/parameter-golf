#!/usr/bin/env bash
# setup_pod.sh — first-time setup on a fresh RunPod pod.
#
# What it does (idempotent — safe to re-run):
#   1. Verifies we're on /workspace (NOT /root — container disk gets wiped on stop).
#   2. Clones the repo if missing, otherwise pulls latest.
#   3. Creates .venv (Python 3.11+) inside /workspace/parameter-golf-ssm/.
#   4. Installs requirements-cuda.txt — assumes the image ships PyTorch + CUDA.
#   5. Downloads the FineWeb sp1024 dataset + tokenizer to data/.
#   6. Imports torch, prints torch + CUDA + GPU info as a smoke test.
#   7. Writes /workspace/.pod_setup_complete with a timestamp + git SHA.
#
# What it does NOT do:
#   - Does NOT launch any training. That's preflight.sh + run_experiment.sh.
#   - Does NOT push to git. That's a git operation the agent does explicitly.
#   - Does NOT install torch. The PG RunPod image ships torch + CUDA. If you
#     are on a non-PG image, install torch first per requirements-cuda.txt.
#
# Usage:
#   cd /workspace
#   bash parameter-golf-ssm/scripts/runpod/setup_pod.sh \
#     [--repo-url https://github.com/potatonyliu/parameter-golf.git] \
#     [--branch autoresearch-ssm]
#
# Prerequisites:
#   - You are inside tmux. (CHECK: $TMUX should be set.)
#   - Network volume is mounted at /workspace.
#   - GitHub access works (SSH key forwarded OR HTTPS public OR PAT in env).

set -euo pipefail

# -------- defaults --------
REPO_URL="${REPO_URL:-https://github.com/potatonyliu/parameter-golf.git}"
BRANCH="${BRANCH:-autoresearch-ssm}"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR_NAME="parameter-golf-ssm"

# -------- arg parsing --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --branch)   BRANCH="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,30p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

REPO_DIR="${WORKSPACE}/${REPO_DIR_NAME}"

# -------- safety: refuse outside /workspace --------
if [[ "${WORKSPACE}" != "/workspace" ]]; then
  echo "WARNING: WORKSPACE=${WORKSPACE} is not /workspace." >&2
  echo "On RunPod, only /workspace is on the network volume — anything else" >&2
  echo "(/root, /tmp) is on container disk and gets WIPED on pod stop." >&2
  read -r -p "Continue anyway? [y/N] " ans
  [[ "${ans:-N}" =~ ^[Yy]$ ]] || exit 1
fi

# -------- safety: warn if not in tmux --------
if [[ -z "${TMUX:-}" ]]; then
  echo "WARNING: not inside tmux." >&2
  echo "Setup downloads ~2 GB of data; SSH disconnect would restart it from scratch." >&2
  echo "Recommended: 'tmux new -s setup' then re-run this script." >&2
  read -r -p "Continue without tmux? [y/N] " ans
  [[ "${ans:-N}" =~ ^[Yy]$ ]] || exit 1
fi

cd "${WORKSPACE}"

# -------- step 1: clone or pull --------
if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[1/6] Cloning ${REPO_URL} into ${REPO_DIR}..."
  git clone "${REPO_URL}" "${REPO_DIR_NAME}"
  cd "${REPO_DIR}"
  git checkout "${BRANCH}"
else
  echo "[1/6] Repo exists at ${REPO_DIR}; fetching + checking out ${BRANCH}..."
  cd "${REPO_DIR}"
  git fetch --all --prune
  # Don't blow away local changes silently.
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "WARNING: working tree has uncommitted changes:" >&2
    git status --short >&2
    echo "Refusing to checkout ${BRANCH}. Stash or commit first." >&2
    exit 1
  fi
  git checkout "${BRANCH}"
  git pull --ff-only
fi

GIT_SHA=$(git rev-parse --short HEAD)
echo "    on branch ${BRANCH} @ ${GIT_SHA}"

# -------- step 2: create venv --------
# Critical: --system-site-packages so the venv inherits the system-installed
# PyTorch + CUDA from the RunPod image. requirements-cuda.txt does NOT pin torch
# (the image's CUDA build is what we want); without --system-site-packages the
# venv is empty and torch import fails.
if [[ ! -d ".venv" ]]; then
  echo "[2/6] Creating .venv with --system-site-packages (inherits image's torch+CUDA)..."
  python3 -m venv .venv --system-site-packages
else
  # If a .venv exists without --system-site-packages, torch import will fail
  # later — flag it loudly. Don't auto-recreate (might destroy state).
  if [[ ! -f .venv/pyvenv.cfg ]] || ! grep -qE "^include-system-site-packages = true" .venv/pyvenv.cfg; then
    echo "[2/6] WARNING: .venv exists but DOES NOT inherit system site packages." >&2
    echo "         If torch import fails later, run:  rm -rf .venv && rerun this script" >&2
  else
    echo "[2/6] .venv exists with system-site-packages; reusing."
  fi
fi

# Activate for the rest of this script.
# shellcheck disable=SC1091
source .venv/bin/activate

# -------- step 3: install requirements --------
echo "[3/6] Upgrading pip + installing requirements-cuda.txt..."
pip install --upgrade pip wheel --quiet
pip install -r requirements-cuda.txt --quiet

# Verify torch + CUDA. If torch is missing (non-PG image), bail with a helpful message.
python3 - <<'PYEOF'
import sys
try:
    import torch
except ImportError:
    print("ERROR: torch is not installed. The PG RunPod image normally ships it preinstalled.", file=sys.stderr)
    print("       If you are on a non-PG image, install torch manually:", file=sys.stderr)
    print("         pip install torch --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
    sys.exit(1)
print(f"    torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    cuda devices: {torch.cuda.device_count()}, name[0]: {torch.cuda.get_device_name(0)}")
    print(f"    vram[0]: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GiB")
else:
    print("    WARNING: CUDA not available — this pod has no GPU or driver mismatch.", file=sys.stderr)
PYEOF

# -------- step 4: data + tokenizer --------
DATA_DIR="data/datasets/fineweb10B_sp1024"
TOKENIZER="data/tokenizers/fineweb_1024_bpe.model"
EXPECTED_TRAIN_SHARDS=80   # default of cached_challenge_fineweb.py

if [[ -f "${TOKENIZER}" && -d "${DATA_DIR}" && \
      $(ls "${DATA_DIR}"/fineweb_train_*.bin 2>/dev/null | wc -l) -ge ${EXPECTED_TRAIN_SHARDS} ]]; then
  echo "[4/6] Data + tokenizer already present at ${DATA_DIR} (>=${EXPECTED_TRAIN_SHARDS} shards). Skipping download."
else
  echo "[4/6] Downloading FineWeb sp1024 (8B tokens / 80 shards, ~2 GB)..."
  python3 data/cached_challenge_fineweb.py --variant sp1024
fi

# -------- step 5: smoke import + GPU sanity --------
echo "[5/6] Smoke importing the project (CUDA path)..."
python3 - <<'PYEOF'
import sys, os
# Smoke: import sentencepiece, load the tokenizer, allocate a small CUDA tensor.
import sentencepiece as spm
import torch
sp = spm.SentencePieceProcessor(model_file="data/tokenizers/fineweb_1024_bpe.model")
assert sp.vocab_size() == 1024, f"unexpected vocab size {sp.vocab_size()}"
if torch.cuda.is_available():
    x = torch.zeros(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T  # forces a kernel launch
    torch.cuda.synchronize()
    print(f"    smoke: tokenizer ok, cuda matmul ok, peak vram: {torch.cuda.max_memory_allocated() // (1024**2)} MiB")
else:
    print("    smoke: tokenizer ok, but no CUDA — anything beyond setup will fail.", file=sys.stderr)
    sys.exit(1)
PYEOF

# -------- step 6: marker file --------
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > "${WORKSPACE}/.pod_setup_complete" <<EOF
setup_pod.sh completed at ${TS}
repo: ${REPO_DIR}
branch: ${BRANCH}
sha: ${GIT_SHA}
EOF
echo "[6/6] Marker written: ${WORKSPACE}/.pod_setup_complete"
echo ""
echo "Setup complete."
echo "  Activate venv:  source ${REPO_DIR}/.venv/bin/activate"
echo "  Next step:      bash scripts/runpod/regression_sentinel.sh"
echo "  Then:           bash scripts/runpod/preflight.sh experiments/<NNNN_slug>"
