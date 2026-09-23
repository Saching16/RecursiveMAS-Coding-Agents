#!/usr/bin/env bash
#
# Gate 0.1 — isolated environment on a GPU box (PLAN.md §5.0.1, §1.2).
#
# Gate 0.1 is "done" when both import checks below print ok AND the exact
# versions are written into a run note. This script does both, then probes the
# things that differ between vendors.
#
# One script, two backends, because the procedure is identical apart from how
# torch is installed — separate files would just let the import checks and the
# run-note format drift apart.
#
#   ROCm: requirements.txt pins `torch==2.9.0`, which resolves to the default
#         PyPI wheel with no ROCm support. The ROCm build must be installed
#         FIRST so it satisfies that pin (PEP 440 `==2.9.0` matches the local
#         version `2.9.0+rocm*`).
#   CUDA: the default PyPI wheel is already CUDA-enabled on Linux, so ordering
#         does not matter — but that wheel is built against one CUDA runtime, so
#         a box with an older driver still needs `--cuda` to pick a matching
#         index.
#
# Either way the script re-checks torch after EVERY install step and fails
# loudly if the backend changed. A silent swap to a CPU wheel would produce a
# run that looks like it worked while quietly invalidating every measurement —
# the worst failure mode available here.
#
# Never "fixes" a resolver conflict by editing requirements.txt (PLAN §5.0.1).
#
# Usage:
#   ./gate0_setup.sh [--backend auto|cuda|rocm] [--force] [--venv-path DIR]
#                    [--rocm 6.4] [--cuda cu124] [--python python3.12]
#
#   --backend    which accelerator to set up; default: autodetect
#   --force      recreate the venv if it exists (destructive to the venv only)
#   --venv-path  where to build the venv; default: <repo>/.venv. Point this at
#                LOCAL disk if the repo lives on a network volume — see below
#   --rocm       ROCm minor series for the wheel index (ROCm only)
#   --cuda       CUDA wheel tag, e.g. cu121/cu124/cu128 (CUDA only; omit to use
#                the default PyPI wheel, which is correct on most modern boxes)
#   --python     interpreter to build the venv from; default: autodetect >=3.11,<4.0
#
# Never build the venv on a network volume. Measured on a RunPod MooseFS mount:
# 387 MB/s sequential (fine for model weights — 5.8 GB landed in 15 s) but
# metadata-bound work crawls, and `pip install --upgrade pip` had not finished
# after several minutes. Local overlay on the same box: 14.1 GB/s, ~36x faster.
# The script warns when the venv path looks networked, but it cannot move it for
# you. Put code + venv on local disk and keep only weights on the volume.

set -euo pipefail

BACKEND="auto"
FORCE=0
ROCM_VER=""
CUDA_TAG=""
PYTHON_BIN=""
VENV_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="${2:?--backend needs auto|cuda|rocm}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --venv-path) VENV_PATH="${2:?--venv-path needs a directory}"; shift 2 ;;
    --rocm) ROCM_VER="${2:?--rocm needs a value, e.g. 6.4}"; shift 2 ;;
    --cuda) CUDA_TAG="${2:?--cuda needs a value, e.g. cu124}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?--python needs a value}"; shift 2 ;;
    -h|--help) sed -n '3,47p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DEEPAGENTS_PKG="$REPO_ROOT/../deepagents/libs/deepagents"
VENV="${VENV_PATH:-$REPO_ROOT/.venv}"
RUNS_DIR="$SCRIPT_DIR/../runs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NOTE="$RUNS_DIR/gate0-$STAMP.md"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   ok    %s\n' "$*"; }
warn() { printf '   WARN  %s\n' "$*"; }
die()  { printf '\n\033[1mFAILED:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
say "Preflight"

[[ -f "$REPO_ROOT/requirements.txt" ]] || die "no requirements.txt at $REPO_ROOT — run this from the RecursiveMAS checkout"
[[ -f "$REPO_ROOT/modeling.py" ]]      || die "no modeling.py at $REPO_ROOT — wrong repo root"
ok "repo root: $REPO_ROOT"

if [[ ! -d "$DEEPAGENTS_PKG" ]]; then
  die "Deep Agents package not found at $DEEPAGENTS_PKG
     PLAN §1 expects the clone symlinked at ../deepagents. On a fresh pod:
       git clone https://github.com/langchain-ai/deepagents \"\$(dirname \"$REPO_ROOT\")/deepagents\""
fi
ok "deepagents package: $DEEPAGENTS_PKG"

# Deep Agents requires >=3.11,<4.0.
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if (3,11) <= sys.version_info < (4,0) else 1)'; then
        PYTHON_BIN="$candidate"; break
      fi
    fi
  done
fi
[[ -n "$PYTHON_BIN" ]] || die "no python >=3.11,<4.0 found; pass --python explicitly"
ok "python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

# ------------------------------------------------------- backend detection
say "Backend detection"

if [[ "$BACKEND" == "auto" ]]; then
  if command -v rocminfo >/dev/null 2>&1 || command -v hipconfig >/dev/null 2>&1; then
    BACKEND="rocm"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    BACKEND="cuda"
  else
    die "could not detect a GPU backend (no rocminfo/hipconfig, no nvidia-smi)
     Pass --backend cuda or --backend rocm explicitly if you know better."
  fi
  ok "autodetected: $BACKEND"
fi

case "$BACKEND" in
  cuda|rocm) ;;
  *) die "--backend must be auto, cuda or rocm; got: $BACKEND" ;;
esac

GPU_NAMES="unknown"
if [[ "$BACKEND" == "rocm" ]]; then
  if command -v rocminfo >/dev/null 2>&1; then
    GPU_NAMES="$(rocminfo 2>/dev/null | awk -F: '/Marketing Name/ {gsub(/^ +/,"",$2); print $2}' | sort -u | grep -v -i 'cpu' | paste -sd'; ' - || true)"
    [[ -n "$GPU_NAMES" ]] && ok "GPUs: $GPU_NAMES" || warn "rocminfo ran but reported no GPU names"
  else
    warn "rocminfo not on PATH — cannot confirm a GPU is visible"
  fi
  if [[ -z "$ROCM_VER" ]] && command -v hipconfig >/dev/null 2>&1; then
    ROCM_VER="$(hipconfig --version 2>/dev/null | cut -d. -f1,2 || true)"
  fi
  if [[ -z "$ROCM_VER" ]]; then
    warn "could not autodetect ROCm; defaulting the wheel index to rocm6.4"
    warn "if that is wrong, re-run with --rocm <major.minor>"
    ROCM_VER="6.4"
  fi
  # Only digits and one dot — this goes into a URL.
  [[ "$ROCM_VER" =~ ^[0-9]+\.[0-9]+$ ]] || die "--rocm must look like 6.4, got: $ROCM_VER"
  TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VER}"
  ACCEL_DESC="ROCm $ROCM_VER"
  ok "torch index: $TORCH_INDEX"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | paste -sd'; ' - || true)"
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    [[ -n "$GPU_NAMES" ]] && ok "GPUs: $GPU_NAMES (driver ${DRIVER:-unknown})" || warn "nvidia-smi ran but reported no GPUs"
  else
    warn "nvidia-smi not on PATH — cannot confirm a GPU is visible"
  fi
  if [[ -n "$CUDA_TAG" ]]; then
    # Only a cuNNN tag — this goes into a URL.
    [[ "$CUDA_TAG" =~ ^cu[0-9]+$ ]] || die "--cuda must look like cu124, got: $CUDA_TAG"
    TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
    ACCEL_DESC="CUDA (wheel tag $CUDA_TAG)"
    ok "torch index: $TORCH_INDEX"
  else
    TORCH_INDEX=""
    ACCEL_DESC="CUDA (default PyPI wheel)"
    ok "using the default PyPI torch wheel (CUDA-enabled on Linux)"
  fi
fi

# ------------------------------------------------------------------- venv
say "Virtual environment"

if [[ -d "$VENV" ]]; then
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$VENV"; ok "removed existing venv (--force)"
  else
    die "venv already exists at $VENV
     Re-run with --force to recreate it, or pass --venv-path elsewhere."
  fi
fi

# A venv on a network mount is the single most expensive mistake available
# here, and it fails silently — pip just appears to hang. Warn loudly rather
# than let it burn GPU-hours. Checks the nearest existing ancestor, since the
# venv directory itself does not exist yet.
VENV_FS_PROBE="$VENV"
while [[ ! -d "$VENV_FS_PROBE" && "$VENV_FS_PROBE" != "/" ]]; do
  VENV_FS_PROBE="$(dirname "$VENV_FS_PROBE")"
done
VENV_FSTYPE="$(df -PT "$VENV_FS_PROBE" 2>/dev/null | awk 'NR==2 {print $2}' || true)"
case "${VENV_FSTYPE:-unknown}" in
  nfs*|cifs|smb*|fuse*|*moosefs*|mfs*|lustre|ceph*|glusterfs|9p|sshfs|afs)
    warn "venv path is on a NETWORK filesystem (${VENV_FSTYPE}): $VENV"
    warn "pip is metadata-bound and crawls here; installs can take 10-100x longer."
    warn "Strongly consider: --venv-path /root/.venv-recursivemas (local disk),"
    warn "keeping only model weights on the network volume."
    ;;
  *) ok "venv filesystem: ${VENV_FSTYPE:-unknown} (not networked)" ;;
esac

mkdir -p "$(dirname "$VENV")"
"$PYTHON_BIN" -m venv "$VENV"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
"$PIP" install --quiet --upgrade pip
ok "created $VENV (pip $("$PIP" --version | awk '{print $2}'))"

# ----------------------------------------------------- torch, backend-aware
say "torch ($ACCEL_DESC)"

TORCH_PIN="$(grep -E '^torch==' "$REPO_ROOT/requirements.txt" | head -1 || true)"
[[ -n "$TORCH_PIN" ]] || die "requirements.txt has no torch== pin; this script assumes one"
ok "pin from requirements.txt: $TORCH_PIN"

if [[ -n "$TORCH_INDEX" ]]; then
  INSTALL_OK=0
  "$PIP" install --index-url "$TORCH_INDEX" "$TORCH_PIN" && INSTALL_OK=1 || true
else
  INSTALL_OK=0
  "$PIP" install "$TORCH_PIN" && INSTALL_OK=1 || true
fi

if [[ "$INSTALL_OK" != "1" ]]; then
  if [[ "$BACKEND" == "rocm" ]]; then
    die "no ROCm wheel for $TORCH_PIN at $TORCH_INDEX
     Pick a different --rocm series, or raise the pin in requirements.txt as a
     deliberate committed change — do not silently install a non-ROCm wheel."
  else
    die "could not install $TORCH_PIN
     If the box has an older driver, retry with an explicit wheel tag, e.g.
       --cuda cu121"
  fi
fi

assert_backend() {
  local phase="$1"
  "$PY" - "$BACKEND" "$phase" <<'PY' || die "torch is not built for the expected backend"
import sys, torch
backend, phase = sys.argv[1], sys.argv[2]
hip = getattr(torch.version, "hip", None)
cuda = getattr(torch.version, "cuda", None)
print(f"   torch {torch.__version__}  hip={hip}  cuda={cuda}  ({phase})")
if backend == "rocm" and not hip:
    print(f"\n   torch has no HIP runtime — CPU/CUDA wheel, not ROCm ({phase}).", file=sys.stderr)
    raise SystemExit(1)
if backend == "cuda":
    if hip:
        print(f"\n   torch is a ROCm build but --backend cuda was requested ({phase}).", file=sys.stderr)
        raise SystemExit(1)
    if not cuda:
        print(f"\n   torch has no CUDA runtime — CPU-only wheel ({phase}).", file=sys.stderr)
        raise SystemExit(1)
PY
}

assert_backend "after torch install"
ok "torch matches the requested backend"

# ------------------------------------------------------- rest of the deps
say "Remaining requirements"

"$PIP" install -r "$REPO_ROOT/requirements.txt"

# The reason torch goes first on ROCm. If pip replaced it here, every later run
# would silently fall back to CPU and still look like it worked.
assert_backend "after requirements.txt"
ok "torch survived requirements.txt"

say "Deep Agents (editable) + Exp 3a dependency"
"$PIP" install -e "$DEEPAGENTS_PKG"
# scikit-learn is needed by the Exp 3a probes and is absent from requirements.txt (PLAN §1).
"$PIP" install scikit-learn
assert_backend "after deepagents"
ok "deepagents + scikit-learn installed, torch backend intact"

# --------------------------------------------------- Gate 0.1 import checks
say "Gate 0.1 import checks (PLAN §5.0.1)"

cd "$REPO_ROOT"
"$PY" -c "import modeling; import deepagents; print('ok')" \
  || die "check 1 failed: import modeling; import deepagents"
ok "import modeling; import deepagents"

"$PY" -c "from inference_utils.inference_mas import autoregressive_latent_rollout, run_outer_adapter; print('ok')" \
  || die "check 2 failed: inference_utils.inference_mas imports"
ok "autoregressive_latent_rollout, run_outer_adapter"

# ----------------------------------------------------- device + attention
say "Device and attention probes"

"$PY" <<'PY'
import torch, transformers
print(f"   transformers {transformers.__version__}")
avail = torch.cuda.is_available()
hip = getattr(torch.version, "hip", None)
print(f"   torch.cuda.is_available() = {avail}" + ("   (HIP, not CUDA)" if hip else ""))
if avail:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"   device {i}: {p.name}  {p.total_memory / 1024**3:.1f} GiB")
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    _ = (x @ x).float().sum().item()
    print("   bf16 matmul on device: ok")
else:
    print("   WARN no device visible — everything would run on CPU")

# The attention implementation changes numerics, and PLAN §11 found the latent
# rollout compounds a ~1e-6 perturbation to cosine 0.761 by step 3. Whatever is
# chosen here must be identical in BOTH arms. Not installed automatically:
# flash-attn often needs a long source build, and sdpa is a fine substitute.
try:
    from transformers.utils import is_flash_attn_2_available
    fa2 = bool(is_flash_attn_2_available())
except Exception:
    fa2 = False
sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
print(f"   flash_attention_2 available: {fa2}")
print(f"   sdpa available:              {sdpa}")
print(f"   RECOMMENDED attn_implementation: {'flash_attention_2' if fa2 else ('sdpa' if sdpa else 'eager')}")
PY

# ------------------------------------------------------------- run note
say "Run note"

mkdir -p "$RUNS_DIR"
{
  echo "# Gate 0.1 run note — $STAMP"
  echo
  echo "Generated by \`integrations/deepagents_latent/scripts/gate0_setup.sh\`."
  echo "Satisfies PLAN.md §5.0.1 (\"the exact versions are written into a run note\")."
  echo
  echo "## Host"
  echo
  echo "- uname: \`$(uname -srm)\`"
  echo "- backend: $ACCEL_DESC"
  echo "- GPUs: $GPU_NAMES"
  echo "- torch index: ${TORCH_INDEX:-default PyPI}"
  echo "- interpreter: \`$PYTHON_BIN\` ($("$PYTHON_BIN" --version 2>&1))"
  echo "- venv: \`$VENV\` (filesystem: ${VENV_FSTYPE:-unknown})"
  echo
  echo "## Repo state"
  echo
  echo "- RecursiveMAS: \`$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)\` on \`$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)\`"
  echo "- deepagents: \`$(git -C "$DEEPAGENTS_PKG" rev-parse --short HEAD 2>/dev/null || echo unknown)\`"
  echo
  echo "## Device and attention"
  echo
  echo '```'
  "$PY" - <<'PY'
import torch
try:
    from transformers.utils import is_flash_attn_2_available
    fa2 = bool(is_flash_attn_2_available())
except Exception:
    fa2 = False
print(f"torch.version.hip      = {getattr(torch.version,'hip',None)}")
print(f"torch.version.cuda     = {getattr(torch.version,'cuda',None)}")
print(f"torch.cuda.is_available= {torch.cuda.is_available()}")
print(f"device_count           = {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
print(f"flash_attention_2      = {fa2}")
print(f"attn_implementation    = {'flash_attention_2' if fa2 else 'sdpa'}   <- use this in BOTH arms")
PY
  echo '```'
  echo
  echo "## Installed versions"
  echo
  echo '```'
  "$PIP" freeze
  echo '```'
} > "$NOTE"

ok "wrote $NOTE"

say "Gate 0.1 GREEN"
cat <<EOF

  Both import checks pass and the run note is written.

  Next, in order:
    1. Gate 0.2 is already decided — Binding A (PLAN §9).
    2. Gate 0.4: snapshot ONLY these three repos, per PLAN §2 —
         RecursiveMAS/Mixture-Code-Qwen2.5-Coder-3B
         RecursiveMAS/Mixture-Summarizer-Qwen3.5-2B
         RecursiveMAS/Mixture-Outerlinks
       Use snapshot_repo per repo. Do NOT call resolve_mas_paths or
       load_mas_system: they pull every repo in the mixture spec (~27 GB,
       including a 7B science model you do not need).
    3. Record the attn_implementation above and use the same one in both arms.

  Fixture check (no GPU needed, cheap sanity that the box is sane):
    cd integrations/deepagents_latent/tasks/golden && python3 verify_fixture.py

EOF
