#!/usr/bin/env bash
#
# Gate 0.1 — isolated environment on AMD/ROCm (PLAN.md §5.0.1, §1.2).
#
# Gate 0.1 is "done" when both import checks below print ok AND the exact
# versions are written into a run note. This script does both, then probes the
# things §1.2 says will bite on ROCm.
#
# The load-bearing step is ordering: requirements.txt pins `torch==2.9.0`,
# which resolves to the default PyPI wheel with no ROCm support. Installing the
# ROCm build FIRST satisfies that pin (PEP 440 `==2.9.0` matches the local
# version `2.9.0+rocm*`), so the later `-r requirements.txt` leaves it alone.
# That is an assumption about pip's resolver, not a guarantee, so the script
# re-checks afterwards and fails loudly if torch got swapped for a CUDA wheel —
# a silent swap would produce a CPU-only run that looks like it worked.
#
# Never "fixes" a resolver conflict by editing requirements.txt (PLAN §5.0.1).
#
# Usage:
#   ./gate0_setup_rocm.sh [--force] [--rocm 6.4] [--python python3.12]
#
#   --force    recreate .venv if it already exists (destructive to the venv only)
#   --rocm     ROCm minor series for the wheel index; default: autodetect
#   --python   interpreter to build the venv from; default: autodetect >=3.11,<4.0

set -euo pipefail

FORCE=0
ROCM_VER=""
PYTHON_BIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --rocm) ROCM_VER="${2:?--rocm needs a value, e.g. 6.4}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?--python needs a value}"; shift 2 ;;
    -h|--help) sed -n '3,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DEEPAGENTS_PKG="$REPO_ROOT/../deepagents/libs/deepagents"
VENV="$REPO_ROOT/.venv"
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
     PLAN §1 expects the clone symlinked at ../deepagents. Clone or symlink it, then re-run."
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

# ------------------------------------------------------------------- ROCm
say "ROCm detection"

if command -v rocminfo >/dev/null 2>&1; then
  GPU_NAMES="$(rocminfo 2>/dev/null | awk -F: '/Marketing Name/ {gsub(/^ +/,"",$2); print $2}' | sort -u | grep -v -i 'cpu' | paste -sd'; ' - || true)"
  [[ -n "$GPU_NAMES" ]] && ok "GPUs: $GPU_NAMES" || warn "rocminfo ran but reported no GPU marketing names"
else
  warn "rocminfo not on PATH — cannot confirm a GPU is visible"
  GPU_NAMES="unknown"
fi

if [[ -z "$ROCM_VER" ]]; then
  if command -v hipconfig >/dev/null 2>&1; then
    ROCM_FULL="$(hipconfig --version 2>/dev/null || true)"
    ROCM_VER="$(printf '%s' "$ROCM_FULL" | cut -d. -f1,2)"
  fi
fi

if [[ -z "$ROCM_VER" ]]; then
  warn "could not autodetect ROCm; defaulting the wheel index to rocm6.4"
  warn "if that is wrong, re-run with --rocm <major.minor>"
  ROCM_VER="6.4"
fi
# Only digits and one dot — this goes into a URL.
[[ "$ROCM_VER" =~ ^[0-9]+\.[0-9]+$ ]] || die "--rocm must look like 6.4, got: $ROCM_VER"
TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VER}"
ok "ROCm series: $ROCM_VER"
ok "torch index: $TORCH_INDEX"

# ------------------------------------------------------------------- venv
say "Virtual environment"

if [[ -d "$VENV" ]]; then
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$VENV"; ok "removed existing .venv (--force)"
  else
    die ".venv already exists at $VENV
     PLAN §1 notes the existing one holds only ipykernel. Re-run with --force to recreate it."
  fi
fi

"$PYTHON_BIN" -m venv "$VENV"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
"$PIP" install --quiet --upgrade pip
ok "created $VENV (pip $("$PIP" --version | awk '{print $2}'))"

# ------------------------------------------------- torch FIRST, from ROCm
say "torch (ROCm build, installed before anything else)"

TORCH_PIN="$(grep -E '^torch==' "$REPO_ROOT/requirements.txt" | head -1 || true)"
[[ -n "$TORCH_PIN" ]] || die "requirements.txt has no torch== pin; this script assumes one"
ok "pin from requirements.txt: $TORCH_PIN"

if ! "$PIP" install --index-url "$TORCH_INDEX" "$TORCH_PIN"; then
  die "ROCm wheel for $TORCH_PIN not available at $TORCH_INDEX
     Check which torch versions that index carries and either pick a different
     --rocm series or raise the pin in requirements.txt as a deliberate,
     committed change — do not silently install a non-ROCm wheel."
fi

assert_rocm_torch() {
  local phase="$1"
  "$PY" - "$phase" <<'PY' || die "torch is not a ROCm build"
import sys, torch
phase = sys.argv[1]
hip = getattr(torch.version, "hip", None)
cuda = getattr(torch.version, "cuda", None)
print(f"   torch {torch.__version__}  hip={hip}  cuda={cuda}  ({phase})")
if not hip:
    print(f"\n   torch has no HIP runtime — this is a CPU/CUDA wheel, not ROCm ({phase}).", file=sys.stderr)
    raise SystemExit(1)
PY
}

assert_rocm_torch "after torch install"
ok "torch is a ROCm build"

# ------------------------------------------------------- rest of the deps
say "Remaining requirements"

"$PIP" install -r "$REPO_ROOT/requirements.txt"

# The whole point of the ordering above. If pip replaced torch here, every
# later run would silently fall back to CPU and still look like it worked.
assert_rocm_torch "after requirements.txt"
ok "torch survived requirements.txt as a ROCm build"

say "Deep Agents (editable) + Exp 3a dependency"
"$PIP" install -e "$DEEPAGENTS_PKG"
# scikit-learn is needed by the Exp 3a probes and is absent from requirements.txt (PLAN §1).
"$PIP" install scikit-learn
assert_rocm_torch "after deepagents"
ok "deepagents + scikit-learn installed, torch still ROCm"

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
say "Device and attention probes (§1.2)"

"$PY" <<'PY'
import torch, transformers
print(f"   transformers {transformers.__version__}")
avail = torch.cuda.is_available()
print(f"   torch.cuda.is_available() = {avail}   (True here means HIP, not CUDA)")
if avail:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"   device {i}: {p.name}  {p.total_memory / 1024**3:.1f} GiB")
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    _ = (x @ x).float().sum().item()
    print("   bf16 matmul on device: ok")
else:
    print("   WARN no device visible — everything below would run on CPU")

# §1.2: flash-attn is CUDA-first. Record which implementation is actually
# usable; both arms must then use the SAME one, because PLAN §11 found the
# latent rollout is chaotically sensitive to numerics.
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
  echo "Generated by \`integrations/deepagents_latent/scripts/gate0_setup_rocm.sh\`."
  echo "Satisfies PLAN.md §5.0.1 (\"the exact versions are written into a run note\")."
  echo
  echo "## Host"
  echo
  echo "- uname: \`$(uname -srm)\`"
  echo "- GPUs: $GPU_NAMES"
  echo "- ROCm series used for the wheel index: $ROCM_VER"
  echo "- torch index: $TORCH_INDEX"
  echo "- interpreter: \`$PYTHON_BIN\` ($("$PYTHON_BIN" --version 2>&1))"
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
       including the 7B science model you do not need).
    3. Record the attn_implementation above and use the same one in both arms.

  Fixture check (no GPU needed, cheap sanity that the box is sane):
    cd integrations/deepagents_latent/tasks/golden && python3 verify_fixture.py

EOF
