#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${STM_PYTHON:-python3.12}"
SKIP_SAGE="${STM_SKIP_SAGE:-0}"

case "${1:-}" in
    --skip-sage) SKIP_SAGE=1 ;;
    -h|--help) echo "Usage: ./install.sh [--skip-sage]"; exit 0 ;;
    "") ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "Usage: ./install.sh [--skip-sage]" >&2; exit 2; }

command -v "$PYTHON" >/dev/null || {
    echo "Python 3.12 is required (set STM_PYTHON if it has another name)." >&2
    exit 1
}
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

"$PYTHON" -m venv "$ROOT/.venv"
PIP="$ROOT/.venv/bin/pip"
"$PIP" install --upgrade pip setuptools wheel
"$PIP" install \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
"$PIP" install -r "$ROOT/requirements.txt"
# rtmlib also installs CPU onnxruntime, which shares and can overwrite the GPU module.
ORT_GPU_REQUIREMENT="$(sed -n '/^onnxruntime-gpu==/p' "$ROOT/requirements.txt")"
"$PIP" install --force-reinstall --no-deps "$ORT_GPU_REQUIREMENT"

CAPABILITY="$("$ROOT/.venv/bin/python" -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else "none")')"
echo "Detected CUDA capability: $CAPABILITY"

if [[ "$SKIP_SAGE" == "1" ]]; then
    echo "Skipping SageAttention; use --attention-backend native."
elif [[ "$CAPABILITY" == "7.5" ]]; then
    if [[ -x "$ROOT/build_sageattention2_sm_75.sh" ]]; then
        "$ROOT/build_sageattention2_sm_75.sh"
    else
        echo "Local sm_75 SageAttention builder not found; using native attention."
    fi
elif [[ "$CAPABILITY" == "none" ]]; then
    echo "No CUDA GPU detected; skipping SageAttention."
else
    "$PIP" install ninja
    if ! command -v nvcc >/dev/null; then
        echo "nvcc not found; skipping SageAttention and using native attention."
    elif ! "$PIP" install --no-build-isolation \
        "sageattention @ git+https://github.com/thu-ml/SageAttention.git@v2.2.0"; then
        echo "SageAttention build failed; using native attention."
    fi
fi

CHECK_ARGS=()
"$ROOT/.venv/bin/python" -c 'import sageattention' 2>/dev/null || CHECK_ARGS+=(--allow-no-sage)
PYTHONPATH="$ROOT/common:$ROOT" "$ROOT/.venv/bin/python" "$ROOT/scripts/check_install.py" \
    "${CHECK_ARGS[@]}"
