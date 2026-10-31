#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$ROOT/.venv/bin/python" ]] || {
    echo "Missing .venv. Run ./install.sh first." >&2
    exit 1
}
export PYTHONPATH="$ROOT/common:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

BACKEND_SET=0
for ARG in "$@"; do
    [[ "$ARG" == "--attention-backend" || "$ARG" == --attention-backend=* ]] && BACKEND_SET=1
done
if [[ "$BACKEND_SET" == "0" ]] && ! "$ROOT/.venv/bin/python" -c 'import sageattention' 2>/dev/null; then
    echo "SageAttention is not installed; using native PyTorch attention."
    set -- --attention-backend native "$@"
fi

exec "$ROOT/.venv/bin/python" -u "$ROOT/inference/scripts/inference_psd.py" "$@"
