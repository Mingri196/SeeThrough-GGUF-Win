#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$ROOT/.venv/bin/python" ]] || {
    echo "Missing .venv. Run ./install.sh first." >&2
    exit 1
}
export PYTHONPATH="$ROOT/common:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/.venv/bin/python" -u "$ROOT/inference/scripts/ragdoll_postprocess.py" "$@"
