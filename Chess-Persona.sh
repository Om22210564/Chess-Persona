#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT_PATH="${CHESS_PERSONA_CHECKPOINT:-$SCRIPT_DIR/best_policy.pt}"

if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

if ! command -v maia3-5m >/dev/null 2>&1; then
    echo "Error: maia3-5m command not found." >&2
    echo "Install Maia3 in your environment, e.g. python -m pip install -e external/maia3" >&2
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: checkpoint not found: $CHECKPOINT_PATH" >&2
    echo "Set CHESS_PERSONA_CHECKPOINT=/path/to/checkpoint.pt or place best_policy.pt in the project root." >&2
    exit 1
fi

exec maia3-5m --checkpoint-path "$CHECKPOINT_PATH"
