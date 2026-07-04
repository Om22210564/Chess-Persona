#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/.venv/bin/activate"

exec maia3-5m \
    --checkpoint-path "$SCRIPT_DIR/best_policy.pt"