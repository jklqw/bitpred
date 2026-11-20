#!/usr/bin/env bash
# Server entry point: resume bootstrap (download + initial train), then serve.
# Any extra arguments are forwarded to bootstrap.py (e.g. --force-train).
set -euo pipefail

cd "$(dirname "$0")"

if [ -x ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

exec "$PYTHON" -u bootstrap.py --start-server "$@"
