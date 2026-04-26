#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v brew >/dev/null 2>&1; then
  echo "brew not found; install Homebrew first: https://brew.sh" >&2
else
  brew list portaudio >/dev/null 2>&1 || brew install portaudio
fi

PY=${PY:-python3}
$PY -m pip install --upgrade pip
$PY -m pip install -r requirements.txt

echo
echo "ok. try a dry run:"
echo "  $PY dance.py /path/to/song.mp3 --dry-run"
