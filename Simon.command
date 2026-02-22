#!/bin/zsh
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ ! -d ".venv" ]; then
  if [ -x "/opt/homebrew/bin/python3.12" ]; then
    /opt/homebrew/bin/python3.12 -m venv .venv
  elif [ -x "/usr/local/bin/python3.12" ]; then
    /usr/local/bin/python3.12 -m venv .venv
  else
    /usr/bin/env python3 -m venv .venv
  fi
fi

source .venv/bin/activate
python simon.py
