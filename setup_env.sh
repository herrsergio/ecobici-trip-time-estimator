#!/usr/bin/env bash
# Creates a Python virtual environment and installs project dependencies.
# TensorFlow currently ships wheels only for Python 3.9 - 3.12, so this
# script picks a compatible interpreter automatically.
# Override with: PYTHON=/path/to/python3.11 ./setup_env.sh
set -euo pipefail

cd "$(dirname "$0")"

# Candidates in preference order: newest TF-supported first.
PY_CANDIDATES=(
  "${PYTHON:-}"
  "$HOME/.pyenv/versions/3.11.14/bin/python3.11"
  "python3.12"
  "python3.11"
  "python3.10"
)

PY=""
for cand in "${PY_CANDIDATES[@]}"; do
  [ -z "$cand" ] && continue
  if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
    ver="$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")"
    case "$ver" in
      3.10|3.11|3.12) PY="$cand"; break;;
    esac
  fi
done

if [ -z "$PY" ]; then
  echo "ERROR: no Python 3.10/3.11/3.12 found. TensorFlow does not yet support 3.13+."
  echo "Install one with pyenv:  pyenv install 3.11.14 && pyenv shell 3.11.14"
  echo "Or rerun with:           PYTHON=python3.11 ./setup_env.sh"
  exit 1
fi

echo "Using $($PY --version) at $(command -v "$PY" || echo "$PY")"

if [ ! -d ".venv" ]; then
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel
pip install -r requirements.txt

echo
echo "Done. Activate the venv with:"
echo "  source .venv/bin/activate"
