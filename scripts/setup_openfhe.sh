#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python3}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python interpreter '$python_bin' was not found." >&2
  exit 1
fi

"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip install openfhe

cat <<'EOM'
OpenFHE installation attempted.
If the installation fails because wheels are unavailable for your platform,
please refer to the official OpenFHE Python documentation for manual build
instructions.
EOM
