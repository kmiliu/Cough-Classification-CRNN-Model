#!/usr/bin/env bash
# Package only allowlisted source files and public evaluation artifacts.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT_DIR/scripts/build_release.py" "${1:-respiratory-results-release.zip}"
