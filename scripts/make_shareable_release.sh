#!/usr/bin/env bash
# scripts/make_shareable_release.sh
# Create a shareable zip of the repository that excludes large data and model artifacts.
# Usage: ./scripts/make_shareable_release.sh [output-name]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT_NAME=${1:-"coronahack-demo-release-$(date +%Y%m%d-%H%M%S).zip"}

# Patterns to exclude (relative to repo root)
EXCLUDES=(
  "data/*"
  "output/*"
  "*.wav"
  "*.webm"
  "*.pt"
  "*.pth"
  "*.tar.gz"
  ".venv/*"
  "__pycache__/*"
)

# Build the zip while excluding patterns
TMPFILE=$(mktemp -u /tmp/release-XXXXXX.zip)

# Use zip while excluding the patterns; fallback to git-archive if zip missing
if command -v zip >/dev/null 2>&1; then
  ZIP_EXCLUDES=( )
  for p in "${EXCLUDES[@]}"; do
    ZIP_EXCLUDES+=( -x "$p" )
  done
  # shellcheck disable=SC2086
  echo "Creating $OUT_NAME (excluding large files)..."
  eval "zip -r $TMPFILE . ${ZIP_EXCLUDES[*]}"
  mv "$TMPFILE" "$OUT_NAME"
  echo "Created $OUT_NAME"
else
  echo "zip not available; falling back to git archive (requires git)"
  if ! command -v git >/dev/null 2>&1; then
    echo "Error: neither zip nor git are available. Install one and retry." >&2
    exit 2
  fi
  echo "Creating $OUT_NAME via git archive..."
  git archive -o "$OUT_NAME" HEAD
  echo "Created $OUT_NAME"
fi

echo "Done. Share $OUT_NAME with collaborators. Note: model binaries and dataset files are NOT included by design."