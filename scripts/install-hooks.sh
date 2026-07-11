#!/bin/bash
# Configure git to use the project's .githooks directory for hooks.
# Run this once after cloning the repository.

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

if [ -z "$REPO_ROOT" ]; then
  echo "Error: Not inside a git repository." >&2
  exit 1
fi

git config core.hooksPath .githooks
echo "Git hooks configured: core.hooksPath = .githooks"
echo "Pre-commit hook is now active."
