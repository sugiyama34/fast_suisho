#!/bin/bash
# SessionStart hook: git hook (.githooks/pre-commit の秘密情報スキャナ) が未インストールなら警告する。
# セッションはブロックしない (常に exit 0)。

cd "$CLAUDE_PROJECT_DIR" || exit 0
HOOKS_PATH=$(git config --get core.hooksPath)
# `.githooks` (相対) / この作業ツリーか元リポジトリ (worktree の場合) の .githooks への絶対パスなら OK
[ "$HOOKS_PATH" -ef .githooks ] && exit 0
[ "$HOOKS_PATH" -ef "$(git rev-parse --git-common-dir)/../.githooks" ] && exit 0

MSG="警告: git hook が未インストールです (core.hooksPath=${HOOKS_PATH:-未設定})。リポジトリのルートで \`bash scripts/install-hooks.sh\` を実行してください (Claude は git config を実行できないため、ユーザーが実行する)。"
jq -n --arg m "$MSG" '{systemMessage: $m, hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $m}}'
exit 0
