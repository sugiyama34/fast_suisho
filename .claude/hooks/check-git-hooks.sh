#!/bin/bash
# Claude Code SessionStart hook: git hook (.githooks/) が未インストールなら警告する。
#
# scripts/install-hooks.sh は core.hooksPath=.githooks を設定し、commit 時に
# .githooks/pre-commit (秘密情報スキャナ) が動くようにする。新しい clone や新サーバーで
# この手順を忘れるとスキャナが黙って無効になる (新サーバー移行時に実際に起きた)。
#
# - Claude は自分で直せない (Bash ツールからの `git config` は block-dangerous-git.sh が
#   ブロックする) ので、ユーザーにスクリプトの実行を依頼する。
#   この hook 自身の `git config --get` は Bash ツール呼び出しではないため、
#   PreToolUse の hook (block-dangerous-git.sh) の対象外。
# - worktree (.git がファイル) でも git に解決させるので正しく動く。
#   core.hooksPath は worktree 間で共有される (.git/config)。
# - 正しく設定済みなら何も出力しない。セッションは絶対にブロックしない (常に exit 0)。

set -uo pipefail
export LC_ALL=C GIT_OPTIONAL_LOCKS=0

INPUT=""
[ -t 0 ] || INPUT=$(cat)

CWD=""
if [ -n "$INPUT" ] && command -v jq >/dev/null 2>&1; then
  CWD=$(jq -r '.cwd // empty' <<<"$INPUT" 2>/dev/null || true)
fi

TOP=""
COMMON_DIR=""
for dir in "$CWD" "${CLAUDE_PROJECT_DIR:-}"; do
  [ -n "$dir" ] && [ -d "$dir" ] || continue
  if out=$(git -C "$dir" rev-parse --show-toplevel \
    --path-format=absolute --git-common-dir 2>/dev/null); then
    TOP=${out%%$'\n'*}
    COMMON_DIR=${out#*$'\n'}
    break
  fi
done
[ -n "$TOP" ] || exit 0

# .githooks/ を同梱しているリポジトリ (= 本プロジェクト) だけが対象
[ -f "$TOP/.githooks/pre-commit" ] || exit 0

HOOKS_PATH=$(git -C "$TOP" config --get core.hooksPath 2>/dev/null || true)

# OK とする値:
# - install-hooks.sh が設定する相対パス `.githooks` (各 worktree のルート基準で解決される)
# - この作業ツリー、または元のリポジトリ (worktree から見た main worktree) の
#   .githooks を指す絶対パス
value=${HOOKS_PATH%/}
case "$value" in
  .githooks | ./.githooks) exit 0 ;;
  /*)
    [ "$value" -ef "$TOP/.githooks" ] && exit 0
    [ "$value" -ef "${COMMON_DIR%/.git}/.githooks" ] && exit 0
    ;;
esac

CURRENT=${HOOKS_PATH:-未設定}
MESSAGE="警告: git hook が未インストールです (core.hooksPath = ${CURRENT}、期待値 .githooks)。

理由:
  .githooks/pre-commit (秘密情報スキャナ) が commit 時に動きません。
  新しい clone / 新サーバーで scripts/install-hooks.sh の実行を忘れると起きます。

対処:
  ユーザー: リポジトリのルート (${TOP}) で \`bash scripts/install-hooks.sh\` を実行してください。
  Claude Code: 自分では設定しない (Bash ツールからの git config は block-dangerous-git.sh が
               ブロックする)。ユーザーに上記コマンドの実行を依頼する。"

# systemMessage はユーザーに、additionalContext は Claude のコンテキストに表示される
if command -v jq >/dev/null 2>&1; then
  jq -n --arg msg "$MESSAGE" '{
    systemMessage: $msg,
    hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $msg}
  }'
else
  # SessionStart の exit 0 の plain-text stdout は Claude のコンテキストに入る
  echo "$MESSAGE"
fi
exit 0
