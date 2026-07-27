#!/bin/bash
# Claude Code PreToolUse hook: WandB 安全ガード (docs/wandb-guide.md 防御層 4)。
# 危険な wandb 操作と、settings.json の env で固定済みの WANDB_* 環境変数の
# インライン上書きをブロックする。
# Exit code 0 = allow/pass-through, exit code 2 = block.

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib/shell-parse.sh"

if ! command -v jq >/dev/null 2>&1; then
  echo "BLOCKED: jq is not installed (required to validate wandb commands)." >&2
  exit 2
fi

INPUT=$(cat)
if ! RAW_COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null); then
  echo "BLOCKED: failed to parse tool input JSON in wandb-guard hook." >&2
  exit 2
fi
[ -z "$RAW_COMMAND" ] && exit 0

block() {
  cat >&2 <<ERRMSG
BLOCKED: $1

Command: $2

Why:
  $3

What to do:
  Claude Code: Follow the conventions in docs/wandb-guide.md (use tools/wandb_utils.py;
               entity/project/API key are managed outside the agent).
  User: Run the command manually in your terminal if it is genuinely needed.
ERRMSG
  exit 2
}

while IFS= read -r segment; do
  segment=$(echo "$segment" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')
  [ -z "$segment" ] && continue

  # 1) wandb login — API キーが ~/.netrc に平文で永続化される
  if echo "$segment" | grep -qE '^(uv run )?(python3? -m )?wandb login( |$)'; then
    block "wandb login is forbidden" "$segment" \
      "wandb login writes the API key to ~/.netrc in plaintext. The key is provided via sandbox credential masking instead."
  fi

  # 2) sweep / agent — WANDB_MODE=offline/disabled を無視してクラウド同期する
  #    既知バグがあり (wandb/wandb#6234)、本プロジェクトでは sweep 全面禁止
  if echo "$segment" | grep -qE '^(uv run )?(python3? -m )?wandb (sweep|agent)( |$)'; then
    block "wandb sweep/agent is forbidden" "$segment" \
      "Sweeps bypass WANDB_MODE sandboxing (wandb/wandb#6234) and are banned in this project. Use explicit per-run configs instead."
  fi

  # 3) artifact 削除
  if echo "$segment" | grep -qE '^(uv run )?(python3? -m )?wandb artifact (delete|rm)( |$)'; then
    block "wandb artifact deletion is forbidden" "$segment" \
      "Artifact deletion is irreversible (no recovery). Retention is managed via artifact TTL in tools/wandb_utils.py."
  fi

  # 4) 固定済み環境変数のインライン設定/上書き
  if echo "$segment" | grep -qE 'WANDB_(API_KEY|ENTITY|PROJECT|BASE_URL|RUN_ID)\+?='; then
    block "inline WANDB_* override is forbidden" "$segment" \
      "WANDB_API_KEY is handled only by sandbox credential masking; WANDB_ENTITY/WANDB_PROJECT are pinned in settings.json env."
  fi
done <<< "$(split_command_segments "$RAW_COMMAND")"

exit 0
