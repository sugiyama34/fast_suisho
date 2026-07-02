#!/bin/bash
# Shared helper: split a compound shell command on unquoted operators.
# Sourced by .claude/hooks/command-allowlist.sh and block-dangerous-git.sh.
#
# Splits on `|`, `||`, `&&`, `;`, and background `&`, while respecting:
#   - single-quoted strings
#   - double-quoted strings
#   - backslash escapes outside single quotes (so `\&`, `\|`, `\;` stay
#     part of their segment instead of being mistaken for operators)
#
# Usage:
#   source "$CLAUDE_PROJECT_DIR/.claude/hooks/lib/shell-parse.sh"
#   split_command_segments "$some_command"   # one segment per line on stdout

split_command_segments() {
  local cmd="$1"
  local in_sq=false
  local in_dq=false
  local escaped=false
  local segment=""
  local i c next prev

  for ((i=0; i<${#cmd}; i++)); do
    c="${cmd:$i:1}"
    next="${cmd:$((i+1)):1}"

    # Previous char was a backslash outside single quotes: treat this char
    # as literal (do not split on it even if it is `&` / `|` / `;`).
    if [[ "$escaped" == true ]]; then
      segment+="$c"
      escaped=false
      continue
    fi

    # Backslash outside single quotes starts an escape.
    if [[ "$c" == "\\" && "$in_sq" == false ]]; then
      segment+="$c"
      escaped=true
      continue
    fi

    # Single-quote toggle (only outside double quotes).
    if [[ "$c" == "'" && "$in_dq" == false ]]; then
      if [[ "$in_sq" == true ]]; then in_sq=false; else in_sq=true; fi
      segment+="$c"
      continue
    fi

    # Double-quote toggle (only outside single quotes).
    if [[ "$c" == '"' && "$in_sq" == false ]]; then
      if [[ "$in_dq" == true ]]; then in_dq=false; else in_dq=true; fi
      segment+="$c"
      continue
    fi

    # Inside any quote: treat separators as literal.
    if [[ "$in_sq" == true || "$in_dq" == true ]]; then
      segment+="$c"
      continue
    fi

    # Outside quotes: split on shell operators.
    prev="${cmd:$((i-1)):1}"
    if [[ "$c" == "|" && "$next" == "|" ]]; then
      echo "$segment"
      segment=""
      ((i++))
    elif [[ "$c" == "&" && "$next" == "&" ]]; then
      echo "$segment"
      segment=""
      ((i++))
    elif [[ "$c" == "|" ]]; then
      echo "$segment"
      segment=""
    elif [[ "$c" == ";" ]]; then
      echo "$segment"
      segment=""
    elif [[ "$c" == "&" && "$prev" != ">" && "$next" != ">" ]]; then
      echo "$segment"
      segment=""
    else
      segment+="$c"
    fi
  done
  echo "$segment"
}
