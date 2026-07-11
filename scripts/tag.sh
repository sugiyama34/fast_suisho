#!/bin/bash
# Create a v-prefixed semver git tag bumped from the latest tag and push it to origin.
#
# Usage: scripts/tag.sh <commit-hash> <--major|--minor|--patch> -m|--message <message>

set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: scripts/tag.sh <commit-hash> <--major|--minor|--patch> -m|--message <message>

Creates a new v-prefixed semver tag (e.g. v1.2.3) at <commit-hash> by
incrementing the chosen component of the latest existing v-tag, then
pushes the tag to origin. The annotation message is required.
EOF
  exit 2
}

commit=""
bump=""
message=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    -m|--message)
      if [ "$#" -lt 2 ]; then
        echo "Error: '$1' requires a message argument." >&2
        usage
      fi
      message="$2"
      shift 2
      ;;
    --major|--minor|--patch)
      if [ -n "$bump" ]; then
        echo "Error: multiple bump flags provided ('$bump' and '$1')." >&2
        usage
      fi
      bump="$1"
      shift
      ;;
    -*)
      echo "Error: unknown option '$1'." >&2
      usage
      ;;
    *)
      if [ -n "$commit" ]; then
        echo "Error: unexpected positional argument '$1' (commit already set to '$commit')." >&2
        usage
      fi
      commit="$1"
      shift
      ;;
  esac
done

if [ -z "$commit" ] || [ -z "$bump" ] || [ -z "$message" ]; then
  echo "Error: <commit-hash>, <--major|--minor|--patch>, and -m/--message are all required." >&2
  usage
fi

if ! git rev-parse --verify --quiet "${commit}^{commit}" >/dev/null; then
  echo "Error: '${commit}' is not a valid commit in this repository." >&2
  echo "Fix: pass a commit hash that exists locally (run 'git log --oneline' to find one)." >&2
  exit 1
fi

# Sync remote tags so the version bump is computed against the authoritative
# set, not just stale local tags. Without this, a local clone missing tags
# could produce a duplicate or lower-than-latest tag when pushed.
if ! git fetch --tags --quiet origin; then
  echo "Error: failed to fetch tags from origin." >&2
  echo "Fix: ensure the 'origin' remote is configured and reachable, then retry." >&2
  exit 1
fi

latest=$(git tag --list 'v[0-9]*.[0-9]*.[0-9]*' \
  | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
  | sort -V \
  | tail -n1 || true)

if [ -z "$latest" ]; then
  major=0
  minor=0
  patch=0
else
  IFS='.' read -r major minor patch <<<"${latest#v}"
fi

case "$bump" in
  --major)
    major=$((major + 1))
    minor=0
    patch=0
    ;;
  --minor)
    minor=$((minor + 1))
    patch=0
    ;;
  --patch)
    patch=$((patch + 1))
    ;;
  *)
    echo "Error: unknown bump flag '${bump}'. Expected --major, --minor, or --patch." >&2
    usage
    ;;
esac

new_tag="v${major}.${minor}.${patch}"

if git rev-parse --verify --quiet "refs/tags/${new_tag}" >/dev/null; then
  echo "Error: tag '${new_tag}' already exists locally." >&2
  echo "Fix: delete the stale tag (git tag -d ${new_tag}) or choose a different bump level." >&2
  exit 1
fi

resolved=$(git rev-parse --verify "${commit}^{commit}")

echo "Creating tag ${new_tag} at ${resolved} (bump: ${bump}, previous: ${latest:-<none>})"
git tag -a "${new_tag}" -m "${message}" "${resolved}"
git push origin "${new_tag}"
echo "Pushed ${new_tag} to origin."
