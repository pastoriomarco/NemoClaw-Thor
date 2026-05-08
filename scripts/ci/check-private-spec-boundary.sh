#!/usr/bin/env bash
# check-private-spec-boundary.sh — boundary guard for the NemoClaw-Thor repo.
#
# NemoClaw-Thor is a deployment/integration repo: vLLM serving + sandbox
# wiring + Thor-platform glue. Operator-facing runbooks must remain
# self-sufficient without manyforge_specs (the private dev repo) checked
# out on disk. This script enforces that boundary.
#
# AGENTS.md and VERSIONS.md are explicitly allowlisted: they're the
# canonical cross-repo authority maps and are EXPECTED to name the
# manyforge_specs repo. Everything else under runtime / operator paths
# is forbidden from referencing it.
#
# Wire into CI:
#   .github/workflows/...   - run as a step; non-zero fails the build.
#   pre-commit              - or invoke from a pre-commit hook.
#   manual                  - bash scripts/ci/check-private-spec-boundary.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Files allowed to reference manyforge_specs. Format: relative-path  # reason.
# Anything else that matches the search pattern is a boundary violation.
ALLOWLIST=(
  "AGENTS.md                                                                  # cross-repo authority map (top-level); names manyforge_specs as authoritative for specs/ADRs"
  "manyforge/AGENTS.md                                                         # cross-repo authority map (subtree-local); names manyforge_specs"
  "VERSIONS.md                                                                 # cross-repo version reference; names manyforge_specs as the spec authority"
  "scripts/ci/check-private-spec-boundary.sh                                   # this file"
)

ALLOW_PATTERN="$(printf '%s\n' "${ALLOWLIST[@]}" | awk -F'#' '{gsub(/[[:space:]]+$/, "", $1); print $1}')"

# Operator/runtime paths that must remain self-sufficient.
RUNTIME_PATHS=(
  "README.md"
  "AGENTS.md"
  "USER_QUICKSTART_MANUAL.md"
  "VERSIONS.md"
  "manyforge"
  "serving/docs"
  "serving/scripts"
  "setup"
  "scripts"
)

VIOLATIONS=()

for path in "${RUNTIME_PATHS[@]}"; do
  [[ -e "${path}" ]] || continue
  while IFS= read -r hit; do
    [[ -z "${hit}" ]] && continue
    file="${hit%%:*}"
    if echo "${ALLOW_PATTERN}" | grep -qxF "${file}"; then
      continue
    fi
    VIOLATIONS+=("${hit}")
  done < <(grep -rHnE "manyforge_specs" "${path}" \
            --include='*.py' --include='*.sh' --include='*.md' \
            --include='*.yaml' --include='*.yml' --include='*.json' \
            --exclude-dir='.git' --exclude-dir='node_modules' \
            --exclude-dir='__pycache__' --exclude-dir='.venv' \
            --exclude-dir='.pytest_cache' --exclude-dir='build' \
            --exclude-dir='dist' 2>/dev/null || true)
done

if (( ${#VIOLATIONS[@]} > 0 )); then
  echo "ERROR: NemoClaw-Thor runtime path(s) reference 'manyforge_specs' (private dev repo)." >&2
  echo "       Operator-facing runbooks must remain self-sufficient." >&2
  echo "" >&2
  echo "Violations:" >&2
  printf '  %s\n' "${VIOLATIONS[@]}" >&2
  echo "" >&2
  echo "Fix by either:" >&2
  echo "  (a) Removing the reference if it's not load-bearing." >&2
  echo "  (b) Pointing at a public manyforge/docs/reference/* doc instead." >&2
  echo "  (c) If this file legitimately describes the cross-repo boundary," >&2
  echo "      add it to the ALLOWLIST in this script with a reason." >&2
  exit 1
fi

echo "OK: no private-spec boundary violations in NemoClaw-Thor runtime paths."
exit 0
