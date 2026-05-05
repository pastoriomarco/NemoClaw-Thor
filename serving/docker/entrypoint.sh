#!/bin/bash
# NemoClaw-Thor vLLM entrypoint
#
# Applies runtime mods (patches to installed packages) before exec-ing
# into the requested command. Mods are specified via the VLLM_MODS
# environment variable as a comma-separated list of mod directory names.
#
# Example:
#   docker run -e VLLM_MODS=fix-qwen3.5-chat-template,fix-qwen3-coder-next ...

set -e

MODS_DIR="/workspace/mods"
WORKSPACE_DIR="/workspace/vllm"
export WORKSPACE_DIR

if [ -n "$VLLM_MODS" ]; then
    IFS=',' read -ra MOD_LIST <<< "$VLLM_MODS"
    for mod in "${MOD_LIST[@]}"; do
        mod=$(echo "$mod" | xargs)  # trim whitespace
        mod_path="${MODS_DIR}/${mod}"
        if [ -d "$mod_path" ] && [ -x "$mod_path/run.sh" ]; then
            echo "[entrypoint] Applying mod: ${mod}"
            (cd "$mod_path" && ./run.sh)
        else
            echo "[entrypoint] WARNING: mod not found or not executable: ${mod}" >&2
        fi
    done
fi

exec "$@"
