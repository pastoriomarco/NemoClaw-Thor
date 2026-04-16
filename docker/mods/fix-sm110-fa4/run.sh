#!/bin/bash

# SM110 (Thor): Route Flash Attention version selection to FA4 instead of FA2.
#
# Problem: vLLM's get_flash_attn_version() in fa_utils.py only checks for
# device_capability.major == 10 when selecting FA4 (CuTe DSL). SM110 has
# major == 11, so it falls through to FA2 — which crashes on SM110 (no PTX).
#
# FA4's CuTe DSL already supports SM110: interface.py dispatches
# arch // 10 in [10, 11] to FlashAttentionForwardSm100. The only blocker
# is this version gate.
#
# Solution: Patch the FA4 selection condition to include major == 11.
# Also patch the head_size fallback to avoid routing to FA2 on SM110
# (FA2 crashes unconditionally on SM110, regardless of head_size).

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FA_UTILS="$SITE_PACKAGES/vllm/v1/attention/backends/fa_utils.py"

echo "Applying SM110 FA4 attention backend patch..."

if [ ! -f "$FA_UTILS" ]; then
    echo "  WARNING: fa_utils.py not found at $FA_UTILS (skipping)"
    exit 0
fi

# Patch 1: FA4 version selection — add major == 11 to the FA4 condition.
# Before: elif device_capability.major == 10 and is_fa_version_supported(4):
# After:  elif device_capability.major in (10, 11) and is_fa_version_supported(4):
if grep -q 'device_capability.major == 10 and is_fa_version_supported(4)' "$FA_UTILS"; then
    sed -i 's/device_capability.major == 10 and is_fa_version_supported(4)/device_capability.major in (10, 11) and is_fa_version_supported(4)/' "$FA_UTILS"
    echo "  Patched FA4 version gate (major == 10 -> major in (10, 11))"
else
    echo "  FA4 version gate already patched or pattern changed (skipping)"
fi

# Patch 2: Head-size fallback guard — on SM110, falling back to FA2 crashes.
# The original code falls back to fa_version=2 when head_size > 128 (and != 192).
# On SM110 this is fatal. Instead, stay on FA4 and let FA4's own head_size
# dispatch handle the TMEM limits (it falls back to non-CuTe path internally).
# This only matters for Gemma 4 (head_dim=256/512) — Qwen3.5 (head_dim=128) is fine.
# For SM110, we replace the FA2 fallback with a warning and keep FA4.
if grep -q 'fa_version = 2  # Fall back to FA2 for unsupported head sizes' "$FA_UTILS" 2>/dev/null; then
    # Only apply if we can find the exact fallback pattern
    python3 - "$FA_UTILS" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

# Find the head_size fallback block and wrap it in a device check
# so SM110 never falls back to FA2 (which would crash)
old = "fa_version = 2  # Fall back to FA2 for unsupported head sizes"
new = """fa_version = 2  # Fall back to FA2 for unsupported head sizes
        # SM110: FA2 crashes unconditionally — keep FA4, let CuTe handle limits
        if device_capability.major == 11:
            import logging
            logging.getLogger(__name__).warning(
                "SM110: head_size %d unsupported by FA4 TMEM, but FA2 crashes "
                "on SM110. Keeping FA4 — expect CuTe internal fallback or error.",
                head_size)
            fa_version = 4"""

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("  Patched head-size FA2 fallback (SM110 safety guard)")
else:
    print("  Head-size fallback pattern not found (skipping — may be fine)")
PYEOF
else
    echo "  Head-size FA2 fallback not found (skipping — may already be handled)"
fi

echo "  FA4 SM110 patches complete."
