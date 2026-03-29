#!/bin/bash

# Detect site-packages path (portable across Python versions)
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

echo "Patching Qwen3-Coder-Next crashing on start"
patch --forward --batch -p1 -d "$SITE_PACKAGES" < fix_crash.diff \
    && echo "  Applied fix_crash.diff" \
    || echo "  Patch not applicable (already fixed upstream), skipping"

echo "Reverting PR #34279 that causes slowness"
patch --reverse --batch -p1 -d "$SITE_PACKAGES" < fix_slowness.diff \
    && echo "  Reverted fix_slowness.diff" \
    || echo "  Can't revert PR #34279 (already reverted upstream), skipping"

echo "Fixing Triton allocator bug"
cp _triton* "$SITE_PACKAGES/"
