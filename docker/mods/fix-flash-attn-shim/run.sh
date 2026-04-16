#!/bin/bash

# Create a flash_attn package symlink → vllm.vllm_flash_attn
#
# FA4 CuTe DSL code is vendored at vllm.vllm_flash_attn.cute but all internal
# imports reference flash_attn.cute.* (the standalone Dao-AILab package).
# When the standalone package isn't installed, these imports fail.
#
# Fix: symlink flash_attn → the vendored copy so Python resolves all imports.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FA_DIR="$SITE_PACKAGES/flash_attn"
VENDORED="$SITE_PACKAGES/vllm/vllm_flash_attn"

if [ -d "$FA_DIR" ] && [ ! -L "$FA_DIR" ]; then
    echo "  flash_attn package already exists (not a symlink), skipping"
    exit 0
fi

if [ -L "$FA_DIR" ]; then
    echo "  flash_attn symlink already exists, skipping"
    exit 0
fi

ln -s "$VENDORED" "$FA_DIR"
echo "  Symlinked flash_attn → vllm/vllm_flash_attn"
