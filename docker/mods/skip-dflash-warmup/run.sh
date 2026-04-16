#!/bin/bash
# Skip DFlash dummy_run during warmup to avoid Xid 43 GPU crash on SM110.
# The DFlash dummy_run triggers a FlashInfer/Triton JIT kernel compilation
# that produces a bad cubin on SM110, crashing the GPU.
# By skipping it, we lose memory profiling accuracy for the drafter but
# the server can start. The drafter will JIT-compile on first real request.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
DFLASH_PY="$SITE_PACKAGES/vllm/v1/spec_decode/dflash.py"

echo "Patching DFlash dummy_run to no-op..."

python3 - "$DFLASH_PY" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

old = """    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:"""

new = """    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # SKIP: DFlash dummy_run causes Xid 43 on SM110 during JIT compilation.
        # The drafter will JIT on first real request instead.
        import logging
        logging.getLogger("dflash.skip").warning(
            f"SKIP dummy_run (num_tokens={num_tokens}) to avoid SM110 Xid 43"
        )
        return"""

if old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("  DFlash dummy_run patched to no-op")
else:
    print("  WARNING: dummy_run pattern not found")
PYEOF

echo "  Done."
