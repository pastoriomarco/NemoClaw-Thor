# NemoClaw-Thor v7 Build-Time Patch Audit

Updated: **2026-04-27** (revised after iterative build attempts)
FlashInfer: tag `v0.6.9`
vLLM: tag `v0.20.0` (commit `b8160878f0`)

This file documents every build-time source modification in v7 and explains why
each is or is not present, with the lessons learned from a series of failed
build attempts.

---

## Final v7 build-time modifications (only these run)

| Mechanism | Target | Fires? | Effect |
|---|---|---|---|
| `docker/patches/flashinfer_cache.patch` | `flashinfer/artifacts.py` (FlashInfer v0.6.9) | **Yes** (fuzzy +50) | Adds checksum-validated local-cache fast path for cubin downloads. Pure build-speed optimization. |
| Inline RUN: CMakeLists.txt 11.0f verification | `vllm/CMakeLists.txt` | **Yes** | Read-only `grep -c` reporting; does NOT modify. Prints expected counts (FP4=1, SCALED=2, MLA=1, total=8) for sanity. |
| Inline RUN: MoE backend verification | `flashinfer_cutlass_moe.py` + 4 `experts/trtllm_*_moe.py` | **Yes** | **Read-only** — confirms CUTLASS MoE has SM110 family check upstream, prints presence info for the 4 TRT-LLM Gen MoE files. Does NOT modify any source. |

**No v7 patch can fail the build.** All three are either fuzz-tolerant or read-only.

---

## Detailed audit

### Patch 1 — `docker/patches/flashinfer_cache.patch` ✓ KEPT

**Target:** `flashinfer/artifacts.py`, hunk header `@@ -203,9 +203,13 @@`.

**Behavior:** Modifies the cubin download loop so that locally-cached cubins are
verified by checksum and reused if they match, instead of unconditionally
re-downloading.

**v0.6.9 reality:** Hunk header is off — actual code lives at lines **253–266**
in v0.6.9 (offset +50). The context-line text matches byte-for-byte. `patch`
applies with `Hunk #1 succeeded at 253 with fuzz 1` (verified empirically in
build attempt 2026-04-27).

**Why fuzz=1, offset=+50 is safe here:**
- The patched lines (`for name, _ in cubin_files`, `source = safe_urljoin(...)`,
  `local_path = FLASHINFER_CUBIN_DIR / name`) are byte-identical to v0.6.9's
  source — only the surrounding line numbers drifted because unrelated code was
  added above the function.
- `checksum` variable is in scope: `cubin_files: list[tuple[str, str]]` carries
  `(name, checksum)` already.
- Post-loop verify (lines 272–276 in v0.6.9) still runs unconditionally, so
  cache hits are validated and corrupt local cubins still get re-downloaded.

**Is it correctness-critical?** No — pure build-speed. Without the patch each
rebuild re-downloads ~10,600 cubins (~10 min wasted bandwidth). Functionality
unchanged.

**v8 hardening (optional):** Re-target the hunk header to `@@ -253,9 +253,13 @@`
to silence the fuzz warning and make the patch resilient to further line drift
above the function.

---

### Patch 2 — `vllm_sm110_no_sm100_cutlass.patch` ✗ DELETED 2026-04-27

**Original behavior (v6):** Removed `11.0f` from `SCALED_MM_ARCHS` so SM110
wouldn't try to compile the SM_100-only CUTLASS scaled_mm kernels (which had
runtime guards rejecting SM110).

**Why deleted:**
1. vLLM PR #39233 made SM_110 a first-class build target. v0.20.0's CMakeLists
   includes `11.0f` natively in 4 arch lists (FP4, SCALED_MM, MLA, MoE). We
   *want* those compiled in.
2. The patch's hunk headers were already stale by +182 lines vs v0.20.0; would
   have fallen through the Dockerfile's `|| echo "skipping"` fallback anyway.
3. Even if it had fired, it would have stripped arches we want kept — wrong
   direction.

The Dockerfile retains a read-only `grep -c` verification block that prints
the 11.0f counts for visibility, useful as a regression detector if a future
upstream change ever drops them.

---

### Patch 3 — Inline TRT-LLM Gen MoE family-gate rewrite ✗ DROPPED 2026-04-27

**Original goal (v6):** Add SM110/SM120 to the gate
`p.is_cuda() and p.is_device_capability_family(100)` in
`flashinfer_trtllm_moe.py`, so vLLM doesn't reject SM110 when selecting the
TRT-LLM Gen MoE backend.

**Why we dropped the rewrite for v7:**

The journey, in order, with each lesson:

1. **Initial attempt:** kept the v6 patch as-is (target file
   `flashinfer_trtllm_moe.py`). **Silent no-op** — the file was renamed in
   v0.20.0. Almost shipped a broken image.

2. **Second attempt:** rewrote to target the four split files
   `experts/trtllm_{bf16,fp8,mxfp4,nvfp4}_moe.py`, made the patch **strict**
   (raise SystemExit(1) on any pattern miss). Build hard-failed at step #19
   with `PATTERN NOT FOUND` for 3 of 4 files. **The strict guard caught what
   would have been a partial-patch silent ship.**

3. **Investigation:** fetched the four upstream files. Discovered:
   - `mxfp4`: single-line gate
     `p.is_cuda() and p.is_device_capability_family(100) and has_flashinfer()`
   - `bf16`, `fp8`, `nvfp4`: **multi-line** gate inside `return (...)`:
     ```python
                 p.is_cuda()
                 and p.is_device_capability_family(100)
     ```
     (12-space indent, `\n` between, in a tuple expression).

4. **Strategic re-think (instead of writing a third patch attempt):**
   - Our active profiles all set `VLLM_USE_FLASHINFER_MOE_FP4=1` and
     `VLLM_FLASHINFER_MOE_BACKEND=latency`. Both env vars verified present in
     v0.20.0's `vllm/envs.py`.
   - Those env vars route NVFP4 MoE to the **CUTLASS** path
     (`flashinfer_cutlass_moe.py`), **not** TRT-LLM Gen.
   - `flashinfer_cutlass_moe.py` already includes SM110+SM120 in its family
     check upstream at lines 131–133 (verified: `is_device_capability_family(110)`).
   - Therefore: TRT-LLM Gen MoE rejecting SM110 has **no effect on us**.
   - The v6 patch, in retrospect, was solving a problem we don't have. It
     was probably operating as a no-op even when it "worked" in v6.

5. **Conclusion:** keep the read-only verification (it confirms the CUTLASS
   path is SM110-ready), drop the rewrite. If a future profile genuinely needs
   TRT-LLM Gen MoE on SM110, add a targeted patch in v8 with **offline
   pre-validation** against the exact upstream source before kicking off the
   multi-hour build.

**Risk of having dropped it:** zero for current profiles. Hypothetical risk for
future profiles that explicitly select TRT-LLM Gen MoE — they would fall back
to CUTLASS, which is already SM110-supported. Worst case: minor perf delta
versus the (unverified) TRT-LLM Gen path.

---

## Lessons learned (process)

1. **Strict guards are worth their cost.** The strict `raise SystemExit(1)` in
   the v7 attempt #2 caught a partial-patch silent failure. Without it, we'd
   have shipped an image where `mxfp4` was patched but `bf16/fp8/nvfp4` weren't.

2. **String-replace patches are brittle to formatting.** A single-line pattern
   doesn't match a multi-line tuple even when the logic is identical. For any
   future patches, either:
   - Validate against the exact upstream source offline before the build, or
   - Use a robust regex / AST-level transformation, or
   - Use a unified-diff `.patch` file with explicit context lines.

3. **Pre-validate before multi-hour builds.** Each retry in v7 cost ~10–15 min
   wasted (build time up to the failure step) plus ~10 min restart prep. The
   "be careful patches actually work" principle should be: never modify a
   build-time patch without also confirming it applies cleanly to the exact
   upstream ref.

4. **Question whether a patch is actually needed before fixing it.** The MoE
   gate patch turned out to be unneeded for our profile set. Asking "what
   happens if we don't patch at all?" surfaced this — and the answer was
   "nothing, our env vars route around it." Always ask that question first.

5. **Use empirical evidence over inherited assumptions.** The v6 inline patch
   pattern was inherited from older vLLM versions. We assumed it still applied
   to v0.20.0. It silently no-op'd. The audit / pre-validation discipline is
   the only way to catch this.

---

## Summary

| Modification | Status | Risk |
|---|---|---|
| `flashinfer_cache.patch` (build-speed) | KEPT, applies fuzzy | None |
| `vllm_sm110_no_sm100_cutlass.patch` | DELETED 2026-04-27 | None — wrong direction for v0.20.0 |
| Inline TRT-LLM Gen MoE rewrite | DROPPED 2026-04-27 | None — env vars route to SM110-ready CUTLASS path |
| Inline CMakeLists.txt verification | KEPT, read-only | None — informational |
| Inline MoE verification | KEPT, read-only | None — informational |

**Net:** v7 image will contain exactly **one** non-upstream source change
(the FlashInfer cubin-cache patch), and it's pure build-speed.
