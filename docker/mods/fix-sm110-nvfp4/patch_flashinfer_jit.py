#!/usr/bin/env python3
"""Patch flashinfer/compilation_context.py: treat SM110 as matching SM100 family
for JIT compilation. SM110 is enterprise Blackwell, same arch as SM100/SM103."""
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

# The get_nvcc_flags_list method filters TARGET_CUDA_ARCHS by major version.
# SM110 (major=11) doesn't match when [10, 12] is requested.
# Patch: also include major 11 when major 10 is requested (SM110 enterprise Blackwell).
old = """\
        if supported_major_versions:
            supported_cuda_archs = [
                major_minor_tuple
                for major_minor_tuple in self.TARGET_CUDA_ARCHS
                if major_minor_tuple[0] in supported_major_versions
            ]"""

new = """\
        if supported_major_versions:
            # SM110 enterprise Blackwell: include major 11 when 10 is requested
            effective_versions = set(supported_major_versions)
            if 10 in effective_versions:
                effective_versions.add(11)
            supported_cuda_archs = [
                major_minor_tuple
                for major_minor_tuple in self.TARGET_CUDA_ARCHS
                if major_minor_tuple[0] in effective_versions
            ]"""

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Patched {path} (SM110 included for SM100 family JIT)")
else:
    if "SM110 enterprise Blackwell" in content:
        print(f"  Already patched (skipping)")
    else:
        print(f"  WARNING: Could not match code in {path}")
        sys.exit(1)
