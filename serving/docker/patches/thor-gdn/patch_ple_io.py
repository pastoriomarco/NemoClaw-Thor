"""Avoid irrelevant readahead for random, disk-backed PLE row lookups.

No tensor layout or numerical changes. MADV_RANDOM is also used in
Saren-Arterius/qwen3.8-Flash-DGX-AutoRound and MiaAI-Lab's Spark recipe.
Keep a runtime opt-out for experiments/storage with different characteristics.
"""
from pathlib import Path
import ast

p = Path('/usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py')
s = p.read_text()
old = '            self.rows_total += rows\n'
new = '''            if _env_int("VLLM_PLE_MMAP_MADV_RANDOM", 1):
                import mmap
                self.mm[idx]._mmap.madvise(mmap.MADV_RANDOM)
            self.rows_total += rows
'''
assert s.count(old) == 1, 'Pinned PLE implementation changed; inspect before patching'
s = s.replace(old, new)
ast.parse(s)
p.write_text(s)
