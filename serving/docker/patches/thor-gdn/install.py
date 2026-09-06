"""Redirect only the missing GDN operator; leave other preview operators intact."""
from pathlib import Path
import ast

p = Path('/usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py')
s = p.read_text()
old = '    torch.ops._C.fused_gdn_decode_post_conv_mtp('
new = '    torch.ops.thor_gdn.fused_gdn_decode_post_conv_mtp('
assert s.count(old) == 1, 'Upstream wrapper changed; review required'
s = s.replace(old, new)
s += '\n# Thor GDN kernel built from pinned upstream source for SM110.\ntorch.ops.load_library("/opt/thor-gdn/build/thor_gdn.so")\n'
ast.parse(s)
p.write_text(s)
