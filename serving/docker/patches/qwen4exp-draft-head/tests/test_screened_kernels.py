"""[screened-head] The real Triton kernels under the Triton CPU interpreter,
against the torch emulation (tests/_screened_common.py).

The interpreter mis-reads bf16 and fp8 tensors, so the kernels get fp32 tensors
that hold the exact bf16 / fp8 values (the kernels accumulate in fp32 either
way). This checks the indexing, masks, block maxima, top-k threshold, atomic
candidate compaction and the exact gather; not the dtype conversions on GPU.

Run: python -B tests/test_screened_kernels.py   (needs `pip install triton`)
"""
import os, sys

os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from _screened_common import make_head_weight, screen_ref, sh  # noqa: E402

torch.manual_seed(0)
V, H = 4096 + 37, 640  # vocab not a multiple of BLOCK_V, hidden not of BLOCK_H
W = make_head_weight(V + 64, H)  # padded rows beyond the vocabulary
head = sh.ScreenedLMHead(W, V, 20, 32)
ref_head = head
fp32 = sh.ScreenedLMHead.__new__(sh.ScreenedLMHead)
fp32.__dict__.update(head.__dict__)
fp32.w_fp8 = head.w_fp8.float()
fp32.weight = head.weight.float()
cases = 0
for n in (1, 2, 5, 16, 17, 20):
    for k in (1, 3, 20, 32):
        x = torch.randn(n, H)
        x[:, :4] *= 5
        x = x.bfloat16()
        ref, keep = screen_ref(ref_head, x, k)
        out = fp32(x.float(), k)
        assert torch.equal(out.isfinite(), keep), (n, k)
        assert torch.equal(fp32.num_cand.long(), keep.sum(1)), (n, k)
        # Exact rows: same products, other summation order: compare in bf16
        # with one-ulp slack, and to the fp32 reference loosely.
        diff = (out[keep].bfloat16().float() - ref[keep].float()).abs()
        assert torch.all(diff <= ref[keep].float().abs() * 2**-7 + 1e-6), (n, k)
        if k == 1:
            assert torch.equal(out.argmax(1), ref.float().argmax(1)), n
        cases += 1
print(f"kernels == torch emulation on {cases} cases (rows 1..20, k 1..32): ALL OK")
