#!/usr/bin/env python3
"""Standalone FlashInfer paged KV test for DFlash drafter dimensions.

Tests whether FlashInfer produces correct attention output when:
1. K/V are written to paged cache via reshape_and_cache_flash (NHD layout)
2. Attention is computed via BatchPrefillWithPagedKVCacheWrapper (reads HND-permuted view)

Compares FlashInfer output with PyTorch SDPA reference.

Drafter config: 32 Q heads, 8 KV heads, head_dim=128, page_size=32
Target config:  16 Q heads, 4 KV heads, head_dim=256, page_size=32
"""

import torch
import sys

def test_flashinfer_paged_attention(
    num_qo_heads, num_kv_heads, head_dim, page_size,
    num_context_tokens, num_query_tokens, label="test",
    causal=False
):
    """Test FlashInfer paged KV attention and compare with SDPA reference."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_pages = (num_context_tokens + num_query_tokens + page_size - 1) // page_size
    total_tokens = num_context_tokens + num_query_tokens

    # Allocate KV cache in NHD layout: [num_blocks, 2, page_size, num_kv_heads, head_dim]
    num_blocks = num_pages + 10  # extra blocks
    kv_cache = torch.zeros(num_blocks, 2, page_size, num_kv_heads, head_dim,
                           dtype=dtype, device=device)

    # Create block table: sequential blocks starting at block 5
    block_start = 5
    block_table = torch.arange(block_start, block_start + num_pages,
                               dtype=torch.int32, device=device).unsqueeze(0)  # [1, num_pages]

    # Generate random Q, K, V
    torch.manual_seed(42)
    # Context K/V (written by precompute)
    ctx_k = torch.randn(num_context_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    ctx_v = torch.randn(num_context_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    # Query Q, K, V
    query_q = torch.randn(num_query_tokens, num_qo_heads, head_dim, dtype=dtype, device=device)
    query_k = torch.randn(num_query_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    query_v = torch.randn(num_query_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)

    # === Step 1: Write K/V to cache using reshape_and_cache_flash ===
    # Context slot mapping
    ctx_slots = torch.arange(num_context_tokens, dtype=torch.int64, device=device)
    ctx_slots = block_table[0, ctx_slots // page_size] * page_size + ctx_slots % page_size

    # Query slot mapping
    qry_positions = torch.arange(num_context_tokens, total_tokens, dtype=torch.int64, device=device)
    qry_slots = block_table[0, qry_positions // page_size] * page_size + qry_positions % page_size

    k_scale = torch.tensor(1.0, device=device)
    v_scale = torch.tensor(1.0, device=device)

    # Write context K/V
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        ctx_k, ctx_v, kv_cache[:, 0], kv_cache[:, 1],
        ctx_slots, "auto", k_scale, v_scale,
    )
    # Write query K/V
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        query_k, query_v, kv_cache[:, 0], kv_cache[:, 1],
        qry_slots, "auto", k_scale, v_scale,
    )

    # Verify cache write
    slot0 = ctx_slots[0].item()
    b_idx, p_idx = slot0 // page_size, slot0 % page_size
    readback_k = kv_cache[b_idx, 0, p_idx]
    diff = (readback_k.float() - ctx_k[0].float()).abs().max().item()
    assert diff == 0, f"Cache write verification failed: diff={diff}"

    # === Step 2: SDPA reference (ground truth) ===
    all_k = torch.cat([ctx_k, query_k], dim=0)  # [total, kv_heads, dim]
    all_v = torch.cat([ctx_v, query_v], dim=0)

    # GQA expansion
    gqa_ratio = num_qo_heads // num_kv_heads
    q_sdpa = query_q.unsqueeze(0).transpose(1, 2)  # [1, qo_heads, num_query, dim]
    k_sdpa = all_k.unsqueeze(0).transpose(1, 2)    # [1, kv_heads, total, dim]
    v_sdpa = all_v.unsqueeze(0).transpose(1, 2)
    if gqa_ratio > 1:
        k_sdpa = k_sdpa.repeat_interleave(gqa_ratio, dim=1)
        v_sdpa = v_sdpa.repeat_interleave(gqa_ratio, dim=1)

    scale = head_dim ** -0.5
    sdpa_out = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, attn_mask=None, scale=scale,
    )
    sdpa_out = sdpa_out.squeeze(0).transpose(0, 1).contiguous()  # [num_query, qo_heads, dim]

    # === Step 3: FlashInfer paged attention ===
    try:
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    except ImportError:
        print(f"  [{label}] FlashInfer not available, skipping")
        return None

    # HND permutation (what vLLM does for SM110)
    # Try both NHD and HND to see which one works
    results = {}
    for layout_name, stride_order in [("NHD", (0, 1, 2, 3, 4)), ("HND", (0, 1, 3, 2, 4))]:
        kv_cache_view = kv_cache.permute(*stride_order)

        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            backend="auto",
        )

        # Plan
        qo_indptr = torch.tensor([0, num_query_tokens], dtype=torch.int32, device="cpu")
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cpu")
        kv_indices = block_table[0, :num_pages].cpu().to(torch.int32)
        last_page_len = torch.tensor([total_tokens - (num_pages - 1) * page_size],
                                     dtype=torch.int32, device="cpu")

        try:
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=kv_indices,
                paged_kv_last_page_len=last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                causal=causal,
                sm_scale=scale,
                q_data_type=dtype,
                kv_data_type=dtype,
            )

            fi_out = wrapper.run(query_q, kv_cache_view)  # [num_query, qo_heads, dim]

            # Compare with SDPA
            diff_max = (fi_out.float() - sdpa_out.float()).abs().max().item()
            diff_mean = (fi_out.float() - sdpa_out.float()).abs().mean().item()

            results[layout_name] = {
                "max_diff": diff_max,
                "mean_diff": diff_mean,
                "output_sample": fi_out[0, 0, :4].tolist(),
            }

            status = "OK" if diff_max < 0.1 else "MISMATCH"
            print(f"  [{label}] {layout_name}: {status} max_diff={diff_max:.6e} mean_diff={diff_mean:.6e}")

        except Exception as e:
            print(f"  [{label}] {layout_name}: ERROR {e}")
            results[layout_name] = {"error": str(e)}

    print(f"  [{label}] SDPA reference sample: {sdpa_out[0, 0, :4].tolist()}")
    return results


def main():
    print("=" * 70)
    print("FlashInfer Paged KV Test — Drafter vs Target dimensions")
    print("=" * 70)

    # Import vLLM ops
    try:
        import vllm._custom_ops  # registers reshape_and_cache_flash
    except:
        print("ERROR: vllm custom ops not available")
        sys.exit(1)

    print("\n--- Test 1: Target dimensions (should work) ---")
    test_flashinfer_paged_attention(
        num_qo_heads=16, num_kv_heads=4, head_dim=256, page_size=32,
        num_context_tokens=20, num_query_tokens=4,
        label="target", causal=True,
    )

    print("\n--- Test 2: Drafter dimensions, causal=True ---")
    test_flashinfer_paged_attention(
        num_qo_heads=32, num_kv_heads=8, head_dim=128, page_size=32,
        num_context_tokens=20, num_query_tokens=4,
        label="drafter-causal", causal=True,
    )

    print("\n--- Test 3: Drafter dimensions, causal=False (DFlash mode) ---")
    test_flashinfer_paged_attention(
        num_qo_heads=32, num_kv_heads=8, head_dim=128, page_size=32,
        num_context_tokens=20, num_query_tokens=4,
        label="drafter-noncausal", causal=False,
    )

    print("\n--- Test 4: Drafter dimensions, small context (1 page) ---")
    test_flashinfer_paged_attention(
        num_qo_heads=32, num_kv_heads=8, head_dim=128, page_size=32,
        num_context_tokens=12, num_query_tokens=4,
        label="drafter-small", causal=False,
    )

    print("\n--- Test 5: Drafter dimensions, page_size=16 ---")
    test_flashinfer_paged_attention(
        num_qo_heads=32, num_kv_heads=8, head_dim=128, page_size=16,
        num_context_tokens=20, num_query_tokens=4,
        label="drafter-ps16", causal=False,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
