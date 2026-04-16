#!/usr/bin/env python3
"""Test FlashInfer paged attention with KNOWN cache pattern.
Write specific values to cache, verify FlashInfer reads them correctly.

Key test: write V = identity-like pattern where each token has a unique value.
If FlashInfer reads from wrong pages, the output won't match the expected pattern.
"""
import torch
import sys

def test():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Match the pipeline's exact parameters
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    num_context = 11  # prompt tokens
    num_query = 9     # 1 bonus + 8 masks
    total_seq = num_context + num_query  # 20
    num_pages = (total_seq + page_size - 1) // page_size  # 2

    # Allocate cache matching pipeline shape: [num_blocks, 2, page_size, kv_heads, head_dim]
    num_blocks = 100
    kv_cache = torch.zeros(num_blocks, 2, page_size, num_kv_heads, head_dim,
                           dtype=dtype, device=device)

    # Block table: pages 5 and 6
    block_ids = [5, 6]

    # Write KNOWN pattern to cache:
    # For each token i, K[i] = i * 0.01, V[i] = i * 0.1
    # This way, attention output should be a weighted sum of token indices
    import vllm._custom_ops

    for token_idx in range(total_seq):
        page_idx = token_idx // page_size
        pos_in_page = token_idx % page_size
        block_id = block_ids[page_idx]

        k_val = torch.full((1, num_kv_heads, head_dim), token_idx * 0.01,
                           dtype=dtype, device=device)
        v_val = torch.full((1, num_kv_heads, head_dim), token_idx * 0.1,
                           dtype=dtype, device=device)

        slot = torch.tensor([block_id * page_size + pos_in_page],
                           dtype=torch.int64, device=device)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            k_val, v_val, kv_cache[:, 0], kv_cache[:, 1],
            slot, "auto", torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device))

    # Verify write
    for check_tok in [0, 5, 10, 15, 19]:
        pi = check_tok // page_size
        pp = check_tok % page_size
        bi = block_ids[pi]
        actual = kv_cache[bi, 1, pp, 0, 0].item()  # V value
        expected = check_tok * 0.1
        print(f"  Token {check_tok}: V[0,0]={actual:.3f} expected={expected:.3f} {'OK' if abs(actual-expected)<0.01 else 'MISMATCH'}")

    # Query: uniform Q (all ones scaled by sm_scale)
    # With uniform Q, attention should be uniform → output = mean(V)
    q = torch.ones(num_query, num_qo_heads, head_dim, dtype=dtype, device=device)

    # Run FlashInfer
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    workspace = torch.empty(128*1024*1024, dtype=torch.uint8, device=device)

    for backend in ["auto", "fa2"]:
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, backend=backend)

        qo_indptr = torch.tensor([0, num_query], dtype=torch.int32, device="cpu")
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cpu")
        kv_indices = torch.tensor(block_ids[:num_pages], dtype=torch.int32, device="cpu")
        last_page_len = torch.tensor([total_seq - (num_pages-1)*page_size],
                                     dtype=torch.int32, device="cpu")

        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=False,
            sm_scale=head_dim**-0.5,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        out = torch.empty(num_query, num_qo_heads, head_dim, dtype=dtype, device=device)
        wrapper.run(q, kv_cache, k_scale=1.0, v_scale=1.0, out=out)

        # With uniform Q and uniform K (all ~0.01*i), the softmax is approximately
        # uniform (all K vectors are similar), so output ≈ mean(V) ≈ mean(i*0.1)
        # mean(0..19)*0.1 = 9.5*0.1 = 0.95
        mean_out = out[0, 0, 0].item()
        expected_approx = sum(range(total_seq)) / total_seq * 0.1
        print(f"  [{backend}] out[0,0,0]={mean_out:.4f} expected≈{expected_approx:.4f} "
              f"norm={out.float().norm():.2f}")

    # SDPA reference
    all_k = torch.zeros(total_seq, num_kv_heads, head_dim, dtype=dtype, device=device)
    all_v = torch.zeros(total_seq, num_kv_heads, head_dim, dtype=dtype, device=device)
    for i in range(total_seq):
        all_k[i] = i * 0.01
        all_v[i] = i * 0.1

    q_s = q.unsqueeze(0).transpose(1,2)
    k_s = all_k.unsqueeze(0).transpose(1,2).repeat_interleave(num_qo_heads//num_kv_heads, dim=1)
    v_s = all_v.unsqueeze(0).transpose(1,2).repeat_interleave(num_qo_heads//num_kv_heads, dim=1)
    sdpa_out = torch.nn.functional.scaled_dot_product_attention(
        q_s, k_s, v_s, scale=head_dim**-0.5)
    sdpa_out = sdpa_out.squeeze(0).transpose(0,1)
    print(f"  [SDPA] out[0,0,0]={sdpa_out[0,0,0].item():.4f} norm={sdpa_out.float().norm():.2f}")

if __name__ == "__main__":
    print("=== FlashInfer Known Pattern Test ===")
    test()
    print("Done.")
