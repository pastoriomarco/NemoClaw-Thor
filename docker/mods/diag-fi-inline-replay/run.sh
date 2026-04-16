#!/bin/bash
# Inline replay: inside the DFlash forward, after FlashInfer .run(),
# create a FRESH wrapper with MANUAL .plan() using the exact same
# kv_cache tensor and query, compare outputs with SDPA reference.
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying inline replay diagnostic..."

python3 - "$QWEN3_DFLASH_PY" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

old = """        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output"""

new = '''        attn_output = self.attn(q, k, v)

        # INLINE REPLAY: compare pipeline FlashInfer vs fresh wrapper vs SDPA
        import sys as _sys
        _rn = getattr(DFlashQwen3Attention, '_rn', 0)
        kv_cache = self.attn.kv_cache
        has_cache = kv_cache is not None and kv_cache.numel() > 0
        num_tokens = q.shape[0]

        # Skip dummy runs: check if forward context has real metadata
        from vllm.forward_context import get_forward_context as _gfc
        _fctx = _gfc()
        _has_meta = (_fctx.attn_metadata is not None and
                     self.attn.layer_name in (_fctx.attn_metadata or {}))
        if _rn < 2 and has_cache and num_tokens > 1 and _has_meta:
            DFlashQwen3Attention._rn = _rn + 1
            p = lambda *a: print(*a, file=_sys.stderr, flush=True)
            p(f"\\n{'='*60}")
            p(f"INLINE-REPLAY step={_rn} layer={self.attn.layer_name}")

            q_h = q.view(num_tokens, self.num_heads, self.head_dim)
            k_h = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            v_h = v.view(num_tokens, self.num_kv_heads, self.head_dim)

            # 1) Pipeline FlashInfer output
            fi_pipe = attn_output.float()
            p(f"  FI-pipeline: norm={fi_pipe.norm():.2f} zero={fi_pipe.abs().max() < 1e-6}")

            from vllm.forward_context import get_forward_context
            fwd_ctx = get_forward_context()
            attn_meta = fwd_ctx.attn_metadata.get(self.attn.layer_name) if fwd_ctx.attn_metadata else None

            # Get slot mapping and write query KV to cache for replay
            slot_mapping = fwd_ctx.slot_mapping
            if isinstance(slot_mapping, dict):
                slot_mapping = slot_mapping.get(self.attn.layer_name)

            # 2) Fresh wrapper replay with SAME kv_cache tensor
            if attn_meta and hasattr(attn_meta, 'prefill') and attn_meta.prefill:
                try:
                    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

                    # Extract plan params from existing wrapper
                    existing = attn_meta.prefill.wrapper
                    page_size = kv_cache.shape[2]  # from cache tensor
                    seq_lens = attn_meta.seq_lens
                    block_table = attn_meta.block_table if hasattr(attn_meta, 'block_table') else None

                    # Get paged_kv info from the metadata
                    # We need qo_indptr, paged_kv_indptr, paged_kv_indices, last_page_len
                    # These are stored internally in the existing wrapper after .plan()
                    # Instead, compute them manually from seq_lens and block_table

                    total_seq = seq_lens[0].item() if seq_lens is not None else num_tokens
                    num_pages = (total_seq + page_size - 1) // page_size

                    p(f"  page_size={page_size} total_seq={total_seq} num_pages={num_pages}")
                    p(f"  kv_cache: {kv_cache.shape} stride={kv_cache.stride()}")

                    # Get block table from CommonAttentionMetadata
                    bt = attn_meta.block_table if hasattr(attn_meta, 'block_table') else None
                    if bt is None and hasattr(attn_meta, 'block_tables'):
                        bt = attn_meta.block_tables
                    if bt is None:
                        # Try to get from paged_kv_indices in the existing wrapper
                        p(f"  No block_table on metadata, checking wrapper internals")
                        for attr in dir(existing):
                            if 'indic' in attr.lower() or 'indptr' in attr.lower() or 'page' in attr.lower():
                                val = getattr(existing, attr, None)
                                if torch.is_tensor(val) and val.numel() < 20:
                                    p(f"    {attr}: {val.tolist()}")
                                elif torch.is_tensor(val):
                                    p(f"    {attr}: shape={val.shape}")
                                elif val is not None:
                                    p(f"    {attr}: {val}")

                    if bt is not None:
                        p(f"  block_table: {bt.shape}, first row[:5]={bt[0, :min(5, bt.shape[1])].tolist()}")
                        page_indices = bt[0, :num_pages].cpu().to(torch.int32)

                        # Create fresh wrapper and plan manually
                        ws = torch.empty(128*1024*1024, dtype=torch.uint8, device=kv_cache.device)
                        fresh = BatchPrefillWithPagedKVCacheWrapper(ws, backend="auto")
                        qo_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device="cpu")
                        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cpu")
                        last_page_len = torch.tensor(
                            [total_seq - (num_pages - 1) * page_size],
                            dtype=torch.int32, device="cpu")

                        p(f"  FRESH plan: qo_indptr={qo_indptr.tolist()} kv_indptr={kv_indptr.tolist()} "
                          f"indices={page_indices.tolist()[:5]} last_page_len={last_page_len.tolist()}")

                        fresh.plan(
                            qo_indptr=qo_indptr,
                            paged_kv_indptr=kv_indptr,
                            paged_kv_indices=page_indices,
                            paged_kv_last_page_len=last_page_len,
                            num_qo_heads=self.num_heads,
                            num_kv_heads=self.num_kv_heads,
                            head_dim_qk=self.head_dim,
                            page_size=page_size,
                            causal=False,
                            sm_scale=self.head_dim ** -0.5,
                            q_data_type=q_h.dtype,
                            kv_data_type=kv_cache.dtype,
                        )

                        fresh_out = torch.empty_like(q_h)
                        fresh.run(q_h, kv_cache, k_scale=1.0, v_scale=1.0, out=fresh_out)
                        fresh_flat = fresh_out.view(num_tokens, -1).float()
                        p(f"  FRESH .run(): norm={fresh_flat.norm():.2f} zero={fresh_flat.abs().max() < 1e-6}")

                        # Compare pipeline vs fresh
                        diff_pipe_fresh = (fi_pipe - fresh_flat).abs().max().item()
                        p(f"  PIPELINE vs FRESH: max_diff={diff_pipe_fresh:.6e}")
                    else:
                        p(f"  Cannot create fresh wrapper: no block_table available")

                except Exception as e:
                    import traceback
                    p(f"  FRESH replay error: {e}")
                    traceback.print_exc(file=_sys.stderr)

            # 3) SDPA reference (read from cache + compute)
            ctx_kv = getattr(self, '_sdpa_ctx_kv', None)
            if ctx_kv is not None:
                ctx_k, ctx_v = ctx_kv
                full_k = torch.cat([ctx_k, k_h], dim=0)
                full_v = torch.cat([ctx_v, v_h], dim=0)
            else:
                full_k = k_h
                full_v = v_h
            q_s = q_h.unsqueeze(0).transpose(1, 2)
            g = self.num_heads // self.num_kv_heads
            k_s = full_k.unsqueeze(0).transpose(1, 2)
            v_s = full_v.unsqueeze(0).transpose(1, 2)
            if g > 1:
                k_s = k_s.repeat_interleave(g, dim=1)
                v_s = v_s.repeat_interleave(g, dim=1)
            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_s, k_s, v_s, attn_mask=None, scale=self.head_dim**-0.5)
            sdpa_flat = sdpa_out.squeeze(0).transpose(0, 1).contiguous().view(num_tokens, -1).float()
            p(f"  SDPA: norm={sdpa_flat.norm():.2f}")

            if fi_pipe.abs().max() > 1e-6:
                diff_fi_sdpa = (fi_pipe - sdpa_flat).abs().max().item()
                p(f"  FI-PIPELINE vs SDPA: max_diff={diff_fi_sdpa:.4f}")

            p(f"{'='*60}")

        output, _ = self.o_proj(attn_output)
        return output'''

if old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("  [1/1] Added inline replay diagnostic")
else:
    print("  [1/1] WARNING: pattern not found")

PYEOF
echo "  Inline replay diagnostic complete."
