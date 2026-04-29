#!/usr/bin/env bash
# lib/launch.sh — Shared vLLM launcher logic for NemoClaw-Thor
#
# Source this file; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced, not executed directly." >&2
    exit 1
fi

prepare_thor_launch_profile() {
    local profile="${1:-${THOR_MODEL_PROFILE:-}}"

    THOR_VLLM_IMAGE="${THOR_VLLM_IMAGE:-nemoclaw-thor/vllm:latest}"
    THOR_VLLM_BIND_HOST="${THOR_VLLM_BIND_HOST:-0.0.0.0}"
    THOR_VLLM_PORT="${THOR_VLLM_PORT:-8000}"
    THOR_HF_CACHE_DIR="${THOR_HF_CACHE_DIR:-$HOME/thor-hf-cache}"
    THOR_VLLM_CACHE_DIR="${THOR_VLLM_CACHE_DIR:-$HOME/thor-vllm-cache}"
    THOR_TORCH_CACHE_DIR="${THOR_TORCH_CACHE_DIR:-$HOME/thor-torch-cache}"
    THOR_FLASHINFER_CACHE_DIR="${THOR_FLASHINFER_CACHE_DIR:-$HOME/thor-flashinfer-cache}"

    THOR_LAUNCH_HOST_MODEL_PATH=""
    THOR_LAUNCH_MODEL_SOURCE=""
    THOR_LAUNCH_GPU_MEMORY_UTILIZATION=""
    THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS=""
    THOR_LAUNCH_SPECULATIVE_CONFIG=""
    THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH=""
    THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH=""
    THOR_CHAT_TEMPLATE_HOST_DIR="${THOR_CHAT_TEMPLATE_HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/templates}"
    THOR_MODS_HOST_DIR="${THOR_MODS_HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../docker/mods" && pwd)}"

    THOR_DOCKER_ENV_ARGS=()
    THOR_VLLM_ARGS=()

    # SM110 (Thor): CUTLASS sm100 kernels are incompatible — disable them.
    # CutlassFp8BlockScaledMMKernel: uses enable_sm100f_only, crashes SM110 (Xid 43).
    # FlashInfer FP8 is re-enabled: JIT cache has sm_110a GEMM kernels.
    #
    # ENABLE_TRIATTENTION=0: disable TriAttention plugin (official off switch).
    # TriAttention auto-registers and crashes at inference time without a
    # sparse_stats_path (TRIATTN_FATAL_TRITON_SCORING_REQUIRED:stats_path_not_set).
    # Qwen3.6-35B-A3B is not in TriAttention's supported-model matrix — it would
    # require porting the CUDA calibration script for DeltaNet hybrid layers.
    # Revisit only if upstream adds Qwen3.6 support.
    THOR_DOCKER_ENV_ARGS+=(
        -e "VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel,CutlassFp8BlockScaledMMKernel"
        -e "ENABLE_TRIATTENTION=0"
    )

    case "${profile}" in
        # minimax-m2.7-139b-a10b-nvfp4 profile removed 2026-04-23.
        # See MINIMAX-M27-INVESTIGATION.md for the why — W4A4 NVFP4 MoE on SM110
        # has no fast kernel path; MARLIN fallback gave degraded output at 12 tok/s.
        # Runtime mod fix-nvfp4-moe-scale-merge is still shipped for potential
        # reuse with other NVFP4 split-scale checkpoints.
        # qwen3.5-122b-a10b-nvfp4 profile removed 2026-04-24 — superseded by qwen3.6.
        # qwen3.5-122b-a10b-nvfp4-resharded removed
        # qwen3.6-35b-a3b-fp8-dflash REMOVED 2026-04-28 — FP8-weights variant
        # of NVFP4 alternative, plus DFlash N=15 was empirically agentic-bad
        # (TEB 46). Heavy-coding workloads now use qwen3.6-35b-a3b-nvfp4-dflash
        # (same RedHatAI/Qwen3.6-35B-A3B-NVFP4 weights as the agentic profiles
        # — saves ~17 GB Qwen FP8 weights on disk — and uses N=8 which retains
        # the burst-throughput edge while staying in the v6 87/100 ★★★★ band).
        # qwen3.6-27b-fp8-dflash REMOVED 2026-04-28 — DFlash N=15 scored TEB 40
        # (★★ Weak); dominated by qwen3.6-27b-fp8-mtp-kvfp8 (TEB 84). 27B-DFlash
        # drafter is gated and adds no value on Thor at this N. See PERFORMANCE-V7.md.
        # qwen3.6-35b-a3b-fp8-mtp-fp8kv REMOVED 2026-04-28 — FP8-weights variant
        # of an NVFP4 profile that's strictly better at every metric. NVFP4 weights
        # available via RedHatAI/Qwen3.6-35B-A3B-NVFP4 → use nvfp4-mtp-fp8kv (TEB
        # 93) or nvfp4-tq-mtp (TEB 90, +1.4× context) instead.
        # qwen3.6-35b-a3b-nvfp4-dflash-vl REMOVED 2026-04-28 — vision support
        # folded into qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge (production profile
        # now serves agentic + vision in one). Same RedHatAI/Qwen3.6-35B-A3B-NVFP4
        # weights. The MTP-2 + TQ KV path beats DFlash-15 on agentic correctness
        # (90 vs 46 TEB) without losing the vision capability.
        cosmos-reason2-2b)
            # NVIDIA Cosmos Reason 2 (2B) — VLM for physical AI reasoning,
            # post-trained from Qwen3-VL-2B-Instruct. Qwen3VLForConditionalGeneration
            # architecture, model_type qwen3_vl.
            # LLM: head_dim=128 → flash_attn works on SM110.
            # ViT: head_dim=64, 24 layers, patch=16, spatial-merge=2.
            # TORCH_SDPA added for ViT as a conservative SM110 workaround
            # (same pattern as the 9B VLM profile; head_dim=64 is small enough
            # that FlashInfer may work too, but TORCH_SDPA is known-safe).
            # No matched drafter available — no speculative decoding.
            # BF16 native weights (no NVFP4/FP8 release yet).
            # Chat template is bundled in the repo (chat_template.json) — vLLM
            # auto-loads it; no --chat-template override needed.
            # Sized for 2×32K concurrent context: FP8 KV + 0.12 gpu_mem_util
            # gives ~15 GB total reservation (weights 4.3 + KV ~8 + buffers ~2.5).
            # Prior measurement at 0.20 / BF16 KV / max_num_seqs=8 used ~22 GB
            # and allocated 140K KV tokens — overkill for BT reasoning loops.
            # Tool-call format: Cosmos emits hermes-style <tool_call>{...}</tool_call>
            # tags (inherited from Qwen3-VL-2B-Instruct post-training), NOT Qwen3.6's
            # XML-attribute format. Must use `hermes` parser, not `qwen3_xml`.
            THOR_LAUNCH_MODEL_SOURCE="nvidia/Cosmos-Reason2-2B"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.12}"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flashinfer"
                "--enforce-eager"
                "--mm-encoder-attn-backend" "TORCH_SDPA"
                "--kv-cache-dtype" "fp8"
                "--max-num-batched-tokens" "8192"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "hermes"
            )
            ;;
        cosmos-reason2-8b)
            # NVIDIA Cosmos Reason 2 (8B) — VLM for physical AI reasoning,
            # post-trained from Qwen3-VL-8B. Qwen3VLForConditionalGeneration,
            # model_type qwen3_vl. LLM: 36 layers, hidden 4096, 32 heads,
            # 8 KV heads, head_dim=128 → flash_attn compatible on SM110
            # (but FP8 KV requires flashinfer regardless).
            # Tool parser: `hermes` (same hermes-format tool calls as the 2B
            # variant — verified empirically 2026-04-19; qwen3_xml does not match).
            # ViT: same pattern as 2B (TORCH_SDPA workaround for SM110).
            # Sized for 3×32K concurrent context: FP8 KV needs ~7 GB for 96K
            # tokens, weights ~16 GB (bf16), ViT ~1.5 GB, activations ~2 GB
            # → ~27 GB → gpu_mem_util 0.25 on Thor. Leaves room to co-serve
            # with the Qwen3.6 manyforge profile (0.32 + 0.25 = 0.57).
            # Gated repo — HF_TOKEN required (start-duo.sh auto-reads it).
            THOR_LAUNCH_MODEL_SOURCE="nvidia/Cosmos-Reason2-8B"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.25}"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flashinfer"
                "--enforce-eager"
                "--mm-encoder-attn-backend" "TORCH_SDPA"
                "--kv-cache-dtype" "fp8"
                "--max-num-batched-tokens" "8192"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "hermes"
            )
            ;;
        nemotron3-nano-omni-30b-a3b-nvfp4)
            # NVIDIA Nemotron 3 Nano Omni — open multimodal reasoning model
            # (released 2026-04-28). 30B-A3B hybrid Mamba-Transformer MoE +
            # C-RADIOv4-H vision encoder + NVIDIA Parakeet audio encoder +
            # EVS frame compression for video, all in a single 20.9 GB NVFP4
            # checkpoint. NVIDIA Open Model License (open, commercial OK).
            #
            # Thor v7 bench results (2026-04-28):
            #   Primary regime (T=0.6, top_p=0.95, max_tokens=512, think=false):
            #     TEB 80/100 ★★★★ Good   IFEval 87.7%
            #   Fallback regime (T=0, think=false):
            #     TEB 75/100 ★★★★ Good   IFEval 86.3%
            # NVIDIA's vendor tool-call recipe wins by +5 TEB and +1.4% IFEval
            # — the OPPOSITE of Qwen3.6 (where T=0 won). Lesson: tool-call-
            # specific vendor recommendations matter when they're explicitly
            # labeled "Tool calling" in the model card / cookbook (Omni's was;
            # Qwen3.6's generic non-thinking sampling was not).
            #
            # vLLM args (per Jetson AI Lab Thor recipe + NVIDIA vllm_cookbook.ipynb):
            #   --reasoning-parser nemotron_v3 (bundled in vLLM 0.20.0)
            #   --tool-call-parser qwen3_coder (bundled)
            #   --trust-remote-code (Nemotron-H custom modeling code)
            #
            # Recommended PER-REQUEST sampling (orchestrator should set):
            #   - Tool calling (winning regime):  T=0.6, top_p=0.95, max_tokens=512
            #   - Reasoning:  T=1.0 (diverse) or 0.6 (structured), max_tokens 1K-2K
            #   - Pure instruct (no tools):  T=0, max_tokens=256
            #
            # Audio support requires `pip install vllm[audio]` extra; not
            # baked into the v7 image. Bake into v8 if voice control is in
            # scope; for text+vision+tool-call workloads it's unnecessary.
            #
            # KNOWN ISSUE in v7 image — fold into v8 rebuild:
            # Two cuDNN installs (apt libcudnn9-cuda-13 9.21.1.3 +
            # pip nvidia-cudnn-cu13 9.20.0.48) cause CUDNN_STATUS_SUBLIBRARY_
            # VERSION_MISMATCH in FlashInfer's fp8_gemm autotuner during boot.
            # LD_LIBRARY_PATH and LD_PRELOAD overrides did NOT help (the
            # nvidia-runtime layer's ld.so.cache wins, plus EngineCore
            # subprocess fork dropped LD_PRELOAD). The working workaround
            # is to disable the FlashInfer JIT autotuner entirely via
            # --kernel-config '{"enable_flashinfer_autotune": false}' so
            # vLLM uses default kernel selection and never invokes the
            # cuDNN-using tactic. Costs ~5-15% throughput vs full autotune
            # but boots reliably. v8 image should drop the apt cuDNN and
            # rely solely on pip's bundled nvidia-cudnn-cu13.
            THOR_LAUNCH_MODEL_SOURCE="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.65}"
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_USE_FLASHINFER_MOE_FP16=0"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--enforce-eager"
                "--trust-remote-code"
                "--max-num-batched-tokens" "8192"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--reasoning-parser" "nemotron_v3"
                "--kernel-config" '{"enable_flashinfer_autotune": false}'
            )
            ;;
        # cosmos-reason2-8b-reasoning REMOVED 2026-04-28 — empirically
        # produced uniform 7-word responses on IFEval-lite vs ~160-word
        # median on the FP8 cosmos-reason2-8b profile (TEB 52 vs 81).
        # The {BF16 KV + max-num-seqs=1 + max-num-batched-tokens=16384}
        # combination triggered some chunked-prefill scheduler edge case
        # in vLLM v0.20.0 that broke generation. The existing
        # `cosmos-reason2-8b` profile (FP8 KV, max-num-seqs=3,
        # max-num-batched-tokens=8192) is the canonical robotics config.
        #
        # nemotron3-nano-30b-a3b-nvfp4 REMOVED 2026-04-28 — NVIDIA's Dec 2025
        # agentic flagship landed at TEB 67/100 ★★★ on Thor, mid-pack vs
        # Qwen3.6-35B-A3B-NVFP4-MTP-FP8KV at 93/100. Worth re-evaluating if
        # NVIDIA ships a v2 with stronger tool-call training, but the Qwen3.6
        # MTP family is empirically dominant at this scale on Thor for now.
        # See PERFORMANCE-V7.md for the cross-bench data.
        # qwen3.6-35b-a3b-fp8-turboquant REMOVED 2026-04-28 — FP8-weights variant
        # of qwen3.6-35b-a3b-nvfp4-tq-mtp (TEB 90, +27% tps, +1.4× ctx). NVFP4
        # alternative is strictly better on every metric.
        # qwen3.6-35b-a3b-prismaquant-dflash REMOVED 2026-04-28 — was the
        # default; default re-pointed to qwen3.6-35b-a3b-nvfp4-mtp-fp8kv. Same
        # DFlash-agentic-weakness pattern; PrismaQuant 4.75bpp is also obscure.
        qwen3.6-35b-a3b-nvfp4-dflash)
            # ★ HEAVY CODING / BURST THROUGHPUT: NVFP4 weights + DFlash N=8.
            # Kept as the canonical DFlash profile after 2026-04-28 cleanup —
            # the v7 agentic bench established that DFlash N=15 (upstream
            # default) tanks tool-call quality (TEB 40-46), but DFlash N=8
            # was the v6 sweet spot (TEB 87/100 ★★★★ Good). At N=8 DFlash is
            # competitive with MTP for tool-calling AND retains the burst-
            # throughput advantage (~3-5× peak vs MTP) for long predictable
            # code stretches. Use this profile when you're doing heavy code
            # generation rather than agentic tool-calling.
            #
            # head_dim=128 → flash_attn works natively on SM110.
            # z-lab/Qwen3.6-35B-A3B-DFlash drafter is gated — requires HF_TOKEN
            # (start-model.sh auto-reads it from ~/.cache/huggingface/token).
            #
            # Same RedHatAI/Qwen3.6-35B-A3B-NVFP4 weights as the agentic
            # profiles, so swapping between this and nvfp4-mtp-fp8kv /
            # nvfp4-tq-mtp doesn't require a new model download.
            THOR_LAUNCH_MODEL_SOURCE="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flash_attn"
                "--enforce-eager"
                "--language-model-only"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--max-num-batched-tokens" "32768"
                "--speculative-config" '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":8}'
            )
            ;;
        qwen3.6-35b-a3b-nvfp4-mtp-fp8kv)
            # ★★ TOOL-EVAL-BENCH WINNER (TEB 93 / IFEval 90.4%): NVFP4 weights +
            # MTP N=2 + FP8 KV. Reproduced on v7 (62 PASS / 4 PARTIAL / 3 FAIL).
            #
            # VLLM_USE_FLASHINFER_MOE_FP16=0: defensive fix against the same
            # non-deterministic FlashInfer-CUTLASS unquantized-MoE autotuner
            # crash that hits the TQ profiles. The autotuner can occasionally
            # pick the SM100-only BF16 tile <128,64,64> on the MTP drafter
            # forward (no SM110 instantiation → engine init crash). Routing
            # the unquantized drafter MoE to Triton avoids the broken tile
            # entirely. Hit this profile for the first time 2026-04-28 during
            # the recommended-sampling re-run.
            #
            # CAVEAT: an earlier version at MTP N=4 was removed because it
            # crashed under 8-concurrent load (CUDA illegal memory at M=128
            # in MoE autotuner). N=2 reduces M growth.
            THOR_LAUNCH_MODEL_SOURCE="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
                "-e" "VLLM_USE_FLASHINFER_MOE_FP4=1"
                "-e" "VLLM_USE_FLASHINFER_MOE_FP16=0"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--enforce-eager"
                "--language-model-only"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--reasoning-parser" "qwen3"
                "--enable-prefix-caching"
                "--max-num-batched-tokens" "32768"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":2}'
            )
            ;;
        # qwen3.6-35b-a3b-nvfp4-mtp-fp8kv-n4 REMOVED 2026-04-28 — variance-
        # probe profile, mission accomplished. TEB 91 confirmed N=2 (TEB 93)
        # is the right pick for FP8 KV; this profile was empirically dominated.
        qwen3.6-35b-a3b-nvfp4-tq-mtp)
            # ★ MAX CONTEXT: 28.0 tok/s, 79% acceptance, 2.22M KV tokens.
            # NVFP4 weights + TurboQuant K8V4 KV + MTP N=4.
            # Requires fix-pr39931-turboquant runtime mod — PR #39931 was NOT
            # merged into v0.20.0 (initial research was wrong); v0.20.0 source
            # still rejects hybrid models. The mod replays the PR idempotently.
            # VLLM_USE_FLASHINFER_MOE_FP16=0: same fix as nvfp4-tq-mtp-manyforge —
            # the FlashInfer-CUTLASS unquantized-MoE oracle non-deterministically
            # picks the SM100-only BF16 tile <128,64,64> on the MTP drafter
            # forward (no SM110 instantiation → engine init crash). Routing the
            # drafter MoE to Triton avoids the broken tile entirely.
            THOR_LAUNCH_MODEL_SOURCE="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
                "-e" "VLLM_USE_FLASHINFER_MOE_FP4=1"
                "-e" "VLLM_FLASHINFER_MOE_BACKEND=latency"
                "-e" "VLLM_USE_FLASHINFER_MOE_FP16=0"
                "-e" "VLLM_MODS=fix-pr39931-turboquant"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--enforce-eager"
                "--language-model-only"
                "--kv-cache-dtype" "turboquant_k8v4"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--max-num-batched-tokens" "8192"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":4}'
            )
            ;;
        # qwen3.6-35b-a3b-nvfp4-tq-mtp-2 REMOVED 2026-04-28 — N=2 hypothesis-
        # test profile, dominated. TEB 87 (vs 90 for nvfp4-tq-mtp at same KV
        # with N=4). With TQ KV, N=4 wins; with FP8 KV, N=2 wins. See full
        # 2×2 KV×N matrix in PERFORMANCE-V7.md.
        # qwen3.6-35b-a3b-nvfp4-mtp-fp8kv removed — crashes under 8-concurrent
        # (MoE autotuner picks invalid SM110 tile at M=128). Superseded by
        # qwen3.6-35b-a3b-nvfp4-tq-mtp which is strictly better on all axes.
        # qwen3.5-35b-a3b-nvfp4 removed — superseded by qwen3.6

        qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge)
            # ★ MANYFORGE PRODUCTION: NVFP4 weights + TurboQuant K8V4 KV + MTP N=2 + VISION.
            # Validated 2026-04-19 (LM-only) via 7-test reliability battery: 100% pass on
            # JSON schema / tool call / multi-turn / 60K needle / 3-concurrent / 2K sustained
            # decode; MTP acceptance 97-99% across 5 consecutive requests.
            # Throughput: 18.5 tok/s single, 46 tok/s at 3-concurrent aggregate.
            # Sized for 3×64K context. KV at TQ K8V4 with vision enabled: ~450K tokens vs
            # 192K needed for 3×64K (still 2.3× headroom).
            # Vision enabled 2026-04-28: removed --language-model-only, added
            # --mm-encoder-attn-backend TORCH_SDPA (SM110 ViT FA2 PTX crash workaround,
            # vllm #38411). ViT (~830 MB BF16 weights + ~1.5 GB activation peak) replaces
            # the now-deleted nvfp4-dflash-vl experimental profile and unifies coding +
            # vision in the production profile.
            # Leaves ~68% of Thor free to co-serve cosmos-reason2-2b (gpu_mem_util 0.12)
            # or cosmos-reason2-8b (0.25).
            THOR_LAUNCH_MODEL_SOURCE="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.32}"
            # NOTE: fix-pr39931-turboquant mod required again on v7 — PR #39931 did NOT
            # actually merge into vLLM v0.20.0; hybrid-rejection guard still in source.
            # VLLM_USE_FLASHINFER_MOE_FP16=0: at max-num-seqs=3, the FlashInfer-CUTLASS
            # unquantized MoE autotuner picks tile <128,64,64> which is BF16+SM100-only
            # and has no SM110 instantiation — drafter forward crashes on engine init.
            # Routing the unquantized drafter MoE to Triton avoids the broken tile.
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
                "-e" "VLLM_USE_FLASHINFER_MOE_FP4=1"
                "-e" "VLLM_FLASHINFER_MOE_BACKEND=latency"
                "-e" "VLLM_USE_FLASHINFER_MOE_FP16=0"
                "-e" "VLLM_MODS=fix-pr39931-turboquant"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--enforce-eager"
                "--mm-encoder-attn-backend" "TORCH_SDPA"
                "--kv-cache-dtype" "turboquant_k8v4"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--max-num-batched-tokens" "8192"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":2}'
            )
            ;;
        qwen3.5-9b-claude-distilled-nvfp4)
            # Qwen3.5-9B VLM: DeltaNet hybrid (linear_attention + full_attention) with visual encoder.
            # Claude 4.6 Opus reasoning-distilled, NVFP4 MLP-only + FP8 KV. Visual encoder kept bf16.
            # Multimodal: vision + text + tools. No --language-model-only.
            # --mm-encoder-attn-backend TORCH_SDPA: workaround for SM110 ViT PTX crash (#38411).
            # --max-num-batched-tokens 4096: MTP speculative decode defaults to 2048 which throttles
            #   throughput. 4096 gives the scheduler enough headroom for 8 seqs + draft tokens.
            # Use the dedicated no-think chat template variant for this fast-control profile.
            # The standard Qwen template opens <think> by default, which causes the 9B distilled
            # model to burn the full token budget reasoning before it emits content/tool calls.
            # Also do not advertise a Qwen reasoning parser for this profile: OpenClaw's current
            # OpenAI-completions path expects ordinary content/tool_calls, and vLLM's split
            # reasoning channel leaves the embedded agent with no final content to consume.
            # Source: Alexzander85/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-NVFP4-MLP-FP8KV
            THOR_LAUNCH_MODEL_SOURCE="Alexzander85/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-NVFP4-MLP-FP8KV"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.4}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat-nothink.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat-nothink.jinja"
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flashinfer"
                "--quantization" "modelopt"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--enable-prefix-caching"
                "--mm-encoder-attn-backend" "TORCH_SDPA"
                "--max-num-batched-tokens" "8192"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":1}'
            )
            ;;
        # qwen3.5-9b-dflash removed
        # qwen3.5-9b-bf16-dflash removed
        # qwen3.5-27b-claude-distilled-nvfp4 removed
        # qwen3.5-27b-claude-distilled-v2-nvfp4 removed 2026-04-24 — superseded by qwen3.6.
        qwen3.6-27b-fp8-mtp-kvfp8)
            # EXPERIMENTAL: Qwen/Qwen3.6-27B-FP8 (official FP8) + MTP + FP8 KV.
            # Official FP8 release preserves the 22 MTP head tensors that all
            # community NVFP4 quantizations strip via llm-compressor.
            # head_dim=256 forces FlashInfer attention. VLLM_DISABLED_KERNELS
            # (set higher up) routes FP8 GEMM through Triton fallback to dodge
            # the Xid 43 CutlassFp8BlockScaledMMKernel crash on SM110.
            THOR_LAUNCH_MODEL_SOURCE="Qwen/Qwen3.6-27B-FP8"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flashinfer"
                "--language-model-only"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--enable-prefix-caching"
                "--max-num-batched-tokens" "32768"
                "--speculative-config" '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
            )
            ;;
        gemma4-e4b-it)
            # Gemma 4 E4B IT — MoE (8B total, ~4B active per token), ~16 GB BF16.
            # Native function calling (gemma4 parser), vision, audio.
            # No NVFP4 quant available — runs at BF16, light enough at 0.4 GPU util.
            # triton_attn: same head_dim=512 FlashInfer limitation as all Gemma 4 models.
            # SWA (sliding window attention) + few global layers = small KV footprint.
            # --mm-encoder-attn-backend TORCH_SDPA: SM110 ViT PTX crash workaround.
            # 128K context (native for E-series, vs 256K for medium models).
            THOR_LAUNCH_MODEL_SOURCE="google/gemma-4-E4B-it"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.4}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH=""
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH=""
            # --max-num-batched-tokens 4096: same MM-encoder-budget reason as 31B
            # (vLLM v0.20.0 enforces ≥ max_tokens_per_mm_item).
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "triton_attn"
                "--reasoning-parser" "gemma4"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "gemma4"
                "--enable-prefix-caching"
                "--mm-encoder-attn-backend" "TORCH_SDPA"
                "--max-num-batched-tokens" "4096"
            )
            ;;
        gemma4-31b-it-nvfp4)
            # Gemma 4 31B IT NVFP4 — dense model, ~17 GB in VRAM.
            # Vision enabled (SigLIP2 ~550M params), tool calling via gemma4 parser.
            # Thinking mode via reasoning-parser deepseek_r1 (<|think|> tokens).
            # --attention-backend triton_attn: FlashInfer kernels crash on head_dim=512
            # (Gemma 4 global attention layers). FlashInfer JIT generates invalid MMA
            # tiling for dim>256. triton_attn handles arbitrary head sizes.
            # See vllm-project/vllm#38887. NVFP4 GEMM still uses flashinfer-cutlass.
            # --mm-encoder-attn-backend TORCH_SDPA: workaround for #38411 — ViT FA2
            # PTX crash on SM110 with CUDA 13.0 host driver.
            THOR_LAUNCH_MODEL_SOURCE="nvidia/Gemma-4-31B-IT-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.80}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH=""
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH=""
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
            )
            # --max-num-batched-tokens 4096: vLLM v0.20.0+ enforces that
            # max_num_batched_tokens >= max_tokens_per_mm_item (2496 for SigLIP2 vision
            # encoder). Default 2048 fails at boot with ValueError. 4096 is the
            # nearest multiple of 1024 that clears it; bump higher if MM throughput
            # becomes an issue.
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "triton_attn"
                "--quantization" "modelopt"
                "--reasoning-parser" "gemma4"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "gemma4"
                "--enable-prefix-caching"
                "--mm-encoder-attn-backend" "TORCH_SDPA"
                "--max-num-batched-tokens" "4096"
            )
            ;;
        gemma4-26b-a4b-it)
            # Gemma 4 26B-A4B IT — MoE (128 total, 8 active, 1 shared), ~52 GB BF16.
            # 3.8B active params per token — inference speed comparable to 4B dense.
            # Vision enabled, tool calling via gemma4 parser, thinking via <|think|>.
            # KV cache is small: hybrid SWA (1024 window) + 5 global attention layers.
            # No NVFP4 quant available — runs at BF16, needs careful memory budgeting.
            # triton_attn: same head_dim=512 FlashInfer limitation as 31B.
            THOR_LAUNCH_MODEL_SOURCE="google/gemma-4-26B-A4B-it"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.80}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH=""
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH=""
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "triton_attn"
                "--reasoning-parser" "gemma4"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "gemma4"
                "--enable-prefix-caching"
                "--mm-encoder-attn-backend" "TORCH_SDPA"
            )
            ;;
        *)
            fail "Unsupported model profile: ${profile}"
            print_supported_model_profiles
            return 1
            ;;
    esac

    THOR_LAUNCH_MAX_MODEL_LEN="${THOR_MAX_MODEL_LEN:-${THOR_TARGET_MAX_MODEL_LEN}}"
    THOR_LAUNCH_KV_CACHE_DTYPE="${THOR_KV_CACHE_DTYPE:-${THOR_TARGET_KV_CACHE_DTYPE}}"
    THOR_LAUNCH_MAX_NUM_SEQS="${THOR_MAX_NUM_SEQS:-${THOR_TARGET_MAX_NUM_SEQS}}"

    THOR_VLLM_ARGS=(
        "${THOR_VLLM_ARGS[@]}"
        "--served-model-name" "${THOR_MODEL_ID}"
        "--host" "${THOR_VLLM_BIND_HOST}"
        "--port" "${THOR_VLLM_PORT}"
        "--gpu-memory-utilization" "${THOR_LAUNCH_GPU_MEMORY_UTILIZATION}"
        "--max-model-len" "${THOR_LAUNCH_MAX_MODEL_LEN}"
        "--kv-cache-dtype" "${THOR_LAUNCH_KV_CACHE_DTYPE}"
        "--max-num-seqs" "${THOR_LAUNCH_MAX_NUM_SEQS}"
        "--compilation-config" '{"custom_ops":["-quant_fp8","-quant_fp8","-quant_fp8"]}'
    )

    if [[ -n "${THOR_LOCAL_VLLM_API_KEY}" && "${THOR_LOCAL_VLLM_API_KEY}" != "dummy" ]]; then
        THOR_VLLM_ARGS+=("--api-key" "${THOR_LOCAL_VLLM_API_KEY}")
    fi

    if [[ -n "${THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS}" ]]; then
        THOR_VLLM_ARGS+=("--max-num-batched-tokens" "${THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS}")
    fi

    if [[ -n "${THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH}" ]]; then
        THOR_VLLM_ARGS+=("--chat-template" "${THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH}")
    fi
}

check_thor_launch_prereqs() {
    if ! command -v docker &>/dev/null; then
        fail "docker command not found"
        fix "Install Docker and the NVIDIA container runtime first."
        return 1
    fi

    if ! docker info &>/dev/null; then
        fail "Docker daemon is not running"
        fix "Run: sudo systemctl start docker"
        return 1
    fi

    if [[ -n "${THOR_LAUNCH_HOST_MODEL_PATH}" && ! -d "${THOR_LAUNCH_HOST_MODEL_PATH}" ]]; then
        fail "Required model path not found: ${THOR_LAUNCH_HOST_MODEL_PATH}"
        info "This profile uses a pre-resharded local model path."
        fix "Follow the download instructions in:"
        fix "  $(cd "$(dirname "${BASH_SOURCE[0]}")/../../thor_llm/models/${THOR_MODEL_PROFILE}" 2>/dev/null && pwd)/README.md"
        return 1
    fi

    if [[ -n "${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}" && ! -f "${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}" ]]; then
        fail "Required chat template not found: ${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}"
        fix "Restore the NemoClaw-Thor templates directory or set THOR_CHAT_TEMPLATE_HOST_DIR."
        return 1
    fi

    if [[ "${THOR_LAUNCH_BACKEND:-vllm}" == "llamacpp" ]]; then
        if [[ ! -f "${THOR_LLAMACPP_MODEL_PATH:-}" ]]; then
            fail "Model file not found: ${THOR_LLAMACPP_MODEL_PATH:-<not set>}"
            fix "Download the GGUF model first."
            return 1
        fi
        if [[ -n "${THOR_LLAMACPP_DRAFT_PATH:-}" && ! -f "${THOR_LLAMACPP_DRAFT_PATH}" ]]; then
            warn "Draft model not found: ${THOR_LLAMACPP_DRAFT_PATH}"
            info "Speculative decoding will be disabled. Download the draft model to enable it."
        fi
    fi

    mkdir -p "${THOR_HF_CACHE_DIR}" "${THOR_VLLM_CACHE_DIR}" "${THOR_TORCH_CACHE_DIR}" "${THOR_FLASHINFER_CACHE_DIR}"
    return 0
}

print_thor_launch_summary() {
    echo "  Profile:            ${THOR_MODEL_PROFILE}"
    echo "  Source:             ${THOR_LAUNCH_MODEL_SOURCE}"
    echo "  Served model id:    ${THOR_MODEL_ID}"
    echo "  Bind:               ${THOR_VLLM_BIND_HOST}:${THOR_VLLM_PORT}"
    echo "  Max context:        ${THOR_LAUNCH_MAX_MODEL_LEN}"
    echo "  Max num seqs:       ${THOR_LAUNCH_MAX_NUM_SEQS}"

    if [[ "${THOR_LAUNCH_BACKEND:-vllm}" == "llamacpp" ]]; then
        echo "  Backend:            llama.cpp (llama-server)"
        echo "  Image:              ${THOR_LLAMACPP_IMAGE}"
        echo "  Model:              ${THOR_LLAMACPP_MODEL_PATH}"
        if [[ -f "${THOR_LLAMACPP_DRAFT_PATH:-}" ]]; then
            echo "  Draft model:        ${THOR_LLAMACPP_DRAFT_PATH}"
            echo "  Draft tokens:       ${THOR_LLAMACPP_DRAFT_N}"
        else
            echo "  Draft model:        (none)"
        fi
        echo "  Context (total):    ${THOR_LLAMACPP_CTX}"
        echo "  Parallel slots:     ${THOR_LLAMACPP_PARALLEL}"
        echo "  KV cache type:      K=${THOR_LLAMACPP_CACHE_TYPE_K} V=${THOR_LLAMACPP_CACHE_TYPE_V}"
    else
        echo "  Image:              ${THOR_VLLM_IMAGE}"
        echo "  KV cache dtype:     ${THOR_LAUNCH_KV_CACHE_DTYPE}"
        echo "  GPU mem util:       ${THOR_LAUNCH_GPU_MEMORY_UTILIZATION}"
        if [[ -n "${THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS}" ]]; then
            echo "  Max batched tokens: ${THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS}"
        fi
        if [[ -n "${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}" ]]; then
            echo "  Chat template:      ${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}"
        fi
        echo "  HF cache:           ${THOR_HF_CACHE_DIR}"
        echo "  vLLM cache:         ${THOR_VLLM_CACHE_DIR}"
        echo "  Torch cache:        ${THOR_TORCH_CACHE_DIR}"
        echo "  FlashInfer cache:   ${THOR_FLASHINFER_CACHE_DIR}"
    fi
}

run_thor_vllm_container() {
    local docker_tty_args=()
    local docker_mount_args=()
    local docker_name_args=()

    # THOR_DETACH=1: run container in background, return immediately.
    # THOR_CONTAINER_NAME=<name>: pin the container name (for duo-serve).
    if [[ "${THOR_DETACH:-0}" == "1" ]]; then
        docker_tty_args=(-d)
    elif [[ -t 0 && -t 1 ]]; then
        docker_tty_args=(-i -t)
    fi

    if [[ -n "${THOR_CONTAINER_NAME:-}" ]]; then
        docker_name_args=(--name "${THOR_CONTAINER_NAME}")
    fi

    if [[ -n "${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}" ]]; then
        docker_mount_args=(-v "${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}:${THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH}:ro")
    fi

    # Mount host mods directory so new/updated mods are available without rebuild
    if [[ -d "${THOR_MODS_HOST_DIR}" ]]; then
        docker_mount_args+=(-v "${THOR_MODS_HOST_DIR}:/workspace/mods:ro")
    fi

    local docker_rm_args=(--rm)
    if [[ "${THOR_NO_RM:-0}" == "1" ]]; then
        docker_rm_args=()
    fi

    docker run "${docker_rm_args[@]}" \
        "${docker_tty_args[@]}" \
        "${docker_name_args[@]}" \
        --runtime nvidia --gpus all \
        --ipc=host --network host \
        -e NVIDIA_DISABLE_REQUIRE=true \
        -e HF_HOME=/data/models/huggingface \
        -e HF_HUB_CACHE=/data/models/huggingface/hub \
        -e TRANSFORMERS_CACHE=/data/models/huggingface/hub \
        ${HF_TOKEN:+-e "HF_TOKEN=${HF_TOKEN}"} \
        -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
        -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor \
        -e "MAX_JOBS=${THOR_MAX_JOBS:-12}" \
        -e "NINJAFLAGS=-j${THOR_MAX_JOBS:-12}" \
        -e "MAKEFLAGS=-j${THOR_MAX_JOBS:-12}" \
        -e "CMAKE_BUILD_PARALLEL_LEVEL=${THOR_MAX_JOBS:-12}" \
        -v "${THOR_HF_CACHE_DIR}:/data/models/huggingface" \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        -v "${THOR_VLLM_CACHE_DIR}:/root/.cache/vllm" \
        -v "${THOR_TORCH_CACHE_DIR}:/root/.cache/torch" \
        -v "${THOR_FLASHINFER_CACHE_DIR}:/root/.cache/flashinfer" \
        "${docker_mount_args[@]}" \
        "${THOR_DOCKER_ENV_ARGS[@]}" \
        "${THOR_VLLM_IMAGE}" \
        vllm serve "${THOR_LAUNCH_MODEL_SOURCE}" "${THOR_VLLM_ARGS[@]}"
}

run_thor_llamacpp_container() {
    local docker_tty_args=()

    if [[ -t 0 && -t 1 ]]; then
        docker_tty_args=(-i -t)
    fi

    # Map host paths under THOR_HF_CACHE_DIR to /data/models inside the container.
    local model_container_path="/data/models/${THOR_LLAMACPP_MODEL_PATH#"${THOR_HF_CACHE_DIR}/"}"

    local draft_args=()
    if [[ -f "${THOR_LLAMACPP_DRAFT_PATH:-}" ]]; then
        local draft_container_path="/data/models/${THOR_LLAMACPP_DRAFT_PATH#"${THOR_HF_CACHE_DIR}/"}"
        draft_args=(-md "${draft_container_path}" --draft "${THOR_LLAMACPP_DRAFT_N}")
    fi

    docker run --rm \
        "${docker_tty_args[@]}" \
        --runtime nvidia --network host \
        -v "${THOR_HF_CACHE_DIR}:/data/models" \
        "${THOR_LLAMACPP_IMAGE}" \
        llama-server \
            -m "${model_container_path}" \
            "${draft_args[@]}" \
            --host "${THOR_VLLM_BIND_HOST}" \
            --port "${THOR_VLLM_PORT}" \
            -np "${THOR_LLAMACPP_PARALLEL}" \
            -c "${THOR_LLAMACPP_CTX}" \
            --cache-type-k "${THOR_LLAMACPP_CACHE_TYPE_K}" \
            --cache-type-v "${THOR_LLAMACPP_CACHE_TYPE_V}" \
            --cache-ram "${THOR_LLAMACPP_CACHE_RAM}" \
            --reasoning-format deepseek \
            --reasoning auto
}
