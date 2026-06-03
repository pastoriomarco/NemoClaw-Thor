#!/usr/bin/env bash
# serving/launch.sh — Shared vLLM launcher logic for NemoClaw-Thor
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
    THOR_CHAT_TEMPLATE_HOST_DIR="${THOR_CHAT_TEMPLATE_HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/templates}"
    THOR_MODS_HOST_DIR="${THOR_MODS_HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/docker/mods" && pwd)}"

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
        # See docs/MINIMAX-M27-INVESTIGATION.md for the why — W4A4 NVFP4 MoE on SM110
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
        # drafter is gated and adds no value on Thor at this N. See docs/PERFORMANCE-V7.md.
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
            # 2026-05-08: bumped 0.25 -> 0.35 to make 256K KV pool viable on
            # Thor (paired with config.sh THOR_TARGET_MAX_MODEL_LEN=262144).
            # The OpenClaw lane needs the larger context to support multi-turn
            # Composer-assistant sessions without hitting the gateway's
            # preemptive overflow guard (~90% of context window). Watch the
            # vLLM boot log for "GPU KV cache size: NNN tokens" to see what
            # the slot count actually allocates after this change — adjust
            # THOR_TARGET_MAX_NUM_SEQS in config.sh if there's room or
            # too-tight headroom.
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.35}"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flashinfer"
                "--enforce-eager"
                "--mm-encoder-attn-backend" "TORCH_SDPA"
                "--kv-cache-dtype" "fp8"
                "--max-num-batched-tokens" "8192"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "hermes"
                # Lane-parity tuning 2026-05-07: deterministic-leaning sampling
                # so the OpenClaw lane (which never forwards per-request sampling
                # fields) gets a tight decode pattern by default.
                # 2026-05-09 (iter 17 experiment): enable thinking by default —
                # Cosmos-Reason2-8B is post-trained on Qwen3-VL with long-CoT
                # reasoning assumed. Running thinking-off is OOD for the model
                # and causes narration-mode collapse on action prompts.
                # Smoke-corpus iter 17 tests whether in-distribution thinking
                # restores accuracy. Latency cost expected: +30-60s per turn.
                # Revert if iter 17 doesn't show clear improvement.
                "--override-generation-config" '{"temperature":0.2,"top_p":0.95}'
                "--default-chat-template-kwargs" '{"enable_thinking":true}'
                # 2026-05-31 (cosmos regression hunt): add qwen3 reasoning parser
                # to extract <think> blocks before hermes parses content. With
                # empty reasoning_parser (the prior state) the model's thinking
                # stays in `content`, confusing hermes's tool-call extraction
                # and producing the ~30% narration / ~47% nodeName-dropped
                # failure modes on v9. Qwen3 reasoning parser is appropriate
                # for cosmos since it's post-trained from Qwen3-VL-8B.
                "--reasoning-parser" "qwen3"
            )
            ;;
        nemotron3-nano-4b-bf16)
            # NVIDIA Nemotron-3-Nano-4B-BF16 — NVIDIA's explicit Jetson
            # Thor / Orin agentic default per HF card: "edge-ready small
            # language model intended for Agentic AI in edge platforms
            # (Jetson Thor, GeForce RTX, DGX Spark)". Hybrid Mamba-2 + 4
            # attention layers, ~8 GB BF16 weights. Tool-call trained on
            # glaive-function-calling-v2 + APIGen + ToolBench + Nemotron-
            # RL-Agentic-Conversational-Tool-Use-Pivot-v1. BFCL v3 = 61.1.
            #
            # 2026-06-01 experiment: previously removed --reasoning-parser
            # nano_v3 because HF discussion #3 (on 30B-A3B sibling) said
            # combining it with --tool-call-parser breaks tool calling.
            # Re-enabling now to test against P1_wrap_root_specific
            # failure where the model emitted a partial tool-call format
            # (`tree_draft_wrap_node <parameter=...></function>` — missing
            # the outer `<tool_call><function=` wrapper). Smoke run will
            # tell us if the warning was outdated or still accurate.
            THOR_LAUNCH_MODEL_SOURCE="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.40}"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--trust-remote-code"
                "--mamba_ssm_cache_dtype" "float32"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--reasoning-parser-plugin" "/workspace/mods/nano_v3_reasoning_parser.py"
                "--reasoning-parser" "nano_v3"
                # Tool-call regime sampling per HF card: T=0.6, top_p=0.95
                "--override-generation-config" '{"temperature":0.6,"top_p":0.95}'
                # Thinking OFF by default for tool-call lane — same
                # rationale as omni instruct profile (reasoning content
                # would otherwise be lost when bridge reads only
                # message.content). Flip ON per request only when the
                # user needs deliberation.
                "--default-chat-template-kwargs" '{"enable_thinking":false}'
                "--max-num-batched-tokens" "8192"
            )
            ;;
        nemotron3-nano-30b-a3b-nvfp4)
            # NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 — NVIDIA's text-only A3B
            # variant. Hybrid Mamba-2 + 6 attn + 128 MoE experts (6 active);
            # 3.5B active per token, 30B total. NVFP4 baked in (~18GB on
            # disk). Positioned by NVIDIA as "specialized sub-agent in
            # long-running multi-step workflows; math, coding, multi-step
            # tool calling" — the primary tool-call A3B Nemotron-3.
            #
            # Per HF discussion #3: tool-call + nano_v3 reasoning parser
            # together breaks tool calling. Keep ONLY tool-call-parser.
            THOR_LAUNCH_MODEL_SOURCE="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.50}"
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_USE_FLASHINFER_MOE_FP16=0"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--trust-remote-code"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--override-generation-config" '{"temperature":0.6,"top_p":0.95}'
                "--default-chat-template-kwargs" '{"enable_thinking":false}'
                "--max-num-batched-tokens" "8192"
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
            # gpu_memory_utilization=0.50 calibrated 2026-04-30 against
            # max_model_len=262144 + max_num_seqs=16 (see config.sh
            # nemotron3-nano-omni profile branch + MANYFORGE-PROFILE-
            # CALIBRATION.md). Yields ~25 GB KV pool, ~32x supportable
            # concurrency at 256K vs the 16 we configure, frees ~14 GB
            # for Isaac ROS / system / cluster gateway vs the prior 0.65.
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.50}"
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_USE_FLASHINFER_MOE_FP16=0"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                # 2026-05-06: dropped --enforce-eager. CUDA graphs reduce
                # kernel-launch overhead by 10-20% on generation; the model
                # was previously enforce-eager because of debugging needs
                # not present in steady-state.
                "--trust-remote-code"
                "--max-num-batched-tokens" "8192"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                # 2026-06-01 (model-bakeoff): kernel-tweak experiment.
                # Tried --moe-backend triton AND flashinfer_cutlass.
                # Both REJECTED on Thor SM110 by vLLM's NvFP4 oracle
                # (triton: not in NvFP4 backend list; flashinfer_cutlass:
                # "kernel does not support current device cuda"). vLLM
                # auto-picks from {FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL,
                # ...} when --moe-backend is unset. Default IS the optimal
                # choice on SM110 NVFP4. Reverted.
                # (--enable-prefix-caching is added below via the
                # THOR_ENABLE_PREFIX_CACHING env-var gate, still active.)
                # 2026-05-06: server-wide sampling defaults, replacing the
                # YAML-driven per-request injection in
                # openclaw_assistant_bridge. Native vLLM 0.20 flags; both
                # lanes (direct via :8100 bridge and OpenClaw via :8200 →
                # gateway → vLLM) inherit these when the client request
                # body omits the field. Client-supplied values still win.
                # Source: NVIDIA's vendor tool-calling recipe for this
                # model (see comment block at the top of this profile —
                # "Tool calling (winning regime): T=0.6, top_p=0.95").
                # Initially tried T=0.2 + top_k=1 (the model's own
                # generation_config.json defaults). With thinking off
                # (enabled below) those greedy settings produced
                # degenerate tool-call loops on multi-step tasks: the
                # model stuck on the same two-tool sequence with null
                # arguments, 80+ calls in one turn, no convergence.
                # top_p=0.95 nucleus-sampling restores the diversity the
                # model needs to escape local minima without re-enabling
                # thinking. max_tokens omitted: --generation-config
                # treats max_new_tokens as a server-wide CAP not a
                # default, and clients (OpenClaw maxTokens=16384,
                # direct bridge) own that knob per-request.
                "--override-generation-config" '{"temperature":0.6,"top_p":0.95}'
                "--default-chat-template-kwargs" '{"enable_thinking":false}'
                # --reasoning-parser nemotron_v3 was here. Removed
                # 2026-05-06 because in thinking-off mode (set by
                # --default-chat-template-kwargs above) the model emits
                # no <think>...</think> envelope, but the parser still
                # operates in "extract thinking" mode by default and
                # buckets the entire response into `reasoning` instead
                # of `content`. The bridges only consume
                # choices[0].message.content, so the OpenClaw lane was
                # returning empty messages until this flag was dropped.
                # Re-enable when/if a profile flips back to thinking on.
                # MTP speculative decoding is **NOT AVAILABLE for this
                # checkpoint** (verified 2026-05-06 on v8.1 / vLLM
                # v0.20.1).
                #
                # Root cause: Nemotron-3-Nano-Omni-30B-A3B-Reasoning-
                # NVFP4 does not ship with MTP head weights. NVIDIA
                # only bundles MTP into the Nemotron-3 *Super* (120B)
                # checkpoints — the Nano and Nano-Omni variants do
                # not. The HuggingFace model card has zero mentions
                # of "MTP", "speculative", or "multi-token", and the
                # Nemotron-3-Nano vLLM cookbook explicitly sets
                # speculative_config=None.
                #
                # Why we kept getting `NotImplementedError`: vLLM
                # v0.20.1's auto-detection path
                # (vllm/config/speculative.py:620) sets
                # `self.method = "mtp"` only after seeing the draft
                # model's hf_config.model_type in MTPModelTypes. With
                # no MTP weights present, no draft_model_config can
                # be built, no auto-detection fires, and the literal
                # "mtp" or "nemotron_h_mtp" passed via --speculative-
                # config falls through the elif chain to
                # `raise NotImplementedError`.
                #
                # NVIDIA's reference MTP config (for Nemotron-3 Super
                # on DGX Spark / GB10) for posterity:
                #
                #   --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
                #
                # Adapt that line if/when this profile switches to a
                # checkpoint that bundles the MTP head (Super 120B,
                # or a future Nano variant with MTP). The lane-parity
                # fix bundle (vendor sampling + MCP wrapper validator
                # + tree unique-name guidance + middleware
                # HTTPException propagation) is independent of MTP
                # and runs fine on the dense path; v8.1 probe 10/10
                # PASS without speculative decoding (LANE-COMPARISON
                # §10).
                # "--speculative-config" '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
                # v7 needed `--kernel-config '{"enable_flashinfer_autotune": false}'`
                # here to dodge a cuDNN sublibrary-version mismatch (apt 9.21.1
                # + pip 9.20.0). v8 drops the apt cuDNN and relies on pip's
                # bundled nvidia-cudnn-cu13==9.20.0.48, so the autotuner can
                # run again. Re-enable verified at v8 boot 2026-04-29.
            )
            if [[ "${THOR_ENABLE_PREFIX_CACHING:-1}" != "0" ]]; then
                THOR_VLLM_ARGS+=("--enable-prefix-caching")
            fi
            ;;
        nemotron3-nano-omni-30b-a3b-nvfp4-reasoning)
            # Reasoning-mode variant. Same weights/KV sizing as the base
            # nemotron3-nano-omni profile (see the long comment block on
            # the case above for the model card / hardware context). The
            # only differences are at the chat-template + parser layer:
            #   - --default-chat-template-kwargs '{"enable_thinking":true}'
            #     (base flips this off for tool-calling regime).
            #   - --reasoning-parser nemotron_v3 re-enabled. With thinking
            #     ON the model emits <think>...</think> envelopes and
            #     this parser splits them into a `reasoning_content`
            #     field on the response. Bridges that route through this
            #     profile must read both `content` and `reasoning_content`
            #     (the openclaw_assistant_bridge as of 2026-05-06 reads
            #     only `content` and would drop the reasoning).
            #   - generation-config overrides intentionally omitted: the
            #     model's own generation_config.json already ships
            #     T=0.2/top_k=1/top_p=0.95/max_tokens=16384, which falls
            #     within NVIDIA's "structured reasoning T=0.6" recipe
            #     tolerance. Higher-T (open-ended exploration, T=1.0) can
            #     be set per-request rather than baked in here.
            THOR_LAUNCH_MODEL_SOURCE="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.50}"
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_USE_FLASHINFER_MOE_FP16=0"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--trust-remote-code"
                "--max-num-batched-tokens" "8192"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--reasoning-parser" "nemotron_v3"
                # Same vendor sampling as the dense base profile. The
                # model's generation_config.json defaults (T=0.2,
                # top_k=1) are greedy enough to trigger token loops
                # even inside <think> blocks; T=0.6/top_p=0.95 (NVIDIA's
                # tool-calling recipe) gives the model the diversity to
                # escape local minima during thinking too.
                "--override-generation-config" '{"temperature":0.6,"top_p":0.95}'
                "--default-chat-template-kwargs" '{"enable_thinking":true}'
            )
            if [[ "${THOR_ENABLE_PREFIX_CACHING:-1}" != "0" ]]; then
                THOR_VLLM_ARGS+=("--enable-prefix-caching")
            fi
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
        # See docs/PERFORMANCE-V7.md for the cross-bench data.
        # qwen3.6-35b-a3b-fp8-turboquant REMOVED 2026-04-28 — FP8-weights variant
        # of qwen3.6-35b-a3b-nvfp4-tq-mtp (TEB 90, +27% tps, +1.4× ctx). NVFP4
        # alternative is strictly better on every metric.
        # qwen3.6-35b-a3b-prismaquant-dflash REMOVED 2026-04-28 — was the
        # default; default re-pointed to qwen3.6-35b-a3b-nvfp4-mtp-fp8kv. Same
        # DFlash-agentic-weakness pattern; PrismaQuant 4.75bpp is also obscure.
        # qwen3.6-35b-a3b-nvfp4-mtp-fp8kv-n4 REMOVED 2026-04-28 — variance-
        # probe profile, mission accomplished. TEB 91 confirmed N=2 (TEB 93)
        # is the right pick for FP8 KV; this profile was empirically dominated.
        qwen3.6-35b-a3b-nvfp4-nvidia)
            # EXPERIMENTAL (2026-05-30 v9 staging): NVIDIA-official NVFP4 quant
            # of Qwen3.6-35B-A3B (MoE 3B active / 35B total). Combines NVIDIA's
            # ModelOpt v0.44.0 quantization with the iter-32 production sampling
            # recipe and the froggeric Qwen3.6 chat-template fix.
            #
            # Rationale: the dense qwen3.6-27b-fp8-mtp-kvfp8 baseline run on
            # cosmos-reason2-8b lineage was too slow for the composer-assistant
            # workload (60-100s per simple case, several full timeouts). 35B-A3B
            # MoE has only ~3B active parameters per token — should be
            # substantially faster than the 27B dense path while gaining the
            # quality benchmarks (τ²-Bench Telecom 94.7 NVFP4 vs 95.5 BF16 per
            # NVIDIA's model card). MTP K=3 with moe_backend:triton is NVIDIA's
            # explicit Spark/DGX recommendation; combined with froggeric's
            # template (which guarantees 100% prefix-cache hit rate), agent loop
            # iterations should be far cheaper than the 27B's were.
            #
            # SM110 / Thor compatibility env vars (mirroring the RedHat quant's
            # working configuration):
            #   VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass — picks the SM110-
            #     compatible NVFP4 GEMM path.
            #   VLLM_USE_FLASHINFER_MOE_FP4=1 — enables FlashInfer NVFP4 MoE.
            #   VLLM_USE_FLASHINFER_MOE_FP16=0 — routes unquantized BF16 MoE
            #     paths (drafter forward) through Triton to dodge the SM100-only
            #     CUTLASS tile <128,64,64> crash that hit on SM110 in v7/v8.
            #
            # Tool-call parser: qwen3_coder. The froggeric template emits native
            # XML tool calls; qwen3_coder parses that format. Do NOT use qwen3_xml
            # (designed for the stock-template format) or hermes (designed for
            # JSON-in-tags).
            #
            # MTP K=3 vs K=2: NVIDIA's Spark recipe specifies K=3; community
            # forum tests on the dense 27B at K=3 hit 85-94% acceptance. The
            # RedHat-quant siblings use K=2 (proven historically at TEB 93).
            # Starting at K=3 to match NVIDIA's testing; drop to K=2 if MTP
            # acceptance falls below ~70% in smoke runs.
            #
            # Proxy caps (active automatically via launch.sh env, not profile):
            #   OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=2048 — iter-21b proved this
            #     is the sweet spot; bigger caps cause model wander → timeouts.
            #   OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=512 — bounds <think> block
            #     within the 2048 cap. Iter-21a proved this is neutral on cosmos
            #     but load-bearing as a thinking-on safety net.
            # V9.1 PHASE 4 TEST (2026-05-31): switched to nvidia/ weights + full
            # NVIDIA Spark recipe to validate PR #42124 (LM head ModelOpt support).
            # The v9 image lacked this fix — load crashed at 67% with
            # lm_head.input_scale ValueError. v9.1 image (vLLM main @ 3fd9d2d35,
            # post #42124) should load cleanly.
            #
            # NVIDIA Spark recipe (from model card, sm_121a → sm_110a for Thor):
            #   env: VLLM_USE_FLASHINFER_MOE_FP4=0 (explicit 0)
            #        VLLM_FP8_MOE_BACKEND=flashinfer_cutlass
            #        FLASHINFER_DISABLE_VERSION_CHECK=1
            #        CUTE_DSL_ARCH=sm_110a
            #   vllm: --quantization modelopt --moe-backend marlin
            #         (rest matches iter-3 RedHat recipe)
            THOR_LAUNCH_MODEL_SOURCE="nvidia/Qwen3.6-35B-A3B-NVFP4"
            # 2026-06-03: dropped from 0.85 to 0.55. The 0.85 default was
            # carried over from the earlier max-concurrency-per-dollar
            # profile but oversubscribed Thor's 122 GiB unified memory
            # for the single-bridge-invocation workflow we actually
            # run in production (smoke harness + composer driving one
            # session at a time). Sizing:
            #   weights:        ~22 GiB (NVFP4 quant)
            #   KV @ 256K FP8:  ~24-32 GiB per sequence
            #   activations:    ~8-10 GiB
            # 0.55 (~67 GiB allocated) leaves comfortable headroom AND
            # supports up to ~2 concurrent 256K sequences at peak. Pair
            # with --max-num-seqs 4 (set in config.sh under
            # THOR_TARGET_MAX_NUM_SEQS) which caps the KV pool to a
            # realistic working set without ever exhausting memory.
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.55}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen-fixed-froggeric.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen-fixed-froggeric.jinja"
            THOR_DOCKER_ENV_ARGS+=(
                # NVIDIA's W4A16_NVFP4 quant has 16-bit activations. ALL
                # FlashInfer NVFP4 MoE backends (TRTLLM/CUTEDSL/CUTLASS) require
                # full NVFP4 (4-bit weights AND 4-bit activations) per the
                # is_supported_config quant_scheme check (kNvfp4Static x
                # kNvfp4Dynamic). Setting =1 causes NotImplementedError at
                # engine init because no backend matches. Marlin handles
                # weight-only int4/NVFP4 correctly via software dequant — keep
                # =0 so the oracle picks MARLIN. (Verified empirically by
                # this Task 4 iteration.)
                "-e" "VLLM_USE_FLASHINFER_MOE_FP4=0"
                "-e" "VLLM_FP8_MOE_BACKEND=flashinfer_cutlass"
                "-e" "FLASHINFER_DISABLE_VERSION_CHECK=1"
                "-e" "CUTE_DSL_ARCH=sm_110a"
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
                "-e" "VLLM_USE_FLASHINFER_MOE_FP16=0"
                # Mod is INERT for NVIDIA's W4A16 path (no FlashInfer NVFP4
                # backend matches the quant scheme regardless of patch). Kept
                # here only to mirror the RedHat profile config — no harm.
                "-e" "VLLM_MODS=sm110a-fp4-dsl-unlock"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--quantization" "modelopt"
                # --moe-backend marlin: NVIDIA Spark recipe. Re-added because
                # the previous iteration removed it expecting the oracle to
                # pick FLASHINFER_CUTEDSL with the patch — but cutedsl needs
                # full-NVFP4 activations which W4A16 doesn't supply. Force
                # marlin to avoid the oracle wasting backends.
                "--moe-backend" "marlin"
                "--kv-cache-dtype" "fp8"
                "--attention-backend" "flashinfer"
                "--enforce-eager"
                "--language-model-only"
                "--enable-prefix-caching"
                "--enable-chunked-prefill"
                "--async-scheduling"
                "--max-num-batched-tokens" "8192"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--default-chat-template-kwargs" '{"enable_thinking":true}'
                # EAGLE-3.1 attempted with Dogacel/specdrift-qwen3.6-35b-a3b-eagle3
                # but vLLM 0.22.1.dev0+g3fd9d2d35 doesn't support the drafter's
                # architecture (EagleLlamaForCausalLMEagle3). vLLM has Eagle3
                # wrappers for Llama, MiniMax, Qwen2.5-VL, Qwen3-VL, DeepSeek but
                # not Qwen3.6-35B-A3B MoE. Reverted to MTP K=3. Revisit when:
                # (a) vLLM adds Eagle3Qwen3_5MoeForCausalLM class, OR
                # (b) drafter is republished with an arch name vLLM supports.
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
                "--trust-remote-code"
            )
            ;;
        # qwen3.6-35b-a3b-nvfp4-tq-mtp-2 REMOVED 2026-04-28 — N=2 hypothesis-
        # test profile, dominated. TEB 87 (vs 90 for nvfp4-tq-mtp at same KV
        # with N=4). With TQ KV, N=4 wins; with FP8 KV, N=2 wins. See full
        # 2×2 KV×N matrix in docs/PERFORMANCE-V7.md.
        # qwen3.6-35b-a3b-nvfp4-mtp-fp8kv removed — crashes under 8-concurrent
        # (MoE autotuner picks invalid SM110 tile at M=128). Superseded by
        # qwen3.6-35b-a3b-nvfp4-tq-mtp which is strictly better on all axes.
        # qwen3.5-35b-a3b-nvfp4 removed — superseded by qwen3.6

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
            #
            # 2026-05-30 (v9 staging): chat-template + tool-call parser
            # corrections. Stock Qwen3.6 chat template ships several bugs
            # (Python-only Jinja filters that crash C++ engines, empty
            # <think>\n</think> injection causing ~80%+ premature <|im_end|>
            # aborts on agentic loops, KV-cache-invalidating history pruning,
            # false-positive error retry loops). Applying
            # froggeric/Qwen-Fixed-Chat-Templates v19 (Apache 2.0,
            # variant-agnostic: covers all Qwen 3.5/3.6 sizes, BF16/FP8/NVFP4
            # alike). Pairs with --tool-call-parser qwen3_coder (the native
            # XML parser the fixed template emits) — replaces the prior
            # qwen3_xml parser which expected the stock template's broken
            # XML format. This combination is the likely root cause of the
            # prior 1/9 lane-comparison Qwen3.6 failure per froggeric's
            # documented symptom set.
            #
            # MTP K bumped 2 → 3. Community single-node recipe (forum
            # @Turrican area) runs K=3 at 19-21 tok/s with 85-94% acceptance
            # on Qwen3.6-27B-FP8. K=2 was the conservative pre-template-fix
            # pick; the chat-template fix removes speculative-decoding
            # mismatches and unlocks higher K without quality drop. Watch
            # agentic-quality smoke for any cliff vs K=2.
            THOR_LAUNCH_MODEL_SOURCE="Qwen/Qwen3.6-27B-FP8"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen-fixed-froggeric.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen-fixed-froggeric.jinja"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flashinfer"
                "--language-model-only"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--enable-prefix-caching"
                "--max-num-batched-tokens" "32768"
                "--speculative-config" '{"method":"qwen3_next_mtp","num_speculative_tokens":3}'
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

    # 2026-05-31 model-bakeoff: alias every profile to "cosmos-reason2-8b"
    # additionally so the composer (hardcoded to that model id) can target
    # any profile without re-config. vLLM --served-model-name accepts
    # multiple values; the first one is the canonical id, the rest are
    # aliases routed to the same engine.
    THOR_VLLM_ARGS=(
        "${THOR_VLLM_ARGS[@]}"
        "--served-model-name" "${THOR_MODEL_ID}" "cosmos-reason2-8b"
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

    # Mount Thor-specific fused-MoE configs to suppress the "Using default MoE
    # config. Performance might be sub-optimal" warning on Qwen3.6-35B-A3B
    # (E=256, N=512). Adapted from NVIDIA_H100_80GB_HBM3's tuned config as a
    # starting point; benchmark_moe.py --tune deadlocks at config ~98 on Thor
    # (Triton autotune sharing-mem / TMEM layout issue) so we can't generate
    # an actually-tuned Thor config until the upstream tuner is fixed.
    local moe_cfg_dir="${THOR_MOE_CONFIGS_HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/docker/moe-configs" 2>/dev/null && pwd)}"
    if [[ -d "${moe_cfg_dir}" ]] && ls "${moe_cfg_dir}"/*.json >/dev/null 2>&1; then
        docker_mount_args+=(-v "${moe_cfg_dir}:/opt/nemoclaw-thor/moe-configs:ro")
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
