# Move serving work to `jetson-llm-recipes` (renamed `thor_llm`)

Status: **planned, not started** (2026-11-15). Starts after the current Flash-Next speed work
([`serving/docs/FLASH-NEXT-SPEED-PLAN.md`](serving/docs/FLASH-NEXT-SPEED-PLAN.md)) is done.

## Why

`serving/` is about half of this repo (160 of ~330 tracked files) and is its own project: which model to serve on
Jetson Thor / Orin and how (vLLM overlays, kernels, benchmarks, quality evals). It does not depend on NemoClaw or
ManyForge; the assistant only needs an OpenAI-compatible endpoint and a model name. A dedicated repo keeps both
repos clean, easier to find and to iterate on.

Target: rename the existing public repo `pastoriomarco/thor_llm` (25 stars, 6 forks, README + `models/`, no
licence, last push 2026-03-21) to `jetson-llm-recipes`. A rename keeps stars, watchers, forks and issues, and GitHub
redirects the old web and git URLs as long as no new repo named `thor_llm` is created.

## What moves, what stays

| Stays in NemoClaw-Thor | Moves to `jetson-llm-recipes` |
|---|---|
| `manyforge/`, `setup/`, assistant plans (sandbox, fine-tune), user quickstart, the assistant side of `ORIN-SETUP.md` | `serving/`: overlays (`docker/patches/`), start scripts, `benchmarks/`, `calibration/`, `agentic-bench/`, `tools/`, `templates/`, model docs and plans (Flash-Next, DS4, Qwen3.8-27B, Cosmos, SM110 MoE, speed plan, accelerator ROI, Tool Eval) |
| A short serving page: which recipe the assistant uses, pinned to a `jetson-llm-recipes` version | The `tps_proxy` (today `manyforge/scripts/debug/tps_proxy.py`), the Tool Eval runner and the agent replay scripts |
| | The Thor-only run scripts and overlay copies (`~/fn-bench/start.sh`, `round*.sh`, `r3env.sh`, `replay.py`, `measure.sh`, `prefill.sh`, `~/fn-cutedsl-moe`, `~/fn-fi-moe`, `~/fn-mxfp8-tactic`), turned into one parameterized launcher per recipe |

## Steps

1. **Maintainer:** rename `thor_llm` to `jetson-llm-recipes` on GitHub.
2. In a local working copy under `worktrees/` (no pushes): tag and branch the current `thor_llm` state as
   `legacy-thor-llm`.
3. Extract `serving/` history from NemoClaw-Thor with `git filter-repo --subdirectory-filter serving`, dropping
   private material in the same pass (`calibration/capture_tasks/omp_tasks.jsonl`, already public in `607c4bf`, so
   the new repo never contains it). Merge it into the renamed repo as an unrelated history line
   (`--allow-unrelated-histories`), so blame and dates carry over.
4. Restructure: `recipes/<model>/<recipe>/` (env + arguments, e.g. `flash-next/r8b`, `flash-next/n3`), `patches/`
   (overlays unchanged), `bench/`, `docs/`; bring in the Thor-only scripts as a launcher; fix doc links.
5. README: "formerly `thor_llm`", scope (LLM serving recipes for Jetson Thor and Orin), results table per device,
   links to recipes. Add a licence (overlays already carry vLLM / FlashInfer Apache-2.0 files).
6. NemoClaw-Thor: remove `serving/` (moved), add the short serving page with the pinned version, update
   `PLANS_INDEX.md`, `INDEX.md`, `README.md`, `AGENTS.md` links.
7. **Maintainer:** review, commit and push both repos.

## Open decisions (maintainer)

- Licence for `jetson-llm-recipes`.
- Whether any non-Flash-Next docs (DS4, Cosmos, MiniMax investigations) move or are archived.
