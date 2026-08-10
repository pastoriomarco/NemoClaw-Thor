# Experimental orchestrated coding workflow

**Status:** exploratory proposal; not a production runbook

**Last updated:** 2026-08-10

**Scope:** development of ManyForge and related repositories using a frontier
Codex supervisor plus sandboxed workers backed by local LAN models

This document proposes an experiment. It records what to test, how to compare
the candidates, and the safety boundary to preserve while testing. It does not
declare a winning harness, authorize autonomous commits, or change the
ManyForge Composer assistant stack.

The existing production-facing OpenClaw workflow remains documented in
[`NEMOCLAW-OPENCLAW-WORKFLOW.md`](NEMOCLAW-OPENCLAW-WORKFLOW.md). The workflow
here is for **writing ManyForge code**, not for serving the assistant inside
ManyForge Composer.

## Goal

Find a repeatable coding setup in which:

1. a frontier model running through the operator's existing **Codex CLI** is
   the sole supervisor and integration authority;
2. one or more local models perform bounded implementation work through
   replaceable worker harnesses;
3. every worker receives an isolated repository copy or worktree;
4. workers return patches and evidence, never merge directly;
5. only Codex runs authoritative builds, compiles the workspace, launches the
   application, and decides whether a worker patch is acceptable; and
6. the experiment can demonstrate whether local workers improve correctness,
   wall-clock time, or frontier-token efficiency.

The initial local inference candidates are:

- **Thor:** DeepSeek-V4-Flash-0731 through Entrpi DS4 on `:8050`.
- **Orin AGX 64 GB:** a smaller coding model such as a Qwen 3.6 27B/35B-class
  profile, after that device receives its own measured serving recipe.

The laptop should run the supervisor and worker harness containers. It should
not need to host the large models.

## Non-goals

- Replacing the ManyForge Composer assistant-provider architecture.
- Giving a local worker access to the operator's integration checkout.
- Letting workers commit, push, merge, publish, or operate robot hardware.
- Letting workers run Docker on the host, launch ROS, or start ManyForge.
- Assuming a benchmark result from a frontier cloud model transfers to DS4.
- Maximizing agent count before single-worker reliability is established.
- Installing experimental harnesses persistently on the laptop or Thor host.

## Proposed topology

```text
Human operator
      |
      v
Frontier Codex CLI supervisor
  - reads specs and integration state
  - decomposes and dispatches tasks
  - reviews every returned patch
  - alone builds, tests, runs, commits, and pushes
      |
      +----------------------+----------------------+-------------------+
      |                      |                      |                   |
      v                      v                      v                   v
 OMP worker             NOOA worker          local Codex CLI     OpenClaw control
 sandbox A              sandbox B            worker sandbox C    sandbox D
      |                      |                      |                   |
      +----------------------+----------------------+-------------------+
                             |
                  OpenAI-compatible LAN inference
                     /                         \
        DS4 on Thor :8050             Qwen-class model on Orin
```

Each worker sandbox gets a different copy of the repository. Comparative runs
must never share a writable tree. The supervisor receives only a patch,
manifest, report, transcript/metrics summary, and the worker's claimed
verification evidence.

## Roles and authority

### 1. Codex CLI with a frontier model — supervisor

This is the control plane and the only trusted integration actor. It should:

- read the applicable `AGENTS.md`, specifications, plans, and implementation;
- freeze the task scope and base commit;
- select the worker model and harness;
- generate a complete task packet;
- monitor progress and stop loops;
- review the returned diff against the task and repository rules;
- apply promising patches to a disposable validation worktree;
- run all authoritative builds, tests, linters, ROS commands, and application
  instances;
- request revisions when evidence is insufficient; and
- present the final diff to the human before any commit or push.

The supervisor should not delegate final architectural judgment, cross-repo
contract interpretation, destructive operations, or release decisions.

### 2. Codex CLI with a local model — worker candidate

Codex CLI itself is worth testing as a local-model worker because it supplies a
strong coding tool loop while keeping the harness constant between frontier and
local-model runs. Current Codex configuration supports custom model providers,
but the custom-provider wire protocol is the **Responses API**. Entrpi DS4
v0.5.6.2 exposes a Responses-compatible endpoint, so the combination is
plausible but still needs a live compatibility test.

Provider settings are machine-local and must not be placed in a repository's
`.codex/config.toml`. For a containerized worker, create a disposable
container-local Codex home and place an experimental profile there. An
illustrative, unvalidated profile is:

```toml
model = "deepseek-v4-flash"
model_provider = "thor_ds4"
model_context_window = 524288
approval_policy = "never"
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = false

[model_providers.thor_ds4]
name = "Thor DS4"
base_url = "http://192.168.1.136:8050/v1"
wire_api = "responses"
requires_openai_auth = false
stream_idle_timeout_ms = 1800000
```

The outer OpenShell/container boundary, not the Codex CLI sandbox alone, is the
security boundary. Network policy must still allow the DS4 endpoint even if
the inner Codex workspace sandbox denies general network access.

Do not promote this candidate until it proves correct handling of streamed
Responses events, tool calls, long idle periods, cancellation, and truncated
outputs against DS4.

### 3. OMP — established local-model baseline

OMP means the `can1357/oh-my-pi` terminal coding harness used in the earlier
DS4 experiments. It is the first practical baseline because it already:

- connects to an OpenAI-compatible local endpoint;
- can inspect and edit a repository;
- has a usable terminal interface and local session artifacts;
- exposes tool progress clearly enough to diagnose stalls; and
- supports model and subagent experimentation.

For this study, disable worker subagents initially. Run one OMP process per
sandbox and task. Automatic approval may be enabled only for a narrow set of
read/search operations inside the outer sandbox. Writes remain bounded to the
task copy; shell actions outside the approved read-only set remain denied or
explicitly enumerated.

OMP is the baseline to beat on setup effort and interactive usability. Its
known risks are long tool loops, commands reported as skipped, context growth,
and sessions that remain in a thinking state after the terminal action has
finished.

### 4. NOOA — high-upside programmatic worker candidate

NVIDIA Labs Object-Oriented Agents (NOOA) is the most interesting experimental
candidate for repeatable, non-interactive workers. It represents an agent as a
Python object: methods are capabilities, fields are state, docstrings are
prompts, and type annotations are validated contracts.

The properties worth testing on local models are:

- **pass by reference:** large tool results can remain live objects instead of
  being serialized into every prompt;
- **typed results:** the worker can be required to return a patch manifest,
  evidence, and unresolved risks in a validated structure;
- **code as action:** Python control flow may be more reliable for DS4 than a
  large catalog of native JSON tool calls;
- **programmable loops:** timeout, retry, inspection, and completion rules can
  be deterministic Python rather than prompt suggestions;
- **tracing:** calls and method relationships are inspectable; and
- **per-method model routing:** a strong model can implement while a smaller
  model performs bounded classification or review.

NOOA is a research preview. Its published SWE-bench results used frontier
models and do not establish DS4 quality. Its in-process Python/AST checks are
also not containment. Every NOOA worker must run inside OpenShell or an
equivalent OS-level sandbox.

The first NOOA implementation should be deliberately small. Start from the
generic `BenchAgent` pattern and expose only these capabilities:

```text
read_file(path, range)
search(pattern, paths)
list_files(path)
write_file(path, content)       # task-allowlisted paths only
apply_patch(patch)              # task-allowlisted paths only
git_diff()
finish(result: WorkerResult)
```

Do not expose arbitrary host shell, Docker, SSH, ROS devices, Git credentials,
or network clients. If Python execution can still reach the filesystem, the
outer sandbox must make everything except the worker copy inaccessible.

### 5. OpenClaw in NemoClaw — historical control

OpenClaw is not the preferred new coding harness, but it is an important
control because the repository contains a previously successful
supervisor-to-OpenClaw coding workflow. Keep one comparable lane so a new
harness must show a real gain over the historical setup rather than merely
look promising in isolation.

Use the current NemoClaw/OpenShell procedure, not commands copied from old Git
history. The current control-plane workflow is
[`NEMOCLAW-OPENCLAW-WORKFLOW.md`](NEMOCLAW-OPENCLAW-WORKFLOW.md).

### 6. Optional second-wave candidates

Do not put every harness into the first matrix. Add these only if the first
wave leaves an unresolved question:

| Candidate | Why it may be useful | Why it is deferred |
|---|---|---|
| OpenCode | Broad provider support, official container image, mature coding UI | Overlaps OMP; adds another variable before the core comparison is stable |
| Hermes Agent | Already available through NemoClaw; memory and skills may help long projects | Less direct evidence for this exact local coding workload |
| Prime-agent | Potentially useful lightweight Pi-family comparison | OMP is the better-established Pi-family baseline here |
| OMP/NOOA subagents | May increase aggregate local throughput | Nested coordination hides single-worker failures and consumes DS4 banks quickly |

## Model allocation hypothesis

This is the initial hypothesis, not a fixed routing policy:

| Work | First-choice model | Reason |
|---|---|---|
| Task decomposition, spec interpretation, final review | Frontier Codex | Highest judgment and full integration context |
| Difficult implementation and cross-file refactors | DS4 on Thor | Strongest currently served local coding model |
| Focused implementation, tests, documentation, independent critique | Qwen-class model on Orin | Preserves Thor capacity and adds model diversity |
| Deterministic validation, builds, app launch | No model decision; Codex executes | Evidence must come from the trusted integration environment |

Do not route solely by advertised context length. Most worker tasks should be
small enough to fit well below 128K. The task packet should contain the
necessary contract excerpts and file pointers instead of a dump of the whole
repository.

### DS4 concurrency starting point

The currently validated Thor profile is 512K context with two continuous
banks. Start with **at most two active DS4 workers**. Additional requests may
queue, but four queued callers are not four independent 512K banks.

The initial experiment should leave DS4 at the documented 512K recipe so that
harness behavior is tested before serving parameters change. A later,
separate experiment may compare:

- 512K allocation / two banks / 4K prefill chunks;
- 256K allocation / candidate higher bank count / 8K chunks; and
- 128K allocation / candidate higher concurrency.

Increase one parameter at a time. A theoretical division of context capacity
does not prove that four 256K or eight 128K banks fit; workspace, drafter,
indexer, admission, and live-memory costs must be measured. Use the stability
gates in [`../serving/docs/DS4-ON-THOR.md`](../serving/docs/DS4-ON-THOR.md).

### Useful total worker count

Begin with one supervisor and **two active local workers**. After the single
worker gates pass, increase to three or four workers split across Thor and
Orin. More than roughly five independently changing workers is unlikely to
help until task decomposition, patch conflict handling, and review automation
are proven.

Use only one orchestration level initially:

```text
Codex supervisor -> worker
```

Do not permit `Codex -> worker main agent -> worker subagents` during baseline
measurement. Nested fan-out may be tested later as its own variable.

## Sandbox and repository boundary

The desired boundary is OpenShell. A disposable non-privileged OCI container
is acceptable for an early harness compatibility smoke, but it must not be
treated as equivalent evidence until the OpenShell deployment also passes.

Every worker environment must satisfy all of the following:

- one task and one repository snapshot per sandbox;
- a pinned base commit recorded before dispatch;
- no writable mount of `~/workspaces/dev_ws`;
- no host Docker socket;
- no GPU or robot device mounts for coding workers;
- no `~/.ssh`, cloud credentials, browser state, host Codex home, or host OMP
  home mounted into the worker;
- no Git push credentials;
- no host package installation;
- model-endpoint-only egress by default;
- no remote-origin fetch by the worker; the supervisor fetches and stages any
  requested comparison commit; and
- an output directory writable by the worker for patch and report artifacts.

A worker may modify only its task copy. Even a read-only audit should use a
read-only repository mount or immutable snapshot so the boundary is testable,
not merely promised in a prompt.

## Task packet contract

Every harness receives semantically identical input. Render harness-specific
syntax only after the common task packet is frozen.

Minimum `TASK.md`:

```markdown
# Task: <short title>

Task ID: <stable-id>
Repository: <repo name>
Base commit: <full SHA>
Mode: read-only | edit

## Objective
<one bounded result>

## Authoritative context
<spec/plan paths and necessary contract excerpts>

## Allowed paths
<explicit files or directories>

## Forbidden actions
- Do not commit, push, fetch, install packages, build, run tests, or launch apps.
- Do not modify files outside Allowed paths.
- Do not change contracts unless the task explicitly says to do so.

## Acceptance criteria
<observable requirements that Codex will validate independently>

## Deliverables
- PATCH.diff
- REPORT.md
- FILES.txt
- METRICS.json if supported by the harness
```

`REPORT.md` must distinguish:

- what the worker observed;
- what it changed;
- what remains inferred or uncertain;
- checks it performed that do not count as authoritative validation; and
- the exact validation it recommends Codex run.

## Standard execution lifecycle

1. **Supervisor orientation**
   - Read repository instructions and the controlling plan/spec.
   - Confirm the integration checkout's branch and dirty state.
   - Freeze the task scope, base SHA, and acceptance criteria.

2. **Create isolated inputs**
   - Create one clean copy/worktree per harness.
   - Remove credentials and host-specific state.
   - Record a hash/manifest of the starting tree.

3. **Dispatch**
   - Start one worker first.
   - Provide only `TASK.md`, the repository snapshot, and the approved local
     model endpoint.
   - Record harness, harness version, model ID, serving profile, sampling,
     start time, and task ID.

4. **Monitor**
   - Observe model requests, tool activity, output growth, DS4 health, memory,
     and queueing.
   - Do not rescue a benchmark run silently. Record every intervention.

5. **Collect artifacts**
   - Stop the worker after completion or a stop condition.
   - Export the patch and reports without applying them to the integration
     checkout.
   - Capture endpoint and harness metrics.

6. **Codex quality gate**
   - Inspect the patch for scope, architecture, and suspicious changes.
   - Apply it to a fresh disposable validation worktree.
   - Run the authoritative formatting, build, unit, integration, and app-level
     checks appropriate to the task.
   - Reject or request a bounded revision when evidence fails.

7. **Human gate**
   - Present the accepted diff and validation evidence.
   - Commit only after explicit authorization.
   - Push only after separate explicit authorization.

## Stop conditions

Terminate and mark a worker run failed when any of these occurs:

- the same failed tool action repeats three times without new evidence;
- no tool, model-stream, transcript, or artifact progress occurs for five
  minutes after endpoint health is confirmed;
- the worker attempts to access a forbidden path, credential, device, Docker
  socket, or remote write;
- malformed tool/code actions persist after two repair attempts;
- context or output grows substantially without progress toward a deliverable;
- the model server falls back unexpectedly, restarts, raises CUDA/Xid/OOM, or
  violates its live-memory floor; or
- the worker claims completion without producing the required patch/report.

Long DS4 prefill is not by itself a stall. Use server metrics/logs to
distinguish active prefill from an idle harness.

## Experimental comparison

### Phase 0 — compatibility smoke

Run each first-wave harness against DS4 on four small tasks:

1. read-only architecture question with exact file citations;
2. one-file mechanical edit;
3. two-file behavioral edit with an explicit invariant; and
4. patch export after a simulated interrupted session.

Pass requirements:

- no malformed or leaked tool syntax in the final answer;
- no edit outside the allowlist;
- valid patch application;
- correct completion/stop signal;
- cancellation leaves the sandbox recoverable; and
- no server restart, fallback, Xid, or OOM.

### Phase 1 — controlled ManyForge task set

Select at least eight historical or newly constructed tasks from fixed commits:

- Python defect repair;
- C++ defect repair;
- ROS/package integration edit;
- test-writing task;
- plan-to-implementation task;
- cross-file refactor;
- read-only architectural audit; and
- adversarial review of an intentionally flawed patch.

Prefer tasks with hidden or independently held acceptance tests. Do not select
only tasks whose answer appears verbatim in plans or Git history.

Run the same task packet through:

1. frontier Codex working alone — quality and token baseline;
2. Codex supervising OMP+DS4;
3. Codex supervising NOOA+DS4;
4. Codex supervising local-Codex-CLI+DS4; and
5. Codex supervising OpenClaw+DS4 as the historical control.

Run candidates sequentially at first so inference contention cannot distort
the harness comparison. Add a concurrent-throughput phase only after the
single-worker matrix is complete.

### Phase 2 — multi-model routing

Take only the two best worker harnesses from Phase 1 and compare:

- DS4 alone;
- the Orin Qwen-class model alone;
- DS4 implementation plus Orin review; and
- parallel non-overlapping tasks split across Thor and Orin.

Do not let both workers edit the same files in the concurrency test. A
same-task ensemble should return independent patches or reviews for Codex to
reconcile.

### Phase 3 — context and concurrency

Only after correctness is stable, measure:

- one versus two DS4 workers on the current 512K/two-bank recipe;
- the separately validated 256K and 128K serving candidates;
- retained-prefix effectiveness across iterative revisions;
- queue latency and aggregate decode throughput; and
- whether more workers reduce or increase supervisor review time.

Change exactly one serving or harness parameter per comparison.

## Metrics

Record both worker output and the supervisor's independent result:

| Category | Metric |
|---|---|
| Correctness | Hidden/independent tests passed; build result; regression count |
| Scope control | Unauthorized files/actions; unnecessary diff size; contract violations |
| Patch quality | Applies cleanly; idiomatic implementation; maintainability; test quality |
| Harness reliability | Tool failures; malformed calls; skipped commands; stalls; retries; interventions |
| Efficiency | Frontier input/output tokens; local prompt/output tokens; calls; wall time; supervisor review time |
| Serving | TTFT; prefill tok/s; decode tok/s; queue time; cache reuse; peak RAM/swap |
| Reproducibility | Same task repeated successfully; output variance; complete artifacts |

Keep **frontier tokens** and **local tokens** separate. Local tokens have no API
charge but consume wall time and serving capacity. A workflow is not more
token-efficient merely because it moves a very large amount of low-quality
generation to a local endpoint.

### Initial promotion criteria

A worker harness becomes a recommended experimental default only if it:

- causes zero forbidden writes or external actions;
- produces valid, applicable artifacts in at least 90% of completed runs;
- matches or improves the OpenClaw/OMP baseline on independently validated
  correctness;
- requires human/supervisor rescue in no more than 10% of runs;
- shows either a meaningful reduction in frontier tokens/review time or a
  meaningful correctness gain; and
- passes repeated OpenShell-sandbox runs, not only an unrestricted container
  demonstration.

No benchmark score overrides a security-boundary failure.

## Expected experiments and decisions

The likely near-term sequence is:

1. **OMP+DS4 baseline:** quantify the setup that already works interactively.
2. **local Codex CLI+DS4 smoke:** determine whether DS4's Responses surface is
   sufficiently compatible with Codex's coding loop.
3. **minimal NOOA+DS4 worker:** test typed output, pass-by-reference, patch
   generation, and cancellation without building a large framework first.
4. **OpenClaw control rerun:** measure the historical approach on the same task
   set and current software.
5. **Select two harnesses:** stop broad exploration and deepen only the best
   interactive and best programmatic candidate.
6. **Add Orin:** evaluate model routing only after its serving recipe and
   single-worker baseline are stable.
7. **Test bounded concurrency:** two DS4 workers first; higher counts only with
   a separately validated lower-context serving profile.

Possible final outcome:

- **Codex CLI frontier** remains the supervisor and validator.
- **OMP** remains the human-facing exploratory local session tool.
- **NOOA** becomes the deterministic programmatic worker/dispatcher if its DS4
  compatibility and task results hold up.
- **OpenShell/NemoClaw** remains the containment and local-inference routing
  substrate.
- **OpenClaw** remains available as a known control or fallback rather than the
  default coding worker.

This is a hypothesis to test, not the conclusion of this document.

## Open questions

- Does DS4 reliably produce NOOA CodeAct Python and typed terminal results?
- Does Codex CLI's Responses loop interoperate fully with DS4 v0.5.6.2,
  especially long SSE idle periods and tool-result continuation?
- Does pass-by-reference materially lower DS4 prefill and retained-context
  growth on real ManyForge work?
- Is OMP's occasional skipped/stuck terminal behavior eliminated by running
  one fresh process per task sandbox?
- Which Qwen-class model and quantization give the best coding reliability on
  Orin without starving Isaac ROS?
- Does an independent second local model improve review quality enough to
  justify its latency?
- At what worker count does Codex spend more effort reconciling patches than it
  saves through delegation?

## References

Repository-local:

- [`NEMOCLAW-OPENCLAW-WORKFLOW.md`](NEMOCLAW-OPENCLAW-WORKFLOW.md) — current
  NemoClaw/OpenClaw onboarding and operation.
- [`../serving/docs/DS4-ON-THOR.md`](../serving/docs/DS4-ON-THOR.md) — current
  DS4 recipe, performance, capacity, and stability evidence.
- [`../VERSIONS.md`](../VERSIONS.md) — verified stack pins.
- [`../serving/docs/KV-CACHE-BUDGET.md`](../serving/docs/KV-CACHE-BUDGET.md) —
  historical model/concurrency measurements, including Qwen3.5-122B-A10B.

External primary sources:

- [OpenAI Codex configuration reference](https://developers.openai.com/codex/config-reference)
- [NVIDIA NOOA repository](https://github.com/NVIDIA-NeMo/labs-OO-Agents)
- [NVIDIA: Six Agent Harness Capabilities for Higher Model Performance](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/)
- [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)
- [OMP / oh-my-pi](https://github.com/can1357/oh-my-pi)
- [OpenCode documentation](https://opencode.ai/docs/)
