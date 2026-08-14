# Experimental orchestrated coding workflow

**Status:** exploratory proposal; not a production runbook

**Last updated:** 2026-08-12

**Scope:** development of ManyForge and related repositories using a frontier
Codex supervisor plus sandboxed workers backed by local LAN models

This document proposes an experiment. Its **primary target** is a hybrid Codex
CLI workflow: native Codex subagents perform parallel read-only work, while
separately launched Codex CLI workers make changes in isolated
containers/worktrees. OMP, NOOA, OpenClaw, and other harnesses remain
comparison or fallback candidates. This document does not declare a proven
winner, authorize autonomous commits, or change the ManyForge Composer
assistant stack.

The existing production-facing OpenClaw workflow remains documented in
[`NEMOCLAW-OPENCLAW-WORKFLOW.md`](NEMOCLAW-OPENCLAW-WORKFLOW.md). The workflow
here is for **writing ManyForge code**, not for serving the assistant inside
ManyForge Composer.

## Goal

Find a repeatable coding setup in which:

1. a frontier model running through the operator's existing **Codex CLI** is
   the sole supervisor and integration authority;
2. local models perform bounded work through Codex CLI first, with other
   harnesses retained for comparison;
3. every write-capable worker receives an isolated repository copy or
   worktree;
4. workers return summaries, patches, and evidence, never merge directly;
5. only the frontier supervisor runs authoritative builds, compiles the
   workspace, launches the application, and decides whether a worker patch is
   acceptable; and
6. the experiment can demonstrate whether local workers improve correctness,
   wall-clock time, or frontier-token efficiency.

The initial local inference targets are:

- **Thor:** a Qwen 3.x 27B-class coding profile through an OpenAI-compatible
  endpoint, using a measured context/concurrency profile.
- **Orin AGX 64 GB:** a compatible Qwen 3.x 27B-class profile, after that
  device receives its own measured serving recipe.
- **Thor comparison:** DeepSeek-V4-Flash-0731 through Entrpi DS4 on `:8050`,
  retained as a quality and long-context reference rather than the primary
  worker target.

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
      +-- Lane A: native Codex subagents
      |      - visible in the same CLI session through /agent
      |      - different model/instructions where supported
      |      - read-only exploration, planning, and review
      |      - no concurrent implementation writes in the baseline
      |
      +-- Lane B: externally launched Codex CLI workers
             - one independent process per task
             - one container and repository copy/worktree per process
             - bounded implementation writes and patch export
             - local Qwen endpoint on Thor or Orin
                         |
              OpenAI-compatible LAN inference
                    /                 \
             Qwen on Thor       Qwen on Orin
```

In simple terms, Lane A is a group of assistants at the supervisor's desk: it
can create, inspect, steer, and collect them directly, but they see the same
working environment, so they are safest when reading. Lane B is a group of
contractors in separate workshops: each is a separate Codex process with its
own disposable tree, so it can edit without colliding with the supervisor or
another worker. Lane B needs a launcher and artifact handoff, but supplies the
stronger containment boundary required for implementation.

Each external worker sandbox gets a different copy of the repository.
Comparative runs must never share a writable tree. The supervisor receives
only a patch, manifest, report, transcript/metrics summary, and the worker's
claimed verification evidence.

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

### 2. Native Codex mixed-model subagents — read-only lane

This is the first lane of the primary target. The frontier supervisor asks
Codex to delegate independent exploration, planning, triage, and review to
native subagents. Codex owns their lifecycle and exposes their threads through
`/agent`, so the supervisor can inspect progress and collect concise results
without filling its own context with raw searches and logs.

Custom Codex agents may use different model configurations and instructions.
The compatibility experiment must determine whether a native custom subagent
can use each local Qwen endpoint reliably. Until this is proven, a hosted
Codex subagent is the control and an external local-model worker is the
fallback.

Native subagents inherit the active sandbox/permission environment. They do
not automatically provide one independent Git worktree per subagent. The
baseline therefore restricts them to:

- repository exploration and evidence gathering;
- plan/spec interpretation;
- proposed change lists and implementation plans;
- independent review of a patch already produced elsewhere; and
- test/log analysis where the supervisor supplies the evidence.

Do not ask multiple native subagents to edit the shared checkout in the first
experiment. If a later Codex release provides a verified one-worktree-per-
subagent workflow, test that as a separate variant before changing this rule.

### 3. External Codex CLI workers — isolated implementation lane

This is the second lane of the primary target. The supervisor launches one
independent non-interactive Codex CLI process per bounded coding task. Each
process runs in its own container and clean repository copy/worktree, uses a
local model endpoint, and returns artifacts rather than integrating changes.

Unlike a native subagent, an external worker is not a child thread managed by
the supervisor's `/agent` view. The supervisor or a small deterministic
launcher must create the sandbox, start `codex exec`, enforce the timeout,
capture the transcript, and collect `PATCH.diff` and `REPORT.md`. The extra
plumbing is intentional: it lets implementation workers write without sharing
the supervisor's checkout or another worker's filesystem.

Codex CLI is the first implementation harness to test because it supplies a
strong coding tool loop while keeping the harness family constant between the
frontier supervisor and local-model workers. Current Codex configuration
supports custom model providers, but the custom-provider wire protocol is the
**Responses API**. Each Qwen serving endpoint must pass a live Responses and
tool-loop compatibility smoke before use. Entrpi DS4 v0.5.6.2 exposes a
Responses-compatible endpoint and remains a comparison target.

Provider settings are machine-local and must not be placed in a repository's
`.codex/config.toml`. For a containerized worker, create a disposable
container-local Codex home and place an experimental profile there. An
illustrative, unvalidated Thor profile is:

```toml
model = "<served-qwen-model-id>"
model_provider = "thor_qwen"
model_context_window = 262144
approval_policy = "never"
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = false

[model_providers.thor_qwen]
name = "Thor Qwen"
base_url = "http://192.168.1.136:8050/v1"
wire_api = "responses"
requires_openai_auth = false
stream_idle_timeout_ms = 1800000
```

The outer OpenShell/container boundary, not the Codex CLI sandbox alone, is the
security boundary. Network policy must still allow the model endpoint even if
the inner Codex workspace sandbox denies general network access.

Do not promote this candidate until it proves correct handling of streamed
Responses events, tool calls, long idle periods, cancellation, and truncated
outputs against the selected Qwen server and DS4 comparison server.

### 4. OMP — established local-model comparison

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

### 5. NOOA — optional programmatic dispatcher candidate

NVIDIA Labs Object-Oriented Agents (NOOA) is a high-upside comparison for
repeatable, non-interactive workers. It represents an agent as a Python object:
methods are capabilities, fields are state, docstrings are prompts, and type
annotations are validated contracts.

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

### 6. OpenClaw in NemoClaw — historical control

OpenClaw is not the preferred new coding harness, but it is an important
control because the repository contains a previously successful
supervisor-to-OpenClaw coding workflow. Keep one comparable lane so a new
harness must show a real gain over the historical setup rather than merely
look promising in isolation.

Use the current NemoClaw/OpenShell procedure, not commands copied from old Git
history. The current control-plane workflow is
[`NEMOCLAW-OPENCLAW-WORKFLOW.md`](NEMOCLAW-OPENCLAW-WORKFLOW.md).

### 7. Optional second-wave candidates

Do not put every harness into the first matrix. Add these only if the first
wave leaves an unresolved question:

| Candidate | Why it may be useful | Why it is deferred |
|---|---|---|
| OpenCode | Broad provider support, official container image, mature coding UI | Overlaps OMP; adds another variable before the core comparison is stable |
| Hermes Agent | Already available through NemoClaw; memory and skills may help long projects | Less direct evidence for this exact local coding workload |
| Prime-agent | Potentially useful lightweight Pi-family comparison | OMP is the better-established Pi-family baseline here |
| OMP/NOOA subagents | May increase aggregate local throughput | Nested coordination hides single-worker failures and consumes serving capacity quickly |

## Model allocation hypothesis

This is the initial hypothesis, not a fixed routing policy:

| Role | First-choice model | Reasoning | Responsibility |
|---|---|---|---|
| Main orchestrator and final integrator | GPT-5.6 Sol | Max | Decompose work, schedule concurrency, resolve architecture, review all results, and alone validate and integrate |
| `architecture_analyst` | GPT-5.6 Sol | High | Interpret specs and architecture, identify invariants and cross-repository constraints, and return evidence to the orchestrator |
| `code_explorer` | Local Qwen 3.x 27B | High | Trace implementation paths, symbols, state, and ownership without editing |
| `implementer` | Local Qwen 3.x 27B | High | Make one bounded change within an explicit path allowlist |
| `test_worker` | Local Qwen 3.x 27B | High | Analyze coverage and write focused tests within a separate test-path allowlist |
| `local_reviewer` | Local Qwen 3.x 27B | High | Independently review worker patches for correctness, scope, regressions, and missing tests |
| Long-context auditor, optional | DS4 on Thor | Low or reasoning off | Handle selected unusually large audits without making DS4 a normal worker dependency |

Do not route solely by advertised context length. Most worker tasks should be
small enough to fit well below 128K. The task packet should contain the
necessary contract excerpts and file pointers instead of a dump of the whole
repository.

The role table records the requested first-test configuration, not installed
agent files. For local Qwen, `High` means the worker profile requests high
reasoning consistently; the compatibility test must verify how the selected
server and model expose that setting and whether it improves the fixed task
set. Do not silently substitute a lower effort during the comparison.

### Serving concurrency starting point

Start the primary Qwen experiment with **one active worker per device**. After
single-worker quality and server stability pass, test two and then four total
workers. Increase one device and one context/bank parameter at a time. A
theoretical maximum such as sixteen request slots is a capacity hypothesis,
not an initial orchestration target: active agents also consume KV cache,
prefill bandwidth, tool-loop time, and supervisor review capacity.

Keep no more than two external implementation workers active initially. Native
read-only subagents can run in parallel when their work is independent, but
their model requests still count against the same serving limits.

For the retained DS4 comparison, the currently validated Thor profile is 512K
context with two continuous banks. Start with **at most two active DS4
workers**. Additional requests may queue, but four queued callers are not four
independent 512K banks.

The DS4 comparison should leave DS4 at the documented 512K recipe so that
harness behavior is tested before serving parameters change. A later,
separate comparison may measure:

- 512K allocation / two banks / 4K prefill chunks;
- 256K allocation / candidate higher bank count / 8K chunks; and
- 128K allocation / candidate higher concurrency.

Increase one parameter at a time. A theoretical division of context capacity
does not prove that four 256K or eight 128K banks fit; workspace, drafter,
indexer, admission, and live-memory costs must be measured. Use the stability
gates in [`../serving/docs/DS4-ON-THOR.md`](../serving/docs/DS4-ON-THOR.md).

### Useful total worker count

Begin with one supervisor, up to **two native read-only subagents**, and up to
**two external implementation workers** split across Thor and Orin. After the
single-worker gates pass, increase to three or four external workers only on
non-overlapping tasks. Six to eight total active helpers may be useful for a
large, cleanly partitioned plan, but more than roughly four simultaneous
writers is unlikely to help until patch conflict handling and review
automation are proven. Treat sixteen agents as a later serving-capacity test,
not a recommended workflow size.

Use only one orchestration level initially:

```text
Codex supervisor -> native read-only subagent
Codex supervisor -> isolated external implementation worker
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

### Architectural test A — five-role mixed-model workflow (run first)

This is the first workflow architecture to test before creating durable custom
agent files or comparing additional harnesses. It tests whether one frontier
Codex CLI conversation can supervise a Sol subagent and a local Qwen worker
pool while preserving authority, evidence, and filesystem boundaries.

Use this exact role/model assignment:

| Role | Model | Reasoning | Initial execution lane |
|---|---|---|---|
| Main orchestrator | GPT-5.6 Sol | Max | Parent Codex CLI conversation |
| `architecture_analyst` | GPT-5.6 Sol | High | Native read-only subagent |
| `code_explorer` | Local Qwen 3.x 27B | High | Native read-only subagent if compatible; otherwise isolated `codex exec` |
| `implementer` | Local Qwen 3.x 27B | High | Isolated write-capable `codex exec` worker |
| `test_worker` | Local Qwen 3.x 27B | High | Isolated write-capable `codex exec` worker with test-only paths |
| `local_reviewer` | Local Qwen 3.x 27B | High | Read-only worker, preferably on the other Jetson from the implementer |

Run one small but representative ManyForge task from a frozen commit:

1. The orchestrator freezes the objective, authoritative files, acceptance
   criteria, dependency graph, and per-worker path ownership.
2. `architecture_analyst` and `code_explorer` run concurrently. The former
   returns architectural constraints; the latter returns the observed code
   path and likely change surface.
3. The orchestrator reconciles those reports and decides what may run at the
   same time. It dispatches `implementer` and `test_worker` concurrently only
   when their writable paths and shared state are disjoint; otherwise it
   serializes them.
4. `local_reviewer` receives the frozen requirements and resulting patches,
   but not the workers' conclusions, and returns an independent findings-first
   review.
5. The Sol Max orchestrator inspects all diffs, runs the authoritative build
   and tests, and decides whether anything is acceptable. No subagent commits,
   pushes, merges, launches the application, or operates hardware.

The architecture test passes only if:

- the parent successfully selects the requested model and High reasoning for
  every subagent/worker;
- Qwen completes the Responses/tool-result loop without malformed tool calls,
  stalls, or leaked tool syntax;
- concurrent workers remain inside their path allowlists and produce patches
  that apply cleanly;
- the reviewer finds seeded defects or meaningful omissions without copying
  the implementer's rationale;
- only the orchestrator performs authoritative validation and integration;
  and
- the run records per-role model, effective reasoning setting, endpoint,
  tokens, wall time, interventions, and serving metrics.

Repeat the exact task once with all local roles on Thor, once with all local
roles on Orin, and once split across both devices. Keep task, prompts, context,
sampling, and concurrency fixed. Do not create permanent project-scoped agent
files until this test establishes that the role boundaries and model routing
are useful.

### Phase 0 — primary-target compatibility smoke

First test the two Codex lanes against one Qwen endpoint on four small tasks:

1. read-only architecture question with exact file citations;
2. one-file mechanical edit;
3. two-file behavioral edit with an explicit invariant; and
4. patch export after a simulated interrupted session.

Use native subagents only for task 1 and an independent review of tasks 2–4.
Use isolated external Codex CLI processes for all edits. Repeat the same smoke
against the second Qwen endpoint, then against DS4 as a model comparison.

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
2. frontier Codex using native read-only subagents;
3. frontier Codex supervising isolated Codex CLI+Qwen workers;
4. the hybrid primary target: native readers plus isolated Qwen writers;
5. the same isolated Codex CLI lane with DS4 for a model comparison; and
6. OMP, NOOA, or OpenClaw controls only where they answer a remaining harness
   question.

Run candidates sequentially at first so inference contention cannot distort
the harness comparison. Add a concurrent-throughput phase only after the
single-worker matrix is complete.

### Phase 2 — multi-device and multi-model routing

Keep Codex CLI as the primary worker harness and compare:

- Thor Qwen alone;
- Orin Qwen alone;
- one device implementing while the other performs an independent review;
- parallel non-overlapping implementation tasks split across Thor and Orin;
  and
- DS4 only on selected long-context or difficult comparison cases.

Do not let both workers edit the same files in the concurrency test. A
same-task ensemble should return independent patches or reviews for Codex to
reconcile.

### Phase 3 — context and concurrency

Only after correctness is stable, measure:

- one, two, and four Qwen workers across Thor and Orin;
- per-device 256K and 128K serving candidates;
- the later six-, eight-, and sixteen-request capacity hypotheses without
  assuming all requests should be implementation writers;
- one versus two DS4 workers on the retained 512K/two-bank recipe;
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

1. **Five-role architectural test:** run the Sol Max orchestrator, Sol High
   `architecture_analyst`, and four High-reasoning Qwen roles on one frozen
   ManyForge task, including the Thor/Orin routing variants above.
2. **Native subagent smoke:** deepen mixed-model configuration, read-only
   delegation, `/agent` visibility, cancellation, and summary-quality checks.
3. **Isolated Codex CLI+Qwen smoke:** verify the Responses/tool loop, patch
   export, timeout handling, and the container/worktree boundary on Thor.
4. **Hybrid workflow task:** native subagents explore and review while an
   external Qwen worker implements; the frontier supervisor validates.
5. **Add Orin:** repeat the same worker contract against Orin and test routing
   between the two devices.
6. **Run model/harness controls:** compare DS4 and, only where useful, OMP,
   NOOA, and historical OpenClaw on the same frozen tasks.
7. **Test bounded concurrency:** scale from two to four local workers; test
   higher counts only after lower-context serving profiles and supervisor
   review capacity are measured.

Primary target to validate:

- **Codex CLI frontier** remains the supervisor and validator.
- **Native Codex subagents** handle parallel read-only exploration, planning,
  triage, and review.
- **External Codex CLI workers** use local Qwen endpoints and isolated
  containers/worktrees for implementation.
- **OpenShell or an equivalent outer sandbox** remains the containment layer.
- **OMP, NOOA, and OpenClaw** remain comparisons or fallbacks unless one shows
  a measured advantage on the fixed task set.

This is a hypothesis to test, not the conclusion of this document.

## Open questions

- Can native custom Codex subagents reliably use both LAN Qwen endpoints, or
  must local-model workers remain external processes?
- Does each Qwen serving stack implement Codex's Responses and tool-result
  continuation behavior completely enough for long coding sessions?
- What is the smallest deterministic launcher that gives every external Codex
  worker a clean container/worktree, timeout, transcript, and artifact bundle?
- Does DS4 remain useful enough on difficult or long-context tasks to justify
  a separate comparison lane?
- Is OMP's occasional skipped/stuck terminal behavior eliminated by running
  one fresh process per task sandbox?
- Which Qwen 3.x 27B profile and quantization give the best coding reliability
  on Thor and Orin without starving Isaac ROS?
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

- [OpenAI Codex subagents](https://developers.openai.com/codex/multi-agent)
- [OpenAI Codex non-interactive mode](https://learn.chatgpt.com/codex/non-interactive-mode)
- [OpenAI Codex Git worktrees](https://learn.chatgpt.com/codex/environments/git-worktrees)
- [OpenAI Codex configuration reference](https://developers.openai.com/codex/config-reference)
- [NVIDIA NOOA repository](https://github.com/NVIDIA-NeMo/labs-OO-Agents)
- [NVIDIA: Six Agent Harness Capabilities for Higher Model Performance](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/)
- [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)
- [OMP / oh-my-pi](https://github.com/can1357/oh-my-pi)
- [OpenCode documentation](https://opencode.ai/docs/)
