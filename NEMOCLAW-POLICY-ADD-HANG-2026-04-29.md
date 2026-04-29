# `configure-local-provider.sh` hang on `nemotron3-nano-omni-30b-a3b-nvfp4`

**Date:** 2026-04-29
**Repo state:** post-served-name alignment commit `55c87b1`
**vLLM:** v8 image, `optimistic_zhukovsky` container, serving
`nemotron3-nano-omni-30b-a3b-nvfp4` on `127.0.0.1:8000` (verified responsive
throughout the incident).

## Symptom

`./configure-local-provider.sh nemotron3-nano-omni-30b-a3b-nvfp4` hangs
indefinitely after the registry-update step. Two independent runs
exhibited the same behaviour: progressed through provider update,
inference-route set, sandbox runtime config sync, runtime-config save,
registry update — then stopped before reaching gateway start + warmup.

## Stage reached on the partial run

```
✓  OpenShell gateway 'nemoclaw' is running
✓  Provider vllm-local updated
✓  Inference route set to vllm-local / nemotron3-nano-omni-30b-a3b-nvfp4 (180s)
✓  Sandbox runtime config synced to nemotron3-nano-omni-30b-a3b-nvfp4
✓  Saved runtime config to /home/tndlux/.config/nemoclaw-thor/config.env
   Registry updated: my-assistant → vllm-local/nemotron3-nano-omni-30b-a3b-nvfp4
   <hangs here>
```

The next step in
[NEMOCLAW-OPENCLAW-WORKFLOW.md](NEMOCLAW-OPENCLAW-WORKFLOW.md) is
`local-inference` policy-preset apply, then gateway start + warmup —
none of which executed.

## Diagnosis

The active child of the configure script was
`nemoclaw my-assistant policy-add local-inference` (a Node.js process).
Process state evidence:

- `State: S (sleeping)`, blocked in `do_epoll_wait`.
- `voluntary_ctxt_switches` low (~36) — process is fully idle, not
  spinning.
- **No active TCP/Unix sockets** open from any of its 7 worker threads
  (`ss -tnp` returned nothing for any thread id).
- File descriptors are exclusively io_uring/eventpoll/eventfd/pipe — pure
  Node.js worker plumbing, no I/O surface to anything external.
- A transient `kubectl` child appeared briefly at 99% CPU during one
  sample but did not persist; subsequent samples (5 over 10 s) showed
  no children at all.

OpenShell cluster pods all `Running` (`agent-sandbox-controller-0`,
`coredns`, `local-path-provisioner`, `metrics-server`, `my-assistant`,
`openshell-0`, `thor-v5`); cluster API responsive; gateway port 8080
listening on host.

In-sandbox state:

- No `openclaw`/`node` processes running inside `my-assistant`.
- Gateway log `/sandbox/openclaw-gateway.log` last touched 2026-04-20
  (Cosmos validation run); no entries for the current session.
- Sandbox port 18789 (gateway) NOT listening.
- `~/.nemoclaw/sandboxes.json` already shows
  `my-assistant → model: nemotron3-nano-omni-30b-a3b-nvfp4, provider:
  vllm-local, policies: ["pypi", "npm", "github"]`.

## Hypothesis

`nemoclaw policy-add` deadlocks on a Promise/event that never resolves
when invoked under the configure script's pty + node version
(`v22.22.1`) combination. The transient kubectl spin suggests it
attempts a kubectl-mediated patch into the sandbox pod, then hangs on
a follow-up await rather than retrying. Nothing in the cluster or vLLM
appears responsible.

The April 20 Cosmos run on the same script went through the same step
without issue (per workflow doc), so the hang is something that has
either drifted into the toolchain since then or is specific to the
new served-name slug.

## Workarounds

1. **Skip `policy-add` for single-agent assistant smoke.** The
   `local-inference` policy is a sandbox egress allowlist. For a
   single-agent OpenClaw dispatch test it is not load-bearing. The
   provider, route, and sandbox runtime config — all set successfully —
   are the load-bearing pieces.

2. **Manually start the gateway** using the documented recipe in
   [NEMOCLAW-OPENCLAW-WORKFLOW.md:117-139](NEMOCLAW-OPENCLAW-WORKFLOW.md):

   ```bash
   docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent -- bash -c '
   for d in /proc/[0-9]*; do
       p="${d##*/}"
       c=$(cat "$d/comm" 2>/dev/null || true)
       case "$c" in openclaw*|node*) kill "$p" 2>/dev/null || true ;; esac
   done
   sleep 2
   HOME=/sandbox nohup openclaw gateway run --auth none --port 18789 \
       > /sandbox/openclaw-gateway.log 2>&1 &
   for i in 1 2 3 4 5 6 7 8; do
       sleep 2
       ss -tlnp 2>/dev/null | grep -q ":18789" && break
   done
   HOME=/sandbox openclaw agent --session-id smoke-omni-$(date +%s) --json --timeout 600 \
       -m "Write '\''hello world'\'' to /tmp/smoke.txt and confirm."
   '
   ```

3. **Try `nemoclaw policy-add` standalone** (outside the configure
   script) to isolate whether the hang is script-environment-related
   or upstream:
   ```bash
   nemoclaw my-assistant policy-add local-inference
   ```

## Open questions for follow-up

- Does `nemoclaw policy-add` hang on other profiles too, or only on
  the Omni one? (Re-test on `cosmos-reason2-8b` would confirm.)
- Is there a NemoClaw CLI version skew? Workflow doc records
  `v0.0.18-10-g946c52b7` as verified; current installed version is
  unrecorded for this incident.
- Does upgrading or downgrading the host Node.js version fix it?
  (`v22.22.1` is current; some Node versions have known undici/epoll
  edge cases.)
- Is the NemoClaw daemon, if any, in a stale state that recycling
  would fix?

## What this means for the assistant-provider live test

Item 2 in the assistant-path checklist (OpenShell `vllm-local` wired to
Omni) is functionally complete: provider, route, and sandbox runtime
config are in place. Items 3 (OpenClaw agent dispatchable) and 4
(ManyForge bridge service) are unblocked via the manual gateway-start
workaround above. The full automated configure script should be revisited
once the policy-add hang is understood.
