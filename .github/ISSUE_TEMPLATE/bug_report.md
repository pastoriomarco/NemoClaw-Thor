---
name: Bug report
about: Report something that's broken or behaves unexpectedly
title: "[bug] "
labels: bug
assignees: ''
---

<!--
Before filing: search existing issues. Security issues should NOT be
filed here — email pastoriomarco@gmail.com (see SECURITY.md).
-->

## What happened

<!-- What did you do, what did you expect, what did you get? -->

## Reproduction

1.
2.
3.

```text
<paste any relevant command output, error message, stack trace>
```

## Environment

- **NemoClaw-Thor version**: <!-- cat VERSION -->
- **Affected scope**: serving / setup / manyforge
- **Model profile** (if relevant): <!-- e.g. cosmos-reason2-8b -->
- **Affected pin from VERSIONS.md** (if relevant): <!-- vLLM v8 / FlashInfer 0.6.10 / OpenClaw v2026.4.24 / etc. -->
- **Thor hardware**: Jetson AGX Thor SM110a / SM110 / other
- **JetPack version**:
- **NemoClaw CLI version** (host): <!-- nemoclaw --version -->
- **OpenShell version**: <!-- openshell version -->

## Logs

<details>
<summary>Container / bridge / smoke-harness logs</summary>

```text
<paste relevant logs: `docker logs manyforge-e2e-vllm`,
 `docker logs openshell-cluster-nemoclaw`, bridge audit log, etc.>
```

</details>

## Additional context

<!-- Anything else: screenshots, related issues, recent changes to
     iptables policy, HF token rotations, etc. -->
