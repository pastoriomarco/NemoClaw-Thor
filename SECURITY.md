# Security policy

## Reporting a vulnerability

Please report security vulnerabilities **privately** by email:

📧 [pastoriomarco@gmail.com](mailto:pastoriomarco@gmail.com)

Include:

- A description of the vulnerability and its impact.
- Reproduction steps, ideally a minimal `serving/config.sh` profile
  or `setup/` invocation that triggers the issue.
- The version of NemoClaw-Thor affected (`cat VERSION` and the
  relevant section of `VERSIONS.md` for the affected component).
- The component scope (serving / setup / manyforge).

I'll acknowledge receipt within 7 days and aim to provide a fix or
mitigation timeline within 30 days for confirmed issues. For severe
vulnerabilities affecting deployed sandboxes — anything that would
let unauthorized code escape the OpenClaw sandbox, exfiltrate the
HuggingFace token, or downgrade the egress policy — flag this
explicitly in your email so the response is prioritized.

Please **do not** open public GitHub issues, draft pull requests, or
post in Discussions for security topics.

## Supported versions

Only the latest 0.x line receives security fixes at this stage:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅ Yes             |
| < 0.1   | ❌ Pre-publication |

When 0.2 ships, 0.1.x will continue to receive critical security
fixes for one minor version cycle (≈ until 0.3 ships).

## Threat model

NemoClaw-Thor is a self-hosted inference stack that intentionally
runs on isolated Thor hardware. The following assumptions are baked
into the design:

- **The host is trusted.** Thor is operator-owned hardware on an
  operator-controlled network. Composer + the OpenClaw gateway +
  vLLM all listen on `127.0.0.1` by default and are not exposed to
  the public internet.
- **The vLLM container runs trusted model weights.** ManyForge does
  not load arbitrary user-supplied weights at runtime; profile
  changes go through `serving/config.sh` and are reviewed.
- **The OpenClaw sandbox is an isolation primitive, not a certified
  safety system.** The Landlock + seccomp + netns layers reduce
  blast radius for the AI-assisted authoring lane; they do not
  certify the resulting robot motions are safe.
- **The HuggingFace token is sensitive.** It can pull gated model
  weights. The default install puts it at
  `~/.cache/huggingface/token` with `0600` permissions; do not
  bake it into container images.
- **iptables egress policies are the network-isolation layer.**
  OpenShell / Kubernetes network policies do NOT enforce egress on
  Thor by default — `setup/policies/` ships iptables presets that
  do.

If your deployment violates any of these assumptions (Thor exposed
to the public internet, untrusted operators with shell access,
egress rules removed), that's a configuration concern that's outside
the scope of upstream security fixes.

## What's NOT a security issue

The following are functional concerns, not security ones, and belong
in regular GitHub issues:

- vLLM container startup failures.
- Model accuracy / refusal patterns.
- Bridge throughput regressions.
- Smoke-corpus score drift across releases.

## Hardening checklist for production deployments

If you're deploying NemoClaw-Thor in a production cell:

- ✅ Run Thor on an isolated network — do not expose ports 8000,
  8050, 8080, 8100, or 8200 to the public internet.
- ✅ Use the `openclaw` lane (sandboxed) — not the direct `nemoclaw`
  lane — for any AI authoring.
- ✅ Apply the iptables egress policy from `setup/policies/`
  (the K3s / OpenShell network policies alone are insufficient).
- ✅ Store the HuggingFace token at `~/.cache/huggingface/token`
  with permissions `0600`. Do NOT commit it to git, do NOT bake
  it into a Docker image, and do NOT pass it via `--build-arg`.
- ✅ Pin the vLLM container by the SHA recorded in `VERSIONS.md`
  rather than a floating tag.
- ✅ Audit `serving/config.sh` profiles — every model profile
  declares the `--served-model-name` and expected weight source;
  reject any profile that downloads weights from a non-HF source.
- ✅ Connect the robot's e-stop and safety chain *outside* the
  ManyForge command path that Thor serves. The AI stack is not a
  certified safety component.
