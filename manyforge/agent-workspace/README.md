# OpenClaw workspace overlay (NemoClaw-Thor)

The workspace `AGENTS.md` injected into the OpenClaw sandbox at every
agent run is **composed** by `setup-manyforge-assistant.sh`:

1. Canonical ManyForge-semantic content (role, vocabulary, tool
   routing, guardrails) is sourced from
   `${MANYFORGE_ROOT}/agent-skills/manyforge-composer/workspace-AGENTS.md`
   (lives in the `manyforge` repo).

2. Optional OpenClaw-mechanical overlay is sourced from
   `openclaw-overlay.md` in this directory — for things that are
   strictly platform-specific to running inside OpenClaw (e.g. the
   `NO_REPLY` empty-message convention, sandbox-side paths under
   `/sandbox/.openclaw/...`, principal-binding mechanics).

If `openclaw-overlay.md` does not exist, the provisioner installs the
canonical file alone — that's the current default since the v7
ManyForge content already covers everything the agent needs.

Add an `openclaw-overlay.md` here only when the OpenClaw runner
introduces a mechanic that the canonical ManyForge content shouldn't
need to know about.
