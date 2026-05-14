<!--
Thank you for opening a pull request to NemoClaw-Thor!

This PR should target `dev` (never `main`). Releases happen via a
maintainer-driven squash-merge from `dev → main`.

If you're an LLM agent: confirm you have explicit human authorization
for the commit(s) on this branch. See AGENTS.md and CONTRIBUTING.md.
-->

## Summary

<!-- One or two sentences. What does this PR do and why? -->

## Scope

- [ ] `serving/` (vLLM container, model profiles, launch scripts)
- [ ] `setup/` (NemoClaw / OpenShell wiring, sandbox runtime)
- [ ] `manyforge/` (OpenClaw bridge, manyforge-composer skill, proxy)
- [ ] `scripts/` (CI, debug harnesses)
- [ ] Top-level docs (README, CHANGELOG, ROADMAP, etc.)

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change (wire-contract or default-profile change)
- [ ] Documentation only
- [ ] Refactor / internal cleanup
- [ ] CI / tooling

## Checklist

- [ ] PR targets `dev` (not `main`); topic branch name `feature/<x>`, `fix/<x>`, or `docs/<x>` if using one
- [ ] Tests added or updated (pytest for bridge; smoke harness if behavior-affecting)
- [ ] `CHANGELOG.md` updated under `[Unreleased]` if user-visible
- [ ] `VERSIONS.md` updated if external dependency pins changed
- [ ] Bash files I touched pass `bash -n`
- [ ] AGENTS.md / CONTRIBUTING.md / README updated if workflow changed
- [ ] For non-trivial diffs: `/codex:review` run and findings addressed
- [ ] **`serving/config.sh` default-profile change**: smoke-corpus retest run and result posted
- [ ] **manyforge boundary**: paired PR opened in `manyforge` if the wire contract or deployment YAML schema is affected

## Test plan

<!-- How you verified this works. Include smoke-harness output for
     bridge/serving changes. -->

## Related issues / PRs

<!-- "Fixes #N", "Refs #N", or a link to a manyforge PR if this change
     crosses the repo boundary. -->
