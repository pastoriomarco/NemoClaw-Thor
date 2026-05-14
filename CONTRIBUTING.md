# Contributing to NemoClaw-Thor

This file is the entry point for both **human contributors** and
**LLM-driven agents** working on NemoClaw-Thor. Both audiences are
first-class — the workflow below works for either.

If you're an LLM, read this file end-to-end before making any change.
You should also read [AGENTS.md](AGENTS.md) which is the durable
machine-readable companion to this doc.

## Short version

1. **`dev` is the integration branch; `main` is for released code only.**
2. **Only the maintainer pushes to upstream `dev`.** External
   contributors fork the repo and open a pull request from their fork
   to upstream `dev`.
3. **Releases**: the maintainer squash-merges `dev → main` when a new
   version lands, then tags. (Mirrors the
   [manymove](https://github.com/pastoriomarco/manymove) policy.)
4. CI must stay green on `dev`.

## Repository structure (for orientation)

- `serving/` — vLLM model serving. `config.sh` is the profile
  registry; `launch.sh` and `start-model.sh` bring up the container;
  the `Dockerfile.vllm` defines the SM110 build.
- `setup/` — control-plane wiring. `configure-local-provider.sh`,
  `sandbox-runtime.sh`, OpenShell + OpenClaw integration, iptables
  policies under `policies/`.
- `manyforge/` — the integration glue. `openclaw_assistant_bridge/`
  hosts the FastAPI service that ManyForge's Composer talks to;
  `setup-manyforge-assistant.sh` provisions the OpenClaw sandbox
  with the manyforge-composer skill; `scripts/proxy/vllm-proxy.py`
  is the mutator proxy that sits between OpenClaw and vLLM.
- `scripts/` — CI + debug harnesses.
- `VERSIONS.md` — single source of truth for the per-component pin
  table. Update it when you bump any external dependency; the repo
  SemVer in `VERSION` moves on a separate cadence.

## For humans

### Setup

Prerequisites: a Jetson Thor (SM110a) host with NemoClaw CLI and
OpenShell already installed. See
[`README.md`](README.md) for the host-prep section. Docker BuildKit
is required for the vLLM container.

```bash
git clone https://github.com/pastoriomarco/NemoClaw-Thor.git
cd NemoClaw-Thor
./serving/start-model.sh cosmos-reason2-8b   # ~10 min first time
./setup/configure-local-provider.sh cosmos-reason2-8b
```

See [`USER_QUICKSTART_MANUAL.md`](USER_QUICKSTART_MANUAL.md) for the
full operator manual.

### Local checks (both flows)

Run the relevant check before committing. These run the same way
whether you're the maintainer or a forked contributor:

- Bridge tests: `cd manyforge/openclaw_assistant_bridge && pytest`
- Smoke harness: `python3 manyforge/smoke-openclaw-assistant-reliability.py --quick`
- vLLM container build: `cd serving && docker build -f Dockerfile.vllm .`
- Bash syntax: `bash -n setup/*.sh serving/*.sh manyforge/*.sh`

Commit with a Conventional-Commits-ish prefix: `feat:`, `fix:`,
`docs:`, `refactor:`, `test:`, `chore:`.

### Maintainer flow (direct push to upstream `dev`)

Only the maintainer has push access to upstream `dev`. The flow is:

1. `git checkout dev && git pull` — start from integration head.
2. Make changes; run the local checks above.
3. `git commit` and `git push origin dev`.
4. CI runs automatically on `dev`: Python tests, bash syntax. Keep
   `dev` green.

### External contributor flow (fork + PR)

1. **Fork** the repository on GitHub.
2. Clone your fork; add upstream as a remote:
   ```bash
   git clone https://github.com/<you>/NemoClaw-Thor.git
   cd NemoClaw-Thor
   git remote add upstream https://github.com/pastoriomarco/NemoClaw-Thor.git
   ```
3. Sync your fork's `dev` with upstream before starting:
   ```bash
   git fetch upstream
   git checkout dev && git reset --hard upstream/dev
   ```
4. Either work on `dev` in your fork, or create a topic branch:
   `git checkout -b feature/<short-desc>` (also `fix/<x>` or
   `docs/<x>`).
5. Make changes; run the local checks above; commit.
6. Push to your fork: `git push origin <branch>` (or `dev`).
7. Open a pull request from your fork → upstream `dev`.
8. CI runs on the PR. Address review comments by pushing more
   commits to the same branch on your fork.

### Cutting a release

Releases happen when `dev` has accumulated enough work to warrant a
version bump. Maintainer flow:

1. Bump `VERSION` (SemVer) and add a section to `CHANGELOG.md` on
   `dev`.
2. Confirm CI is green on `dev`.
3. **Squash-merge `dev → main`** — one commit per release, as
   manymove does. The squash message is the release notes.
4. Tag the merge commit: `git tag v0.1.1 && git push --tags`.

If you spot something that should go in the next release, mention it
in your commit / PR description so the maintainer can pull it into
the changelog.

### Model profile changes — special rules

Modifying `serving/config.sh` (adding, removing, or retuning a model
profile) is a high-impact change. Profile changes interact with the
entire downstream lane: bridge calibration, assistant sampling, smoke
corpus scores.

**Rules:**

- Do not modify the default profile (`cosmos-reason2-8b` as of
  2026-05-07) without operator approval. Default-swap requires the
  full smoke-corpus retest documented in `manyforge/docs/MANYFORGE-PROFILE-CALIBRATION.md`.
- New profiles are additive — add a new entry; do not edit existing
  entries unless you've measured the impact.
- Update `README.md` model-profiles table when adding a profile.
- Document the calibration in
  `manyforge/docs/MANYFORGE-PROFILE-CALIBRATION.md`.

## For LLM agents

These rules are operational, non-negotiable, and load-bearing. They
mirror the manyforge rules — both repos use the same workflow for
consistency.

### Branch + commit rules

1. **`dev` is where work lands; `main` is off-limits.** Only the
   maintainer pushes to upstream `dev`. If you are running in the
   maintainer's checkout, an authorized push goes to `origin/dev` (or
   to a topic branch if the operator explicitly asks). If you are
   running in a contributor's fork, an authorized push goes to the
   fork's `dev` or topic branch — never to upstream. Never push to
   `main` from anywhere; only the maintainer does that, via the
   release squash-merge.
2. **`git commit` requires explicit human authorization.** Standing
   rule, regardless of how clearly the change is requested. Stage
   the diff, present it, wait for the literal word `commit`.
   Multi-step instructions like "fix everything" do not transitively
   authorize commits. Reference: priority memory rule
   `feedback_commits_explicit_only.md`.
3. **`git push` is a separate authorization** from `git commit`.
   Wait for an explicit `push` reply.
4. **Never open a pull request automatically.** Do not run
   `gh pr create` unless explicitly instructed.
5. **Releases are operator-driven.** Do not bump `VERSION`, do not
   write `[X.Y.Z]` sections in `CHANGELOG.md`, do not tag, do not
   squash-merge `dev → main`.

### Special rules for this repo

NemoClaw-Thor sits between manyforge (orchestration) and the vLLM
inference stack. LLM agents working here should:

- **Not modify `serving/config.sh` model profiles** without operator
  approval. Even adding a new profile requires the calibration step
  documented in `manyforge/docs/MANYFORGE-PROFILE-CALIBRATION.md`;
  the default-profile swap is a multi-day operation including a
  full smoke-corpus retest.
- **Update `VERSIONS.md` when bumping external dependencies.** The
  table is the source of truth for "what does Thor v0.1.0 actually
  contain." Don't silently bump vLLM in a Dockerfile without
  updating the table.
- **Coordinate cross-repo changes with manyforge.** If your change
  modifies the OpenClaw assistant bridge wire contract, the
  deployment YAML's `assistant` block, or the runtime status surface,
  open a paired PR in `manyforge` so the two repos move together.

### Before any non-trivial change

Read the relevant existing code first. NemoClaw-Thor has a strong
"reproducibility over cleverness" rule: changes that affect
serving / bridge behavior must be paired with a smoke-harness run
showing the change is neutral or improves the relevant metric.

Before opening a PR with a substantial diff, run
[`/codex:review`](https://github.com/openai/codex) and iterate until
it returns clean or the only remaining suggestions are ones you've
deliberately declined.

## Reporting issues

- **Bugs**: open a GitHub issue using the bug-report template.
  Include the affected scope (serving / setup / manyforge), the model
  profile, and the smoke-harness output if relevant.
- **Security**: do NOT open a public issue. Email
  [pastoriomarco@gmail.com](mailto:pastoriomarco@gmail.com); see
  [SECURITY.md](SECURITY.md) for the disclosure process.
- **New model profile requests**: open an issue with the
  `enhancement` label describing the upstream model + expected
  use case.

## Cross-repo coordination

NemoClaw-Thor depends on [manyforge](https://github.com/pastoriomarco/manyforge)
for the assistant-provider contract. If your change affects the bridge
HTTP contract or the deployment YAML, read manyforge's
[CONTRIBUTING.md](https://github.com/pastoriomarco/manyforge/blob/main/CONTRIBUTING.md)
and open a paired PR.

## License

By contributing to NemoClaw-Thor you agree that your contributions
will be licensed under the [MIT License](LICENSE).
