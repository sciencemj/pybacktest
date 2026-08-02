# README Future Work Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a concise, prioritized, evidence-backed future-work roadmap in the root README.

**Architecture:** Replace the existing short extension-boundary section with one directional roadmap ordered from core readiness through separate StrategySpec/MCP projects to ecosystem and research work. Keep all extensions outside the core dependency graph, link the existing plans with an explicit boundary-revision warning, and make every committed stage end in an observable completion condition.

**Tech Stack:** Markdown, Git, pytest, Ruff, ty, uv

## Global Constraints

- Modify documentation only; do not change runtime code, packaging metadata, or dependencies.
- Preserve the rule that future packages consume only Pybacktest's public typed API and never add their dependencies to the core.
- Treat the roadmap as directional; do not promise dates or releases.
- Separate verified repository gaps from exploratory or explicitly unscheduled ideas.
- Do not expose volatile internal counts in the public README.
- Do not describe protocol `NotImplementedError` bodies or accepted low-value refactors as unfinished work.
- Do not add brittle tests that police exact prose.
- Every shell command must begin with `rtk`.

---

### Task 1: Publish the prioritized README roadmap

**Files:**
- Modify: `README.md:870`
- Reference: `docs/superpowers/specs/2026-08-02-readme-future-work-roadmap-design.md`
- Reference: `docs/superpowers/plans/2026-07-29-pybacktest-v2-strategy-spec.md`
- Reference: `docs/superpowers/plans/2026-07-29-pybacktest-v2-mcp.md`
- Test: `tests/integration/test_readme.py`

**Interfaces:**
- Consumes: the existing public types named in `Building on top of the core`, including `SimulationRequest`, `BacktestRequest`, `Observation`, `StepResult`, `OrderIntent`, `BacktestResult`, `RunManifest`, persisted artifacts, and `SimulationSession`.
- Produces: a root README `Roadmap` section with four ordered stages, task checkboxes, observable completion conditions, valid relative plan links, and explicit unscheduled boundaries.

- [ ] **Step 1: Confirm the edit boundary and references**

Run:

```bash
rtk sed -n '850,910p' README.md
rtk ls docs/superpowers/plans/2026-07-29-pybacktest-v2-strategy-spec.md
rtk ls docs/superpowers/plans/2026-07-29-pybacktest-v2-mcp.md
```

Expected: `Building on top of the core` appears immediately before
`Performance`, and both referenced plans exist.

- [ ] **Step 2: Replace the extension note with the approved roadmap**

Replace `## Building on top of the core` and its paragraphs with this content:

```markdown
## Roadmap

Pybacktest 0.2 core is complete. This roadmap is directional: it orders the
work that is currently justified, but it does not promise release dates. Every
extension remains a separate package that consumes the public typed API; no
MCP, model, training, UI, or adapter dependency enters the core.

### Next — Core readiness

- [ ] Automate the supported Python matrix, core and optional-dependency tests,
  Ruff, ty, executable README examples, and package-build checks in CI. Keep the
  reference performance gate on a controlled manual or scheduled runner.
- [ ] Remove inherited repository-wide formatting debt in a formatting-only
  change, then enforce `ruff format --check` in CI.
- [ ] Restore the intended dependency direction by moving engine-consumed
  request and provenance contracts to a neutral layer with compatibility
  re-exports.
- [ ] Add reusable contract suites for data, broker, and artifact adapters.

This stage is complete when every quality gate runs automatically and a new
adapter can prove compatibility without copying implementation-specific tests.

### Next project — StrategySpec and MCP

- [ ] Revise the existing [StrategySpec plan](docs/superpowers/plans/2026-07-29-pybacktest-v2-strategy-spec.md)
  and [MCP plan](docs/superpowers/plans/2026-07-29-pybacktest-v2-mcp.md) so they
  produce separate consumer packages instead of modules inside the core.
- [ ] Build StrategySpec first: versioned strict schemas, an allowlisted
  component registry, semantic validation, an `eval`-free compiler, parity with
  Python strategies, and risk-policy intersection.
- [ ] After StrategySpec is stable, build a bounded local-stdio MCP server with
  allowlisted datasets and components, request/run/artifact quotas, sanitized
  failures, and no arbitrary code, URL, or filesystem-path execution.

This stage is complete when both packages are independently installable, depend
only on the public Pybacktest contract, and pass parity, quota, and security
tests.

### Later — Ecosystem and operations

- [ ] Add standalone market-data adapters, including a real yfinance adapter.
- [ ] Add standalone plotting/reporting and artifact-storage adapters; the
  current `yfinance` and `plot` extras install dependencies but do not provide
  these integrations.
- [ ] Automate package publication, release notes, and compatibility checks for
  the core and extension packages.

Each item is complete only with a reusable contract test and an end-to-end
example. Paper/live brokers, remote multi-user MCP hosting, and distributed
execution remain unscheduled rather than implied commitments.

### Research — AI training and reinforcement learning

- [ ] Record reproducible experiment lineage across StrategySpec, validation
  feedback, dataset/configuration fingerprints, metrics, artifacts, and
  parent-child runs; export training trajectories without hidden model
  reasoning.
- [ ] Wrap `SimulationSession` in a separate Gym-style package with explicit
  observation, action, and reward contracts and parity with engine results.
- [ ] Explore walk-forward, parameter-sweep, and Monte Carlo workflows after the
  preceding contracts are stable.

Automatic FX accounting, derivatives, tick/order-book simulation, and other
explicitly excluded domains stay outside the committed roadmap until they have
separate approved designs.
```

Leave `## Performance` and all following content unchanged.

- [ ] **Step 3: Inspect the documentation diff and links**

Run:

```bash
rtk git diff -- README.md
rtk rg -n '^## Roadmap$|^### (Next|Next project|Later|Research)' README.md
rtk ls docs/superpowers/plans/2026-07-29-pybacktest-v2-strategy-spec.md
rtk ls docs/superpowers/plans/2026-07-29-pybacktest-v2-mcp.md
rtk proxy git diff --check HEAD
```

Expected: only the intended README section changes, all four stages occur once,
both relative link targets exist, and the whitespace check exits zero.

- [ ] **Step 4: Run the executable README tests**

Run:

```bash
rtk proxy uv run pytest tests/integration/test_readme.py -q
```

Expected: all README integration tests pass. Do not add an exact-word roadmap
test; the existing suite validates the document's executable contracts.

- [ ] **Step 5: Run the full verification gates**

Run:

```bash
rtk proxy uv run pytest -q
rtk proxy uv run ruff check src tests benchmarks
rtk proxy uv run ty check src/pybacktest
rtk proxy git diff --check HEAD
```

Expected: 861 or more tests pass, Ruff and ty report no errors, and the Git
whitespace check exits zero.

- [ ] **Step 6: Commit the README change**

Run:

```bash
rtk git add README.md
rtk proxy git diff --cached --check
rtk git commit -m "docs: add future work roadmap"
```

Expected: one documentation commit containing only `README.md` is created after
the already committed design and plan documents.

- [ ] **Step 7: Review and push the documentation branch**

Run:

```bash
rtk git status --short --branch --untracked-files=all
rtk git log --oneline origin/main..HEAD
rtk git push -u origin docs/future-work-roadmap
rtk git rev-parse HEAD
rtk git rev-parse origin/docs/future-work-roadmap
```

Expected: the worktree is clean, the branch contains the design, plan, and
README commits, the push succeeds, and local and remote HEAD values match.
