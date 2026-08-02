# README Future Work Roadmap Design

## Status

- Approved direction: a prioritized roadmap in the root README
- Target branch base: merged Pybacktest V2 core on `main`
- Scope: documentation only; no runtime, packaging, or dependency changes

## Purpose

The README currently explains how future MCP, AI, and reinforcement-learning
projects consume the core, but it does not say what should be built next or how
to tell when a roadmap stage is complete. Add a concise, evidence-backed roadmap
that gives contributors an execution order without promising release dates.

## Placement and format

Replace the short `Building on top of the core` section with a `Roadmap` section
immediately before `Performance`. Preserve the existing architectural rule that
extensions depend on the core through public typed APIs and never introduce
their dependencies into the core.

Use four ordered subsections with task-list checkboxes:

1. `Next — Core readiness`
2. `Next project — StrategySpec and MCP`
3. `Later — Ecosystem and operations`
4. `Research — AI training and reinforcement learning`

Each subsection must state its completion condition in observable terms. Link
to existing plans where they remain useful, while warning that the older
StrategySpec and MCP plans must be revised from in-core modules to separate
consumer packages before implementation.

## Roadmap content

### Next — Core readiness

- Add continuous integration for the supported Python matrix, core and optional
  dependency tests, Ruff, ty, README examples, packaging, and a separately run
  performance gate.
- Remove the inherited repository-wide formatting debt in a formatting-only
  change, then enforce `ruff format --check` in CI.
- Restore the intended dependency direction by moving request and provenance
  contracts needed by the engine to a neutral layer with compatibility
  re-exports.
- Build reusable contract suites for data, broker, and artifact adapters.
- Define completion as all gates running automatically and new adapters being
  verifiable without copying implementation-specific tests.

### Next project — StrategySpec and MCP

- Implement a separate StrategySpec package first: versioned strict schemas,
  allowlisted component registry, semantic validation, an `eval`-free compiler,
  Python-strategy parity, and risk-policy intersection.
- Build a separate local-stdio MCP package only after StrategySpec is stable.
  It must use allowlisted datasets and components, bounded requests and runs,
  artifact quotas, sanitized failures, and no arbitrary code, URL, or path
  execution.
- Define completion as independently installable packages consuming only the
  public Pybacktest API, with contract, parity, quota, and security tests.

### Later — Ecosystem and operations

- Add optional, standalone market-data, plotting/reporting, and artifact-storage
  adapters. The existing `yfinance` and `plot` extras do not yet provide these
  integrations.
- Automate package publication, release notes, and compatibility checks for the
  core and extension packages.
- Treat paper/live brokers, remote multi-user MCP hosting, and distributed
  execution as unscheduled rather than implied commitments.
- Define completion per adapter or operational capability through its public
  contract suite and an end-to-end example.

### Research — AI training and reinforcement learning

- Record reproducible experiment lineage: StrategySpec, validation feedback,
  dataset and configuration fingerprints, metrics, artifacts, and parent-child
  relationships. Export training trajectories without storing hidden model
  reasoning.
- Wrap `SimulationSession` in a separate Gym-style environment with explicit
  observation, action, and reward contracts and parity with engine episode
  results.
- Explore walk-forward, parameter-sweep, and Monte Carlo workflows only after
  the preceding contracts are stable.
- Keep automatic FX accounting, derivatives, tick/order-book simulation, and
  other explicitly excluded domains outside the committed roadmap until they
  receive separate designs.

## Accuracy and maintenance rules

- Describe the roadmap as directional and do not assign dates or release
  promises.
- Distinguish verified repository gaps from exploratory ideas.
- Do not present protocol `NotImplementedError` bodies as unfinished work.
- Do not promote accepted low-value refactors, such as small private validator
  duplication or defensive feature copies, without new evidence.
- Keep exact volatile counts, such as the current number of unformatted files,
  out of the public roadmap text; track them in issues or implementation plans.

## Validation

- Run the README integration tests so every Python and shell example remains
  parseable and executable under its documented dependency class.
- Run the full test suite because the README is executable documentation.
- Run Ruff, ty, and Git whitespace checks to prove that the documentation-only
  change does not disturb the merged core.
- Review the final README diff for concise wording, working relative links, and
  a clear separation between core commitments and external projects.
