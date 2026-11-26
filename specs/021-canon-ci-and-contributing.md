# Spec 021: Canon CI and Contributor Guide

> Phase 6 — The Canon (docs/CANON.md, "Collaboration model")
> Priority: P1
> Estimated effort: 1 session
> Dependency: Specs 017 (schema), 020 (canon tools, duplicate detection)
> Network: none for parts 1–2. Part 3 edits `.github/workflows/`, which the
> runner's session is not allowed to do; a permitted session applies it.

## Objective

Make the Mathlib-style rules enforceable by a machine and learnable by a
person: one command that a pull request must pass before a pattern lands,
and one page that tells a scientist with a laptop how to add a pattern,
evidence, an edge or a check.

## Part 1 — `tools/canon_ci.py` (offline, deterministic)

```
uv run --package kinship-orchestrator python tools/canon_ci.py canon/patterns/
```

Exits non-zero on any failure, prints one line per finding. Checks:

1. **Schema and cross-node edges** — everything `tools/canon_validate.py` does.
2. **One pattern, one node** — no two nodes share an id, name or alias
   (normalised the same way as `canon_tools`; reuse it).
3. **Stubs are honest** — every stub is referenced by at least one edge, and
   no stub id is also a written node (remove it from `_stubs.yaml` when the
   node lands).
4. **Checks point at real runners** — each `checks[].runner` path exists.
5. **Status matches results** — a node whose only support is a check must
   have a stored `pass: true` result to be `supported`; a stored
   `pass: false` with status `supported` is an error ("flip to contested and
   ask a reviewer").
6. **Presence-only guard** — no `checks[].data.adapter` in
   {gbif, inaturalist} for a node whose `regime.conditions` says counts are
   required.

It does **not** run check runners (slow, networked). That is the nightly
job's work (Part 3).

## Part 2 — `docs/CONTRIBUTING-canon.md`

Plain language, about two screens:

- What a node is; copy-paste skeleton from `canon/patterns/taylors-power-law.md`.
- How to check a draft before opening a PR: the `canon_propose` tool or
  `tools/canon_ci.py`.
- Status rules: new nodes enter as `conjecture`; `supported` needs a passing
  check or two independent evidence entries **and** a named human reviewer.
- Edges: the ten types, when to use `ANALOG_OF`, and stubs.
- Checks: what a runner must do (`--offline` from cached data, write a result
  JSON, update the node), and that presence-only data cannot test count laws.
- CARE: `care_status`, and never committing data whose holders have not
  agreed.
- Review: one domain reviewer + green CI; discussion on the PR.

Link it from `docs/CANON.md` and README.

## Part 3 — Workflows (needs a session allowed to edit `.github/workflows`)

- `test.yml` (pull requests): run the offline test command from CLAUDE.md
  (with `-m "not network"`) **and** `tools/canon_ci.py canon/patterns/`.
- `nightly.yml` (schedule + manual): full suite, then each check runner
  against live data; open an issue labelled `canon-drift` when a stored
  result flips. Needs `NEON_API_TOKEN` as a repository secret (it exists).

This part also completes the CI split Christine approved on 2026-11-14
(Radar `kinship-ci-timeout`), which unsticks PR #17.

## Tests — `shared/tests/test_canon_ci.py` (offline, temp canon dirs)

1. The real `canon/patterns/` passes.
2. Duplicate alias across two nodes fails with both ids named.
3. Unreferenced stub fails; stub that is also a node fails.
4. Missing runner path fails.
5. `supported` with stored `pass: false` fails.
6. Presence-only adapter on a counts-required node fails.

## Commit

```
Canon CI checks and contributor guide (spec 021, parts 1–2)

- tools/canon_ci.py: schema, duplicates, stubs, runners, status vs results,
  presence-only guard
- docs/CONTRIBUTING-canon.md
- 6 tests
```
