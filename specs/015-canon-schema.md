# Spec 015: Canon Schema and Graph Extension

> Phase 6 — The Canon (see docs/CANON.md)
> Priority: P0
> Estimated effort: 1 session
> Dependency: None

## Objective

Create the `canon/` directory, a Pydantic model and validator for canon node files, and extend the knowledge-graph ontology so canon nodes and their evidential edges can be loaded into `graph_store.py`. No MCP tools yet (spec 017). No real nodes yet beyond one fixture (spec 016 ships the first real one).

## What to Build

### 1. Directory layout

```
canon/
  README.md            ← 20 lines: what a node is, link to docs/CANON.md
  patterns/            ← one Markdown file per node, YAML front matter
  checks/              ← executable check runners (Python)
  snapshots/           ← cached data + check results, committed (small CSV/JSON)
  schema/node.py       ← Pydantic model (re-exported from kinship_shared)
```

### 2. Node model — `shared/src/kinship_shared/canon_schema.py`

Implement the front-matter schema from docs/CANON.md as Pydantic models:

- `CanonNode` with fields: `id` (slug, `^[a-z0-9-]+$`), `name`, `aliases: list[str]`, `kind: Literal["empirical_law","theory","model","conjecture","open_problem","rule_of_thumb"]`, `status: Literal["conjecture","supported","contested","refuted_in_regime","refuted"]`, `layer: Literal["organism","population","community","ecosystem","macro","cross_field"]`, `statement_plain`, `statement_formal: Optional[str]`, `variables: list[Variable]`, `regime: Regime`, `known_exceptions: list[str]`, `evidence: list[Evidence]`, `checks: list[Check]`, `edges: list[Edge]`, `open_questions: list[str]`, `contributors: list[str]`, `care_status` (reuse the Literal from `Provenance`).
- `Variable {symbol, meaning, unit, typical: Optional[str], source: Optional[str]}`
- `Regime {taxa: list[str], scales: list[str], conditions: list[str]}`
- `Evidence {type: Literal["paper","dataset","check_result","expert"], doi: Optional[str], ref: Optional[str], source: Optional[str], product: Optional[str], role: Literal["origin","support","contradiction","check","cross_field_review"], note: Optional[str]}`
- `Check {id, runner: str (path), data: dict, expect: dict, last_run: Optional[datetime], result: Optional[dict]}`
- `Edge {to: str, type: CanonEdgeType, field: Optional[str], note: Optional[str], confidence: Literal["established","conjectural"] = "established"}`
- Validation rules:
  - `status == "supported"` requires ≥1 check with a non-null passing `result`, or ≥2 evidence entries with role `support`/`origin`.
  - `kind in {"conjecture","open_problem"}` requires ≥3 evidence entries (the "what's been tried" list) and empty `checks`.
  - Every `edges[].to` must resolve to an existing node id OR be listed in `canon/patterns/_stubs.yaml` (placeholder ids for nodes not yet written). Fail otherwise.

### 3. Loader — `shared/src/kinship_shared/canon_loader.py`

- `load_node(path) -> CanonNode`: parse YAML front matter (use `python-frontmatter`), validate, attach body text as `narrative`.
- `load_canon(dir) -> dict[str, CanonNode]`: load all, then run cross-node edge validation.
- `canon_to_graph(nodes, store: GraphStore)`: upsert each node as a `GraphEntity(entity_type="pattern")` and each edge as a `GraphRelationship`. Evidence of type `dataset` creates `EVIDENCED_BY` edges to `data_source` entities (already an EntityType).

### 4. Graph ontology — `graph_schema.py`

- Add `"pattern"` to `EntityType`.
- Add to `RelationshipType`: `DERIVES_FROM, GENERALIZES, SPECIAL_CASE_OF, SUPPORTS, CONTRADICTS, REQUIRES, ANALOG_OF, TESTED_BY, EVIDENCED_BY`.
- Define `CanonEdgeType` as the Literal of those nine; import it in `canon_schema.py`.

### 5. Validator CLI — `tools/canon_validate.py`

`uv run python tools/canon_validate.py canon/patterns/` exits non-zero on any validation error, prints one line per node (`OK taylors-power-law (supported, 2 edges, 1 check)`).

### 6. Fixture node

`shared/tests/fixtures/canon/example-law.md` — a minimal valid node used only by tests. Not a real pattern.

## Tests — `shared/tests/test_canon_schema.py`

1. Fixture node loads and validates.
2. `status: supported` with no check and one evidence entry → validation error.
3. `kind: open_problem` with a check → validation error.
4. Dangling edge target not in stubs → validation error; present in stubs → OK.
5. `canon_to_graph` on the fixture creates one `pattern` entity and the right number of relationships in an in-memory `GraphStore`.
6. Existing graph tests still pass after the ontology change.

## Do Not

- Do not write real pattern nodes in this spec.
- Do not touch adapters or the orchestrator.
- Do not modify `EcologicalObservation`.

## Commit

```
Add canon schema, loader, and pattern entity type (spec 015)

- CanonNode Pydantic model + validation rules
- canon_loader: front matter → node → graph
- graph_schema: pattern entity, 9 evidential edge types
- tools/canon_validate.py CLI
- 6 tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
```
