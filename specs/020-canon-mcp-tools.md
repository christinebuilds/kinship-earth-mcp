# Spec 020: Canon MCP Tools — let an agent read, follow and propose patterns

> Phase 6 — The Canon (see docs/CANON.md, "MCP tools")
> Priority: P1
> Estimated effort: 1 session
> Dependency: Spec 017 (schema, loader) — done. Spec 018's node file exists on the runner branch.
> Network: none. Fully buildable and testable offline.

## Objective

Today the canon is files an agent can only read by opening them, and
`ecology_memory_recall` cannot find a pattern by name (found while building
spec 018). Add five small tools so a chat can ask "what patterns does
Kinship Earth track?", "tell me about Taylor's law", "what is it connected
to?", "when was it last checked, and did it pass?", and "draft a new pattern
for me", with every answer tied to a node file.

## What to Build

### 1. Pure functions — `shared/src/kinship_shared/canon_tools.py`

All take a `canon_dir: Path` (default: repo `canon/patterns`, overridable by
env `KINSHIP_CANON_DIR`) and use `canon_loader.load_canon` (cached per
process, reloaded when any file's mtime changes).

```python
def canon_list(status: str | None = None, kind: str | None = None,
               layer: str | None = None) -> list[dict]
    # id, name, kind, status, layer, statement_plain, n_checks, last_check_pass

def canon_get(node_id_or_name: str) -> dict | None
    # Full node (model_dump, json mode) plus "file": relative path.
    # Matches id, name or any alias, case-insensitive; also accepts
    # punctuation-free forms ("taylors law" -> taylors-power-law).

def canon_neighbors(node_id: str, depth: int = 1) -> dict
    # Outgoing and incoming edges up to depth, each with type, field, note,
    # and whether the other end is a real node or a stub.

def canon_check_status(node_id: str) -> list[dict]
    # Per check: id, runner, expect, last_run, result, passed.
    # Reads the node file only. Does NOT run the check (checks can be slow
    # and need network; running is a CI/nightly job, spec 021).

def canon_propose(node_yaml: str, narrative: str = "") -> dict
    # Validates a proposed node with CanonNode. Returns
    # {valid: bool, errors: [str], node_id, file_path, file_text, duplicates: [ids]}.
    # duplicates = existing nodes whose id, name or alias matches the
    # proposal's id, name or aliases (the canonization rule: one pattern, one
    # node). Never writes a file and never opens a pull request.
```

Status vocabulary in answers: report `status` exactly as stored. In
`statement_plain` and any summary, never upgrade `conjecture` to
`supported`.

### 2. MCP tools — `servers/orchestrator/src/kinship_orchestrator/server.py`

Thin `@mcp.tool()` wrappers named `canon_list`, `canon_get`,
`canon_neighbors`, `canon_check_status`, `canon_propose`. Docstrings in
plain language, one example each. Wrap errors the same way other tools do.

### 3. Graph recall by name

`ecology_memory_recall` gains an optional `pattern: str | None = None`
argument: when given, resolve it with `canon_get` and return the pattern
entity (`pattern:<id>`) and its graph neighbours if the graph has it, or the
node summary if not. No change to existing arguments.

### 4. Docs

- `docs/CANON.md` "MCP tools": mark the five as built and note that
  `canon_check` became `canon_check_status` (reads results; does not run).
- README tool table: one row per tool.

## Tests — `shared/tests/test_canon_tools.py` (offline, temp canon dir)

Build a temp canon with two small valid nodes (one `supported` with a
passing check, one `conjecture`), a `_stubs.yaml`, and one edge to a stub.

1. `canon_list()` returns both; `canon_list(status="conjecture")` returns one.
2. `canon_get` matches by id, by name, by alias and by "taylors law" style input; returns `None` for unknown.
3. `canon_neighbors` lists the edge and marks the stub end as a stub; incoming edges appear on the target.
4. `canon_check_status` reports `passed` true / false correctly and never imports or runs the runner.
5. `canon_propose` with a valid new node: `valid` true, no duplicates, `file_text` round-trips through `load_node`.
6. `canon_propose` reusing an existing alias: `duplicates` names the existing node.
7. `canon_propose` with `status: supported` and no evidence or check: `valid` false with the schema's message.
8. `ecology_memory_recall(pattern="Taylor's law")` against a temp graph returns the pattern (in `servers/orchestrator/tests/test_memory_tools.py`).
9. Cache reloads after a node file is edited (touch mtime).

## Do Not

- Do not run check runners from a tool call.
- Do not write node files or open pull requests from `canon_propose`.
- Do not change node statuses from these tools.

## Definition of Done

- Five tools listed by the server, each with a passing test.
- "Tell me about Taylor's law" in a chat resolves to the node via `canon_get` or `ecology_memory_recall(pattern=...)`.

## Commit

```
Canon MCP tools: list, get, neighbors, check status, propose (spec 020)

- canon_tools.py: pure functions over canon/patterns with mtime cache
- five MCP tools; ecology_memory_recall(pattern=...)
- 9 tests
```
