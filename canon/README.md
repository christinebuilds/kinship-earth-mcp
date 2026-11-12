# The Kinship Canon

One known ecological pattern per file, written once, with its regime of
validity, its evidence, and an executable check against open data.
The design and the reasons for it are in [docs/CANON.md](../docs/CANON.md).

## Layout

- `patterns/` — one Markdown file per node (`<id>.md`), YAML front matter
  plus a plain-language narrative. `_stubs.yaml` lists ids that edges may
  point to before their node exists.
- `checks/` — Python runners that test a node against federation data.
- `snapshots/` — small cached inputs and check results, committed.
- `schema/node.py` — the node model (defined in
  `shared/src/kinship_shared/canon_schema.py`).

## Validate

```bash
uv run --package kinship-shared python tools/canon_validate.py canon/patterns/
```

A node cannot be `supported` without a passing check or two supporting
evidence entries. Conjectures and open problems carry no checks and need
at least three "what's been tried" citations.
