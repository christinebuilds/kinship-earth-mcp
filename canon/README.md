# The Kinship Canon

A machine-readable, human-editable library of ecology's known patterns.

One file per pattern, in `patterns/<slug>.md` with YAML front matter.
Every quantitative node carries executable checks that run against the
Kinship Earth data federation (11+ open sources).

See [docs/CANON.md](../docs/CANON.md) for the full format, contribution
model, and seed list of the first 20 nodes.

## Structure

```
canon/
  patterns/      one .md file per node; _stubs.yaml for placeholder IDs
  checks/        Python check runners (called from node front matter)
  snapshots/     cached check results and data excerpts (committed)
  schema/        re-exports CanonNode from kinship_shared
```

## Validate

```bash
uv run python tools/canon_validate.py canon/patterns/
```
