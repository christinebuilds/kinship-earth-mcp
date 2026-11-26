# Contributing to the Kinship Canon

The canon is a library of ecological patterns: regularities that hold in
some places, at some scales, for some decades. Each pattern is one file,
each claim cites its evidence, and wherever possible a check re-tests it on
open data. You need a laptop and a browser, not a lab.

## What a pattern file looks like

One Markdown file per pattern in `canon/patterns/<id>.md`: YAML front matter
(the structured part), then a short plain-language write-up. Copy
`canon/patterns/taylors-power-law.md` as your starting point. The id is
lowercase with hyphens and matches the file name.

The write-up answers four things, in plain words: what it says, why people
believe it, where it breaks, and what it connects to. Add the open question
if there is one.

## Check your draft before opening a pull request

Either ask an agent connected to Kinship Earth to run `canon_propose` on
your YAML, or run:

```bash
uv run --package kinship-orchestrator python tools/canon_ci.py canon/patterns/
```

Both check the schema, look for an existing pattern with the same name or
alias, and make sure every edge points somewhere real. The same command runs
on every pull request.

## Status: start humble

| Status | Means |
|---|---|
| `conjecture` | Stated, not yet confirmed here. **Every new pattern starts here.** |
| `supported` | A passing check, or at least two independent evidence entries, **and** a named human reviewer has agreed. |
| `contested` | Good evidence on both sides, or a check that used to pass now fails. |
| `refuted_in_regime` | Fails in a stated setting (taxon, scale, place) but may hold elsewhere. |
| `refuted` | Fails where it was claimed to hold. |

A pattern does not become `supported` because it is famous. If its check
starts failing, CI asks for it to move to `contested`, and a reviewer decides.

## One pattern, one node

If your pattern already exists under another name, do not add a second
file. Add your name as an alias, add evidence, or add an edge. Duplicate
names or aliases fail CI.

## Edges

| Type | Use when |
|---|---|
| `DERIVES_FROM` | This pattern follows from that one. |
| `GENERALIZES` / `SPECIAL_CASE_OF` | One is a broader form of the other. |
| `SUPPORTS` / `CONTRADICTS` | Evidence for one bears on the other. |
| `REQUIRES` | This only holds if that does. |
| `ANALOG_OF` | The same form in another field (statistics, physics, finance). Set `field:`. This is the edge no other database draws. |
| `INFORMS` | Knowing this helps interpret that, without a formal link. |
| `TESTED_BY` / `EVIDENCED_BY` | Drawn automatically from checks and dataset evidence. |

If an edge points at a pattern nobody has written yet, add its id to
`canon/patterns/_stubs.yaml`. Remove it from the stubs when the file lands.

## Checks

A check is a script under `canon/checks/` that re-tests the pattern on open
data. It must:

- run from cached data with `--offline`, so CI and other people can repeat it;
- write a result JSON under `canon/snapshots/results/` with `pass: true` or `false`;
- update the pattern file's `checks[].last_run` and `result`;
- come with tests that run on synthetic data.

**Count-based patterns need true counts.** Presence-only sightings (GBIF,
iNaturalist) record that something was seen, not how many or where nobody
looked, so they cannot test laws about abundance. CI rejects that pairing.

## Care and consent

Set `care_status` (`public`, `research`, `restricted` or `sovereign`). Do
not commit data whose holders have not agreed to share it, and never commit
personal details or exact private locations. Indigenous and community data
follow the CARE principles: the people who hold it decide how it is used.

## Review

A pull request lands with one domain reviewer's approval and green CI.
Discuss on the pull request; why a pattern is believed matters as much as
the pattern itself.
