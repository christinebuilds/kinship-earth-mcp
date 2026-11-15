---
id: example-law
name: Example Law
aliases: [example, test-law]
kind: empirical_law
status: supported
layer: population
statement_plain: >
  An example quantitative pattern used only for testing the canon schema.
  Not a real ecological law.
variables:
  - {symbol: N, meaning: example count per sample, unit: individuals}
regime:
  taxa: [all]
  scales: [sample-to-site]
  conditions: []
known_exceptions: []
evidence:
  - {type: paper, doi: "10.0000/example.1961", ref: "Example 1961, Nature", role: origin}
  - {type: paper, doi: "10.0000/example.2000", ref: "Example 2000, Ecology", role: support}
  - {type: dataset, source: neon, product: DP1.00001.001, role: check, note: "test dataset"}
checks: []
edges:
  - {to: stub-node-a, type: ANALOG_OF, field: testing, note: "test edge to stub", confidence: conjectural}
open_questions:
  - "Is this a real law? (No — it is only a test fixture.)"
contributors: [testuser]
care_status: public
---

This is a minimal example node used only by the canon schema tests. It is not a real
ecological pattern. The id, evidence, and edges are chosen to exercise the validation
rules in `CanonNode` and `load_canon`.
