---
id: example-law
name: Example law
aliases: [test law]
kind: empirical_law
status: supported
layer: population
statement_plain: >
  A made-up regularity used only by the test suite. Not a real pattern.
statement_formal: "y = a · x^b"
variables:
  - {symbol: x, meaning: predictor, unit: dimensionless}
  - {symbol: b, meaning: exponent, unit: dimensionless, typical: "1.0–2.0", source: Nobody 1900}
regime:
  taxa: [all]
  scales: [site]
  conditions: ["fixture only"]
known_exceptions: []
evidence:
  - {type: paper, ref: "Nobody 1900", role: origin}
  - {type: paper, ref: "Somebody 1950", role: support}
  - {type: dataset, source: neonscience, product: DP1.00000.001, role: check}
checks:
  - id: example-check
    runner: canon/checks/example.py
    data: {adapter: neonscience, product: DP1.00000.001}
    expect: {b_between: [1.0, 2.0]}
    last_run: null
    result: null
edges:
  - {to: example-stub, type: ANALOG_OF, field: statistics, note: "same form"}
  - {to: example-stub, type: REQUIRES, confidence: conjectural}
open_questions:
  - "None; this is a fixture."
contributors: [test]
care_status: public
---

A fixture node. It exists so the schema, loader, and graph projection have
something valid to parse.
