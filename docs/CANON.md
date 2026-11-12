# The Kinship Canon — Foundation

> Draft 0.1 · 2026-11-06 · Christine Su with Claude
> Status: proposal. Read alongside STRATEGY.md and ROADMAP.md; this document changes both.
>
> **Updated 2026-11-12 — read docs/DIRECTION.md first.** Three things changed after this was written: (1) entries are *patterns*, not laws — each records where it holds and how that is drifting; (2) the canon is the tested evidence behind the existing baseline → anomaly → event pipeline, not a separate library; (3) watershed work (specs 015–016) now comes before the canon schema and the Taylor check, which are specs 017–018. Spec numbers in the table below are renumbered in docs/DIRECTION.md.

## One sentence

Ecology's known patterns have never been written down once, in one place, in a form that both humans and machines can reason over and test — so Kinship Earth builds that library, and uses the data federation it already has as the verifier.

## Why now

In October 2026 AI systems produced proofs of the quasi-Riemann hypothesis, a Navier–Stokes blow-up, and several hundred other results. The part that matters for ecology is *why* that worked. Those systems could formalize whole textbooks only when the material sat one or two steps away from Mathlib, the curated Lean library where every lemma exists once, in its most general form, with its assumptions built in. Three steps out, definitions clashed and nothing stitched. Mathlib is ten years of unglamorous human work that Alex Kontorovich calls "canonization," and he says AI is still bad at it.

Ecology has no Mathlib. Its laws are real and surprisingly few, but they live in incompatible notation across thousands of papers, with their regimes of validity buried in discussion sections. Nobody can see which results are adjacent, which contradict, or which are the same law wearing two names. That opacity is the thing Steve Brunton described as the fog over the convex hull. Lifting it is a prerequisite for everything else: cross-field discovery, cheap models for people without sensors, and any honest "listen to an ecosystem" tool.

Two further facts from that episode shape the design:

1. **Answers are now cheap; the right variables are not.** Brunton's open problem for turbulence is a coordinate system — the handful of numbers that describe the state the way temperature, pressure and density describe a gas. Ecology has the same gap, and a canon of known relationships is how you find out which variables already carry most of the signal.
2. **Formal verification is what made the AI results trustworthy.** Nobody trusted the natural-language proofs; they trusted the Lean checks. Ecology cannot be formalized in Lean, but every quantitative pattern *can* be re-tested against open observational data. Kinship Earth already has eleven data sources behind one schema. That federation is ecology's verifier.

## What changes from STRATEGY.md

STRATEGY.md (April 2026) names the moat as a proprietary conversation corpus and a memory graph that compounds from user queries. That framing is retired. The reasons:

- A moat built on hoarding researchers' questions is the extraction model Kinship Earth exists to oppose (see CARE principles already in `schema.py`).
- Ankur Moitra's worry in the episode was that mathematicians will become *more* secretive now that a single published idea can be exhausted by AI. The field that wins is the one that stays open and drops the first-through-the-gate race. Kinship Earth should be infrastructure for that choice.
- The conversation corpus is not useless. It becomes *evidence* — "what questions get asked about which patterns" — feeding the canon rather than being the product.

What stays, and matters more than before:

| Existing asset | New role |
|---|---|
| 11 source adapters + `EcologicalObservation` schema | The verifier. Every quantitative node in the canon carries executable checks that run against this federation. |
| `graph_schema.py` ontology + `graph_store.py` | Extended with a `pattern` entity type and evidential edges (below). The graph becomes the canon's index. |
| Provenance + CARE fields | Mandatory on every node and every piece of evidence. |
| Spec queue + nightly autonomous sessions | The build engine. Canon work ships as specs 015+. |
| `ecology_memory_store` / `ecology_memory_recall` | Repurposed: store and recall *findings about patterns*, not user histories. |

What pauses: new data-source adapters, continuous-monitoring expansion, and the per-user memory product. Not deleted — paused. The canon decides which sources are needed next (a scaling law needs a body-mass trait source before it needs camera traps).

## The object: one node

A node is one known pattern. Not a dataset, not a paper. One file per node, Markdown with YAML front matter, so a scientist can edit it in a browser and an agent can parse it. Directory: `canon/patterns/<slug>.md`.

```yaml
id: taylors-power-law
name: Taylor's power law
aliases: [fluctuation scaling, Taylor's law, TPL]
kind: empirical_law            # empirical_law | theory | model | conjecture | open_problem | rule_of_thumb
status: supported               # conjecture | supported | contested | refuted_in_regime | refuted
layer: population               # organism | population | community | ecosystem | macro | cross_field
statement_plain: >
  For a species counted repeatedly across sites or times, the variance of
  counts grows as a power of the mean, with an exponent between 1 and 2.
statement_formal: "Var(N) = a · E[N]^b,  1 ≤ b ≤ 2 (typically)"
variables:
  - {symbol: N, meaning: abundance count per sample, unit: individuals}
  - {symbol: b, meaning: scaling exponent, unit: dimensionless, typical: "1.0–2.0", source: Taylor 1961}
regime:
  taxa: [all]                   # where it is claimed to hold
  scales: [sample-to-site, site-to-region]
  conditions: ["requires true counts, not presence-only records"]
known_exceptions:
  - "b > 2 reported for some strongly aggregated or epidemic-like populations"
evidence:
  - {type: paper, doi: 10.1038/189732a0, ref: "Taylor 1961, Nature", role: origin}
  - {type: paper, doi: 10.1080/00018730801893043, ref: "Eisler, Bartos & Kertész 2008, Adv. Phys.", role: cross_field_review}
  - {type: dataset, source: neon, product: DP1.10003.001, role: check, note: "breeding landbird point counts"}
checks:
  - id: neon-landbird-tpl
    runner: canon/checks/taylors_power_law.py
    data: {adapter: neonscience, product: DP1.10003.001}
    expect: {b_between: [1.0, 2.0], min_species: 20}
    last_run: null
    result: null
edges:
  - {to: tweedie-distributions, type: ANALOG_OF, field: statistics, note: "same variance–mean form"}
  - {to: fluctuation-scaling-physics, type: ANALOG_OF, field: physics, note: "Eisler et al. 2008 review spans ecology, finance, networks"}
  - {to: density-dependence, type: INFORMS}
open_questions:
  - "Is b a property of the species, the sampling scale, or the statistical process? (unresolved since the 1980s)"
contributors: [christinesu]
care_status: public
```

Body text below the front matter: a short narrative in plain language (what it says, why people believe it, where it breaks, what it connects to). Think encyclopedia entry written by someone who has actually fit the curve.

### Node kinds

- **empirical_law** — a quantitative regularity (Kleiber, Taylor, species–area).
- **theory** — a mechanism that explains laws (metabolic theory, neutral theory, island biogeography).
- **model** — a named mathematical form (Lotka–Volterra, Holling type II).
- **conjecture / open_problem** — stated, not settled (May's stability–complexity paradox, paradox of the plankton, 3/4 vs 2/3). These are first-class nodes. Brunton's advice was to find questions smart people asked in the 1970s and hit a computational wall; the canon lists them with what has been tried.
- **rule_of_thumb** — field heuristics with weaker evidence (Lindeman's 10%, Bergmann's rule). Included so they can be tested rather than repeated.

### Edge types (added to `graph_schema.py`)

`DERIVES_FROM`, `GENERALIZES`, `SPECIAL_CASE_OF`, `SUPPORTS`, `CONTRADICTS`, `REQUIRES`, `ANALOG_OF` (cross-field), `TESTED_BY` (node → check), `EVIDENCED_BY` (node → dataset/paper).

`ANALOG_OF` is the product. It is the edge no existing database draws, and the one that makes the convex hull transparent.

## Why checks are the differentiator

GloBI indexes species interactions. TRY and the Open Traits Network hold traits. COMPADRE holds demographic matrices. BioTIME holds time series. Wikidata holds identifiers. None of them hold *claims about patterns*, and none re-test a claim when new data arrives.

A canon node with an executable check is a living claim. CI runs the check against a cached snapshot on every PR; a nightly session re-runs it against live adapters. If Taylor's exponent drifts outside [1, 2] on NEON data, the node's status flips to `contested` automatically and a human reviewer is pinged. That is as close to Lean as observational science gets, and it is only possible because the federation already exists.

For the scientist with no sensors and no compute budget: every node tells them which free dataset tests it, how, and what the current answer is. They can fork the check, point it at their own region, and submit the result as evidence. That is the contribution path for someone with a laptop and a field notebook.

## Seed canon (first 20 nodes)

Ordered by how little data each needs and how many cross-field edges it carries. Nodes marked ✓ can be checked today with existing adapters; ◐ need one trait or body-mass source; ○ are open problems (no check, but a curated "what's been tried" list).

**Scaling (organism → population)**
1. ✓ Taylor's power law — NEON landbird counts, eBird as second check. Edges: Tweedie, fluctuation scaling in physics/finance.
2. ◐ Kleiber's law (metabolism ∝ M^3/4) — needs AnAge/PanTHERIA or TRY. Edges: West–Brown–Enquist fractal networks → Murray's law (fluid transport). *This is the direct bridge to Brunton's field.*
3. ◐ Damuth's law (density ∝ M^−3/4) — GBIF/iNat densities + body mass. Edges: energetic equivalence.
4. ◐ Hatton et al. 2019 linkage — metabolism and abundance scale reciprocally (±3/4 within groups, ±1 across); growth and mortality at ±1/4. Edges: contradicts a pure metabolic basis for growth scaling. Status: supported, contested mechanism.
5. ✓ Species–area relationship (S = cA^z, z ≈ 0.25) — GBIF/iNat occurrence counts in nested areas, with a sampling-effort caveat baked into the check. Edges: fractal geometry, island biogeography.
6. ◐ Metabolic temperature dependence (Boltzmann–Arrhenius) — ERA5 temperature × trait rates. Edges: chemistry (Arrhenius), climate response.

**Community**
7. ○ May's stability–complexity criterion (σ√(SC) < 1) — open problem. Edges: random matrix theory / Wigner semicircle; Allesina & Tang 2012. *A direct edge into the mathematics in the episode.*
8. Holling functional responses (types I–III) — model node. Edges: Michaelis–Menten enzyme kinetics.
9. Lotka–Volterra — model node. Edges: chemical oscillators, Kuramoto-style coupling (weak edge, mark as `conjectural`).
10. MacArthur–Wilson island biogeography — theory node; check against OBIS/GBIF for real islands.
11. Hubbell's neutral theory — theory node, `contested`. Edges: Kimura neutral theory in population genetics, random walks.
12. ○ Hutchinson's paradox of the plankton (1961) — open problem.
13. Competitive exclusion (Gause) — status `supported` with many exceptions; list them.

**Macroecology**
14. ✓ Latitudinal diversity gradient — GBIF/OBIS species counts by latitude band. Status: supported; mechanism contested.
15. ✓ Rapoport's rule (range size increases with latitude) — `contested`; a good first "test a shaky rule" node.
16. ◐ Bergmann's rule — `rule_of_thumb`, needs body mass.

**Ecosystem**
17. Harte et al. 2022 "equation of state" (diversity, productivity, abundance, biomass) — theory node. *This is the ecological analogue of PV = nRT and the closest thing to Brunton's coordinate system.* Edges: thermodynamics, maximum entropy.
18. Metabolic theory of ecology (Brown et al. 2004) — theory node tying 2, 3, 6 together.
19. Lindeman's 10% trophic efficiency — `rule_of_thumb`; check against FLUXNET productivity where possible.
20. ○ Holling 1973 resilience — open problem node: "what is the minimal observable set that predicts regime shift?" Edges: control theory, the insect-wing problem (partial information, robust control).

Twenty nodes is a weekend of writing for someone who knows the literature and a month of checks. It is also enough to show the shape of the thing to collaborators.

## Collaboration model: Mathlib, not Wikipedia

Mathlib works because the unit of contribution is a pull request, review is public, duplication is refused, and the library has maintainers whose job is canonization rather than authorship. Copy that.

- **Unit of contribution: a PR.** Three kinds — new node, new evidence or check on an existing node, new edge. Small PRs merge fast.
- **Canonization rule.** One pattern, one node. A PR adding a node that duplicates an existing one is redirected to add aliases or edges instead. Reviewers hold this line; it is the whole point.
- **Review = one domain reviewer + green checks.** A node cannot flip to `supported` without a passing check or two independent evidence entries. Open problems need a "what's been tried" list with at least three citations.
- **Public review.** Discussion lives on the PR. The narrative of *why* a node is believed is as important as the node (Alex's point: math is the why, not the fact).
- **Credit without the race.** Every node lists contributors; releases get a Zenodo DOI (as GloBI does) and CITATION.cff cites the release, not a person. Nobody is first through the gate; everybody is in the author list of the release.
- **Licenses.** Content CC-BY 4.0; code MIT (already). Evidence inherits the license of its source and the node says so.
- **CARE.** Nodes derived from Indigenous or local ecological knowledge carry `care_status` and a consent record, exactly as observations do today. Shared memory stays opt-in, as already decided.

### Who to connect to, and how

Connect by linking identifiers and using their data as evidence, never by merging or re-hosting.

- **GloBI** (Jorrit Poelen) — interaction data as evidence for community-layer nodes; their GitHub-plus-Zenodo contribution model is the one to imitate.
- **Open Traits Network / TRY / AnAge** — body-mass and rate traits for the ◐ nodes.
- **COMPADRE/COMADRE** — demographic matrices for growth/mortality scaling.
- **BioTIME** — long time series for Taylor's law and stability nodes.
- **TDWG** — Darwin Core alignment already exists; propose a "pattern" extension there once 20 nodes exist, not before.
- **Santa Fe Institute orbit** (Hatton, Harte, West) — the people who wrote the scaling nodes; invite them to review their own nodes.
- **Ecological Forecasting Initiative** — the community most likely to use checks as benchmarks.

## Agent interface (new MCP tools)

- `canon_list(layer?, status?, kind?)` — browse nodes.
- `canon_get(id)` — full node with evidence, checks, edges.
- `canon_neighbors(id, edge_type?)` — walk the graph; `ANALOG_OF` across fields is the headline query.
- `canon_check(id, region?, adapter?)` — run a node's check against live data, optionally scoped to a place; returns fitted parameters and whether the regime held.
- `canon_propose(node_yaml)` — draft a PR from an agent session, for human review.

An agent asked "what do we know about how abundance scales with body size, and where does it break?" should be able to answer from the canon with citations and a fresh check, not from training-data vibes.

## First node, end to end

Taylor's power law on NEON breeding-landbird point counts (DP1.10003.001).

Why this one: no API key, research-grade standardized sampling, needs only counts, the law is well-attested so a failure means a bug not a discovery, and it carries a clean cross-field edge. It is small enough to finish in one evening and big enough to force every piece of the pipeline to exist: node file, check runner, CI, graph edge, MCP tool.

Steps (this is spec 018):
1. Pull a season of point counts for ~10 NEON sites via the existing adapter.
2. For each species with ≥ 20 samples, compute mean and variance of counts across points.
3. Fit log Var = log a + b · log mean (OLS, with bootstrap CI on b).
4. Assert 1 ≤ b ≤ 2 for the pooled fit; record per-species b.
5. Write results back into the node's `checks[].result`, commit the snapshot, render one figure.

If b lands around 1.5 the pipeline works. Then write node 2.

## Specs for the queue

| # | Spec | What ships |
|---|---|---|
| 015 | `canon-schema` | `canon/` directory, node YAML schema + validator, `pattern` entity type and new edge types in `graph_schema.py`, loader into `graph_store.py` |
| 016 | `first-node-taylor` | Taylor's law node, check runner, result snapshot, figure |
| 017 | `canon-mcp-tools` | the five tools above, registered in the orchestrator |
| 018 | `canon-ci-and-contributing` | CI job that validates node files and runs checks on cached snapshots; CONTRIBUTING.md section for node/evidence/edge PRs; CODE_OF_CONDUCT |
| 019 | `seed-nodes-batch-1` | nodes 5, 14, 15 (the other ✓ nodes) with checks |
| 020 | `trait-source-adapter` | AnAge or TRY adapter — the first adapter the canon *asked for* |

## Open decisions for Christine

1. **Name.** "Kinship Canon" inside the Kinship Earth repo, or a separate repo (`kinship-canon`) that depends on `kinship-earth-mcp`? Separate repo is cleaner for contributors who only write nodes; monorepo is simpler to build. Recommendation: `canon/` inside the monorepo for the first 20 nodes, split when a non-engineer contributor shows up.
2. **Status of STRATEGY.md.** Rewrite it, or add this document and mark the moat section superseded? Recommendation: rewrite the "Competitive Moat" section to read "the moat is being the canon; everything else is the verifier."
3. **Web app.** It was kept as a showroom. The showroom now shows the canon: a node page with its figure and live check status is a better landing page than a chat box.
4. **Review bench.** Who are the first two domain reviewers? A node cannot become `supported` without one. This is the first ask of the network, and it is a specific, small ask.

## Sources behind this document

- Lex Fridman Podcast #503 (Brunton, Moitra, Kontorovich), transcript: https://lexfridman.com/math-ai-crisis-transcript/
- Hatton et al. 2019, *Linking scaling laws across eukaryotes*, PNAS 116(43). doi:10.1073/pnas.1900492116
- Harte et al. 2022, *An equation of state unifies diversity, productivity, abundance and biomass*, Comms Biology. https://www.nature.com/articles/s42003-022-03817-8
- Dodds 2009, *Laws, Theories, and Patterns in Ecology* (a human-readable predecessor to this list)
- Marquet & Quiñones 2005, *Scaling and power-laws in ecological systems*
- Eisler, Bartos & Kertész 2008, *Fluctuation scaling in complex systems*, Adv. Phys. 57(1)
- GloBI contribution model: https://globalbioticinteractions.org/contribute
