---
id: taylors-power-law
name: Taylor's power law
aliases: [fluctuation scaling, Taylor's law, TPL]
kind: empirical_law
status: conjecture
layer: population
statement_plain: >
  For a species counted repeatedly across sites or times, the variance of
  counts grows as a power of the mean, with an exponent between 1 and 2.
statement_formal: "Var(N) = a · E[N]^b,  1 ≤ b ≤ 2 (typically)"
variables:
  - {symbol: N, meaning: abundance count per sample, unit: individuals}
  - {symbol: b, meaning: scaling exponent, unit: dimensionless, typical: "1.0–2.0", source: Taylor 1961}
regime:
  taxa: [all]
  scales: [sample-to-site, site-to-region]
  conditions: ["requires true counts, not presence-only records"]
known_exceptions:
  - "b > 2 reported for some strongly aggregated or epidemic-like populations"
evidence:
  - {type: paper, doi: 10.1038/189732a0, ref: "Taylor 1961, Nature 189:732", role: origin}
  - {type: paper, doi: 10.1080/00018730801893043, ref: "Eisler, Bartos & Kertész 2008, Adv. Phys. 57:89", role: cross_field_review}
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
---

**Status note.** Marked `conjecture` in this canon until a human reviewer
signs off and the NEON check has run on live data. The literature treats the
law as well supported; the status here records only what this canon has
checked itself.

## What it says

Count the same species again and again, at many points or on many days, and
work out the average count and how much the counts spread. Across species,
the spread (the variance) rises with the average in a straight line on a
log–log plot. The slope, *b*, usually sits between 1 and 2. A slope of 1 is
what pure chance gives (a Poisson process); a slope near 2 means individuals
come in clumps, so a good spot is very good and a bad spot is empty.

## Why people believe it

L. R. Taylor described it in 1961 from insect and other counts, and it has
since been found in hundreds of species, from aphids to fish to birds. Eisler,
Bartos and Kertész (2008) showed the same form, under the name fluctuation
scaling, in physics, finance and computer networks.

## Where it breaks

Some strongly clumped or outbreak-like populations give *b* above 2. The law
needs true counts: presence-only records such as GBIF or iNaturalist
sightings cannot test it, because they do not record how many were there or
where nobody looked. Do not run this check on presence-only data.

## What it connects to

Tweedie distributions in statistics have exactly this variance–mean form.
Fluctuation scaling in physics and finance is the same pattern by another
name. It also informs density dependence: how crowding changes the spread of
a population.

## Open question

Is *b* a fact about the animal, or about how we sample it? Kendal (2004)
derived the law from a family of statistical distributions, and Cohen and Xu
(2015, PNAS 112:7749, doi:10.1073/pnas.1503824112) showed it can arise from
sampling alone. Both readings remain open.

TODO: a second check on eBird counts (spec 018 defers it).
