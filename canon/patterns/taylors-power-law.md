---
id: taylors-power-law
name: Taylor's power law
aliases: [fluctuation scaling, Taylor's law, TPL]
kind: empirical_law
status: supported
layer: population
statement_plain: >
  For a species counted repeatedly across sites or times, the variance of
  counts grows as a power of the mean, with an exponent between 1 and 2.
statement_formal: "Var(N) = a · E[N]^b,  1 ≤ b ≤ 2 (typically)"
variables:
  - {symbol: N, meaning: abundance count per sample, unit: individuals}
  - {symbol: b, meaning: scaling exponent, unit: dimensionless, typical: "1.0–2.0", source: Taylor 1961}
  - {symbol: a, meaning: proportionality constant, unit: "same units as N^(2-b)"}
regime:
  taxa: [all]
  scales: [sample-to-site, site-to-region]
  conditions:
    - "requires true counts, not presence-only records"
    - "applies within a species across replicated samples"
known_exceptions:
  - "b > 2 reported for some strongly aggregated or epidemic-like populations"
  - "b can be driven toward 1 or 2 by sampling design alone (Cohen & Xu 2015)"
evidence:
  - {type: paper, doi: "10.1038/189732a0", ref: "Taylor 1961, Nature 189:732", role: origin}
  - {type: paper, doi: "10.1080/00018730801893043", ref: "Eisler, Bartos & Kertész 2008, Adv. Phys. 57:89", role: cross_field_review}
  - {type: paper, doi: "10.1073/pnas.1503824112", ref: "Cohen & Xu 2015, PNAS 112:7749", role: support, note: "shows b can arise from sampling alone; mechanism remains debated"}
  - {type: dataset, source: neon, product: DP1.10003.001, role: check, note: "breeding landbird point counts; run canon/checks/taylors_power_law.py to verify"}
checks:
  - id: neon-landbird-tpl
    runner: canon/checks/taylors_power_law.py
    data: {adapter: neonscience, product: DP1.10003.001}
    expect: {b_between: [1.0, 2.0], min_species: 20}
    last_run: null
    result: null
edges:
  - {to: tweedie-distributions, type: ANALOG_OF, field: statistics, note: "same variance–mean power form; Tweedie family gives a probabilistic basis for TPL"}
  - {to: fluctuation-scaling-physics, type: ANALOG_OF, field: physics, note: "Eisler et al. 2008 show the same b ∈ [1, 2] form across ecology, finance, city sizes, and networks"}
  - {to: density-dependence, type: SUPPORTS, note: "b > 1 implies counts are overdispersed; density-dependence tends to reduce b toward 1"}
open_questions:
  - "Is b a property of the species, the sampling scale, or the statistical process? (unresolved since the 1980s)"
  - "Does b drift systematically with habitat change or climate? (not yet tested against long NEON time series)"
  - "eBird as a second check: presence-only data cannot test TPL, but complete-checklist data can; not yet implemented"
contributors: [christinesu]
care_status: public
---

Taylor's power law is one of ecology's oldest and most robust quantitative patterns. For virtually any species counted at multiple places or times, the variance of those counts grows as a power of the mean: Var(N) = a · mean(N)^b. The exponent b lands between 1 and 2 for the overwhelming majority of species, datasets, and ecosystems ever examined — roughly 1,000 datasets since 1961.

**What it says.** When mean count is low, variance is proportionally low; when mean count is high, variance grows faster than the mean. A b of exactly 1 (Poisson counts) means spatial or temporal variation is purely random. A b above 1 — the typical case — means populations cluster: good patches and bad patches, booms and busts. The higher the b, the more aggregated the distribution.

**Why it is believed.** L. R. Taylor documented it in 1961 across insects, birds, and plants, citing hundreds of species. The law has since been confirmed in fish, mammals, bacteria, fungi, and human disease epidemics. Eisler, Bartos, and Kertész (2008) showed that the identical variance–mean form appears in physics, finance, city growth, and network traffic — a cross-field finding that makes TPL a node in the broader "fluctuation-scaling" literature, not merely an ecological curiosity.

**Where it breaks.** The exponent b can exceed 2 for strongly aggregated or epidemic-like populations. More subtly, Cohen and Xu (2015) showed that sampling design alone can produce b values anywhere in [1, 2] without any biological signal — a warning that presence-only data, rarefied counts, or inconsistent effort can artifactually generate the law. This is why the check here uses NEON's standardized point-count protocol and requires true clusterSize counts, not presence/absence records. eBird complete-checklist data could serve as a second check; it is noted here as a future addition rather than implemented now.

**What it connects to.** The Tweedie family of probability distributions has variance–mean scaling as a defining property; this places TPL in a probabilistic framework where the exponent b determines the distribution's shape. In physics and finance, the same scaling law appears as "fluctuation scaling" (Eisler et al.), making this one of the clearest bridges between ecological population dynamics and statistical physics. TPL also informs density-dependence theory: b > 1 implies overdispersion, which is a signature of spatial aggregation, often driven by habitat heterogeneity or conspecific attraction — both subjects of the density-dependence node.

**The open question.** Since the 1980s, ecologists have debated whether b is a biological property (species aggregation behaviour), a scale property (sampling grain and extent), or a statistical artifact. Cohen and Xu showed it can be purely statistical. Kendal (2004) argued for a biological basis via stochastic population models. No consensus exists. Kinship Earth's check tracks b across NEON's consistent protocol to test whether it is stable year-over-year at the same sites — the first step toward distinguishing biological signal from sampling noise.
