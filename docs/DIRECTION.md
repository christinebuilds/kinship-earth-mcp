# Kinship Earth — Direction (November 2026)

> Written 2026-11-12 by Christine Su with Claude, as the handoff from a strategy conversation (Nov 6–12) into Claude Code.
> **If you are a Claude Code session picking this up: read this file first.** It supersedes the moat and pricing sections of STRATEGY.md and reorders docs/CANON.md. Where this file and an older doc disagree, this file wins.

## One sentence

Kinship Earth answers "is this normal for here, and what usually follows?" for any place, from open data, with a published record of how often its answers have been right.

## What was decided

| Decision | Detail |
|---|---|
| **Non-commercial** | Kinship Earth stays non-commercial. Drop the free/pro tier and pricing work. BYOK stays only as a way for users to cover their own API costs. |
| **Open, not a private moat** | The "proprietary conversation corpus" moat in STRATEGY.md is retired. The federation is worth more as a public verifier than as a private memory. Per-user memory stays opt-in. |
| **Patterns, not rules** | Christine rejects "law/rule" language. Ecology is emergent patterns that hold in some places, at some scales, for some decades. Every pattern records where it holds, how strongly, and how that has changed. Drift is the signal. |
| **Data must be easy to get at scale** | Build on satellite, open sightings (eBird, iNaturalist) and public science databases. No dependence on manual field collection. (Christine spent 8 years building PastureMap and does not want to repeat manual, uncalibrated data capture.) |
| **Rangeland is not required to be first** | Any ecosystem with open data and a hard, connected problem qualifies. |
| **Mathlib-style collaboration** | Contribution by pull request, public review, one pattern in one place, credit by release (see docs/CANON.md, "Collaboration model"). Still stands. |

## What the repo already has (verified by reading the code, 2026-11-11)

The "is this normal" pipeline already exists from Phase 5:

`baselines.py` → `anomaly_detect.py` → `event_classify.py` → `analog_match.py`, with `state_builder.py` and `monitoring.py` around it.

The design centre is watersheds: the `EcosystemState` example ID is `watershed:russian-river`, streamflow is its first signal, and the roadmap's examples are coho spawning and the Klamath drought.

So the canon is not a separate library. It is the tested evidence behind this pipeline.

## Where the pipeline is weak today

These are the first things to fix. Each was confirmed in the source.

1. **Streamflow "normal" is the last 30 days.** `compute_baselines_from_usgs` pools every gauge within 50 km over the 30 days before the target date. A river is compared to its own last month and to neighbours of different size, not to its history.
2. **Climate "normal" is the last 5 years.** `compute_baselines_from_era5` defaults to `years_back=5`. In a warming trend that hides the drift we want to detect, and 5 years × 15 days is a noisy sample.
3. **Flow anomalies use percent of mean.** `FlowDetector` fires at ±50% of the baseline mean. Streamflow is strongly skewed, so percent-of-mean over-flags wet spikes and under-flags low flow. Percentile of the day-of-year record is the standard measure.
4. **Past events are six hand-written entries** in `KNOWN_ANALOGS`, matched by type, distance and season.
5. **Event rules are hand-written and never scored.** `DROUGHT_CASCADE` and the other three patterns in `event_classify.py` have never been run against history to see how often they fire correctly or falsely.

## First use case: rivers and watersheds

**Why**

- Best open data in ecology after satellites: thousands of US stream gauges, live, public domain.
- A river physically integrates what happens upstream (snow, rain, fire, land use, fish), so connecting datasets pays off.
- Every user Christine named sits in a watershed (below).
- A community benchmark exists in California: the EFI-USGS river chlorophyll challenge in the Sacramento–San Joaquin Delta, about 3,600 forecasts scored (https://projects.ecoforecast.org/usgsrc4cast-ci/).

**First proof (specs 015 and 016)**

1. Honest baselines: each gauge compared to its own long record for that time of year; climate normals from 30 years.
2. Replay history: run the detector over past decades and publish a scorecard against documented events. Did it flag the Klamath in 2021? How often did it raise a false alarm?

**Open: which watershed first.** Default is the Russian River because the code already names it. Christine has not confirmed. Ask her before hard-coding a site list beyond what spec 016 needs.

## Who it is for

Christine named these users. None are scientists with large budgets.

- Cities planning for El Niño winters (flooding, storm response)
- Rural fire departments in Northern California planning staffing and response teams
- Grassfed ranchers planning the grazing season and stocking rate
- Point Blue Conservation Science, mapping birds and showing gradual change across an ecosystem
- Scientists without sensors or compute budgets, anywhere

They are connected by one chain: ocean state → winter rain → streamflow and grass growth → forage (ranchers), fine fuel (fire districts), flood and low-flow risk (cities), bird response (Point Blue).

**Later use cases, in rough order**

1. Seasonal timing (first bloom, first arrival, green-up) — global, and where open sightings data is strongest.
2. California rangeland forage — "your grass is at 70–95% of normal for this date." Grass-Cast lists only the Great Plains and Southwest. Point Blue's Rangeland Monitoring Network (145 ranches, 28 counties, since 2014) is the natural calibration partner. Their data is not confirmed open; ask, with ownership terms.
3. Coast and ocean — OBIS is rich, but sea temperature data is not yet federated.

**Not doing:** forecasting El Niño itself, AMOC, or operational wildfire prediction. These are well-funded fields where Kinship Earth adds little. What it can add: an honest record of how reliable a link has been locally (for example, how often El Niño has meant a wet winter in a given city).

## Design principles

1. **One product question:** "Is this normal for here?" Every answer shows expected, observed, and how sure we are.
2. **"What usually follows."** Cross-system links are chains with time lags. Each link is tested on years the model never saw, because such chains are easy to imagine and often false.
3. **Three data layers.** Satellite as the consistent base. A small rigorous set (NEON, USGS, a partner such as Point Blue) to calibrate. Many rough local observations to make the answer right for one place. Weight each by proven reliability; model the observer instead of discarding messy data.
4. **Guard against false discovery.** Write the prediction down first. Find a pattern in one set of sites, confirm it in sites never seen. New candidates enter as `conjecture` and stay there until a second dataset and a human reviewer agree.
5. **New patterns come from residuals.** Look where a known pattern fails consistently. Do not mine blind correlations.
6. **Publish the scorecard.** Hit rate and false-alarm rate for every detector and every "what follows" link.
7. **Local observers own what they record.** CARE status and consent records as in `schema.py`.

## Data policy

| Source | Terms | Use |
|---|---|---|
| USGS, NEON, ERA5, Landsat, Sentinel | Public / open | Base layers. No restriction. |
| eBird | Free for research and education; commercial use needs Cornell's written permission | Fine now that the project is non-commercial. Do not redistribute raw data. |
| iNaturalist | Default CC BY-NC | Fine for non-commercial use. Respect per-record licences. |
| Seek | Records stay on the phone by default | Not a data source. |
| Merlin | Sound detections are not published as a dataset as far as we know (unverified) | Not a data source. |
| Point Blue, UC forage records | Not confirmed open | Ask first. |

**First source to scout: satellite.** It is the one base layer the federation lacks. Candidates: live greenness from Sentinel-2/Landsat; the Rangeland Analysis Platform (16-day and annual forage estimates, western US, back to 1986; accuracy on California annual grassland unverified); Google's AlphaEarth satellite embeddings (10 m, annual 2017–2024, CC-BY 4.0) for mapping from a few dozen labelled points. Also needed for watersheds: snowpack (SNOTEL).

## The long-term loop

Christine's goal is for Claude to scout data sources and surface new patterns. The loop:

1. **Scout** — for each pattern, find open data that would test it; write a one-page source card (licence, method, coverage).
2. **Check** — re-run every pattern's test on live data.
3. **Hunt** — investigate consistent failures as candidate patterns.
4. **Brief** — a weekly page for Christine: what was tested, what broke, two or three candidates, decisions needed.

The hunt step needs five or six working checks first.

## Spec order

| # | Spec | Status |
|---|---|---|
| 015 | `015-honest-baselines.md` — per-gauge day-of-year flow record, 30-year climate normals, percentile-based flow anomalies | ready |
| 016 | `016-history-replay-scorecard.md` — replay the detector over past decades, score against documented events | ready |
| 017 | `017-canon-schema.md` — pattern files, validator, graph extension (was 015) | ready |
| 018 | `018-first-node-taylor.md` — first pattern check on NEON bird counts (was 016) | ready |
| 019 | satellite source cards + first greenness adapter | to write |
| 020 | canon MCP tools | to write |
| 021 | canon CI + contributor guide | to write |

## Open decisions for Christine

1. **Which watershed first?** Default: Russian River.
2. **Two domain reviewers.** A pattern cannot be marked `supported` without a human reviewer. Point Blue is the first place to ask.
3. **Rewrite STRATEGY.md in full?** It currently carries a "superseded" banner pointing here.
4. **Web app.** The repo's CLAUDE.md says the web app stays as a demo client. A citizen field guide was drafted as a Claude artifact ("Kinship Canon"); decide whether it becomes the web app's landing page.

## Things made outside the repo

Private Claude artifacts in Christine's account, not in this repo: the "Kinship Canon" field-guide page and the ten-slide "Kinship Earth" deck. Both were written before the watershed decision and still lead with the pattern library and Taylor's law. They need a pass before sharing. The children's game concept is in `docs/game-concept.md`.

## Sources

- Lex Fridman Podcast #503 transcript: https://lexfridman.com/math-ai-crisis-transcript/
- AMOC and rate of change (Utrecht): https://www.uu.nl/en/news/rate-of-climate-change-affects-stability-of-the-amoc
- NEON Ecological Forecasting Challenge: https://projects.ecoforecast.org/neon4cast-ci/
- EFI-USGS river chlorophyll challenge: https://projects.ecoforecast.org/usgsrc4cast-ci/
- Point Blue Rangeland Monitoring Network: https://pointblue.org/?p=35
- UC ANR, "Average year, what's that?": https://www.ucanr.edu/blog/ranching-sierra-foothills/article/average-year-whats
- Grass-Cast: https://grasscast.unl.edu/
- Rangeland Analysis Platform products: https://rangelands.app/products
- eBird data terms: https://www.birds.cornell.edu/home/ebird-data-access-terms-of-use
- iNaturalist licences: https://help.inaturalist.org/en/support/solutions/articles/151000173511
- Seek privacy policy: https://www.inaturalist.org/pages/seek_privacy_policy
- AlphaEarth satellite embeddings: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
