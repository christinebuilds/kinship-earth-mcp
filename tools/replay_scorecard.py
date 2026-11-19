"""Replay the flow detector over each gauge's history and write a scorecard (spec 016).

    uv run --package kinship-orchestrator python tools/replay_scorecard.py \
        --events canon/events/reference_events.yaml --out canon/scorecards/

Writes flow-detector.json, flow-detector.md and figures/{site_id}.png.
Low-flow thresholds 5, 10 and 20 are all reported (high = 100 - low); no
single threshold is picked as "the" result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

from kinship_shared.replay import (
    Episode, ReferenceEvent, Scorecard, group_episodes, replay_flow, score,
)

THRESHOLDS = [5, 10, 20]
HEADLINE = 10  # used for the unmatched list and the figures, not as "the" answer


def load_events(path: Path) -> list[ReferenceEvent]:
    data = yaml.safe_load(path.read_text())
    items = data["events"] if isinstance(data, dict) else data
    return [ReferenceEvent(**{k: v for k, v in e.items() if k in ReferenceEvent.model_fields})
            for e in items]


async def fetch_records(site_ids: list[str]) -> dict[str, list[tuple[date, float]]]:
    from usgs_nwis_mcp.adapter import USGSNWISAdapter
    adapter = USGSNWISAdapter()
    return {s: await adapter.get_daily_record(s) for s in site_ids}


def run(records, events, min_low_days=14, min_high_days=2):
    """Returns {threshold: {site_id: (Scorecard, days, episodes)}}."""
    out: dict[int, dict] = {}
    for low in THRESHOLDS:
        high = 100 - low
        per_site = {}
        for site, rec in records.items():
            if not rec:
                continue
            days = replay_flow(rec, rec[0][0], rec[-1][0], low_pct=low, high_pct=high)
            eps = group_episodes(days, min_low_days=min_low_days, min_high_days=min_high_days)
            per_site[site] = (score(site, days, eps, events, low_pct=low, high_pct=high), days, eps)
        out[low] = per_site
    return out


def pooled(cards: list[Scorecard]) -> dict:
    hits = sum(c.events_hit for c in cards)
    total = sum(c.events_total for c in cards)
    leads = sorted(v for c in cards for v in c.hits.values() if v is not None)
    mid = None
    if leads:
        n = len(leads)
        mid = float(leads[n // 2]) if n % 2 else (leads[n // 2 - 1] + leads[n // 2]) / 2
    return {
        "events_hit": hits, "events_total": total, "lead_days_median": mid,
        "episodes_total": sum(c.episodes_total for c in cards),
        "episodes_unmatched": sum(c.episodes_unmatched for c in cards),
        "years_replayed": sum(c.years_replayed for c in cards),
    }


def _per_decade(n: int, years: int) -> str:
    return f"{10 * n / years:.1f}" if years else "–"


def _lead(v) -> str:
    return "–" if v is None else f"{v:+.0f} d"


def write_markdown(results, events, path: Path, fig_dir_rel: str = "figures") -> None:
    lines = [
        "# How often is the flow detector right?",
        "",
        "Each gauge's full daily record was replayed day by day. Each day was judged only",
        "against earlier years for the same two weeks of the year (no look-ahead). A run of",
        "at least 14 low days, or 2 high days, counts as an episode. Episodes were then",
        "matched against a short list of documented, cited events",
        "(`canon/events/reference_events.yaml`).",
        "",
        "| Gauge | Low / high threshold (percentile) | Years replayed | Events hit / total | Median lead | Unmatched episodes per decade |",
        "|---|---|---|---|---|---|",
    ]
    for low, per_site in results.items():
        for site, (card, _, _) in sorted(per_site.items()):
            lines.append(
                f"| {site} | {low} / {100 - low} | {card.years_replayed} | "
                f"{card.events_hit} / {card.events_total} | {_lead(card.lead_days_median)} | "
                f"{_per_decade(card.episodes_unmatched, card.years_replayed)} |"
            )
        p = pooled([c for c, _, _ in per_site.values()])
        lines.append(
            f"| **All gauges** | {low} / {100 - low} | {p['years_replayed']} | "
            f"**{p['events_hit']} / {p['events_total']}** | {_lead(p['lead_days_median'])} | "
            f"{_per_decade(p['episodes_unmatched'], p['years_replayed'])} |"
        )
    lines += [
        "",
        "Lead is the first flagged day minus the documented start; negative means the",
        "detector flagged it before the documented start.",
        "",
        "## What this shows and what it does not",
        "",
        "It shows whether the detector, using only data it would have had at the time,",
        "noticed each documented event, and how early. It does not show a false-alarm rate:",
        "the reference list is short, so an *unmatched* episode is often a real dry spell or",
        "flood that nobody has listed yet, not a mistake. All three thresholds are reported",
        "so that no single setting is presented as the answer. A short, hand-picked event",
        "list also flatters any detector, so treat these numbers as a first record, not a",
        "verdict.",
        "",
        f"## Unmatched episodes longer than 60 days (threshold {HEADLINE})",
        "",
        "Candidates for the reference list, if a source documents them:",
        "",
    ]
    long_ones = [(site, ep) for site, (card, _, _) in sorted(results[HEADLINE].items())
                 for ep in card.unmatched if ep.days > 60]
    if long_ones:
        for site, ep in long_ones:
            lines.append(f"- {site}: {ep.kind} flow, {ep.start} to {ep.end} ({ep.days} days)")
    else:
        lines.append("- None.")
    lines += ["", "## Figures", ""]
    for site in sorted(results[HEADLINE]):
        lines.append(f"- [{site}]({fig_dir_rel}/{site}.png)")
    lines.append("")
    path.write_text("\n".join(lines))


def write_figures(results, events, fig_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib not installed; figures skipped"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    for site, (card, days, eps) in results[HEADLINE].items():
        xs = [d.day for d in days if d.percentile is not None]
        ys = [d.percentile for d in days if d.percentile is not None]
        fig, ax = plt.subplots(figsize=(12, 3.2), dpi=120)
        ax.plot(xs, ys, lw=0.3, color="#555")
        for ev in events:
            if ev.site_id == site:
                ax.axvspan(ev.start, ev.end, color="#d9a441" if ev.kind == "low_flow" else "#4a7fb5",
                           alpha=0.35, lw=0)
        for ep in eps:
            ax.plot([ep.start, ep.end], [-4, -4] if ep.kind == "low" else [104, 104],
                    color="#b5482f" if ep.kind == "low" else "#2f5fb5", lw=3, solid_capstyle="butt")
        ax.set_ylim(-8, 108)
        ax.set_ylabel("Flow percentile")
        ax.set_title(f"{site}: flow percentile vs earlier years "
                     f"(shaded = documented events; bars = flagged episodes, threshold {HEADLINE})",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{site}.png")
        plt.close(fig)
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=Path("canon/events/reference_events.yaml"))
    ap.add_argument("--out", type=Path, default=Path("canon/scorecards"))
    args = ap.parse_args()

    events = load_events(args.events)
    sites = sorted({e.site_id for e in events})
    records = asyncio.run(fetch_records(sites))
    results = run(records, events)

    args.out.mkdir(parents=True, exist_ok=True)
    notes = write_figures(results, events, args.out / "figures")
    payload = {
        "generated": date.today().isoformat(),
        "detector": "flow (day-of-year percentile, spec 015)",
        "episode_rule": {"min_low_days": 14, "min_high_days": 2},
        "thresholds": {
            str(low): {
                "per_gauge": {s: c.model_dump(mode="json") for s, (c, _, _) in sorted(per.items())},
                "pooled": pooled([c for c, _, _ in per.values()]),
            }
            for low, per in results.items()
        },
        "notes": notes,
    }
    (args.out / "flow-detector.json").write_text(json.dumps(payload, indent=2))
    write_markdown(results, events, args.out / "flow-detector.md")
    for low, per in results.items():
        p = pooled([c for c, _, _ in per.values()])
        print(f"threshold {low}: {p['events_hit']}/{p['events_total']} hit, "
              f"median lead {p['lead_days_median']}, unmatched {p['episodes_unmatched']}")


if __name__ == "__main__":
    main()
