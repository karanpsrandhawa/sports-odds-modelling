"""Build a market-consensus probability for every outcome and flag mispriced quotes.

For each event, every bookmaker's prices are de-vigged independently, then the
fair probabilities are combined across bookmakers (median by default, which is
robust to a single stale or erroneous quote). Each bookmaker's offered price is
then compared with the consensus: a quote whose expected value under the
consensus probability exceeds a threshold is flagged as potentially mispriced.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

import pandas as pd

from .probability import expected_value, fair_probabilities, kelly_fraction, overround


@dataclass
class Flag:
    event_id: str
    sport: str
    commence_time: str
    match: str
    outcome: str
    bookmaker: str
    price: float
    best_alt_price: float
    consensus_prob: float
    implied_prob: float
    edge: float          # expected value per unit at the consensus probability
    kelly: float
    n_books: int
    dispersion: float    # std dev of fair probability across books

    def to_dict(self) -> dict:
        return asdict(self)


def tidy(events: list[dict], market: str = "h2h") -> pd.DataFrame:
    """Flatten the nested API response into one row per (event, bookmaker, outcome)."""
    rows = []
    for ev in events:
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != market:
                    continue
                for oc in mk.get("outcomes", []):
                    rows.append(
                        {
                            "event_id": ev["id"],
                            "sport": ev.get("sport_key", ""),
                            "commence_time": ev.get("commence_time", ""),
                            "match": f"{ev.get('home_team')} vs {ev.get('away_team')}",
                            "bookmaker": bk["key"],
                            "outcome": oc["name"],
                            "price": float(oc["price"]),
                        }
                    )
    return pd.DataFrame(rows)


def consensus_table(df: pd.DataFrame, method: str = "shin") -> pd.DataFrame:
    """Add a per-bookmaker fair probability and overround column."""
    out = []
    for (_, _), g in df.groupby(["event_id", "bookmaker"], sort=False):
        prices = g["price"].tolist()
        g = g.copy()
        g["fair_prob"] = fair_probabilities(prices, method)
        g["overround"] = overround(prices)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df


def find_mispricings(
    events: list[dict],
    method: str = "shin",
    min_edge: float = 0.03,
    min_books: int = 4,
    max_overround: float = 0.15,
) -> list[Flag]:
    df = tidy(events)
    if df.empty:
        return []
    df = consensus_table(df, method)
    # Drop books with incomplete markets or extreme margins; they distort consensus
    df = df[df["overround"].between(-0.02, max_overround)]

    flags: list[Flag] = []
    for (event_id, outcome), g in df.groupby(["event_id", "outcome"], sort=False):
        if len(g) < min_books:
            continue
        probs = g["fair_prob"].tolist()
        p = statistics.median(probs)
        disp = statistics.pstdev(probs)
        for _, row in g.iterrows():
            ev = expected_value(p, row["price"])
            if ev < min_edge:
                continue
            others = g.loc[g["bookmaker"] != row["bookmaker"], "price"]
            flags.append(
                Flag(
                    event_id=event_id,
                    sport=row["sport"],
                    commence_time=row["commence_time"],
                    match=row["match"],
                    outcome=outcome,
                    bookmaker=row["bookmaker"],
                    price=row["price"],
                    best_alt_price=float(others.max()) if len(others) else float("nan"),
                    consensus_prob=round(p, 4),
                    implied_prob=round(1 / row["price"], 4),
                    edge=round(ev, 4),
                    kelly=round(kelly_fraction(p, row["price"]), 4),
                    n_books=len(g),
                    dispersion=round(disp, 4),
                )
            )
    return sorted(flags, key=lambda f: f.edge, reverse=True)
