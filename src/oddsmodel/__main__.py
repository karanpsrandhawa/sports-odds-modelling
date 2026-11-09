"""Command-line entry point.

    python -m oddsmodel --sport soccer_epl                # live odds
    python -m oddsmodel --snapshot examples/sample_odds.json --no-llm   # offline
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .client import OddsAPIClient, load_snapshot, save_snapshot
from .consensus import find_mispricings
from .probability import METHODS


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Flag mispriced sports markets")
    ap.add_argument("--sport", default="soccer_epl", help="Odds API sport key")
    ap.add_argument("--regions", default="uk,eu")
    ap.add_argument("--snapshot", help="Use a saved JSON response instead of the live API")
    ap.add_argument("--method", default="shin", choices=sorted(METHODS))
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--min-books", type=int, default=4)
    ap.add_argument("--no-llm", action="store_true", help="Skip the Claude review step")
    ap.add_argument("--out", default="output", help="Directory for CSV results")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out)

    if args.snapshot:
        events = load_snapshot(args.snapshot)
    else:
        events = OddsAPIClient().odds(args.sport, regions=args.regions)
        save_snapshot(events, out / "snapshots" / f"{args.sport}_{stamp}.json")

    flags = find_mispricings(
        events, method=args.method, min_edge=args.min_edge, min_books=args.min_books
    )
    logging.info("%d events, %d flagged quotes", len(events), len(flags))

    if flags and not args.no_llm:
        from .llm_review import review

        rows = review(flags)
    else:
        rows = [f.to_dict() for f in flags]

    df = pd.DataFrame(rows)
    if df.empty:
        print("No quotes above the edge threshold.")
        return
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"flags_{stamp}.csv"
    df.to_csv(path, index=False)

    cols = [c for c in ["match", "outcome", "bookmaker", "price", "consensus_prob",
                        "edge", "n_books", "verdict"] if c in df.columns]
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df[cols].to_string(index=False))
    print(f"\nSaved {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
