from pathlib import Path

from oddsmodel.client import load_snapshot
from oddsmodel.consensus import find_mispricings, tidy
from oddsmodel.evaluation import brier, expected_calibration_error, log_loss
from oddsmodel.llm_review import _parse

SAMPLE = Path(__file__).parents[1] / "examples" / "sample_odds.json"


def _event(prices_by_book):
    return {
        "id": "e1",
        "sport_key": "test",
        "commence_time": "2026-01-01T15:00:00Z",
        "home_team": "A",
        "away_team": "B",
        "bookmakers": [
            {
                "key": bk,
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "A", "price": a},
                            {"name": "Draw", "price": d},
                            {"name": "B", "price": b},
                        ],
                    }
                ],
            }
            for bk, (a, d, b) in prices_by_book.items()
        ],
    }


def test_outlier_price_is_flagged():
    books = {f"book{i}": (2.00, 3.40, 3.80) for i in range(5)}
    books["generous"] = (2.00, 3.40, 4.60)
    flags = find_mispricings([_event(books)], min_edge=0.03)
    assert [(f.bookmaker, f.outcome) for f in flags] == [("generous", "B")]
    assert flags[0].edge > 0.03


def test_consistent_market_not_flagged():
    books = {f"book{i}": (2.00, 3.40, 3.80) for i in range(6)}
    assert find_mispricings([_event(books)]) == []


def test_min_books():
    books = {"x": (2.0, 3.4, 3.8), "generous": (2.0, 3.4, 4.6)}
    assert find_mispricings([_event(books)], min_books=4) == []


def test_sample_snapshot_runs():
    events = load_snapshot(SAMPLE)
    df = tidy(events)
    assert len(df) == len(events) * 8 * 3
    flags = find_mispricings(events)
    assert {f.bookmaker for f in flags} >= {"betvictor", "skybet"}


def test_scoring_rules():
    assert brier([1, 0], [1, 0]) == 0
    assert brier([0.5, 0.5], [1, 0]) == 0.25
    assert log_loss([0.5], [1]) > log_loss([0.9], [1])
    assert expected_calibration_error([0.2] * 5, [0, 0, 0, 0, 1]) < 1e-9


def test_parse_llm_json():
    text = 'Here you go:\n[{"index": 0, "verdict": "likely_artefact", "confidence": 0.7, "reason": "stale"}]'
    assert _parse(text)[0]["verdict"] == "likely_artefact"
