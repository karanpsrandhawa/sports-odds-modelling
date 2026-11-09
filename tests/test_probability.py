import math

import pytest

from oddsmodel.probability import (
    METHODS,
    expected_value,
    fair_probabilities,
    implied,
    kelly_fraction,
    overround,
)

BOOK = [1.80, 3.60, 4.50]  # typical three-way football market, ~5.5% margin


@pytest.mark.parametrize("method", sorted(METHODS))
def test_probabilities_sum_to_one(method):
    p = fair_probabilities(BOOK, method)
    assert math.isclose(sum(p), 1.0, abs_tol=1e-9)
    assert all(0 < x < 1 for x in p)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_fair_book_unchanged(method):
    fair = [2.0, 4.0, 4.0]  # sums to exactly 1
    assert fair_probabilities(fair, method) == pytest.approx([0.5, 0.25, 0.25])


def test_overround():
    assert overround(BOOK) == pytest.approx(1 / 1.8 + 1 / 3.6 + 1 / 4.5 - 1)


def test_shin_and_power_shade_longshots_more_than_multiplicative():
    mult = fair_probabilities(BOOK, "multiplicative")
    for m in ("shin", "power"):
        p = fair_probabilities(BOOK, m)
        assert p[0] > mult[0]  # favourite gets more probability back
        assert p[2] < mult[2]  # long shot gets less


def test_preserves_ordering():
    for m in METHODS:
        p = fair_probabilities(BOOK, m)
        assert p[0] > p[1] > p[2]


def test_invalid_odds():
    with pytest.raises(ValueError):
        implied([1.0, 2.0])
    with pytest.raises(ValueError):
        fair_probabilities(BOOK, "nope")


def test_ev_and_kelly():
    assert expected_value(0.5, 2.2) == pytest.approx(0.1)
    assert kelly_fraction(0.5, 2.2) == pytest.approx(0.1 / 1.2)
    assert kelly_fraction(0.4, 2.0) == 0.0
