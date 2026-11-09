"""Implied probabilities and bookmaker-margin (overround) removal.

A bookmaker's decimal odds o_i imply raw probabilities q_i = 1 / o_i whose sum
B = sum(q_i) exceeds 1. The excess (B - 1) is the overround. Recovering "fair"
probabilities means distributing that margin back across outcomes, and the
choice of method matters: margin is not applied uniformly in practice
(favourite-longshot bias), so several standard de-vigging methods are provided.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

Method = Callable[[Sequence[float]], list[float]]


def implied(odds: Sequence[float]) -> list[float]:
    """Raw implied probabilities 1/o for decimal odds."""
    if any(o <= 1.0 for o in odds):
        raise ValueError("decimal odds must be > 1.0")
    return [1.0 / o for o in odds]


def overround(odds: Sequence[float]) -> float:
    """Bookmaker margin: sum of implied probabilities minus 1."""
    return sum(implied(odds)) - 1.0


def multiplicative(odds: Sequence[float]) -> list[float]:
    """Normalise q_i / B. Assumes margin is proportional to probability."""
    q = implied(odds)
    b = sum(q)
    return [x / b for x in q]


def additive(odds: Sequence[float]) -> list[float]:
    """Subtract an equal share of the margin from each outcome."""
    q = implied(odds)
    share = (sum(q) - 1.0) / len(q)
    p = [x - share for x in q]
    if min(p) <= 0:  # degenerate for long shots; fall back
        return multiplicative(odds)
    return p


def _bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-12) -> float:
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def power(odds: Sequence[float]) -> list[float]:
    """Find k >= 1 with sum(q_i ** k) = 1. Penalises long shots more heavily."""
    q = implied(odds)
    if sum(q) <= 1.0:
        return multiplicative(odds)
    k = _bisect(lambda k: sum(x ** k for x in q) - 1.0, 1.0, 10.0)
    return [x ** k for x in q]


def shin(odds: Sequence[float]) -> list[float]:
    """Shin (1993) model: margin arises from a fraction z of insider trading.

    p_i = (sqrt(z^2 + 4(1 - z) q_i^2 / B) - z) / (2(1 - z)),  solve sum p_i = 1 for z.
    """
    q = implied(odds)
    b = sum(q)
    if b <= 1.0:
        return multiplicative(odds)

    def probs(z: float) -> list[float]:
        return [(math.sqrt(z * z + 4 * (1 - z) * x * x / b) - z) / (2 * (1 - z)) for x in q]

    z = _bisect(lambda z: sum(probs(z)) - 1.0, 0.0, 0.5)
    return probs(z)


METHODS: dict[str, Method] = {
    "multiplicative": multiplicative,
    "additive": additive,
    "power": power,
    "shin": shin,
}


def fair_probabilities(odds: Sequence[float], method: str = "shin") -> list[float]:
    try:
        return METHODS[method](odds)
    except KeyError as exc:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}") from exc


def expected_value(prob: float, odds: float) -> float:
    """Expected return per unit staked: p * o - 1."""
    return prob * odds - 1.0


def kelly_fraction(prob: float, odds: float) -> float:
    """Kelly-optimal fraction of capital, f* = (p*o - 1) / (o - 1), floored at 0."""
    return max(0.0, (prob * odds - 1.0) / (odds - 1.0))
