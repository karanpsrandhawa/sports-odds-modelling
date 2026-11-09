"""Scoring rules and calibration checks for probability forecasts.

Used to compare de-vigging methods against realised results: a better method
should produce probabilities with lower Brier score / log loss and a reliability
curve closer to the diagonal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def brier(probs, outcomes) -> float:
    p, y = np.asarray(probs, float), np.asarray(outcomes, float)
    return float(np.mean((p - y) ** 2))


def log_loss(probs, outcomes, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(probs, float), eps, 1 - eps)
    y = np.asarray(outcomes, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(probs, outcomes, bins: int = 10) -> pd.DataFrame:
    """Mean forecast vs observed frequency per probability bin."""
    df = pd.DataFrame({"p": probs, "y": outcomes})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = df.groupby("bin", observed=True).agg(
        mean_forecast=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    return out.reset_index()


def expected_calibration_error(probs, outcomes, bins: int = 10) -> float:
    r = reliability(probs, outcomes, bins)
    w = r["n"] / r["n"].sum()
    return float((w * (r["mean_forecast"] - r["observed"]).abs()).sum())
