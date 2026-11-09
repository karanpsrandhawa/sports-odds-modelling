"""Second-pass review of statistically flagged markets using the Claude API.

The quantitative screen finds prices that disagree with the market consensus,
but many of those disagreements have mundane explanations (a stale quote, a
mismatched team name, a market that has already moved, a known late team-news
change). The LLM is asked to triage each flag and return structured JSON, so
its output can be filtered and logged like any other model output. It does not
produce the probabilities; those come from the consensus model.
"""
from __future__ import annotations

import json
import os
import re

from .consensus import Flag

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM = """You review candidate pricing discrepancies in sports odds markets.
Each candidate comes from a quantitative screen: one bookmaker's decimal price
implies an expected value above threshold relative to the de-vigged median
probability across several bookmakers.

For each candidate, judge whether the discrepancy is more likely a genuine
mispricing or an artefact (stale quote, thin market, outlier book, data error,
consensus itself uncertain given high dispersion). Be sceptical: most screens
produce false positives.

Respond with ONLY a JSON array, one object per candidate, in input order:
[{"index": int, "verdict": "likely_mispriced" | "likely_artefact" | "uncertain",
  "confidence": float between 0 and 1, "reason": short string}]"""


def _format(flags: list[Flag]) -> str:
    lines = []
    for i, f in enumerate(flags):
        lines.append(
            f"[{i}] {f.match} ({f.sport}, kicks off {f.commence_time}) | outcome: {f.outcome} | "
            f"{f.bookmaker} @ {f.price} (implied {f.implied_prob:.3f}) | next best {f.best_alt_price} | "
            f"consensus p={f.consensus_prob:.3f} from {f.n_books} books, dispersion {f.dispersion:.3f} | "
            f"EV {f.edge:+.3f}"
        )
    return "\n".join(lines)


def _parse(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        raise ValueError(f"no JSON array in model output: {text[:200]}")
    return json.loads(match.group(0))


def review(flags: list[Flag], model: str = DEFAULT_MODEL, max_flags: int = 25) -> list[dict]:
    """Return flags merged with the model's verdict, confidence and reason."""
    if not flags:
        return []
    import anthropic  # imported lazily so the quant pipeline runs without it

    flags = flags[:max_flags]
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": _format(flags)}],
    )
    text = "".join(block.text for block in msg.content if block.type == "text")
    verdicts = {v["index"]: v for v in _parse(text)}

    merged = []
    for i, f in enumerate(flags):
        v = verdicts.get(i, {})
        merged.append(
            {
                **f.to_dict(),
                "verdict": v.get("verdict", "uncertain"),
                "llm_confidence": v.get("confidence"),
                "reason": v.get("reason", ""),
            }
        )
    return merged
