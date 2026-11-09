# Sports Odds Modelling

![tests](https://github.com/karanpsrandhawa/sports-odds-modelling/actions/workflows/tests.yml/badge.svg)

A Python pipeline that pulls live sports odds from [The Odds API](https://the-odds-api.com/), converts bookmaker prices into fair probabilities, builds a cross-market consensus, and flags quotes that look mispriced relative to that consensus. A second stage uses the Claude API to triage the flagged quotes and separate likely genuine discrepancies from data artefacts.

The project treats bookmaker odds as noisy probability forecasts. The core questions are statistical: how to remove the bookmaker's margin, how to combine several noisy estimates into one, and how to tell a real pricing error from noise.

## Pipeline

```
The Odds API  ──►  tidy (event × bookmaker × outcome)
                     │
                     ▼
              remove margin per bookmaker   (multiplicative / additive / power / Shin)
                     │
                     ▼
              consensus probability          (median across bookmakers + dispersion)
                     │
                     ▼
              screen: EV = p·o − 1 > threshold
                     │
                     ▼
              Claude API review              (structured JSON verdict per flag)
                     │
                     ▼
              CSV output + raw API snapshot saved for reproducibility
```

## Methods

**Implied probability and overround.** Decimal odds $o_i$ imply $q_i = 1/o_i$. For a full market, $B = \sum_i q_i > 1$, and $B - 1$ is the bookmaker's margin.

**Removing the margin.** The margin is not spread evenly in practice: long shots are typically overpriced (the favourite–longshot bias). Four methods are implemented in [`probability.py`](src/oddsmodel/probability.py):

| Method | Fair probability | Assumption |
|---|---|---|
| Multiplicative | $p_i = q_i / B$ | margin proportional to probability |
| Additive | $p_i = q_i - (B-1)/n$ | equal margin on every outcome |
| Power | $p_i = q_i^k$, with $k$ such that $\sum q_i^k = 1$ | margin weighted towards long shots |
| Shin (1993) | $p_i = \dfrac{\sqrt{z^2 + 4(1-z)q_i^2/B} - z}{2(1-z)}$, solve $\sum p_i = 1$ for $z$ | margin protects against a fraction $z$ of informed money |

Power and Shin are solved numerically by bisection. Shin is the default.

**Consensus.** Each bookmaker's market is de-vigged on its own. The fair probabilities are then combined using the median, which is robust to a single stale or wrong quote. The standard deviation across bookmakers is recorded as a measure of how uncertain the consensus is. Bookmakers with incomplete markets or margins above 15% are excluded.

**Screening.** A quote is flagged when its expected value under the consensus probability, $\text{EV} = p \cdot o - 1$, is above a threshold (3% by default) and at least 4 bookmakers price the market. The Kelly fraction $f^* = (po - 1)/(o - 1)$ is also reported, as a measure of how large the edge is relative to the odds.

**LLM review.** A statistical screen produces false positives (stale prices, thin markets, team-news moves the consensus hasn't caught up with). Flagged quotes are sent to Claude with their context (consensus, dispersion, next-best price, number of bookmakers). Claude returns a JSON verdict (`likely_mispriced` / `likely_artefact` / `uncertain`) with a confidence and a reason. The LLM does not produce the probabilities. It only triages what the quantitative model has already found.

**Evaluation.** [`evaluation.py`](src/oddsmodel/evaluation.py) provides the Brier score, log loss, reliability tables and expected calibration error. These are for comparing de-vigging methods against realised results.

## Quick start

```bash
git clone https://github.com/karanpsrandhawa/sports-odds-modelling.git
cd sports-odds-modelling
pip install -e ".[llm,dev]"
cp .env.example .env          # add ODDS_API_KEY and ANTHROPIC_API_KEY

# Offline demo on a bundled synthetic snapshot (no API keys needed)
python -m oddsmodel --snapshot examples/sample_odds.json --no-llm

# Live: Premier League head-to-head markets, UK + EU bookmakers, with Claude review
python -m oddsmodel --sport soccer_epl --method shin --min-edge 0.03
```

Example output from the offline demo:

```
                                    match        outcome bookmaker  price  consensus_prob   edge  n_books
Wolverhampton Wanderers vs Crystal Palace Crystal Palace    skybet   3.28          0.3357 0.1010        8
          Newcastle United vs Aston Villa    Aston Villa betvictor   3.95          0.2748 0.0855        8
```

`examples/sample_odds.json` is **synthetic**. It was generated with realistic margins and noise, and two outlier prices were planted on purpose. The demo and the tests check that the screen recovers exactly those two.

Run the tests with `pytest`.

## Project structure

```
src/oddsmodel/
  client.py        The Odds API client, snapshot save/load
  probability.py   implied probability, overround, de-vigging methods, EV, Kelly
  consensus.py     flatten API data, cross-book consensus, mispricing screen
  llm_review.py    Claude API triage with structured JSON output
  evaluation.py    Brier score, log loss, calibration
  __main__.py      CLI
tests/             pytest suite
examples/          synthetic sample snapshot
```

## Limitations and next steps

- The consensus is a market-implied estimate, not an independent model. Next step: an independent model to compare against it (e.g. a Poisson / Dixon–Coles goals model).
- The screen is cross-sectional (one snapshot). Line movement over time is not used yet.
- Comparing the de-vigging methods needs a dataset of historical closing odds and results to score with `evaluation.py`.
- Only head-to-head (match result) markets are handled so far. Totals and handicaps would need their own outcome matching.

## Licence

MIT
