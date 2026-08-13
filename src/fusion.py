"""Station 3 (extension) - fuse sentiment into the funds.

Tilt or factor: combine your sentiment signal with the portfolio weights,
look-ahead safe, then test whether it adds value. An honest negative result,
explained, is good work.

Design (documented so the report can cite it):

- Fusion rule: multiplicative tilt on a cross-sectional standardised signal,
      z_i  = (s_i - mean(s)) / std(s)          over the sentiment-bearing assets
      w'_i = clip(w_i * (1 + tilt_strength * z_i), 0, inf)
      w''  = w' / sum(w')
  Multiplicative (not additive) so the tilt scales each name relative to its
  own base weight and never flips the base ordering. Standardising z makes
  tilt_strength comparable across models with different dispersions (VADER
  ticker-day std ~0.18 vs FinVADER ~0.14), so one lambda means the same tilt
  intensity for either lexicon.
- Mean-zero z per date: the tilt is a reallocation - high-sentiment names gain
  weight funded by low-sentiment names - never an unhedged market bet, and the
  fund stays fully invested after renormalisation.
- Long-only, fully invested: weights must be non-negative and sum to 1 (an
  investable fund a user buys into). Clipping + renormalisation enforces this.
- Crypto and any asset without news get z = 0 (sentiment applies to equity
  only - crypto is price-only), so their weights pass through untouched.
- Combined-fund weights (EQ_/CR_ prefixes) are supported: an EQ_<ticker>
  column is looked up as <ticker> in the sentiment panel, so the equity leg of
  a combined fund is tilted; CR_<ticker> columns never match and stay at z=0.
- Look-ahead safety is owned by apply_sentiment, not the caller: it lags the
  raw ticker-day signal by ONE TRADING DAY internally (shift on the sentiment
  calendar) before aligning each rebalance date with the most recent sentiment
  value ON OR BEFORE that date (forward-fill reindex). The tilt at date t
  therefore uses only news aligned to trading day t-1 or earlier. Passing an
  already-lagged signal is conservative (an extra day of lag), never look-ahead;
  passing a sector-level index fails loudly (it has no ticker column). If the
  whole portfolio clips to zero the baseline weights are returned unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_sentiment(weights: pd.DataFrame, sentiment: pd.DataFrame,
                    score_col: str = "finvader_sentiment",
                    tilt_strength: float = 0.5) -> pd.DataFrame:
    """Tilt long-only portfolio weights by a lagged sentiment signal.

    Parameters
    ----------
    weights : wide DataFrame, index = dates, columns = asset tickers,
        each row non-negative and summing to ~1 (fully invested, long only).
    sentiment : long DataFrame with columns [ticker, date, score_col] - the
        shape of results/data/ticker_day_sentiment.csv. RAW ticker-day
        sentiment (value on day t is same-day news); apply_sentiment lags it
        one trading day internally before aligning to the weights dates.
    score_col : which sentiment column to use ('vader_sentiment' or
        'finvader_sentiment') - this is how the same rule runs either model.
    tilt_strength : lambda in w*(1 + lambda*z). 0 returns the baseline weights.

    Returns
    -------
    A wide DataFrame of the same index/columns as `weights`, renormalised to
    sum to 1 per date.
    """
    if score_col not in sentiment.columns:
        raise ValueError(f"sentiment has no column {score_col!r}; "
                         f"available: {list(sentiment.columns)}")
    if not isinstance(weights.index, pd.DatetimeIndex):
        raise TypeError("weights.index must be a DatetimeIndex")

    # Wide, model-agnostic sentiment on the equity tickers.
    sent_wide = sentiment.pivot_table(index="date", columns="ticker",
                                       values=score_col)
    # Lag one trading day on the sentiment calendar (raw ticker-day input is
    # same-day news; shift makes the tilt at date t use news from t-1 or
    # earlier). Forward-fill alignment onto the weights dates then picks the
    # latest value on or before each rebalance date.
    sent_wide = sent_wide.shift(1)
    sent_wide = sent_wide.reindex(weights.index, method="ffill")  # latest <= rebalance date

    # Tilt only the assets that carry news. Weight columns are mapped back to
    # sentiment tickers: bare equity tickers map to themselves (equity-only
    # funds); EQ_<ticker> columns map to <ticker> (combined funds); CR_<ticker>
    # and anything else have no news and get z = 0.
    tilt_map = {}
    for c in weights.columns:
        if c in sent_wide.columns:
            tilt_map[c] = c
        elif c.startswith("EQ_") and c[3:] in sent_wide.columns:
            tilt_map[c] = c[3:]

    z = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    if tilt_map:
        bare = [tilt_map[c] for c in tilt_map]
        sub = sent_wide[bare]
        demeaned = sub.sub(sub.mean(axis=1), axis=0)
        scale = sub.std(axis=1).replace(0.0, np.nan)  # guard: no cross-sectional spread
        z_bare = demeaned.div(scale, axis=0).fillna(0.0)
        z_bare.columns = list(tilt_map)
        z[list(tilt_map)] = z_bare

    tilted = weights.mul(1.0 + tilt_strength * z, axis=0).clip(lower=0.0)
    total = tilted.sum(axis=1)

    # If a date clips to all-zero (pathological signal), fall back to baseline.
    zero_days = total <= 0
    if zero_days.any():
        tilted.loc[zero_days] = weights.loc[zero_days]
        total = tilted.sum(axis=1)

    return tilted.div(total, axis=0)
