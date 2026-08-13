"""Station 3 - your sentiment model and index from news headlines.

This is the model step: score each headline, aggregate to a daily per-ticker score,
then to an equal-weight sector index. Headlines are a noisy proxy, so lag to avoid
look-ahead.

Design notes (so the report can cite them):
- Two lexicons are compared: plain VADER and the FinVADER finance lexicon
  (SentiBignomics*0.1 merged with Henry on top of the base VADER lexicon). Plain
  VADER leaves about half of finance headlines neutral (false neutrals), so the
  finance lexicon is expected to reclassify some of them.
- Headline text is kept UNMODIFIED (casing, punctuation, negation intact) because
  VADER's mechanics rely on them - we only lowercase for the LEXICON build below.
- We replicate FinVADER's exact lexicon merge with a single pre-built analyzer
  rather than calling finvader.finvader() per headline: the package rebuilds the
  analyzer and lexicons on every call (~35x slower over 147k headlines). The
  replicated analyzer is verified numerically identical to finvader.finvader().
- Ticker-days with no headlines are treated as neutral (0.0), justified in the
  script: no news is its own state, and carrying stale headlines forward would
  make the index sticky. The index is therefore defined on the full equity
  trading calendar.
- The index is lagged one trading day AFTER sector averaging (mean commutes with
  shift, so averaging first is equivalent) so the value on day t uses only
  headlines aligned to trading day t-1 or earlier - no look-ahead.
- Additional option: a COVERAGE-WEIGHTED sector index (see
  coverage_weighted_sector_sentiment_index). The number of headlines backing a
  ticker-day reading is used as a confidence weight: index(s,t) =
  sum_i n(i,t)*score(i,t) / sum_i n(i,t). Zero-headline ticker-days get weight
  0 (no evidence -> no contribution), unlike the equal-weight index which
  counts them as neutral 0.0. Sector-days with no headlines at all have no
  reading (NaN, dropped after the lag) - downstream fusion's forward-fill holds
  the last trustworthy reading rather than a false neutral. The original
  equal-weight VADER/FinVADER indices are unchanged.
"""
from __future__ import annotations

import pandas as pd

from src import etl


def _base_analyzer():
    """Plain VADER analyzer (nltk). One-time nltk.download('vader_lexicon') is
    a build step, not part of the deployed app."""
    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def _finvader_analyzer():
    """VADER analyzer with FinVADER's finance lexicons merged in.

    Replicates finvader.finvader(use_sentibignomics=True, use_henry=True):
    SentiBignomics scaled by 0.1, then Henry merged on top (Henry wins on key
    conflicts, matching FinVADER's dict merge). Built once instead of per call.
    """
    from finvader import lexicon1, lexicon2

    sentibignomics = lexicon1()
    sentibignomics.update((word, val * 0.1) for word, val in sentibignomics.items())
    merged = {**sentibignomics, **lexicon2()}

    analyzer = _base_analyzer()
    analyzer.lexicon.update(merged)
    return analyzer


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score every headline in the aligned text panel with plain VADER and with
    FinVADER's finance lexicon.

    Adds two compound columns:
    - vader_compound: plain VADER compound score in [-1, 1].
    - finvader_compound: VADER + finance-lexicon compound score in [-1, 1].

    `panel` must already be the trading-day-aligned panel from
    etl.build_text_panel (columns ticker, date, sector, title, ..., trading_day).
    The title is used as-is - no casing/punctuation stripping (VADER needs them).
    """
    scored = panel.copy()
    scored["vader_compound"] = _score_titles(scored["title"], _base_analyzer())
    scored["finvader_compound"] = _score_titles(scored["title"], _finvader_analyzer())
    return scored


def _score_titles(titles: pd.Series, analyzer) -> pd.Series:
    """Compound score of each title under a given analyzer."""
    return titles.map(lambda t: analyzer.polarity_scores(t)["compound"])


def ticker_day_sentiment(panel: pd.DataFrame, calendar_dates: pd.DatetimeIndex,
                         fill_missing: str = "neutral") -> pd.DataFrame:
    """Aggregate per-headline scores to one (ticker, trading_day) score.

    A ticker-day score is the simple mean of its headline compounds. The result
    is reindexed onto the full equity trading calendar so the panel is complete
    (one row per ticker per trading day).

    fill_missing: 'neutral' -> ticker-days with no headlines score 0.0;
    'drop' -> keep only ticker-days that actually had headlines.

    Returns a long DataFrame [ticker, date, vader_sentiment, finvader_sentiment].
    """
    agg = (
        panel.groupby(["ticker", "trading_day"], as_index=False)[
            ["vader_compound", "finvader_compound"]
        ]
        .mean()
        .rename(columns={
            "trading_day": "date",
            "vader_compound": "vader_sentiment",
            "finvader_compound": "finvader_sentiment",
        })
    )

    if fill_missing == "neutral":
        tickers = sorted(panel["ticker"].unique())
        full_index = pd.MultiIndex.from_product(
            [tickers, pd.DatetimeIndex(calendar_dates).sort_values()],
            names=["ticker", "date"],
        )
        agg = agg.set_index(["ticker", "date"]).reindex(full_index, fill_value=0.0).reset_index()
    elif fill_missing != "drop":
        raise ValueError(f"fill_missing must be 'neutral' or 'drop', got {fill_missing!r}")

    return agg.sort_values(["ticker", "date"]).reset_index(drop=True)


def sector_sentiment_index(ticker_day: pd.DataFrame, sector_map: pd.DataFrame,
                           lag: int = 1) -> pd.DataFrame:
    """Equal-weight sector sentiment index from ticker-day scores.

    Each sector's index on day t is the mean of its constituents' scores that
    day (equal-weight the tickers - each of the 5 names counts equally). The
    result is then shifted forward by `lag` trading days so the value on day t
    uses only headlines aligned to day t-lag or earlier. The first `lag` days of
    each sector are dropped (no prior news to form a lagged signal from).

    sector_map: DataFrame with ticker and sector columns (one row per ticker).

    Returns a long DataFrame [date, sector, vader_sentiment, finvader_sentiment].
    """
    df = ticker_day.merge(sector_map, on="ticker", how="left")
    idx = (
        df.groupby(["date", "sector"], as_index=False)[
            ["vader_sentiment", "finvader_sentiment"]
        ]
        .mean()
        .sort_values(["sector", "date"])
    )

    if lag > 0:
        idx[["vader_sentiment", "finvader_sentiment"]] = (
            idx.groupby("sector")[["vader_sentiment", "finvader_sentiment"]].shift(lag)
        )
        idx = idx.dropna(subset=["vader_sentiment"])

    return idx.reset_index(drop=True)


def ticker_day_coverage(panel: pd.DataFrame,
                        calendar_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Number of headlines per (ticker, trading_day), the confidence weight
    used by coverage_weighted_sector_sentiment_index.

    Reindexed onto the full equity trading calendar, so no-news ticker-days
    get 0 (same grid as ticker_day_sentiment - one row per ticker per day).

    Returns a long DataFrame [ticker, date, n_headlines].
    """
    cov = (
        panel.groupby(["ticker", "trading_day"], as_index=False)
        .size()
        .rename(columns={"trading_day": "date", "size": "n_headlines"})
    )

    tickers = sorted(panel["ticker"].unique())
    full_index = pd.MultiIndex.from_product(
        [tickers, pd.DatetimeIndex(calendar_dates).sort_values()],
        names=["ticker", "date"],
    )
    cov = (
        cov.set_index(["ticker", "date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return cov.sort_values(["ticker", "date"]).reset_index(drop=True)


def coverage_weighted_sector_sentiment_index(
    ticker_day: pd.DataFrame, coverage: pd.DataFrame, sector_map: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    """Coverage-weighted sector sentiment index from ticker-day scores.

    The number of headlines backing each ticker-day reading is used as a
    confidence weight when aggregating to the sector index:

        index(s, t) = sum_i n(i, t) * score(i, t) / sum_i n(i, t)

    A ticker-day reading backed by more headlines is treated as more
    trustworthy and contributes proportionally more to its sector. The result
    is then lagged `lag` trading days exactly like the equal-weight index (the
    lag is a per-sector date shift applied after aggregation, so it commutes
    with any per-date weighting - same no-look-ahead guarantee).

    Differences vs the equal-weight index (deliberate, documented in the
    report):
    - Zero-headline ticker-days get weight 0 -> excluded (no evidence -> no
      contribution), instead of counting as neutral 0.0 with full weight.
    - A sector-day with NO headlines at all has no reading (NaN) and is
      dropped after the lag. Downstream fusion's forward-fill then holds the
      last trustworthy reading instead of manufacturing a false neutral.

    coverage: DataFrame from ticker_day_coverage (ticker, date, n_headlines).

    Returns a long DataFrame [date, sector, vader_sentiment, finvader_sentiment].
    """
    df = (
        ticker_day.merge(sector_map, on="ticker", how="left")
        .merge(coverage, on=["ticker", "date"], how="left")
    )
    df["n_headlines"] = df["n_headlines"].fillna(0)

    score_cols = ["vader_sentiment", "finvader_sentiment"]
    for col in score_cols:
        df[col + "_weighted"] = df[col] * df["n_headlines"]

    weighted_sum = (
        df.groupby(["date", "sector"], as_index=False)[
            [c + "_weighted" for c in score_cols]
        ]
        .sum()
        .rename(columns={c + "_weighted": c for c in score_cols})
    )
    total_coverage = (
        df.groupby(["date", "sector"], as_index=False)["n_headlines"]
        .sum()
        .rename(columns={"n_headlines": "n_headlines_total"})
    )
    idx = weighted_sum.merge(total_coverage, on=["date", "sector"])

    for col in score_cols:
        idx[col] = idx[col] / idx["n_headlines_total"].replace(0.0, pd.NA)
    idx = idx.drop(columns=["n_headlines_total"]).sort_values(["sector", "date"])

    if lag > 0:
        idx[score_cols] = idx.groupby("sector")[score_cols].shift(lag)
        idx = idx.dropna(subset=score_cols)

    return idx.reset_index(drop=True)


def build_sector_sentiment_index(calendar_dates: pd.DatetimeIndex,
                                 fill_missing: str = "neutral", lag: int = 1,
                                 ) -> pd.DataFrame:
    """End-to-end convenience: clean news, align to trading days, score with
    VADER + FinVADER, aggregate to ticker-day then to a lagged sector index.

    Uses the same Part A building blocks (etl.clean_news, etl.build_text_panel)
    so the headline alignment is identical to Project A.
    """
    raw = etl.load_raw()
    news_clean, _ = etl.clean_news(raw["news"])
    panel = etl.build_text_panel(news_clean, calendar_dates)
    scored = score_headlines(panel)

    equity = raw["equity"]
    sector_map = equity[["ticker", "sector"]].drop_duplicates()

    ticker_day = ticker_day_sentiment(scored, calendar_dates, fill_missing=fill_missing)
    return sector_sentiment_index(ticker_day, sector_map, lag=lag)
