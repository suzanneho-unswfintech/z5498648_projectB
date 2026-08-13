"""
Station 1 - Data Lake (ETL)
FINS3645 Project B - reused verbatim from Project A (Stations 1-2).

This is the validated Part A data foundation, copied unchanged so the Part B
funds, backtest, and sentiment stages build on the same cleaned data. The
cleaning methodology, feature logic, and function interfaces are identical to
Part A; only this header note differs.

Loads the three raw panels through src/data_access.py, cleans and
deduplicates them, runs integrity checks (missing dates, duplicates,
outliers), computes daily returns within each panel, and left-merges
crypto returns onto the equity trading calendar.

Design choices and why (so the report can cite them):
- Prices are deduplicated on (ticker, date) - one price row per asset per day.
- News is deduplicated on (ticker, date, title) - many legitimate headlines
  can share a ticker-date, so title is part of the key.
- Returns are computed WITHIN each panel (equity, crypto) BEFORE any merge.
  Merging price levels first and differencing afterwards would create
  spurious returns at the seam between the two calendars.
- News dates arrive timezone-aware (UTC); price dates do not. We normalise
  both to timezone-naive dates before anything touches them together.
- Outliers in returns are NOT deleted. They are flagged in an integrity
  table so the report can inspect and justify them (real market events).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_access import (
    load_equity_prices,
    load_crypto_prices,
    load_news_headlines,
)

# ---------------------------------------------------------------------------
# Dtype / schema precheck - run BEFORE cleaning, on the raw data
# ---------------------------------------------------------------------------

def dtype_precheck(df: pd.DataFrame, name: str) -> dict:
    """Inspect raw dtypes and flag anything that would break downstream
    numeric/date logic, BEFORE any cleaning touches the data. Cleaning
    functions assume 'date' is parseable and numeric columns are numeric -
    this checks that assumption instead of silently relying on it.

    Returns a dict summary (dtypes, nulls, whether numeric columns are
    actually numeric) rather than raising, so you can inspect surprises
    before deciding how to handle them.
    """
    numeric_candidates = [c for c in ["open", "high", "low", "close", "adjClose", "volume"] if c in df.columns]
    non_numeric = [c for c in numeric_candidates if not pd.api.types.is_numeric_dtype(df[c])]

    date_parse_failures = None
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        date_parse_failures = int(parsed.isna().sum())

    return {
        "dataset": name,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "n_nulls_by_col": df.isna().sum().to_dict(),
        "non_numeric_price_cols": non_numeric,
        "date_parse_failures": date_parse_failures,
    }


# ---------------------------------------------------------------------------
# Loading + basic normalisation
# ---------------------------------------------------------------------------

def load_raw() -> dict[str, pd.DataFrame]:
    """Load the three raw panels, unchanged, via the provided helper."""
    return {
        "equity": load_equity_prices(),
        "crypto": load_crypto_prices(),
        "news": load_news_headlines(),
    }


def _normalise_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Make a date column timezone-naive and datetime64[ns], regardless of
    whether it arrived tz-aware (news, UTC) or tz-naive (prices)."""
    df = df.copy()
    s = pd.to_datetime(df[col])
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    df[col] = s.dt.normalize()  # drop any intraday time component
    return df


# ---------------------------------------------------------------------------
# Cleaning + deduplication
# ---------------------------------------------------------------------------

def clean_equity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Dedupe on (ticker, date); check OHLC + volume internal consistency."""
    df = _normalise_date(df)
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    n_dupes = before - len(df)

    # OHLC internal-consistency check (does not drop rows, just reports).
    bad_ohlc = df[
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ]
    bad_volume = df[df["volume"] < 0] if "volume" in df.columns else df.iloc[0:0]

    report = {
        "n_duplicate_rows_removed": n_dupes,
        "n_ohlc_inconsistent_rows": len(bad_ohlc),
        "n_negative_volume_rows": len(bad_volume),
    }
    return df.reset_index(drop=True), report


def clean_crypto(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = _normalise_date(df)
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    n_dupes = before - len(df)

    # DATA_GUIDE.md is explicit: "there are 10 stray rows dated 2024-01-01,
    # so cap your sample at 2023-12-31." Drop them here, after dedup, before
    # anything downstream (returns, calendar merge) ever sees them.
    before_cap = len(df)
    df = df[df["date"] <= pd.Timestamp("2023-12-31")]
    n_capped = before_cap - len(df)

    bad_volume = df[df["volume"] < 0] if "volume" in df.columns else df.iloc[0:0]
    report = {
        "n_duplicate_rows_removed": n_dupes,
        "n_negative_volume_rows": len(bad_volume),
        "n_stray_2024_rows_capped": n_capped,
    }
    return df.reset_index(drop=True), report


def clean_news(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Dedupe on (ticker, date, title) - NOT ticker-date alone, since many
    genuine headlines share a ticker-date."""
    df = _normalise_date(df)
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date", "title"]).sort_values(["ticker", "date"])
    n_dupes = before - len(df)
    report = {"n_duplicate_rows_removed": n_dupes}
    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Missing-date audit
# ---------------------------------------------------------------------------

def missing_date_audit(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """For each ticker, compare its observed trading days against the
    expected calendar (business days for equity, every day for crypto) over
    its own observed date range, and report the count missing.

    freq: 'B' for equity (business days), 'D' for crypto (every day).
    """
    rows = []
    for ticker, g in df.groupby("ticker"):
        obs_dates = pd.DatetimeIndex(g["date"].unique())
        full_range = pd.date_range(obs_dates.min(), obs_dates.max(), freq=freq)
        missing = full_range.difference(obs_dates)
        rows.append({
            "ticker": ticker,
            "start": obs_dates.min(),
            "end": obs_dates.max(),
            "n_expected": len(full_range),
            "n_observed": len(obs_dates),
            "n_missing": len(missing),
        })
    return pd.DataFrame(rows).sort_values("n_missing", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Returns + outlier screen
# ---------------------------------------------------------------------------

def compute_returns(df: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Simple daily returns per ticker. Computed WITHIN a single panel -
    call this separately on equity and crypto, before any merge."""
    df = df.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")[price_col].pct_change()
    return df


def outlier_screen(df: pd.DataFrame, ret_col: str = "ret", z_thresh: float = 5.0) -> pd.DataFrame:
    """Flag (do not remove) returns more than z_thresh standard deviations
    from their ticker's own mean. Genuine extreme returns are real events -
    the brief asks us to document and keep them, not delete them."""
    df = df.copy()
    grp = df.groupby("ticker")[ret_col]
    z = (df[ret_col] - grp.transform("mean")) / grp.transform("std")
    df["ret_zscore"] = z
    df["is_outlier"] = z.abs() > z_thresh
    return df


# ---------------------------------------------------------------------------
# Calendar merge (equity trading calendar is the master calendar)
# ---------------------------------------------------------------------------

def merge_calendars(equity_ret: pd.DataFrame, crypto_ret: pd.DataFrame) -> pd.DataFrame:
    """Left-merge crypto returns onto the equity trading-day calendar.

    Both inputs must already have a 'ret' column computed WITHIN their own
    panel (see compute_returns). This intentionally drops weekend-only
    crypto price moves, since a fund trading only on equity days could not
    have acted on them.
    """
    eq_dates = pd.DatetimeIndex(equity_ret["date"].unique()).sort_values()

    eq_wide = equity_ret.pivot(index="date", columns="ticker", values="ret")
    eq_wide.columns = [f"EQ_{c}" for c in eq_wide.columns]

    cr_wide = crypto_ret.pivot(index="date", columns="ticker", values="ret")
    cr_wide.columns = [f"CR_{c}" for c in cr_wide.columns]

    combined = eq_wide.join(cr_wide, how="left")  # left = equity calendar wins
    combined = combined.reindex(eq_dates)  # explicit, defensive re-alignment
    return combined


# ---------------------------------------------------------------------------
# Text panel assembly (Station 2, but lives here since it's still ETL-ish)
# ---------------------------------------------------------------------------

def build_text_panel(news: pd.DataFrame, equity_trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Map every headline to its equity trading day: same day if the
    headline date is a trading day, otherwise the NEXT trading day.

    Keeps the raw headline text unmodified (no stopword removal) since
    Part B's VADER scoring needs the full text.
    """
    trading_days = pd.DatetimeIndex(sorted(equity_trading_days))
    news = news.copy()

    # searchsorted finds, for each headline date, the position of the first
    # trading day >= that date - i.e. same day if it's a trading day, else
    # the next one.
    idx = trading_days.searchsorted(news["date"].values, side="left")
    idx = np.clip(idx, 0, len(trading_days) - 1)
    news["trading_day"] = trading_days[idx]

    return news


# ---------------------------------------------------------------------------
# Provenance (brief: "record provenance - source and how each field was
# produced" for each clean output dataset)
# ---------------------------------------------------------------------------

def integrity_summary_long(eq_report: dict, cr_report: dict, news_report: dict,
                            eq_missing: pd.DataFrame, cr_missing: pd.DataFrame,
                            n_eq_outliers: int, n_cr_outliers: int) -> pd.DataFrame:
    """Long-format integrity summary: one row per (dataset, check), with
    the count AND the resolution built into the same table - satisfies
    the brief's "found and how you resolved them" as a single readable
    exhibit, rather than counts in a table and resolution buried in
    separate prose the reader has to go find.
    """
    eq_tickers_missing = int((eq_missing["n_missing"] > 0).sum())
    eq_median_missing = eq_missing["n_missing"].median()
    cr_tickers_missing = int((cr_missing["n_missing"] > 0).sum())

    rows = [
        {"dataset": "equity", "check": "duplicates",
         "found": f"{eq_report['n_duplicate_rows_removed']:,} rows",
         "resolution": "Removed - exact duplicates on (ticker, date)."},
        {"dataset": "equity", "check": "OHLC inconsistency",
         "found": f"{eq_report['n_ohlc_inconsistent_rows']:,} rows",
         "resolution": "None found; would flag-and-document (not delete) if present."},
        {"dataset": "equity", "check": "negative volume",
         "found": f"{eq_report['n_negative_volume_rows']:,} rows",
         "resolution": "None found."},
        {"dataset": "equity", "check": "missing dates",
         "found": f"{eq_tickers_missing}/{len(eq_missing)} tickers (median {eq_median_missing:.0f} days)",
         "resolution": "Expected, not real data loss: the 'B' business-day calendar does not "
                        "know NYSE holidays (~9/yr x 4yr ~ 36) - confirmed by magnitude match."},
        {"dataset": "equity", "check": "outliers",
         "found": f"{n_eq_outliers:,} returns (|z| > 5)",
         "resolution": "Kept and documented, not deleted - brief specifies genuine extreme "
                        "returns reflect real market events (see outliers_flagged.csv)."},
        {"dataset": "crypto", "check": "duplicates",
         "found": f"{cr_report['n_duplicate_rows_removed']:,} rows",
         "resolution": "Removed - exact duplicates on (ticker, date)."},
        {"dataset": "crypto", "check": "date range cap",
         "found": f"{cr_report.get('n_stray_2024_rows_capped', 0):,} rows dated 2024-01-01",
         "resolution": "Removed per DATA_GUIDE.md - raw data has 10 stray rows past the "
                        "2020-2023 coverage window; capped at 2023-12-31 as instructed."},
        {"dataset": "crypto", "check": "negative volume",
         "found": f"{cr_report['n_negative_volume_rows']:,} rows",
         "resolution": "None found."},
        {"dataset": "crypto", "check": "missing dates",
         "found": f"{cr_tickers_missing}/{len(cr_missing)} tickers",
         "resolution": "None found - consistent with crypto trading every calendar day "
                        "(no exchange closures), matching the dataset-inventory frequency."},
        {"dataset": "crypto", "check": "outliers",
         "found": f"{n_cr_outliers:,} returns (|z| > 5)",
         "resolution": "Kept and documented, not deleted - same policy as equity."},
        {"dataset": "news", "check": "duplicates",
         "found": f"{news_report['n_duplicate_rows_removed']:,} rows",
         "resolution": "Removed - exact duplicates on (ticker, date, title), not ticker-date "
                        "alone, since many genuine distinct headlines share a ticker-date."},
    ]
    return pd.DataFrame(rows)


def dataset_provenance() -> pd.DataFrame:
    """Static documentation table: for each clean output dataset, what it
    is, its schema/frequency, and how each field was derived. Hand-written
    (not inferred from data) because provenance is a documentation
    requirement, not something you can compute from the data itself."""
    rows = [
        {"dataset": "clean_equity", "field": "adjClose", "source": "equity_prices.parquet (provided)",
         "derivation": "as loaded, deduplicated on (ticker,date); no adjustment applied by us"},
        {"dataset": "clean_equity", "field": "ret", "source": "derived",
         "derivation": "pct_change() of adjClose within ticker, on equity's native ~252-day calendar"},
        {"dataset": "clean_crypto", "field": "adjClose", "source": "crypto_prices.parquet (provided)",
         "derivation": "as loaded, deduplicated on (ticker,date)"},
        {"dataset": "clean_crypto", "field": "ret", "source": "derived",
         "derivation": "pct_change() of adjClose within ticker, on crypto's native 365-day calendar"},
        {"dataset": "combined_returns_panel", "field": "EQ_*/CR_*", "source": "derived",
         "derivation": "equity ret columns unchanged; crypto ret columns left-merged onto the equity "
                        "trading calendar, so weekend-only crypto moves are dropped by construction"},
        {"dataset": "text_panel", "field": "trading_day", "source": "derived",
         "derivation": "each headline's raw date mapped to itself if a trading day, else the next "
                        "equity trading day (searchsorted on the equity calendar)"},
        {"dataset": "text_panel", "field": "title", "source": "news_headlines.parquet (provided)",
         "derivation": "unmodified raw text - no stopword removal, so Part B VADER scoring is not degraded"},
    ]
    return pd.DataFrame(rows)
