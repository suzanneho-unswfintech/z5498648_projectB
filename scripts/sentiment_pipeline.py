"""Baseline sentiment pipeline (Station 3, model step only).

Scores the Part A trading-day-aligned headline panel with VADER and FinVADER,
aggregates to ticker-day and then to a lagged equal-weight sector index, and
writes the required app artifact plus comparison tables and figures.

Run from the project root:

    python scripts/sentiment_pipeline.py

Outputs:
  results/data/sector_sentiment_index.csv       (required filename, app reads it)
  results/data/ticker_day_sentiment.csv         (per ticker-day, for analysis/fusion)
  results/data/coverage_weighted_sentiment_index.csv  (third index, for fusion)
  results/tables/sentiment_model_comparison.csv
  results/tables/sector_sentiment_summary.csv
  results/figures/sentiment_sector_index_over_time_vader.png   (FT style)
  results/figures/sentiment_sector_index_over_time_finvader.png (FT style)
  results/figures/sentiment_sector_grid_finvader.png  (2x5 grid, publication)
  results/figures/sentiment_vader_vs_finvader.png              (FT style)
  results/figures/sentiment_by_sector.png                      (FT style)

Assumptions and choices (documented for the report):
1. No-headline ticker-days are treated as NEUTRAL (0.0), not dropped or carried
   forward: absence of news is its own state, and carrying stale headlines
   forward would make the index sticky and double-count old information. With
   neutral fill the index is defined on the full equity trading calendar.
2. The signal is lagged ONE TRADING DAY after sector averaging: the value on
   trading day t uses only headlines aligned to t-1 or earlier. A Saturday or
   Monday headline (both aligned to Monday) first becomes usable on Tuesday.
   Because an equal-weight mean commutes with a lag, applying the lag after the
   sector mean is equivalent to applying it to each ticker first.
3. Ticker-day scores are the simple mean of that ticker's headline compounds.
   Headlines are noisy and there is no weighting scheme implied by the data, so
   equal weighting per headline within a ticker-day is the neutral choice.
4. Sector index equal-weights the 5 constituent tickers, per the brief.
5. Compound scores are used ([-1, 1] with sign = direction), the same scale for
   both models so they are directly comparable.
6. No portfolio fusion here: this is the standalone sentiment index only.
7. A third, COVERAGE-WEIGHTED sector index is also produced: each ticker's
   daily reading enters its sector index weighted by the number of headlines
   backing it (headline count = confidence). Zero-headline ticker-days are
   excluded (weight 0), so unlike the equal-weight index a sector-day with no
   news at all has no reading - fusion's forward-fill then holds the last
   trustworthy reading instead of a false neutral. Same one-day lag, same
   two model columns as the equal-weight index.
"""
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import etl
from src import plotting as ft
from src import sentiment as sent

matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "data"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

SECTOR_ORDER = ["Tech", "Financials", "Energy", "Consumer", "Industrials",
                "Healthcare", "Comm", "Materials", "Utilities", "RealEstate"]

SECTOR_DISPLAY = {
    "Comm": "Communication Services",
    "Consumer": "Consumer",
    "Energy": "Energy",
    "Financials": "Financials",
    "Healthcare": "Healthcare",
    "Industrials": "Industrials",
    "Materials": "Materials",
    "RealEstate": "Real Estate",
    "Tech": "Technology",
    "Utilities": "Utilities",
}

SECTOR_GRID_ORDER = [
    "Comm", "Consumer", "Energy", "Financials", "Healthcare",
    "Industrials", "Materials", "RealEstate", "Tech", "Utilities",
]


def save_outputs(sector_index: pd.DataFrame, coverage_index: pd.DataFrame,
                 ticker_day: pd.DataFrame, scored: pd.DataFrame,
                 sector_map: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- app-readable artifacts -------------------------------------------------
    sector_index.to_csv(DATA_DIR / "sector_sentiment_index.csv", index=False)
    coverage_index.to_csv(DATA_DIR / "coverage_weighted_sentiment_index.csv", index=False)
    ticker_day.to_csv(DATA_DIR / "ticker_day_sentiment.csv", index=False)

    # --- comparison tables ------------------------------------------------------
    headline_cmp = headline_comparison_table(scored)
    headline_cmp.to_csv(TABLE_DIR / "sentiment_model_comparison.csv", index=False)
    sector_summary = sector_summary_table(sector_index)
    sector_summary.to_csv(TABLE_DIR / "sector_sentiment_summary.csv", index=False)

    # --- figures ---------------------------------------------------------------
    fig_over_time(sector_index)
    fig_sector_grid_finvader(sector_index)
    fig_scatter(ticker_day)
    fig_by_sector(sector_summary)

    print(f"saved {DATA_DIR / 'sector_sentiment_index.csv'}")
    print(f"saved {DATA_DIR / 'coverage_weighted_sentiment_index.csv'}")
    print(f"saved {DATA_DIR / 'ticker_day_sentiment.csv'}")
    print(f"saved {TABLE_DIR / 'sentiment_model_comparison.csv'}")
    print(f"saved {TABLE_DIR / 'sector_sentiment_summary.csv'}")
    print(f"saved {FIG_DIR / 'sentiment_sector_index_over_time_vader.png'}")
    print(f"saved {FIG_DIR / 'sentiment_sector_index_over_time_finvader.png'}")
    print(f"saved {FIG_DIR / 'sentiment_sector_grid_finvader.png'}")
    print(f"saved {FIG_DIR / 'sentiment_vader_vs_finvader.png'}")
    print(f"saved {FIG_DIR / 'sentiment_by_sector.png'}")


def headline_comparison_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-headline model comparison: how often each model is neutral, and how
    much FinVADER reclassifies VADER's neutrals into a signed signal."""
    n = len(scored)
    v = scored["vader_compound"]
    f = scored["finvader_compound"]

    vader_neutral = (v.abs() < 1e-9).mean()
    finvader_neutral = (f.abs() < 1e-9).mean()
    false_neutral_fixed = ((v.abs() < 1e-9) & (f.abs() >= 1e-9)).mean()
    flipped_sign = (v * f < 0).mean()

    rows = [
        {"statistic": "n_headlines", "vader": n, "finvader": n},
        {"statistic": "pct_neutral", "vader": round(vader_neutral, 4),
         "finvader": round(finvader_neutral, 4)},
        {"statistic": "pct_nonneutral", "vader": round(1 - vader_neutral, 4),
         "finvader": round(1 - finvader_neutral, 4)},
        {"statistic": "mean_compound", "vader": round(v.mean(), 4),
         "finvader": round(f.mean(), 4)},
        {"statistic": "std_compound", "vader": round(v.std(), 4),
         "finvader": round(f.std(), 4)},
    ]
    summary = pd.DataFrame(rows)
    corr = pd.DataFrame([{
        "statistic": "pearson_corr_vader_finvader",
        "vader": round(v.corr(f), 4), "finvader": round(v.corr(f), 4),
    }])
    reclass = pd.DataFrame([{
        "statistic": "vader_neutral_finvader_signed",
        "vader": round(false_neutral_fixed, 4), "finvader": round(false_neutral_fixed, 4),
    }, {
        "statistic": "sign_flip_between_models",
        "vader": round(flipped_sign, 4), "finvader": round(flipped_sign, 4),
    }])
    return pd.concat([summary, corr, reclass], ignore_index=True)


def sector_summary_table(sector_index: pd.DataFrame) -> pd.DataFrame:
    """Per-sector summary of the LAGGED index: mean, std, and % positive days."""
    out = []
    for sector, g in sector_index.groupby("sector"):
        out.append({
            "sector": sector,
            "n_days": int(g["vader_sentiment"].notna().sum()),
            "vader_mean": round(g["vader_sentiment"].mean(), 4),
            "vader_std": round(g["vader_sentiment"].std(), 4),
            "vader_pct_positive": round((g["vader_sentiment"] > 0).mean(), 4),
            "finvader_mean": round(g["finvader_sentiment"].mean(), 4),
            "finvader_std": round(g["finvader_sentiment"].std(), 4),
            "finvader_pct_positive": round((g["finvader_sentiment"] > 0).mean(), 4),
        })
    order = {s: i for i, s in enumerate(SECTOR_ORDER)}
    return (pd.DataFrame(out)
              .sort_values("sector", key=lambda s: s.map(order))
              .reset_index(drop=True))


def fig_over_time(sector_index: pd.DataFrame) -> None:
    """Lagged sector index over time, one FT PNG per model.

    Ten sector lines per model - a compact borderless FT legend sits BELOW
    the chart (loc='upper left' would cover the lines, which all hover
    around zero in the early part of the window).
    """
    for col, model, label in [
            ("vader_sentiment", "vader", "VADER"),
            ("finvader_sentiment", "finvader", "FinVADER")]:
        fig, ax = ft.ft_figure(figsize=(9.4, 5.0))
        for i, (sector, g) in enumerate(sector_index.groupby("sector")):
            ax.plot(g["date"], g[col],
                    color=ft.FT_PALETTE[i % len(ft.FT_PALETTE)], lw=1.1,
                    label=sector)
        ax.axhline(0, color=ft.FT_TEXT, lw=0.8, ls="--")
        leg = ax.legend(ncol=5, fontsize=7.5, loc="upper center",
                        bbox_to_anchor=(0.5, -0.10), frameon=False)
        for t in leg.get_texts():
            t.set_color(ft.FT_TEXT)
        ft.ft_title(ax, f"Sector sentiment index ({label}), lagged 1 day "
                        "(2020-2023)")
        ax.set_ylabel("Sentiment index (lagged 1 day)",
                      color=ft.FT_TEXT, fontsize=9)
        fig.savefig(FIG_DIR / f"sentiment_sector_index_over_time_{model}.png",
                    dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)


def fig_sector_grid_finvader(sector_index: pd.DataFrame,
                             window: int = 21) -> None:
    """Publication 2x5 grid: FinVADER lagged sector index, 21-day average.

    One small panel per equity sector (2 rows x 5 columns), all sharing the
    same symmetric y-limits so every sector reads on one scale, a horizontal
    line at sentiment = 0 per panel, and years on the x-axis. The dark
    burgundy line on the cream report background is the FT_PALETTE[0]/FT_BG
    pairing used across the report. Plots the 21-day rolling average of the
    pipeline's lagged FinVADER index; the purpose is to show how sentiment
    evolves across all equity sectors over 2020-2023, not to compare
    portfolio performance.
    """
    import matplotlib.dates as mdates

    series = {}
    for sector, g in sector_index.groupby("sector"):
        g = g.sort_values("date")
        s = pd.Series(g["finvader_sentiment"].values,
                      index=pd.to_datetime(g["date"]))
        series[sector] = s.rolling(window, min_periods=1).mean()

    span = max(pd.concat(series.values()).abs().max(), 0.05) * 1.08

    fig, axes = plt.subplots(2, 5, figsize=(14, 7.2))
    fig.patch.set_facecolor(ft.FT_BG)
    for ax in axes.flat:
        ax.set_facecolor(ft.FT_BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, colors=ft.FT_TEXT, labelsize=8)
        ax.yaxis.grid(True, color=ft.FT_GRID, linewidth=0.6)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_ylim(-span, span)
        ax.axhline(0, color=ft.FT_TEXT, lw=0.8, alpha=0.55)

    for ax, sector in zip(axes.flat, SECTOR_GRID_ORDER):
        s = series[sector]
        ax.plot(s.index, s.values, color=ft.FT_PALETTE[0], lw=1.1)
        ax.set_title(SECTOR_DISPLAY[sector], fontsize=9.5, color=ft.FT_TEXT,
                     loc="left", pad=4)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xticks(pd.date_range("2020-01-01", periods=4, freq="YS"))

    fig.suptitle("Sector sentiment over time (FinVADER, 21-day average, "
                 "2020\u20132023)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=ft.FT_TEXT)
    fig.supylabel("Sentiment index (21-day average of the lagged index)",
                  fontsize=9, color=ft.FT_TEXT)
    fig.subplots_adjust(hspace=0.24, wspace=0.20, top=0.88, bottom=0.07,
                        left=0.07, right=0.985)
    fig.savefig(FIG_DIR / "sentiment_sector_grid_finvader.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_scatter(ticker_day: pd.DataFrame) -> None:
    """Ticker-day VADER vs FinVADER: hexbin + difference histogram, FT frame."""
    v = ticker_day["vader_sentiment"]
    f = ticker_day["finvader_sentiment"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(ft.FT_BG)
    for ax in axes:
        ax.set_facecolor(ft.FT_BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, colors=ft.FT_TEXT, labelsize=9)
    hb = axes[0].hexbin(v, f, gridsize=40, cmap="viridis", mincnt=1,
                        extent=(-1, 1, -1, 1))
    axes[0].plot([-1, 1], [-1, 1], color=ft.FT_TEXT, ls="--", lw=1,
                 label="identity")
    axes[0].set_xlabel("VADER compound (ticker-day)", color=ft.FT_TEXT, fontsize=9)
    axes[0].set_ylabel("FinVADER compound (ticker-day)", color=ft.FT_TEXT, fontsize=9)
    ft.ft_title(axes[0], f"Ticker-day scores, r = {v.corr(f):.3f}")
    leg = axes[0].legend(frameon=False)
    for t in leg.get_texts():
        t.set_color(ft.FT_TEXT)
    fig.colorbar(hb, ax=axes[0], label="ticker-days")
    axes[1].hist(f - v, bins=60, color=ft.FT_PALETTE[1], alpha=0.85)
    axes[1].axvline(0, color=ft.FT_TEXT, ls="--", lw=0.8)
    axes[1].set_xlabel("FinVADER - VADER (ticker-day)", color=ft.FT_TEXT, fontsize=9)
    axes[1].set_ylabel("ticker-days", color=ft.FT_TEXT, fontsize=9)
    ft.ft_title(axes[1], "Score differences: finance lexicon vs plain VADER")
    fig.suptitle("VADER vs FinVADER on ticker-day sentiment (2020-2023)",
                 color=ft.FT_TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sentiment_vader_vs_finvader.png", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_by_sector(sector_summary: pd.DataFrame) -> None:
    """Grouped bar of mean lagged index by sector, VADER vs FinVADER (FT)."""
    fig, ax = ft.ft_figure(figsize=(9.4, 5.0))
    x = np.arange(len(sector_summary))
    width = 0.38
    ax.bar(x - width / 2, sector_summary["vader_mean"], width,
           label="VADER", color=ft.FT_PALETTE[0])
    ax.bar(x + width / 2, sector_summary["finvader_mean"], width,
           label="FinVADER", color=ft.FT_PALETTE[1])
    ax.axhline(0, color=ft.FT_TEXT, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sector_summary["sector"], rotation=30, ha="right",
                       fontsize=8, color=ft.FT_TEXT)
    ax.set_ylabel("Mean lagged sentiment index", color=ft.FT_TEXT, fontsize=9)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=False)
    for t in leg.get_texts():
        t.set_color(ft.FT_TEXT)
    ft.ft_title(ax, "Mean sector sentiment by model (lagged index, 2020-2023)")
    fig.savefig(FIG_DIR / "sentiment_by_sector.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    raw = etl.load_raw()
    equity_clean, _ = etl.clean_equity(raw["equity"])
    trading_days = pd.DatetimeIndex(equity_clean["date"].unique()).sort_values()

    print(f"equity trading days: {len(trading_days)} "
          f"({trading_days.min().date()} to {trading_days.max().date()})")

    sector_map = equity_clean[["ticker", "sector"]].drop_duplicates()
    print("building trading-day-aligned headline panel (reuses Part A etl)...")
    panel = etl.build_text_panel(etl.clean_news(raw["news"])[0], trading_days)
    print(f"panel: {panel.shape} headlines aligned to "
          f"{panel['trading_day'].nunique()} trading days")

    print("scoring headlines with VADER and FinVADER (may take ~30s)...")
    scored = sent.score_headlines(panel)

    ticker_day = sent.ticker_day_sentiment(scored, trading_days, fill_missing="neutral")
    print(f"ticker-day panel: {ticker_day.shape} "
          f"({ticker_day['ticker'].nunique()} tickers x {ticker_day['date'].nunique()} days)")

    sector_index = sent.sector_sentiment_index(ticker_day, sector_map, lag=1)
    print(f"sector index: {sector_index.shape} rows, {sector_index['sector'].nunique()} sectors, "
          f"lagged 1 trading day, first date {sector_index['date'].min().date()}")

    coverage = sent.ticker_day_coverage(panel, trading_days)
    coverage_index = sent.coverage_weighted_sector_sentiment_index(
        ticker_day, coverage, sector_map, lag=1)
    print(f"coverage-weighted index: {coverage_index.shape} rows, "
          f"first date {coverage_index['date'].min().date()}")

    save_outputs(sector_index, coverage_index, ticker_day, scored, sector_map)
    print("done.")


if __name__ == "__main__":
    main()
