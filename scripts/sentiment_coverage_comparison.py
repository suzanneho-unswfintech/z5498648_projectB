"""Equal-weight vs coverage-weighted sector sentiment index diagnostic.

Rebuilds the sentiment pipeline from raw data, then compares the two sector
indices with a focus on the three thin sectors the DATA_GUIDE flags as having
sparser news coverage (Materials, Utilities, RealEstate). Also checks grid
completeness of the pre-lag sector panel and whether the positional lag on the
coverage-weighted index ever skips trading days.

Run from the project root:

    python scripts/sentiment_coverage_comparison.py

Outputs:
  results/tables/sentiment_equal_vs_coverage.csv    (per sector x model stats)
  results/tables/sector_headline_coverage.csv       (per sector headline counts)
  results/figures/sentiment_equal_vs_coverage_thin_sectors.png  (2x3 grid)
"""
import itertools
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import etl
from src import plotting as ft
from src import sentiment as sent

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
THIN_SECTORS = ["Materials", "Utilities", "RealEstate"]


def trading_day_positions(trading_days: pd.DatetimeIndex) -> dict:
    return {d: i for i, d in enumerate(trading_days)}


def _rolling_series(frame: pd.DataFrame, sector: str, col: str,
                    window: int) -> pd.Series:
    """21-day rolling mean of one sector's lagged index column, indexed by date."""
    g = frame[frame["sector"] == sector].sort_values("date")
    s = pd.Series(g[col].values, index=pd.to_datetime(g["date"]))
    return s.rolling(window, min_periods=1).mean()


def fig_equal_vs_coverage(eq_idx: pd.DataFrame, cov_idx: pd.DataFrame,
                          window: int = 21) -> pathlib.Path:
    """2x3 publication grid: equal-weight vs coverage-weighted lagged index.

    Rows = VADER / FinVADER, columns = the three thin sectors (Materials,
    Utilities, Real Estate). Plots the 21-day rolling average of the lagged
    index so the two weighting schemes read on one scale; all panels share
    symmetric y-limits, a y=0 line, and the report's FT cream styling.
    """
    import matplotlib.dates as mdates

    models = [("vader_sentiment", "VADER"), ("finvader_sentiment", "FinVADER")]
    series = {
        (col, sector, kind): _rolling_series(idx, sector, col, window)
        for kind, idx in [("equal", eq_idx), ("coverage", cov_idx)]
        for (col, model) in models
        for sector in THIN_SECTORS
    }
    span = max(max(s.abs().max(), 0.05) for s in series.values()) * 1.08

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8))
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

    for row, (col, model) in enumerate(models):
        for ax, sector in zip(axes[row], THIN_SECTORS):
            ax.plot(series[(col, sector, "equal")].index,
                    series[(col, sector, "equal")].values,
                    color=ft.FT_PALETTE[0], lw=1.1, label="Equal-weight")
            ax.plot(series[(col, sector, "coverage")].index,
                    series[(col, sector, "coverage")].values,
                    color=ft.FT_PALETTE[1], lw=1.1, label="Coverage-weighted")
            ax.set_title(f"{sector} ({model})", fontsize=9.5, color=ft.FT_TEXT,
                         loc="left", pad=4)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_xticks(pd.date_range("2020-01-01", periods=4, freq="YS"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03),
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Equal-weight vs coverage-weighted sector sentiment "
                 "(21-day average of the lagged index, 2020\u20132023)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold",
                 color=ft.FT_TEXT)
    fig.supylabel("Sentiment index (21-day average of the lagged index)",
                  fontsize=9, color=ft.FT_TEXT)
    fig.subplots_adjust(hspace=0.30, wspace=0.20, top=0.90, bottom=0.13,
                        left=0.07, right=0.985)
    out = FIG_DIR / "sentiment_equal_vs_coverage_thin_sectors.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main() -> None:
    raw = etl.load_raw()
    equity_clean, _ = etl.clean_equity(raw["equity"])
    trading_days = pd.DatetimeIndex(equity_clean["date"].unique()).sort_values()
    positions = trading_day_positions(trading_days)
    sector_map = equity_clean[["ticker", "sector"]].drop_duplicates()
    sectors = sorted(sector_map["sector"].unique())

    news_clean, _ = etl.clean_news(raw["news"])
    panel = etl.build_text_panel(news_clean, trading_days)
    print(f"headline panel: {panel.shape[0]} headlines, "
          f"{panel['trading_day'].nunique()} trading days")

    scored = sent.score_headlines(panel)
    ticker_day = sent.ticker_day_sentiment(scored, trading_days, fill_missing="neutral")
    print(f"ticker-day panel: {ticker_day.shape}, columns {list(ticker_day.columns)}")

    # --- 1. pre-lag grid completeness of the equal-weight sector aggregation ---
    pre = (
        ticker_day.merge(sector_map, on="ticker", how="left")
        .groupby(["date", "sector"], as_index=False)[
            ["vader_sentiment", "finvader_sentiment"]
        ]
        .mean()
    )
    full_grid = pd.MultiIndex.from_product(
        [trading_days, sectors], names=["date", "sector"]
    )
    pre_grid = pd.MultiIndex.from_frame(pre[["date", "sector"]])
    missing = full_grid.difference(pre_grid)
    print(f"\npre-lag sector grid: {len(pre_grid)} of {len(full_grid)} "
          f"expected ({len(missing)} missing)")
    if len(missing):
        counts = missing.to_frame().groupby("sector").size()
        print("missing (date, sector) combos by sector:")
        print(counts.to_string())
        counts.to_csv(TABLE_DIR / "sector_grid_missing.csv")

    # --- 2. coverage-weighted index (lag=0 pre-lag reading grid) ---
    coverage = sent.ticker_day_coverage(panel, trading_days)
    cov_pre = sent.coverage_weighted_sector_sentiment_index(
        ticker_day, coverage, sector_map, lag=0
    )
    print(f"coverage index pre-lag readings: {len(cov_pre)} "
          f"of {len(full_grid)} sector-days")

    # 2a. effective lag of the positional shift on the gapped coverage grid
    lag_rows = []
    for sector, g in cov_pre.groupby("sector"):
        dates = sorted(g["date"])
        for prev, cur in itertools.pairwise(dates):
            eff_lag = positions[cur] - positions[prev]
            if eff_lag > 1:
                lag_rows.append({"sector": sector, "date": cur,
                                 "effective_lag_days": eff_lag})
    lag_df = pd.DataFrame(lag_rows)
    if len(lag_df):
        print("\npositional lag on the gapped coverage grid (effective lag > 1 "
              "trading day):")
        print(lag_df.groupby("sector")["effective_lag_days"].agg(
            ["count", "max"]).to_string())
        print("-> thin sectors only:")
        print(lag_df[lag_df["sector"].isin(THIN_SECTORS)]
              .groupby("sector")["effective_lag_days"].agg(["count", "max"])
              .to_string())

    # --- 3. post-lag indices as the pipeline produces them ---
    eq_idx = sent.sector_sentiment_index(ticker_day, sector_map, lag=1)
    cov_idx = sent.coverage_weighted_sector_sentiment_index(
        ticker_day, coverage, sector_map, lag=1
    )
    print(f"\nequal-weight index: {len(eq_idx)} rows | "
          f"coverage-weighted index: {len(cov_idx)} rows")

    # 3a. headline coverage per sector (for the report)
    hl = (
        panel.groupby(["sector", "trading_day"], as_index=False)
        .size()
        .groupby("sector")["size"]
        .agg(total_headlines="sum", n_days_with_news="count")
        .reset_index()
    )
    hl["n_days"] = len(trading_days)
    hl["pct_days_with_news"] = (hl["n_days_with_news"] / hl["n_days"]).round(4)
    hl["pct_of_headlines"] = (hl["total_headlines"] / hl["total_headlines"].sum()).round(4)
    hl = hl.sort_values("total_headlines", ascending=False).reset_index(drop=True)
    print("\nheadline coverage by sector:")
    print(hl.to_string(index=False))
    hl.to_csv(TABLE_DIR / "sector_headline_coverage.csv", index=False)

    # 3b. equal vs coverage stats per sector x model
    models = ["vader_sentiment", "finvader_sentiment"]
    rows = []
    for sector in sectors:
        a = eq_idx[eq_idx["sector"] == sector][["date", *models]].sort_values("date")
        b = cov_idx[cov_idx["sector"] == sector][["date", *models]].sort_values("date")
        m = a.merge(b, on="date", suffixes=("_eq", "_cov"))
        for col in models:
            x = m[f"{col}_eq"].astype(float)
            y = m[f"{col}_cov"].astype(float)
            diff = (y - x).abs()
            rows.append({
                "sector": sector,
                "model": col.replace("_sentiment", ""),
                "eq_n_days": len(a),
                "cov_n_days": len(b),
                "shared_n_days": len(m),
                "corr": round(x.corr(y), 4),
                "eq_mean": round(x.mean(), 4),
                "cov_mean": round(y.mean(), 4),
                "eq_std": round(x.std(), 4),
                "cov_std": round(y.std(), 4),
                "mean_abs_diff": round(diff.mean(), 4),
                "rmse": round(np.sqrt((diff ** 2).mean()), 4),
                "max_abs_diff": round(diff.max(), 4),
                "sign_flip_pct": round((np.sign(x) != np.sign(y)).mean(), 4),
            })
    stats = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(TABLE_DIR / "sentiment_equal_vs_coverage.csv", index=False)

    print("\nequal-weight vs coverage-weighted comparison (all sectors):")
    print(stats.to_string(index=False))
    print("\n-- thin sectors only --")
    thin = stats[stats["sector"].isin(THIN_SECTORS)]
    print(thin.to_string(index=False))

    fig_path = fig_equal_vs_coverage(eq_idx, cov_idx)
    print(f"\nsaved {TABLE_DIR / 'sentiment_equal_vs_coverage.csv'}")
    print(f"saved {TABLE_DIR / 'sector_headline_coverage.csv'}")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
