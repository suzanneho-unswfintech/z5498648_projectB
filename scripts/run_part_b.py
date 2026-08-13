"""Reproduce all Part B results. Run from the project root:

    python scripts/run_part_b.py

Single entry point for every report/app artifact. Order of work:

1. Sentiment pipeline (Station 3 model step): VADER + FinVADER headline
   scoring, ticker-day panel, equal-weight and coverage-weighted sector
   indexes, comparison tables, and sentiment figures. Reuses the exact
   functions from scripts/sentiment_pipeline.py so there is one scoring path.
2. Walk-forward OOS backtests for 19 funds. Initial window 2020-01-01 to
   2022-12-31; out-of-sample is 2023. Funds are (family, method) pairs:
     - 12 BASELINE funds: 3 families (Equity, Crypto, Combined) x 4 methods
       (Equal Weight, Minimum Variance, Maximum Sharpe, Risk Parity)
     - 3 CVaR EXTENSION funds: one per family (Minimum CVaR), kept separate
       so the extension is never labelled a baseline method
     - 4 FUSED funds: Equity/Combined Minimum Variance tilted by VADER and
       by FinVADER. Fused returns are recomputed from the fused weights
       (weights held flat between monthly rebalances) - never copied from the
       unfused baseline series.
3. Transaction-cost layering (5 bps per unit of one-way turnover, charged on
   rebalance days) on every fund -> gross and net return series.
4. Required artifacts plus exhibits: fund_returns.csv, fund_weights.csv,
   performance_metrics.csv (gross AND net basis rows), fusion before/after
   table, lambda-sensitivity table, and the report figures.

Annualisation: sqrt(252) for equity and combined, sqrt(365) for crypto-only.
The combined universe keeps the 252 factor - the merged panel is on the
equity trading calendar (see src.portfolios module docstring).

Risk-free rate: 0 (see src.portfolios module docstring). Cost model: 5 bps
one-way, initial allocation trade excluded so every fund is charged for the
same thing (rebalancing).

Outputs (all under results/):
  data/fund_returns.csv                     [fund, date, gross_return, net_return]
  data/fund_weights.csv                     [fund, date, ticker, weight]
  data/sector_sentiment_index.csv           (regenerated, required app filename)
  data/ticker_day_sentiment.csv             (regenerated)
  data/coverage_weighted_sentiment_index.csv (regenerated)
  tables/performance_metrics.csv            [fund, fund_type, basis, n_days,
                                             ann_return, ann_vol, sharpe,
                                             max_drawdown, mean_turnover, total_cost]
  tables/fusion_before_after.csv
  tables/fusion_lambda_sensitivity.csv
  tables/sentiment_model_comparison.csv     (regenerated)
  tables/sector_sentiment_summary.csv       (regenerated)
  figures/growth_of_dollar_equity.png     (FT style, from Project A's plotting)
  figures/growth_of_dollar_crypto.png     (FT style)
  figures/growth_of_dollar_combined.png   (FT style)
  figures/drawdown_equity.png             (FT style)
  figures/drawdown_crypto.png             (FT style)
  figures/drawdown_combined.png           (FT style)
  figures/weights_over_time_combined_min_variance.png (FT style)
  figures/weights_over_time_combined_max_sharpe.png   (FT style)
  figures/sharpe_by_fund.png             (FT style, grouped by family)
  figures/sharpe_cost_combined.png       (FT style, combined gross vs net)
  figures/fusion_before_after_equity.png  (FT style, gross)
  figures/fusion_sharpe_equity.png        (FT style, gross vs net)
  figures/fusion_weight_tilt_equity.png   (FT style, weight-change tilt)
  figures/turnover_cost_equity.png        (FT style)
  figures/turnover_cost_crypto.png        (FT style)
  figures/turnover_cost_combined.png      (FT style)
  figures/sentiment_sector_index_over_time_vader.png   (regenerated, FT)
  figures/sentiment_sector_index_over_time_finvader.png (regenerated, FT)
  figures/sentiment_vader_vs_finvader.png (regenerated, FT)
  figures/sentiment_by_sector.png         (regenerated, FT)
"""
import importlib.util
import pathlib
import sys
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    etl,
    fusion,
)
from src import (  # noqa: E402
    plotting as ft,
)
from src import portfolios as pf  # noqa: E402
from src import sentiment as sent  # noqa: E402

# Load the sentiment pipeline as a module so this runner reuses its exact
# save_outputs()/figure functions instead of duplicating them.
_spec = importlib.util.spec_from_file_location(
    "sentiment_pipeline", ROOT / "scripts" / "sentiment_pipeline.py")
_sentpipe = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_sentpipe)

TRAIN_END = pf.TRAIN_END_DEFAULT

FAMILIES = ["equity", "crypto", "combined"]
BASE_METHODS = ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]
CVAR_METHOD = "cvar_min"
MODELS = ["vader_sentiment", "finvader_sentiment"]
TILT_STRENGTH = 0.5
SENSITIVITY_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
COST_RATE = 0.0005

FAMILY_LABELS = {"equity": "Equity", "crypto": "Crypto", "combined": "Combined"}
METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
    "cvar_min": "Minimum CVaR",
}
MODEL_LABELS = {"vader_sentiment": "VADER", "finvader_sentiment": "finVADER"}
PERIODS_PER_YEAR = {"equity": 252, "crypto": 365, "combined": 252}

COLORS = {"Equal Weight": "#7f7f7f", "Minimum Variance": "#1f77b4",
          "Maximum Sharpe": "#d62728", "Risk Parity": "#2ca02c",
          "Minimum CVaR": "#9467bd"}


def fund_name(family, method, model=None):
    label = f"{FAMILY_LABELS[family]} {METHOD_LABELS[method]}"
    if model is not None:
        label += f" + {MODEL_LABELS[model]}"
    return label


def run_one(returns, method, ticker_day=None, score_col=None,
            tilt_strength=TILT_STRENGTH):
    """Run the walk-forward backtest for one (family, method, model) fund.

    For a fused fund the monthly-rebalanced baseline weights are tilted by
    the lagged sentiment signal (apply_sentiment, look-ahead safe by
    construction), held flat between rebalances, and the fused daily returns
    are recomputed from those fused weights - the unfused baseline series is
    never reused.

    Returns a dict with gross/net daily returns, the weights frame (fused
    weights when score_col is given), and the turnover/cost series.
    """
    res = pf.oos_backtest(returns, method, train_end=TRAIN_END)
    gross = res["daily_returns"]
    weights = res["weights"]
    if score_col is not None:
        oos_dates = returns.index[returns.index > TRAIN_END]
        rebal = list(pf.get_rebalance_dates(oos_dates))
        if not rebal or rebal[0] != oos_dates[0]:
            rebal = [oos_dates[0], *rebal]
        rebal = pd.DatetimeIndex(rebal)
        fused_rebal = fusion.apply_sentiment(weights.loc[rebal], ticker_day,
                                             score_col=score_col,
                                             tilt_strength=tilt_strength)
        weights = fused_rebal.reindex(oos_dates, method="ffill")
        gross = (weights * returns.loc[oos_dates].fillna(0.0)).sum(axis=1)
    costs = pf.apply_transaction_costs(gross, weights, cost_rate=COST_RATE)
    return {
        "gross": gross,
        "net": costs["net_return"],
        "weights": weights,
        "turnover": costs["turnover"],
        "cost": costs["cost"],
    }


def build_funds(universes, ticker_day):
    """Run every fund and return records in report/app display order."""
    funds = []
    for fam in FAMILIES:
        returns = universes[fam]
        for m in BASE_METHODS:
            out = run_one(returns, m)
            funds.append({"name": fund_name(fam, m), "family": fam,
                          "method": m, "kind": "baseline", **out})
        out = run_one(returns, CVAR_METHOD)
        funds.append({"name": fund_name(fam, CVAR_METHOD), "family": fam,
                      "method": CVAR_METHOD, "kind": "extension", **out})
    for fam in ("equity", "combined"):
        for col in MODELS:
            out = run_one(universes[fam], "min_variance", ticker_day=ticker_day,
                          score_col=col)
            funds.append({"name": fund_name(fam, "min_variance", col),
                          "family": fam, "method": "min_variance",
                          "kind": "fusion", "model": col, **out})
    return funds


def build_metrics(funds):
    """One row per (fund, basis) for the required performance_metrics.csv."""
    rows = []
    for f in funds:
        for basis in ("gross", "net"):
            r = f[basis].dropna()
            m = pf.performance_metrics(r, periods_per_year=PERIODS_PER_YEAR[f["family"]])
            rows.append({
                "fund": f["name"],
                "fund_type": f["kind"],
                "basis": basis,
                "n_days": len(r),
                "ann_return": m["ann_return"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
                "mean_turnover": float(f["turnover"].mean()),
                "total_cost": float(f["cost"].sum()),
            })
    return pd.DataFrame(rows)


def build_fund_returns(funds):
    frames = []
    for f in funds:
        idx = f["gross"].index
        frames.append(pd.DataFrame({
            "fund": f["name"],
            "date": idx,
            "gross_return": f["gross"].values,
            "net_return": f["net"].values,
        }))
    return pd.concat(frames, ignore_index=True)


def build_fund_weights(funds):
    frames = []
    for f in funds:
        w = f["weights"].reset_index(names="date").melt(
            id_vars="date", var_name="ticker", value_name="weight")
        w.insert(0, "fund", f["name"])
        frames.append(w)
    return pd.concat(frames, ignore_index=True)


def build_fusion_before_after(funds):
    """Baseline vs fused metrics for the four fusion funds."""
    rows = []
    for f in funds:
        if f["kind"] != "fusion":
            continue
        base = next(x for x in funds if x["kind"] == "baseline"
                    and x["family"] == f["family"] and x["method"] == f["method"])
        ppy = PERIODS_PER_YEAR[f["family"]]
        for basis in ("gross", "net"):
            bm = pf.performance_metrics(base[basis].dropna(), ppy)
            fm = pf.performance_metrics(f[basis].dropna(), ppy)
            rows.append({
                "fund": f["name"], "family": f["family"],
                "model": MODEL_LABELS[f["model"]], "basis": basis,
                "baseline_ann_return": bm["ann_return"],
                "fused_ann_return": fm["ann_return"],
                "delta_ann_return": fm["ann_return"] - bm["ann_return"],
                "baseline_sharpe": bm["sharpe"], "fused_sharpe": fm["sharpe"],
                "delta_sharpe": fm["sharpe"] - bm["sharpe"],
                "baseline_max_drawdown": bm["max_drawdown"],
                "fused_max_drawdown": fm["max_drawdown"],
                "baseline_turnover": float(base["turnover"].mean()),
                "fused_turnover": float(f["turnover"].mean()),
            })
    return pd.DataFrame(rows)


def build_lambda_sensitivity(universes, ticker_day):
    """Net metrics for both lexicons at several tilt strengths (lambda).

    lambda = 0 reproduces the unfused baseline and is included as an
    internal consistency check. The lambda=0.5 row matches the headline fused
    funds; the sensitivity table is reported separately from the headline
    results because lambda was set as a fixed assumption, not fitted.
    """
    rows = []
    for fam in ("equity", "combined"):
        returns = universes[fam]
        for col in MODELS:
            for lam in SENSITIVITY_LAMBDAS:
                out = run_one(returns, "min_variance", ticker_day=ticker_day,
                              score_col=col, tilt_strength=lam)
                ppy = PERIODS_PER_YEAR[fam]
                for basis in ("gross", "net"):
                    m = pf.performance_metrics(out[basis].dropna(), ppy)
                    rows.append({
                        "fund": fund_name(fam, "min_variance", col),
                        "family": fam, "model": MODEL_LABELS[col],
                        "lambda": lam, "basis": basis,
                        "ann_return": m["ann_return"], "ann_vol": m["ann_vol"],
                        "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                        "mean_turnover": float(out["turnover"].mean()),
                        "total_cost": float(out["cost"].sum()),
                    })
    return pd.DataFrame(rows)


def _drawdown(series):
    wealth = (1.0 + series).cumprod()
    return wealth / wealth.cummax() - 1.0


def _ft_legend(ax, ncol=1, loc="upper left"):
    leg = ax.legend(loc=loc, frameon=False, fontsize=8, ncol=ncol)
    for t in leg.get_texts():
        t.set_color(ft.FT_TEXT)
    return leg


def _method_series(funds, fam, series_fn):
    """{method label -> transformed series} for the 5 funds in one family."""
    out = {}
    for m in [*BASE_METHODS, CVAR_METHOD]:
        f = next(x for x in funds if x["family"] == fam and x["method"] == m
                 and x["kind"] in ("baseline", "extension"))
        out[METHOD_LABELS[m]] = series_fn(f)
    return out


def fig_growth(funds, fig_dir):
    """One FT-style growth-of-$1 PNG per family (Equity / Crypto / Combined).

    FT house style from Project A (src.plotting, copied from
    fins2026/z5498648_projectA): cream background, no axis box, faint
    horizontal gridlines, bold left-aligned title, and each line's last value
    labelled in its own colour instead of a legend. The CVaR extension fund is
    drawn dashed so it reads as an add-on, not one of the core methods.
    """
    for fam in FAMILIES:
        series = _method_series(funds, fam,
                                lambda f: (1.0 + f["gross"]).cumprod())
        fig, ax = ft.ft_figure(figsize=(8.6, 4.6))
        ft.ft_multi_line_end_labels(ax, series, max_end_labels=5)
        ax.lines[-1].set_linestyle("--")  # CVaR extension reads as an add-on
        ft.ft_title(ax, f"{FAMILY_LABELS[fam]} funds: growth of $1, 2023 "
                        "(gross of costs)")
        ax.set_ylabel("Growth of $1 invested", color=ft.FT_TEXT, fontsize=9)
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.22)
        fig.tight_layout()
        fig.savefig(fig_dir / f"growth_of_dollar_{fam}.png", dpi=150,
                    facecolor=fig.get_facecolor())
        plt.close(fig)


def fig_drawdown(funds, fig_dir):
    """One FT-style drawdown PNG per family.

    All five lines end near 0 (funds recover by end-2023), so end-labelling
    would overlap; a borderless FT legend sits BELOW the chart instead of
    over the lines.
    """
    for fam in FAMILIES:
        series = _method_series(funds, fam, lambda f: _drawdown(f["net"]))
        fig, ax = ft.ft_figure(figsize=(8.6, 4.8))
        ft.ft_multi_line_end_labels(ax, series, max_end_labels=4)  # legend fallback
        ax.lines[-1].set_linestyle("--")
        ax.axhline(0, color=ft.FT_TEXT, lw=0.8)
        ax.legend(handles=ax.lines[:5], labels=list(series), ncol=5,
                  loc="upper center", bbox_to_anchor=(0.5, -0.14),
                  frameon=False, fontsize=8)
        for t in ax.get_legend().get_texts():
            t.set_color(ft.FT_TEXT)
        ft.ft_title(ax, f"{FAMILY_LABELS[fam]} funds: drawdown, 2023 "
                        "(net of costs)")
        ax.set_ylabel("Drawdown", color=ft.FT_TEXT, fontsize=9)
        fig.savefig(fig_dir / f"drawdown_{fam}.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def fig_weights(funds, fig_dir):
    """One FT-style stacked-weights PNG per combined method."""
    top_n = 6
    colors = [ft.FT_PALETTE[i % len(ft.FT_PALETTE)] for i in range(top_n)]
    for m in ("min_variance", "max_sharpe"):
        f = next(x for x in funds if x["family"] == "combined"
                 and x["method"] == m and x["kind"] == "baseline")
        w = f["weights"]
        top = w.mean().sort_values(ascending=False).head(top_n).index
        stacked = w[top].copy()
        stacked["Other"] = w.drop(columns=top).sum(axis=1)
        fig, ax = ft.ft_figure(figsize=(9.6, 4.8))
        stacked.plot.area(ax=ax, stacked=True, color=[*colors, "#C9C2B8"],
                          alpha=0.85, linewidth=0)
        ax.set_ylim(0, 1)
        _ft_legend(ax, ncol=1, loc="center left").set_bbox_to_anchor((1.01, 0.5))
        ft.ft_title(ax, f"Combined {METHOD_LABELS[m]}: portfolio weights "
                        "over time, 2023")
        ax.set_ylabel("Weight", color=ft.FT_TEXT, fontsize=9)
        fig.savefig(fig_dir / f"weights_over_time_combined_{m}.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def fig_sharpe(metrics, fig_dir):
    """One FT-style grouped-bar PNG: OOS gross Sharpe by method within family.

    Single chart (replaces the three sharpe_by_fund_{family}.png exhibits):
    Equity / Crypto / Combined on the x-axis, five adjacent bars per family
    (the four required methods plus the Minimum-CVaR extension), gross returns
    only - no VADER/fused funds, no net-vs-gross split, no costs. Bars are
    coloured consistently by method across all three families, with a
    borderless FT legend below the chart and a subtitle noting that higher
    Sharpe is better. 2023 out-of-sample, FT styling as elsewhere.
    """
    methods = [*BASE_METHODS, CVAR_METHOD]
    n_methods = len(methods)
    method_labels = [METHOD_LABELS[m] for m in methods]
    fam_labels = [FAMILY_LABELS[f] for f in FAMILIES]

    g = metrics[metrics["basis"] == "gross"].set_index("fund")
    by_family = {}
    for fam in FAMILIES:
        names = [fund_name(fam, m) for m in methods]
        by_family[fam] = [float(g.loc[name, "sharpe"]) for name in names]

    fig, ax = ft.ft_figure(figsize=(9.8, 5.6))
    width = 0.15
    x = np.arange(len(FAMILIES))
    for i, m in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, [by_family[fam][i] for fam in FAMILIES],
               width=width, color=COLORS[method_labels[i]],
               label=method_labels[i])

    ax.axhline(0, color=ft.FT_TEXT, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fam_labels, fontsize=10)
    ax.set_xlabel("Asset family", color=ft.FT_TEXT, fontsize=9)
    ax.set_ylabel("Out-of-sample Sharpe Ratio (rf = 0)",
                  color=ft.FT_TEXT, fontsize=9)
    ft.ft_title(ax, "Out-of-sample Sharpe Ratio by Asset Family and "
                    "Optimisation Method, 2023", pad=26)
    ax.text(0, 1.0, "Higher Sharpe ratio is better", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9.5, color="#7A7264")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    ncol=n_methods, frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(ft.FT_TEXT)
    fig.savefig(fig_dir / "sharpe_by_fund.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_sharpe_cost_combined(metrics, fig_dir):
    """One FT-style grouped-bar PNG: Combined funds' gross vs net Sharpe.

    X-axis = the five Combined optimisation methods, two adjacent bars each
    (Gross / Net) so the transaction-cost drag on the 2023 OOS Sharpe ratio
    reads directly. Baseline/extension funds only, FT styling as elsewhere,
    borderless legend below the chart.
    """
    methods = [*BASE_METHODS, CVAR_METHOD]
    labels = [METHOD_LABELS[m] for m in methods]
    names = [fund_name("combined", m) for m in methods]
    g = metrics[metrics["basis"] == "gross"].set_index("fund")
    n = metrics[metrics["basis"] == "net"].set_index("fund")
    gross_vals = [float(g.loc[nm, "sharpe"]) for nm in names]
    net_vals = [float(n.loc[nm, "sharpe"]) for nm in names]

    hi = max([*gross_vals, *net_vals, 0.0])
    pad = max(hi, 0.1) * 0.12

    fig, ax = ft.ft_figure(figsize=(9.6, 5.0))
    width = 0.34
    x = np.arange(len(labels))
    ax.bar(x - width / 2, gross_vals, width=width,
           color=ft.FT_PALETTE[1], label="Gross")
    ax.bar(x + width / 2, net_vals, width=width,
           color=ft.FT_PALETTE[3], label="Net")
    ax.axhline(0, color=ft.FT_TEXT, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(-pad, hi + pad)
    ax.set_ylabel("Out-of-sample Sharpe Ratio (rf = 0)",
                  color=ft.FT_TEXT, fontsize=9)
    ft.ft_title(ax, "Impact of Transaction Costs on Out-of-Sample Sharpe "
                    "Ratio (Combined Funds)")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                    ncol=2, frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(ft.FT_TEXT)
    fig.tight_layout()
    fig.savefig(fig_dir / "sharpe_cost_combined.png", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _fusion_funds(funds, family="equity"):
    """(baseline, {model: fused fund}) for one family's Minimum-Variance fusion."""
    base = next(x for x in funds if x["kind"] == "baseline"
                and x["family"] == family and x["method"] == "min_variance")
    fused = {col: next(x for x in funds if x["kind"] == "fusion"
                       and x["family"] == family and x["model"] == col)
             for col in MODELS}
    return base, fused


def fig_fusion(funds, fig_dir):
    """Equity Minimum Variance fusion exhibits (no Combined, no other methods).

    Two FT-style PNGs for the Equity Minimum Variance fund and its two
    sentiment-fused variants (+ VADER, + finVADER):
      1) fusion_before_after_equity.png - growth of $1, gross of costs
      2) fusion_sharpe_equity.png       - gross vs net Sharpe grouped bars
    The growth exhibit is gross-only to isolate the sentiment-tilt effect; the
    Sharpe chart pairs gross and net side by side per portfolio so the
    transaction-cost impact reads directly. Backtest/fusion logic is untouched
    - only the exhibits are regenerated.
    """
    base, fused = _fusion_funds(funds)
    labels = ["Baseline", *[f"+ {MODEL_LABELS[c]}" for c in MODELS]]
    gross = {"Baseline": base["gross"]}
    net = {"Baseline": base["net"]}
    for col in MODELS:
        gross[f"+ {MODEL_LABELS[col]}"] = fused[col]["gross"]
        net[f"+ {MODEL_LABELS[col]}"] = fused[col]["net"]

    # 1) Growth of $1, gross of costs (isolates the fusion effect).
    growth = {k: (1.0 + s).cumprod() for k, s in gross.items()}
    fig, ax = ft.ft_figure(figsize=(8.6, 4.6))
    ft.ft_multi_line_end_labels(ax, growth, max_end_labels=4)
    ft.ft_title(ax, "Equity Minimum Variance: sentiment fusion, 2023 "
                    "(gross of costs)")
    ax.set_ylabel("Growth of $1 invested", color=ft.FT_TEXT, fontsize=9)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.22)
    fig.tight_layout()
    fig.savefig(fig_dir / "fusion_before_after_equity.png", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2) Gross vs net Sharpe grouped bars, side by side for direct comparison.
    def _sharpe(series):
        return pf.performance_metrics(series.dropna(),
                                      periods_per_year=PERIODS_PER_YEAR["equity"])["sharpe"]

    gross_vals = [_sharpe(gross[k]) for k in labels]
    net_vals = [_sharpe(net[k]) for k in labels]
    all_vals = [*gross_vals, *net_vals, 0.0]
    lo, hi = min(all_vals), max(all_vals)
    pad = max(hi - lo, 0.1) * 0.12

    fig, ax = ft.ft_figure(figsize=(8.6, 4.8))
    width = 0.34
    x = np.arange(len(labels))
    ax.bar(x - width / 2, gross_vals, width=width,
           color=ft.FT_PALETTE[1], label="Gross")
    ax.bar(x + width / 2, net_vals, width=width,
           color=ft.FT_PALETTE[3], label="Net")
    ax.axhline(0, color=ft.FT_TEXT, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("Sharpe ratio (rf = 0)", color=ft.FT_TEXT, fontsize=9)
    ft.ft_title(ax, "Equity Minimum Variance: fusion Sharpe ratio, 2023 "
                    "(gross vs net)")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                    ncol=2, frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(ft.FT_TEXT)
    fig.tight_layout()
    fig.savefig(fig_dir / "fusion_sharpe_equity.png", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_fusion_weight_tilt(funds, fig_dir):
    """Horizontal grouped-bar PNG: mean weight change vs the baseline MinVar.

    Shows the sentiment-tilt mechanism (not performance): for the Equity
    Minimum Variance baseline and its two fused variants (+ VADER, + finVADER)
    the daily weights (held flat between monthly rebalances) are sampled at
    each out-of-sample rebalance date, per-stock delta w = w_fused - w_base is
    computed at each rebalance, and the deltas are averaged across all 2023
    rebalances. The 10 stocks with the largest |mean delta| (max across the
    two lexicons) are plotted as horizontal grouped bars straddling zero:
    positive = overweight vs the baseline, negative = underweight.
    """
    base, fused = _fusion_funds(funds)
    oos_dates = base["weights"].index
    rebal = list(pf.get_rebalance_dates(oos_dates))
    if not rebal or rebal[0] != oos_dates[0]:
        rebal = [oos_dates[0], *rebal]
    rebal = pd.DatetimeIndex(rebal)
    w_base = base["weights"].loc[rebal]

    deltas = {}
    for col in MODELS:
        w_fused = fused[col]["weights"].loc[rebal]
        deltas[MODEL_LABELS[col]] = (w_fused - w_base).mean(axis=0)

    sel = pd.DataFrame(deltas)
    sel["mag"] = sel.abs().max(axis=1)
    top = sel.sort_values("mag", ascending=False).head(10)

    vals = np.concatenate([top["VADER"].values, top["finVADER"].values])
    lo, hi = min(vals.min(), 0.0), max(vals.max(), 0.0)
    pad = (hi - lo) * 0.15 if hi > lo else 1.0
    label_off = (hi - lo) * 0.02

    fig, ax = ft.ft_figure(figsize=(8.6, 5.6))
    y = np.arange(len(top))
    bh = 0.36
    ax.barh(y - bh / 2, 100 * top["VADER"].values, height=bh,
            color=ft.FT_PALETTE[1], label="VADER")
    ax.barh(y + bh / 2, 100 * top["finVADER"].values, height=bh,
            color=ft.FT_PALETTE[2], label="finVADER")
    ax.axvline(0, color=ft.FT_TEXT, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(100 * lo - 100 * pad, 100 * hi + 100 * pad)
    ax.set_xlabel("Average Weight Change (%)", color=ft.FT_TEXT, fontsize=9)
    ax.set_title("Average Weight Changes after Sentiment Fusion (Relative "
                 "to\nthe Baseline Minimum Variance Portfolio)",
                 loc="left", fontsize=11, fontweight="bold", color=ft.FT_TEXT,
                 pad=12)

    for i, tkr in enumerate(top.index):
        for model, yoff in (("VADER", -bh / 2), ("finVADER", bh / 2)):
            v = 100 * top[model].loc[tkr]
            ax.text(v + (label_off if v >= 0 else -label_off) * 100,
                    i + yoff, f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8,
                    color=ft.FT_TEXT)

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07),
                    ncol=2, frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(ft.FT_TEXT)
    fig.tight_layout()
    fig.savefig(fig_dir / "fusion_weight_tilt_equity.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_turnover_cost(metrics, funds, fig_dir):
    """One FT-style turnover PNG per family (bars coloured by method)."""
    for fam in FAMILIES:
        fam_funds = [f for f in funds if f["family"] == fam]
        names = [f["name"] for f in fam_funds]
        n = metrics[metrics["basis"] == "net"].set_index("fund").loc[names]
        colors = [COLORS[METHOD_LABELS[f["method"]]] for f in fam_funds]
        fig, ax = ft.ft_figure(figsize=(8.8, max(3.4, 0.42 * len(names))))
        ft.ft_bar(ax, names, n["mean_turnover"], color=colors)
        ax.set_xlabel("Mean one-way turnover per rebalance",
                      color=ft.FT_TEXT, fontsize=9)
        ft.ft_title(ax, f"{FAMILY_LABELS[fam]} funds: mean turnover, 2023")
        fig.tight_layout()
        fig.savefig(fig_dir / f"turnover_cost_{fam}.png", dpi=150,
                    facecolor=fig.get_facecolor())
        plt.close(fig)


def verify_report(funds, metrics):
    kinds = {k: sum(1 for f in funds if f["kind"] == k)
             for k in ("baseline", "extension", "fusion")}
    print(f"fund count: {len(funds)} = {kinds['baseline']} baseline + "
          f"{kinds['extension']} CVaR extension + {kinds['fusion']} fusion")
    for f in funds:
        if len(f["gross"]) != len(f["net"]):
            raise AssertionError(f"{f['name']}: gross/net length mismatch")
        if (f["weights"].sum(axis=1) - 1.0).abs().max() > 1e-6:
            raise AssertionError(f"{f['name']}: weights don't sum to 1")
        if not (f["gross"] != f["net"]).any():
            raise AssertionError(f"{f['name']}: gross == net everywhere "
                                 "(transaction costs not applied)")
    combined = [f for f in funds if f["family"] == "combined"
                and f["kind"] == "baseline"]
    dists = sorted(
        (float((a["weights"].mean() - b["weights"].mean()).abs().sum()),
         a["name"].split(" ", 1)[1], b["name"].split(" ", 1)[1])
        for a, b in combinations(combined, 2))
    print(f"closest mean-weight L1 distance between combined methods: "
          f"{dists[0][0]:.4f} ({dists[0][1]} vs {dists[0][2]})")

    fused = next(f for f in funds if f["kind"] == "fusion")
    base = next(x for x in funds if x["kind"] == "baseline"
                and x["family"] == fused["family"] and x["method"] == "min_variance")
    tilt_active = (fused["weights"].iloc[0] != base["weights"].iloc[0]).any()
    print(f"first OOS/rebalance date: {fused['weights'].index.min().date()} "
          f"(tilt active from day one: {tilt_active})")

    net = metrics[metrics["basis"] == "net"].sort_values("sharpe", ascending=False)
    print("top 5 net Sharpe:\n" + net[["fund", "sharpe"]].head(5).to_string(index=False))
    print("bottom 2 net Sharpe:\n" + net[["fund", "sharpe"]].tail(2).to_string(index=False))


def main():
    matplotlib.use("Agg")

    data_dir = ROOT / "results" / "data"
    table_dir = ROOT / "results" / "tables"
    fig_dir = ROOT / "results" / "figures"
    for d in (data_dir, table_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=== Part B: sentiment pipeline ===")
    raw = etl.load_raw()
    equity_clean, _ = etl.clean_equity(raw["equity"])
    trading_days = pd.DatetimeIndex(equity_clean["date"].unique()).sort_values()
    sector_map = equity_clean[["ticker", "sector"]].drop_duplicates()
    panel = etl.build_text_panel(etl.clean_news(raw["news"])[0], trading_days)
    print(f"scoring {panel.shape[0]} headlines (VADER + FinVADER)...")
    scored = sent.score_headlines(panel)
    ticker_day = sent.ticker_day_sentiment(scored, trading_days, fill_missing="neutral")
    sector_index = sent.sector_sentiment_index(ticker_day, sector_map, lag=1)
    coverage = sent.ticker_day_coverage(panel, trading_days)
    coverage_index = sent.coverage_weighted_sector_sentiment_index(
        ticker_day, coverage, sector_map, lag=1)
    _sentpipe.save_outputs(sector_index, coverage_index, ticker_day, scored, sector_map)

    print("=== Part B: funds ===")
    universes = {fam: pf.build_universe_returns(fam) for fam in FAMILIES}
    print(f"equity OOS days: {(universes['equity'].index > TRAIN_END).sum()}, "
          f"crypto OOS days: {(universes['crypto'].index > TRAIN_END).sum()}, "
          f"combined OOS days: {(universes['combined'].index > TRAIN_END).sum()}")
    print("running backtests (15 funds) + fusion (4 funds)...")
    funds = build_funds(universes, ticker_day)

    metrics = build_metrics(funds)
    fund_returns = build_fund_returns(funds)
    fund_weights = build_fund_weights(funds)
    fusion_ba = build_fusion_before_after(funds)
    sensitivity = build_lambda_sensitivity(universes, ticker_day)

    fund_returns.to_csv(data_dir / "fund_returns.csv", index=False)
    fund_weights.to_csv(data_dir / "fund_weights.csv", index=False)
    metrics.to_csv(table_dir / "performance_metrics.csv", index=False)
    fusion_ba.to_csv(table_dir / "fusion_before_after.csv", index=False)
    sensitivity.to_csv(table_dir / "fusion_lambda_sensitivity.csv", index=False)

    print("=== Part B: figures ===")
    fig_growth(funds, fig_dir)
    fig_drawdown(funds, fig_dir)
    fig_weights(funds, fig_dir)
    fig_sharpe(metrics, fig_dir)
    fig_sharpe_cost_combined(metrics, fig_dir)
    fig_fusion(funds, fig_dir)
    fig_fusion_weight_tilt(funds, fig_dir)
    fig_turnover_cost(metrics, funds, fig_dir)

    for p in (data_dir / "fund_returns.csv", data_dir / "fund_weights.csv",
              data_dir / "sector_sentiment_index.csv",
              table_dir / "performance_metrics.csv"):
        print(f"saved {p}")

    verify_report(funds, metrics)
    print("done.")


if __name__ == "__main__":
    main()
