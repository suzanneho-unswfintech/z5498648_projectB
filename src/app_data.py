"""EverVest data layer: cached readers over precomputed results/ artifacts.

The app never recomputes backtests or sentiment. Everything comes from the
results/ CSV and PNG artifacts, cached in memory, and every loader is written
defensively so the app still works if a column is renamed or removed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

_METRIC_COLS = [
    "fund",
    "fund_type",
    "basis",
    "n_days",
    "ann_return",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "total_cost",
]

METHODS = ["Equal Weight", "Minimum Variance", "Maximum Sharpe", "Risk Parity", "Minimum CVaR"]
FAMILIES = ["Equity", "Crypto", "Combined"]

FUND_KIND_LABEL = {
    "baseline": "Core",
    "extension": "CVaR extension",
    "fusion": "Sentiment-fused",
}

# Investor-friendly naming (display only - fund IDs, columns and files unchanged).
UNIVERSE_LABEL = {"Equity": "Equity", "Crypto": "Crypto", "Combined": "Diversified"}
UNIVERSE_SUPPORT = {"Combined": "Equity + Crypto"}
FAMILY_ORDER = ["Equity", "Crypto", "Diversified"]

METHOD_FRIENDLY = {
    "Equal Weight": "Diversified Core",
    "Minimum Variance": "Capital Preservation",
    "Maximum Sharpe": "Risk-Adjusted Growth",
    "Risk Parity": "Balanced Risk",
    "Minimum CVaR": "Downside Protection",
}

STRATEGY_DESCRIPTION = {
    "Equal Weight": (
        "Owns every asset in the universe at an equal share and rebalances "
        "monthly - a simple, low-cost, broadly diversified starting point."
    ),
    "Minimum Variance": (
        "Builds the portfolio with the lowest predicted volatility, "
        "prioritising steadiness and smaller drawdowns over chasing returns."
    ),
    "Maximum Sharpe": (
        "Targets the best return per unit of risk on the efficient frontier - "
        "the strongest risk-adjusted performance the data supports."
    ),
    "Risk Parity": (
        "Splits the portfolio so every asset contributes an equal amount of "
        "risk, so no single holding or sector dominates the ride."
    ),
    "Minimum CVaR": (
        "Minimises the average of the worst 5% of monthly losses - built to "
        "cushion deep drawdowns, not just volatility."
    ),
}


@st.cache_data(show_spinner=False)
def load_returns() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "data" / "fund_returns.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_weights() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "data" / "fund_weights.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "tables" / "performance_metrics.csv")
    cols = [c for c in _METRIC_COLS if c in df.columns]
    return df[cols].copy()


@st.cache_data(show_spinner=False)
def load_sector_sentiment() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "data" / "sector_sentiment_index.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_fusion_before_after() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "tables" / "fusion_before_after.csv")


@st.cache_data(show_spinner=False)
def load_fusion_lambda() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "tables" / "fusion_lambda_sensitivity.csv")
    df = df[df["basis"] == "net"].copy()
    return df


@st.cache_data(show_spinner=False)
def load_sentiment_comparison() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "tables" / "sentiment_model_comparison.csv")


@st.cache_data(show_spinner=False)
def load_sector_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "tables" / "sector_sentiment_summary.csv")


def all_funds() -> list[str]:
    """All fund names, in a stable family->method order."""
    df = load_returns()
    funds = list(df["fund"].unique())
    return sorted(funds, key=lambda f: _fund_key(f))


def _fund_key(name: str) -> tuple:
    p = parse_fund(name)
    fam = FAMILIES.index(p["family"]) if p["family"] in FAMILIES else 9
    method = p["method"] if p["method"] else ""
    mi = METHODS.index(method) if method in METHODS else 9
    return (fam, mi, p["model"] or "")


def parse_fund(name: str) -> dict:
    """Split a fund name into family, method and optional sentiment model."""
    family = next((f for f in FAMILIES if name.startswith(f)), None)
    rest = name[len(family) :].strip() if family else name
    model = None
    if " + " in rest:
        base, model = rest.split(" + ", 1)
    else:
        base = rest
    method = next((m for m in METHODS if base.startswith(m)), None)
    if method is None:
        base = base.strip()
        method = next((m for m in METHODS if base == m), base)
    return {"family": family, "method": method, "model": model}


def universe_label(family: str) -> str:
    """Display name for a fund universe ('Combined' -> 'Diversified')."""
    return UNIVERSE_LABEL.get(family, family)


def universe_support(family: str) -> str | None:
    """Supporting text for a universe ('Equity + Crypto' for Diversified)."""
    return UNIVERSE_SUPPORT.get(family)


def friendly_name(fund: str) -> str:
    """Investor-friendly strategy name (e.g. 'Capital Preservation')."""
    p = parse_fund(fund)
    base = METHOD_FRIENDLY.get(p["method"], p["method"])
    if p["model"]:
        base = "News-Aware " + base
    return base


def display_lines(fund: str) -> tuple[str, str]:
    """(Friendly name, technical strategy · asset universe) for one fund."""
    p = parse_fund(fund)
    fam = universe_label(p["family"])
    head = f"{fam} {friendly_name(fund)}"
    tech = p["method"] + (f" + {p['model']}" if p["model"] else "")
    uni = universe_label(p["family"])
    if universe_support(p["family"]):
        uni = f"{uni} ({universe_support(p['family'])})"
    return head, f"{tech} · {uni}"


def dropdown_label(fund: str) -> str:
    """Compact label for dropdowns: 'Friendly Name (model) - Universe'."""
    p = parse_fund(fund)
    name = friendly_name(fund)
    if p["model"]:
        name = f"{name} ({p['model']})"
    return f"{name} - {universe_label(p['family'])}"


def strategy_description(fund: str) -> str:
    """Plain-English objective of the strategy, for catalogue cards."""
    p = parse_fund(fund)
    if p["model"]:
        return (
            "A capital-preservation portfolio whose monthly weights are "
            "gently tilted by one-day-lagged company news sentiment, so the "
            "fund responds to what is happening in the headlines."
        )
    return STRATEGY_DESCRIPTION.get(p["method"], "")


def catalogue_group(fund: str) -> str | None:
    """Which Home catalogue section a fund belongs to (None = not showcased)."""
    if fund == "Equity Minimum Variance + VADER":
        return "Research-enhanced strategies"
    if fund_kind(fund) == "fusion":
        return None
    return "Core strategies"


def catalogue() -> dict[str, list[dict]]:
    """Home catalogue grouped into Core / Research-enhanced strategies."""
    groups: dict[str, list[dict]] = {
        "Core strategies": [],
        "Research-enhanced strategies": [],
    }
    for fund in all_funds():
        grp = catalogue_group(fund)
        if grp is None:
            continue
        head, sub = display_lines(fund)
        groups[grp].append(
            {"id": fund, "name": head, "subtitle": sub, "desc": strategy_description(fund)}
        )
    return groups


def _ret_col(basis: str) -> str:
    return f"{basis}_return"


def _pivot_returns(basis: str = "net") -> pd.DataFrame:
    df = load_returns()[["date", "fund", _ret_col(basis)]]
    return df.pivot_table(index="date", columns="fund", values=_ret_col(basis))


def growth_series(fund: str, basis: str = "net") -> pd.Series:
    p = _pivot_returns(basis)
    return (1 + p[fund]).cumprod()


def drawdown_series(fund: str, basis: str = "net") -> pd.Series:
    g = growth_series(fund, basis)
    return g / g.cummax() - 1


def metrics_from_series(s: pd.Series, periods_per_year: int) -> dict:
    """Return / vol / Sharpe (rf=0) / max drawdown for one return series."""
    s = s.dropna()
    n = len(s)
    if n == 0:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    ann_return = (1 + s).prod() ** (periods_per_year / n) - 1
    ann_vol = s.std(ddof=1) * math.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    wealth = (1 + s).cumprod()
    max_drawdown = (wealth / wealth.cummax() - 1).min()
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def blend_funds(funds: list[str], weights: dict[str, float], basis: str = "net") -> pd.Series:
    """Capital-weighted blend of fund return series aligned on common dates."""
    p = _pivot_returns(basis)[list(funds)].dropna()
    w = np.array([weights[f] for f in p.columns], dtype=float)
    w = w / w.sum()
    return (p * w).sum(axis=1)


def annualising_weights(funds: list[str]) -> int:
    """365 for crypto-heavy blends, 252 otherwise (matches report rule)."""
    crypto_share = sum(1 for f in funds if parse_fund(f)["family"] == "Crypto") / len(funds)
    return 365 if crypto_share >= 0.5 else 252


def current_holdings(fund: str, top_n: int = 10) -> pd.DataFrame:
    """Latest rebalance weights for a fund, biggest holdings first."""
    w = load_weights()
    sub = w[w["fund"] == fund]
    last = sub["date"].max()
    hold = sub[sub["date"] == last].sort_values("weight", ascending=False).head(top_n)
    hold["leg"] = hold["ticker"].map(
        lambda t: (
            "Crypto"
            if str(t).startswith("CR_")
            else ("Equity" if str(t).startswith("EQ_") else "Single")
        )
    )
    hold["display"] = hold["ticker"].str.replace("^(EQ|CR)_", "", regex=True)
    return hold[["display", "leg", "weight"]].rename(
        columns={"display": "Holding", "leg": "Leg", "weight": "Weight"}
    )


def metric_row(fund: str, basis: str = "net") -> pd.Series:
    """One fund's metrics row; falls back to any row or a from-returns
    computation so a missing basis can never crash the fact sheet."""
    m = load_metrics()
    exact = m[(m["fund"] == fund) & (m["basis"] == basis)]
    if len(exact):
        return exact.iloc[0]
    any_row = m[m["fund"] == fund]
    if len(any_row):
        return any_row.iloc[0]
    s = _pivot_returns(basis)[fund].dropna()
    ppy = annualising_weights([fund])
    stats = metrics_from_series(s, ppy)
    stats["fund"] = fund
    stats["basis"] = basis
    stats["fund_type"] = "baseline"
    stats["n_days"] = len(s)
    return pd.Series(stats)


def fund_kind(fund: str) -> str:
    m = load_metrics()
    rows = m[m["fund"] == fund]["fund_type"]
    return rows.mode().iloc[0] if len(rows) else "baseline"


def fund_kind_label(fund: str) -> str:
    return FUND_KIND_LABEL.get(fund_kind(fund), "Core")


def compare_table(basis: str = "net") -> pd.DataFrame:
    """Rankable metrics table for the Compare page (friendly display names)."""
    m = load_metrics()
    m = m[m["basis"] == basis].copy()
    out = []
    for _, row in m.iterrows():
        head, sub = display_lines(row["fund"])
        p = parse_fund(row["fund"])
        out.append(
            {
                "_id": row["fund"],
                "Fund": head,
                "Strategy": sub,
                "Objective": METHOD_FRIENDLY.get(p["method"], p["method"]),
                "Universe": universe_label(p["family"]),
                "Type": FUND_KIND_LABEL.get(row["fund_type"], "Core"),
                "Ann. return": row.get("ann_return", np.nan),
                "Ann. volatility": row.get("ann_vol", np.nan),
                "Sharpe": row.get("sharpe", np.nan),
                "Max drawdown": row.get("max_drawdown", np.nan),
                "Days": int(row.get("n_days", np.nan)) if pd.notna(row.get("n_days")) else None,
            }
        )
    df = pd.DataFrame(out)
    return df.sort_values("Sharpe", ascending=False).reset_index(drop=True)
